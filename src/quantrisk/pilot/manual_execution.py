"""
MANUAL_LIVE：人工录入经券商确认的真实交易事件

本模块不连接券商、不调用任何下单接口。交易建议（suggestions.jsonl）只是参考，是否下单、以什么价格下单由本人在
券商 App 中决定；之后把券商 App / 交割单上的事件逐条录入这里，账本只按录入的真实记录重放。

记录类型（都写入同一本 MANUAL_LIVE 流水，来源固定为 BROKER_CONFIRMED_MANUAL_ENTRY；系统自己不会写任何一条）：
- CAPITAL            登记实验本金（≤ 配置本金 ≤ 100 元，只能一条）
- BROKER_ORDER       委托（券商委托编号唯一）
- BROKER_FILL        成交（券商成交编号唯一，必须引用已存在且未终态的委托；不携带入账费用）
- BROKER_CANCEL      撤单（可部分撤单）
- BROKER_REJECT      券商/交易所拒单（只能在无成交、无撤单时录入）
- BROKER_EXPIRED     收盘未成交作废（数量必须等于剩余数量；只能在当日 15:00 之后录入）
- BROKER_SETTLEMENT  交割单费用（每个委托每个交易日一条；费用唯一入账来源）
- BROKER_DIVIDEND    券商实际到账的现金分红
- VOID               作废一条错录（原记录保留在流水中，重放时跳过；同一自然键允许重录一次）

委托状态不存储，每次由事件序列推导：Q = 委托数量，F = 有效成交数量之和，C = 有效撤单数量之和，E = 收盘作废数量，
R = Q − F − C − E（录入 EXPIRED 时要求 E 等于录入前的 R，所以作废后 R = 0）。
费用只有一个入账来源（BROKER_SETTLEMENT），按成交数量比例分摊到各笔成交，按分四舍五入、尾差归最后一笔；
交割单未录入前该委托费用按 0 计并标 FEES_PENDING，不用情景估算代替。

校验分两级：ERROR 直接拒绝写入；FLAG 如实写入并固化在记录的 flags 字段（真实发生的交易必须进账本）。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..strategies.backtest.ledger import Account
from .etf_rules import explicit_fee_total, load_rules
from .experiment_config import ExperimentConfig
from .ledgers import MANUAL, Journal
from .market_data import CST

SOURCE = "BROKER_CONFIRMED_MANUAL_ENTRY"
CHILD_TYPES = ("BROKER_FILL", "BROKER_CANCEL", "BROKER_REJECT", "BROKER_EXPIRED", "BROKER_SETTLEMENT")
FEE_COMPONENTS = ("commission", "transfer_fee", "exchange_fee", "other")
EVIDENCE_KINDS = ("APP_SCREENSHOT", "SETTLEMENT_STATEMENT", "BROKER_MESSAGE")
EXPIRED_ENTRY_AFTER = (15, 0)        # 收盘未成交作废只能在当日 15:00（北京时间）之后录入
LATE_ENTRY_CUTOFF = (15, 30)         # 晚于 trade_date 次日 15:30 录入 → LATE_ENTRY 标记
TERMINAL = ("FILLED", "PARTIAL_CLOSED", "CANCELLED", "EXPIRED", "REJECTED")


class ManualEntryError(ValueError):
    """录入的真实记录未通过校验（ERROR 级）。"""


def _journal(exp_dir) -> Journal:
    return Journal(exp_dir, MANUAL)


def _now(now_utc: Optional[datetime]) -> datetime:
    return now_utc or datetime.now(timezone.utc)


def _cst(trade_date: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{trade_date} {hhmm}").tz_localize(CST)


def _cents(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _require_real(cfg: ExperimentConfig):
    if not cfg.fee_scenario.actual_broker:
        raise ManualEntryError(f"实验 {cfg.experiment_id()} 的佣金情景 {cfg.fee['name']} 不是真实券商费率"
                               "（actual_broker=false），不接受任何真实账户记录；真实轨道只能挂在真实费率实验下")


def _check_evidence(evidence: dict) -> dict:
    if not isinstance(evidence, dict) or evidence.get("kind") not in EVIDENCE_KINDS or not str(evidence.get("ref", "")).strip():
        raise ManualEntryError(f"每条真实记录必须附凭据 evidence={{kind ∈ {EVIDENCE_KINDS}, ref: 截图文件名/交割单编号}}")
    return {"kind": evidence["kind"], "ref": str(evidence["ref"]).strip(), "captured_at": evidence.get("captured_at")}


# ---------- 流水视图 ----------

def active_records(recs: List[dict]) -> List[dict]:
    """排除 VOID 本身与被 VOID 作废的记录。"""
    voided = {r["target_hash"] for r in recs if r["type"] == "VOID"}
    return [r for r in recs if r["type"] != "VOID" and r["hash"] not in voided]


def order_status(q: int, f: int, c: int, rejected: bool, e: int = 0) -> Tuple[str, bool]:
    r = q - f - c - e
    if rejected:
        return "REJECTED", True
    if r == 0 and f == q:
        return "FILLED", True
    if r == 0 and 0 < f < q:
        return "PARTIAL_CLOSED", True
    if r == 0 and f == 0:
        return ("EXPIRED" if e else "CANCELLED"), True
    if f == 0 and c == 0:
        return "SUBMITTED", False
    return "OPEN", False


def apportion_fees(fees: Dict[str, float], fills: List[dict]) -> List[Dict[str, float]]:
    """交割单费用按成交数量比例分摊到各笔成交：各笔按分四舍五入，尾差归最后一笔，总额不变。"""
    if not fills:
        return []
    total_qty = sum(x["qty"] for x in fills)
    out = [{} for _ in fills]
    for k, v in fees.items():
        v = _cents(v)
        parts = [_cents(v * x["qty"] / total_qty) for x in fills[:-1]]
        parts.append(round(v - sum(parts), 2))
        for i, p in enumerate(parts):
            out[i][k] = p
    return out


def order_book(recs: List[dict]) -> Dict[str, dict]:
    """由有效记录推导每个委托的状态（§8 状态机）。"""
    act = active_records(recs)
    book: Dict[str, dict] = {}
    for r in act:
        if r["type"] == "BROKER_ORDER":
            book[r["broker_order_id"]] = {"order": r, "fills": [], "cancels": [], "reject": None, "expired": None,
                                          "settlement": None}
    for r in act:
        if r["type"] not in CHILD_TYPES:
            continue
        o = book.get(r["broker_order_id"])
        if o is None:
            continue                                     # 引用缺失：录入时拒绝，独立复算会报告
        if r["type"] == "BROKER_FILL":
            o["fills"].append(r)
        elif r["type"] == "BROKER_CANCEL":
            o["cancels"].append(r)
        elif r["type"] == "BROKER_REJECT":
            o["reject"] = r
        elif r["type"] == "BROKER_EXPIRED":
            o["expired"] = r
        else:
            o["settlement"] = r
    for oid, o in book.items():
        o["fills"].sort(key=lambda x: (x["fill_time"], x["seq"]))
        q = o["order"]["qty"]
        f = sum(x["qty"] for x in o["fills"])
        c = sum(x["cancelled_qty"] for x in o["cancels"])
        e = o["expired"]["expired_qty"] if o["expired"] else 0
        status, terminal = order_status(q, f, c, o["reject"] is not None, e)
        o.update(Q=q, F=f, C=c, E=e, R=q - f - c - e, status=status, terminal=terminal,
                 vwap=(sum(x["qty"] * x["price"] for x in o["fills"]) / f) if f else None,
                 fees_pending=bool(o["fills"]) and o["settlement"] is None)
        o["fill_fees"] = apportion_fees(o["settlement"]["fees"], o["fills"]) if o["settlement"] else [{} for _ in o["fills"]]
    return book


@dataclass
class ManualState:
    capital: Optional[dict]
    recs: List[dict]
    book: Dict[str, dict]
    acct: Optional[Account]
    ledger_events: List[dict] = field(default_factory=list)     # 按时间排序的入账事件（成交 + 分红）

    @property
    def open_orders(self) -> List[dict]:
        return [o for o in self.book.values() if not o["terminal"]]


def _ledger_events(cap: dict, book: Dict[str, dict], act: List[dict]) -> List[dict]:
    ev = []
    for oid, o in book.items():
        for x, fees in zip(o["fills"], o["fill_fees"]):
            ev.append({"kind": "FILL", "trade_date": x["trade_date"], "time": x["fill_time"], "seq": x["seq"],
                       "side": x["side"], "qty": x["qty"], "price": x["price"], "fees": fees,
                       "fee_total": explicit_fee_total(fees) if fees else 0.0, "broker_trade_id": x["broker_trade_id"],
                       "broker_order_id": oid, "fees_pending": o["fees_pending"]})
    for r in act:
        if r["type"] == "BROKER_DIVIDEND":
            ev.append({"kind": "DIVIDEND", "trade_date": r["trade_date"], "time": "23:59:59", "seq": r["seq"],
                       "amount": r["amount"], "broker_ref": r["broker_ref"]})
    return sorted(ev, key=lambda e: (e["trade_date"], e["time"], e["seq"]))


def manual_state(exp_dir, cfg: ExperimentConfig, calendar: pd.DatetimeIndex, through=None) -> ManualState:
    """从 MANUAL_LIVE 流水重放真实账本；没有登记本金时 acct 为 None。through：只重放到该交易日（含）。"""
    recs = _journal(exp_dir).records()
    act = active_records(recs)
    caps = [r for r in act if r["type"] == "CAPITAL"]
    book = order_book(recs)
    if not caps:
        return ManualState(None, recs, book, None)
    cap = caps[0]
    rules = load_rules(cfg.etf)
    pos = {d: i for i, d in enumerate(calendar)}
    acct = Account(cap["amount"], settlement_lag=rules.settlement_lag)
    events = _ledger_events(cap, book, act)
    if through is not None:
        events = [e for e in events if pd.Timestamp(e["trade_date"]) <= pd.Timestamp(through)]
    for e in events:
        if e["kind"] == "FILL":
            if e["side"] == "BUY":
                acct.buy(e["trade_date"], cfg.etf, e["qty"], e["price"], e["fees"], e["broker_trade_id"])
            else:
                acct.sell(e["trade_date"], cfg.etf, e["qty"], e["price"], e["fees"], e["broker_trade_id"],
                          lambda a, b: pos[pd.Timestamp(b)] - pos[pd.Timestamp(a)])
        else:
            acct.cash += e["amount"]
            acct._log(e["trade_date"], "DIVIDEND", symbol=cfg.etf, amount=e["amount"], note="券商入账（人工录入）")
    return ManualState(cap, recs, book, acct, events)


def replay_manual(exp_dir, cfg: ExperimentConfig, calendar: pd.DatetimeIndex):
    """兼容旧接口：返回 (Account | None, 全部记录)。"""
    st = manual_state(exp_dir, cfg, calendar)
    return st.acct, st.recs


def daily_positions(exp_dir, cfg: ExperimentConfig, store) -> pd.DataFrame:
    """真实账本逐交易日：现金、持仓、按当日收盘估值的权益、费用是否待录（无 K 线的日子沿用最近收盘价）。"""
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    if st.capital is None:
        return pd.DataFrame()
    bars = store.bars()
    first = pd.Timestamp(st.capital["trade_date"])
    days = cal[(cal >= first) & (cal <= bars.index.max())]
    cash, qty, close, fees, rows = st.capital["amount"], 0, None, 0.0, []
    pending_orders = {e["broker_order_id"] for e in st.ledger_events if e["kind"] == "FILL" and e["fees_pending"]}
    for d in days:
        key = str(d.date())
        for e in st.ledger_events:
            if e["trade_date"] != key:
                continue
            if e["kind"] == "DIVIDEND":
                cash += e["amount"]
            elif e["side"] == "BUY":
                cash -= e["qty"] * e["price"] + e["fee_total"]
                qty += e["qty"]
                fees += e["fee_total"]
            else:
                cash += e["qty"] * e["price"] - e["fee_total"]
                qty -= e["qty"]
                fees += e["fee_total"]
        if d in bars.index:
            close = float(bars.at[d, "close"])
        pend = any(e["kind"] == "FILL" and e["trade_date"] <= key and e["broker_order_id"] in pending_orders
                   for e in st.ledger_events)
        rows.append({"date": key, "close": close, "cash": round(cash, 6), "holding": qty,
                     "equity": round(cash + qty * (close or 0.0), 6), "capital": st.capital["amount"],
                     "cum_return": (cash + qty * (close or 0.0)) / st.capital["amount"] - 1,
                     "fees_cum": round(fees, 2), "fees_pending": pend})
    return pd.DataFrame(rows)


# ---------- 幂等与自然键 ----------

def _natural_key(rec: dict) -> Optional[tuple]:
    t = rec["type"]
    if t == "CAPITAL":
        return ("CAPITAL",)
    if t == "BROKER_ORDER":
        return (t, rec["broker_order_id"])
    if t == "BROKER_FILL":
        return (t, rec["broker_trade_id"])
    if t == "BROKER_CANCEL":
        return (t, rec["broker_cancel_id"])
    if t in ("BROKER_REJECT", "BROKER_EXPIRED"):
        return (t, rec["broker_order_id"])
    if t == "BROKER_SETTLEMENT":
        return (t, rec["broker_order_id"], rec["trade_date"])
    if t == "BROKER_DIVIDEND":
        return (t, rec["broker_ref"])
    if t == "VOID":
        return (t, rec["target_hash"])
    return None


_VOLATILE = ("seq", "prev_hash", "hash", "ledger", "source", "written_at_utc", "entered_at_utc", "flags", "warnings",
             "reentry_of")


def _duplicate(recs: List[dict], key: tuple, given: dict) -> Optional[dict]:
    """
    业务校验之前的幂等检查：同一自然键已有有效记录时，录入内容（given 各字段）相同 → 返回已有记录；不同 → 拒绝。
    必须先于业务校验执行，否则原样重录会因状态已推进（剩余数量已减少、委托已终态、交割已存在）被误拒。
    """
    alive = [r for r in active_records(recs) if _natural_key(r) == key]
    if not alive:
        return None
    same = lambda a, b: json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    if all(same(alive[0].get(k), v) for k, v in given.items()):
        return alive[0]
    raise ManualEntryError(f"{key[0]} {key[1:] if len(key) > 1 else ''} 已存在且内容不同：不允许覆盖，如录错请先 VOID 再重录")


def _same_content(a: dict, b: dict) -> bool:
    strip = lambda r: {k: v for k, v in r.items() if k not in _VOLATILE}
    return json.dumps(strip(a), sort_keys=True, default=str) == json.dumps(strip(b), sort_keys=True, default=str)


def _append(exp_dir, payload: dict, recs: List[dict], now_utc: datetime) -> dict:
    """
    幂等写入：同一自然键已有有效记录且内容相同 → 返回已有记录，不新增；内容不同 → 拒绝（修正只能 VOID + 重录）。
    自然键曾被 VOID 作废 → 允许重录一次（带 reentry_of），第二次拒绝。
    """
    key = _natural_key(payload)
    act_hashes = {r["hash"] for r in active_records(recs)}
    same_key = [r for r in recs if r["type"] == payload["type"] and _natural_key(r) == key]
    alive = [r for r in same_key if r["hash"] in act_hashes]
    if alive:
        if _same_content(alive[0], payload):
            return alive[0]
        raise ManualEntryError(f"{payload['type']} {key[1:] if len(key) > 1 else ''} 已存在且内容不同：不允许覆盖，"
                               "如录错请先 VOID 再重录")
    voided = [r for r in same_key if r["hash"] not in act_hashes]
    if voided:
        if len(voided) >= 2:
            raise ManualEntryError(f"{payload['type']} {key[1:]} 已被作废并重录过一次，不允许再次重录")
        payload = {**payload, "reentry_of": voided[-1]["hash"]}
    late = _cst(payload["trade_date"], "00:00") + pd.Timedelta(days=1, hours=LATE_ENTRY_CUTOFF[0], minutes=LATE_ENTRY_CUTOFF[1])
    flags = list(payload.get("flags", []))
    if payload["type"] != "VOID" and pd.Timestamp(now_utc).tz_convert(CST) > late and "LATE_ENTRY" not in flags:
        flags.append("LATE_ENTRY")
    payload = {**payload, "flags": flags, "entered_at_utc": now_utc.isoformat(timespec="seconds")}
    return _journal(exp_dir).append(payload, SOURCE)


# ---------- 公共校验 ----------

def _trading_day(cal: pd.DatetimeIndex, trade_date: str) -> pd.Timestamp:
    d = pd.Timestamp(trade_date)
    if d not in cal:
        raise ManualEntryError(f"{trade_date} 不是交易日")
    return d


def _order_or_fail(book: Dict[str, dict], broker_order_id: str) -> dict:
    o = book.get(broker_order_id)
    if o is None:
        raise ManualEntryError(f"委托 {broker_order_id} 不存在（或已被作废），请先录入 BROKER_ORDER")
    return o


def _not_terminal(o: dict):
    if o["terminal"]:
        raise ManualEntryError(f"委托 {o['order']['broker_order_id']} 已是终态 {o['status']}，不能再录入事件")


def _same_day(o: dict, trade_date: str):
    if o["order"]["trade_date"] != trade_date:
        raise ManualEntryError(f"事件日期 {trade_date} 与委托日期 {o['order']['trade_date']} 不同：限价委托当日有效，不支持隔夜委托")


def _limit_band(store, rules, d: pd.Timestamp):
    bars = store.bars()
    prior = bars.index[bars.index < d]
    if not len(prior):
        return None
    prev_close = float(bars.loc[prior[-1], "close"])
    divs = store.dividends()
    ex = float(divs.at[d, "cash"]) if d in divs.index else None
    ref = rules.ex_dividend_reference(prev_close, ex) if ex else prev_close
    return rules.limit_prices(ref)


def _suggestion(exp_dir, suggestion_id: Optional[str]) -> Optional[dict]:
    p = Path(exp_dir) / "suggestions.jsonl"
    if not suggestion_id or not p.exists():
        return None
    for line in p.read_text("utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("suggestion_id") == suggestion_id:
            return rec
    return None


def order_flags(exp_dir, cfg: ExperimentConfig, cal: pd.DatetimeIndex, trade_date: str, order_time: str, side: str,
                qty: int, limit_price: float, suggestion_id: Optional[str]) -> Tuple[List[str], Optional[dict]]:
    """§10 的 FLAG 级标记（建议相关）；ORDER_BEFORE_SUGGESTION 为 ERROR，在此直接抛出。"""
    flags: List[str] = []
    sug = _suggestion(exp_dir, suggestion_id)
    if sug is None:
        return ["NO_SUGGESTION"], None
    d = pd.Timestamp(trade_date)
    pos = cal.get_loc(d)
    prev = str(cal[pos - 1].date()) if pos > 0 else None
    if sug.get("decision_day") != prev:
        flags.append("SUGGESTION_STALE")
    gen = pd.Timestamp(sug["generated_at_utc"])
    gen = (gen if gen.tzinfo else gen.tz_localize("UTC")).tz_convert(CST)
    if _cst(trade_date, order_time) < gen:
        raise ManualEntryError(f"ORDER_BEFORE_SUGGESTION：委托时间 {trade_date} {order_time} 早于建议 {suggestion_id} 的生成时刻 "
                               f"{gen.isoformat()}。如果这笔单不是按建议下的，请不要填 suggestion_id（将记为 NO_SUGGESTION）")
    prop = sug.get("proposal") or {}
    if prop.get("action") == "REJECT":
        flags.append("SUGGESTION_REJECTED")
    order = prop.get("order") or {}
    if not order:
        if sug.get("intent") is None:
            flags.append("NO_SUGGESTION")
        return flags, sug
    if order.get("side") != side:
        flags.append("SIDE_MISMATCH")
    if int(order.get("qty", -1)) != int(qty):
        flags.append("QTY_MISMATCH")
    if abs(float(order.get("limit_price", -1)) - float(limit_price)) > 1e-9:
        flags.append("LIMIT_MISMATCH")
    return flags, sug


# ---------- 录入 ----------

def record_capital(exp_dir, cfg: ExperimentConfig, amount: float, date: str, evidence: dict, note: str = "",
                   now_utc: Optional[datetime] = None) -> dict:
    _require_real(cfg)
    recs = _journal(exp_dir).records()
    ev = _check_evidence(evidence)
    try:
        dup = _duplicate(recs, ("CAPITAL",), {"trade_date": date, "amount": float(amount), "evidence": ev})
    except ManualEntryError:
        raise ManualEntryError("实验本金已登记，不允许追加资金")
    if dup:
        return dup
    if not (0 < amount <= cfg.capital):
        raise ManualEntryError(f"登记本金 {amount} 必须在 (0, {cfg.capital}] 元内")
    payload = {"type": "CAPITAL", "day": date, "trade_date": date, "amount": float(amount), "note": note, "evidence": ev}
    return _append(exp_dir, payload, recs, _now(now_utc))


def record_order(exp_dir, cfg: ExperimentConfig, store, trade_date: str, order_time: str, side: str, qty: int,
                 limit_price: float, broker_order_id: str, evidence: dict, suggestion_id: Optional[str] = None,
                 order_kind: str = "LIMIT", code: Optional[str] = None, note: str = "",
                 now_utc: Optional[datetime] = None) -> dict:
    _require_real(cfg)
    side, code = side.upper(), code or cfg.etf
    rules = load_rules(cfg.etf)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    dup = _duplicate(st.recs, ("BROKER_ORDER", str(broker_order_id)),
                     {"trade_date": trade_date, "order_time": order_time, "code": code, "side": side, "qty": int(qty),
                      "limit_price": float(limit_price), "order_kind": order_kind, "suggestion_id": suggestion_id, "evidence": ev})
    if dup:
        return dup
    if st.acct is None:
        raise ManualEntryError("尚未登记实验本金（record-capital），不能录入委托")
    if code != cfg.etf:
        raise ManualEntryError(f"代码 {code} 与实验标的 {cfg.etf} 不一致")
    if side not in ("BUY", "SELL"):
        raise ManualEntryError("方向只能是 BUY 或 SELL")
    if order_kind != "LIMIT":
        raise ManualEntryError("第一阶段只接受限价委托（LIMIT）")
    d = _trading_day(cal, trade_date)
    if not rules.on_tick(limit_price):
        raise ManualEntryError(f"限价 {limit_price} 不在 {rules.tick_size} 元报价单位上")
    flags: List[str] = []
    band = _limit_band(store, rules, d)
    if band is None:
        flags.append("PRICE_UNVERIFIED")
    elif not (band[0] - 1e-9 <= limit_price <= band[1] + 1e-9):
        raise ManualEntryError(f"OUTSIDE_LIMIT_BAND：限价 {limit_price} 超出 {trade_date} 涨跌停价区间 {band}")
    holding = st.acct.quantity(cfg.etf)
    open_buy = sum(o["R"] * o["order"]["limit_price"] for o in st.open_orders if o["order"]["side"] == "BUY")
    open_sell = sum(o["R"] for o in st.open_orders if o["order"]["side"] == "SELL")
    if side == "BUY":
        if not rules.valid_buy_qty(qty):
            raise ManualEntryError(f"买入数量 {qty} 不是 {rules.lot_step} 份的整数倍")
        if open_buy + qty * limit_price > st.acct.cash + 1e-9:
            raise ManualEntryError(f"买入委托按限价需 {qty * limit_price:.2f} 元（另有未成交买单 {open_buy:.2f} 元），"
                                   f"账本现金只有 {st.acct.cash:.2f} 元")
    else:
        if not rules.valid_sell_qty(qty, holding):
            raise ManualEntryError(f"卖出数量 {qty} 不符合申报单位或超过持仓 {holding}")
        pos = {x: i for i, x in enumerate(cal)}
        sellable = st.acct.sellable(cfg.etf, pos[d], lambda x: pos[pd.Timestamp(x)])
        if open_sell + qty > sellable:
            raise ManualEntryError(f"T+1：{trade_date} 可卖 {sellable} 份（另有未成交卖单 {open_sell} 份），不能再卖 {qty} 份")
    sflags, _ = order_flags(exp_dir, cfg, cal, trade_date, order_time, side, qty, limit_price, suggestion_id)
    flags += sflags
    payload = {"type": "BROKER_ORDER", "day": trade_date, "trade_date": trade_date, "order_time": order_time,
               "broker_order_id": str(broker_order_id), "code": code, "side": side, "qty": int(qty),
               "limit_price": float(limit_price), "order_kind": order_kind, "suggestion_id": suggestion_id,
               "flags": flags, "note": note, "evidence": ev}
    return _append(exp_dir, payload, st.recs, _now(now_utc))


def record_fill(exp_dir, cfg: ExperimentConfig, store, trade_date: str, fill_time: str, broker_order_id: str,
                qty: int, price: float, broker_trade_id: str, evidence: dict, fee_hint: Optional[Dict[str, float]] = None,
                note: str = "", now_utc: Optional[datetime] = None) -> dict:
    _require_real(cfg)
    rules = load_rules(cfg.etf)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    dup = _duplicate(st.recs, ("BROKER_FILL", str(broker_trade_id)),
                     {"trade_date": trade_date, "fill_time": fill_time, "broker_order_id": str(broker_order_id),
                      "qty": int(qty), "price": float(price), "evidence": ev})
    if dup:
        return dup
    if st.acct is None:
        raise ManualEntryError("尚未登记实验本金（record-capital），不能录入成交")
    o = _order_or_fail(st.book, str(broker_order_id))
    _not_terminal(o)
    _same_day(o, trade_date)
    d = _trading_day(cal, trade_date)
    od = o["order"]
    if fill_time < od["order_time"]:
        raise ManualEntryError(f"成交时间 {fill_time} 早于委托时间 {od['order_time']}")
    if int(qty) <= 0 or int(qty) > o["R"]:
        raise ManualEntryError(f"成交数量 {qty} 超过委托 {broker_order_id} 的剩余数量 {o['R']}")
    if not rules.on_tick(price):
        raise ManualEntryError(f"成交价 {price} 不在 {rules.tick_size} 元报价单位上")
    lim = od["limit_price"]
    if (od["side"] == "BUY" and price > lim + 1e-9) or (od["side"] == "SELL" and price < lim - 1e-9):
        raise ManualEntryError(f"成交价 {price} 违反限价 {lim}（{od['side']}）")
    flags: List[str] = []
    bars = store.bars()
    if d in bars.index:
        lo, hi = float(bars.at[d, "low"]), float(bars.at[d, "high"])
        if not (lo - 1e-9 <= price <= hi + 1e-9):
            raise ManualEntryError(f"成交价 {price} 不在 {trade_date} 的最低价 {lo} 与最高价 {hi} 之间，请核对券商成交回报")
        market_check = "已核对"
    else:
        market_check, flags = "行情未核对（行情存储中还没有该日 K 线）", ["PRICE_UNVERIFIED"]
    if od["side"] == "BUY" and qty * price > st.acct.cash + 1e-9:
        raise ManualEntryError(f"买入 {qty} 份 × {price} 需 {qty * price:.2f} 元，账本现金只有 {st.acct.cash:.2f} 元")
    if fee_hint is not None:
        if any(v < 0 for v in fee_hint.values()):
            raise ManualEntryError("fee_hint 各项不能为负")
        fee_hint = {k: round(float(v), 2) for k, v in fee_hint.items()}
    payload = {"type": "BROKER_FILL", "day": trade_date, "trade_date": trade_date, "fill_time": fill_time,
               "broker_trade_id": str(broker_trade_id), "broker_order_id": str(broker_order_id), "code": cfg.etf,
               "side": od["side"], "qty": int(qty), "price": float(price), "fee_hint": fee_hint,
               "market_check": market_check, "flags": flags, "note": note, "evidence": ev}
    return _append(exp_dir, payload, st.recs, _now(now_utc))


def record_cancel(exp_dir, cfg: ExperimentConfig, store, trade_date: str, cancel_time: str, broker_order_id: str,
                  cancelled_qty: int, evidence: dict, broker_cancel_id: Optional[str] = None, note: str = "",
                  now_utc: Optional[datetime] = None) -> dict:
    _require_real(cfg)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    cancel_id = str(broker_cancel_id or f"{broker_order_id}#{cancel_time}")
    dup = _duplicate(st.recs, ("BROKER_CANCEL", cancel_id),
                     {"trade_date": trade_date, "cancel_time": cancel_time, "broker_order_id": str(broker_order_id),
                      "cancelled_qty": int(cancelled_qty), "evidence": ev})
    if dup:
        return dup
    o = _order_or_fail(st.book, str(broker_order_id))
    _not_terminal(o)
    _same_day(o, trade_date)
    _trading_day(cal, trade_date)
    if cancel_time < o["order"]["order_time"]:
        raise ManualEntryError(f"撤单时间 {cancel_time} 早于委托时间 {o['order']['order_time']}")
    if int(cancelled_qty) <= 0 or int(cancelled_qty) > o["R"]:
        raise ManualEntryError(f"撤单数量 {cancelled_qty} 超过委托 {broker_order_id} 的剩余数量 {o['R']}")
    payload = {"type": "BROKER_CANCEL", "day": trade_date, "trade_date": trade_date, "cancel_time": cancel_time,
               "broker_order_id": str(broker_order_id),
               "broker_cancel_id": cancel_id, "cancelled_qty": int(cancelled_qty), "flags": [], "note": note, "evidence": ev}
    return _append(exp_dir, payload, st.recs, _now(now_utc))


def record_reject(exp_dir, cfg: ExperimentConfig, store, trade_date: str, reject_time: str, broker_order_id: str,
                  reason_text: str, evidence: dict, note: str = "", now_utc: Optional[datetime] = None) -> dict:
    _require_real(cfg)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    dup = _duplicate(st.recs, ("BROKER_REJECT", str(broker_order_id)),
                     {"trade_date": trade_date, "reject_time": reject_time, "reason_text": str(reason_text), "evidence": ev})
    if dup:
        return dup
    o = _order_or_fail(st.book, str(broker_order_id))
    _not_terminal(o)
    _same_day(o, trade_date)
    if o["F"] or o["C"]:
        raise ManualEntryError(f"委托 {broker_order_id} 已有成交或撤单记录，不能再录入拒单")
    payload = {"type": "BROKER_REJECT", "day": trade_date, "trade_date": trade_date, "reject_time": reject_time,
               "broker_order_id": str(broker_order_id), "reason_text": str(reason_text), "flags": [], "note": note,
               "evidence": ev}
    return _append(exp_dir, payload, st.recs, _now(now_utc))


def record_expired(exp_dir, cfg: ExperimentConfig, store, trade_date: str, broker_order_id: str, expired_qty: int,
                   evidence: dict, note: str = "", now_utc: Optional[datetime] = None) -> dict:
    """收盘未成交作废。系统不会自动生成：必须在券商 App 核实委托状态后录入，且只能在当日 15:00 之后。"""
    _require_real(cfg)
    now = _now(now_utc)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    dup = _duplicate(st.recs, ("BROKER_EXPIRED", str(broker_order_id)),
                     {"trade_date": trade_date, "expired_qty": int(expired_qty), "evidence": ev})
    if dup:
        return dup
    o = _order_or_fail(st.book, str(broker_order_id))
    _not_terminal(o)
    _same_day(o, trade_date)
    if int(expired_qty) != o["R"] or o["R"] <= 0:
        raise ManualEntryError(f"作废数量 {expired_qty} 必须等于委托 {broker_order_id} 的剩余未成交数量 {o['R']}（防止漏录成交）")
    earliest = _cst(trade_date, f"{EXPIRED_ENTRY_AFTER[0]:02d}:{EXPIRED_ENTRY_AFTER[1]:02d}")
    if pd.Timestamp(now).tz_convert(CST) < earliest:
        raise ManualEntryError(f"收盘未成交作废只能在 {trade_date} 15:00（北京时间）之后录入")
    payload = {"type": "BROKER_EXPIRED", "day": trade_date, "trade_date": trade_date,
               "broker_order_id": str(broker_order_id), "expired_qty": int(expired_qty), "flags": [], "note": note,
               "evidence": ev}
    return _append(exp_dir, payload, st.recs, now)


def settlement_flags(cfg: ExperimentConfig, store, o: dict, fees: Dict[str, float]) -> Tuple[List[str], dict]:
    """OVER_BUDGET_ACTUAL：(实收费用 + 实际价差成本) / 成交额 > 预算；实际价差成本按成交价相对决策日（前一交易日）收盘价。"""
    od = o["order"]
    f_total = explicit_fee_total(fees)
    notional = sum(x["qty"] * x["price"] for x in o["fills"])
    bars = store.bars()
    d = pd.Timestamp(od["trade_date"])
    prior = bars.index[bars.index < d]
    ref = float(bars.loc[prior[-1], "close"]) if len(prior) else None
    slip = None
    if ref is not None:
        signed = sum(x["qty"] * (x["price"] - ref) for x in o["fills"])
        slip = max(0.0, signed if od["side"] == "BUY" else -signed)
    ratio = ((f_total + (slip or 0.0)) / notional) if notional else 0.0
    flags = ["OVER_BUDGET_ACTUAL"] if notional and ratio > cfg.friction_budget + 1e-12 else []
    est = cfg.fee_scenario.breakdown(o["vwap"], o["F"]) if o["F"] else {}
    hint_total = sum(sum((x.get("fee_hint") or {}).values()) for x in o["fills"])
    if any(x.get("fee_hint") for x in o["fills"]) and abs(hint_total - f_total) > 0.005:
        flags.append("FEE_HINT_MISMATCH")
    audit = {"notional": round(notional, 4), "fee_total": f_total, "slippage_vs_decision_close": None if slip is None else round(slip, 6),
             "decision_close": ref, "actual_ratio": round(ratio, 6), "budget": cfg.friction_budget,
             "scenario_estimate": est, "scenario_estimate_total": explicit_fee_total(est) if est else 0.0,
             "diff_vs_scenario": round(f_total - (explicit_fee_total(est) if est else 0.0), 2)}
    return flags, audit


def _clean_fees(fees: Dict[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for k, v in fees.items():
        v = float(v)
        if k == "stamp_duty":
            if v > 0:
                raise ManualEntryError("ETF 二级市场交易不收印花税（规则已核实）：出现 stamp_duty > 0，请核对是否录错品种")
            continue
        if k not in FEE_COMPONENTS:
            raise ManualEntryError(f"费用项 {k} 不在 {FEE_COMPONENTS} 内")
        if v < 0:
            raise ManualEntryError(f"费用项 {k} 不能为负")
        clean[k] = _cents(v)
    for k in FEE_COMPONENTS:
        clean.setdefault(k, 0.0)
    return clean


def record_settlement(exp_dir, cfg: ExperimentConfig, store, trade_date: str, broker_order_id: str, settled_qty: int,
                      settled_amount: float, fees: Dict[str, float], statement_ref: str, evidence: dict, note: str = "",
                      now_utc: Optional[datetime] = None) -> dict:
    """交割单费用：每个委托每个交易日一条；费用唯一入账来源。"""
    _require_real(cfg)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    ev = _check_evidence(evidence)
    clean = _clean_fees(fees)
    try:
        dup = _duplicate(st.recs, ("BROKER_SETTLEMENT", str(broker_order_id), trade_date),
                         {"settled_qty": int(settled_qty), "settled_amount": round(float(settled_amount), 4), "fees": clean,
                          "statement_ref": str(statement_ref), "evidence": ev})
    except ManualEntryError:
        raise ManualEntryError(f"委托 {broker_order_id} 在 {trade_date} 已有交割记录；如录错请先 VOID")
    if dup:
        return dup
    o = _order_or_fail(st.book, str(broker_order_id))
    _same_day(o, trade_date)
    if o["settlement"] is not None:
        raise ManualEntryError(f"委托 {broker_order_id} 在 {trade_date} 已有交割记录；如录错请先 VOID")
    if not o["fills"]:
        raise ManualEntryError(f"委托 {broker_order_id} 没有成交记录，交割单无从对应")
    if int(settled_qty) != o["F"]:
        raise ManualEntryError(f"交割数量 {settled_qty} ≠ 该委托累计成交 {o['F']}")
    amt = sum(x["qty"] * x["price"] for x in o["fills"])
    if abs(float(settled_amount) - amt) > 0.005:
        raise ManualEntryError(f"交割金额 {settled_amount} ≠ Σ 成交价 × 数量 = {amt:.4f}")
    flags, audit = settlement_flags(cfg, store, o, clean)
    warnings = []
    if clean["transfer_fee"] > 0:
        warnings.append("过户费 > 0：规则文件中 transfer_fee 仍为 UNVERIFIED，本条交割单是首个实证，待复盘后更新规则文件")
    payload = {"type": "BROKER_SETTLEMENT", "day": trade_date, "trade_date": trade_date,
               "broker_order_id": str(broker_order_id), "settled_qty": int(settled_qty),
               "settled_amount": round(float(settled_amount), 4), "fees": clean, "fee_total": explicit_fee_total(clean),
               "statement_ref": str(statement_ref), "audit": audit, "flags": flags, "warnings": warnings, "note": note,
               "evidence": ev}
    return _append(exp_dir, payload, st.recs, _now(now_utc))


def record_dividend(exp_dir, cfg: ExperimentConfig, trade_date: str, amount: float, evidence: dict,
                    broker_ref: Optional[str] = None, note: str = "", now_utc: Optional[datetime] = None) -> dict:
    """券商实际入账的现金分红（trade_date 为到账日；真实账本不按行情数据自动计分红）。"""
    _require_real(cfg)
    if amount <= 0:
        raise ManualEntryError("分红金额必须为正")
    recs = _journal(exp_dir).records()
    ev = _check_evidence(evidence)
    ref = str(broker_ref or f"{trade_date}#{round(float(amount), 2)}")
    dup = _duplicate(recs, ("BROKER_DIVIDEND", ref), {"trade_date": trade_date, "amount": round(float(amount), 2), "evidence": ev})
    if dup:
        return dup
    if not any(r["type"] == "CAPITAL" for r in active_records(recs)):
        raise ManualEntryError("尚未登记实验本金，不能录入分红")
    payload = {"type": "BROKER_DIVIDEND", "day": trade_date, "trade_date": trade_date, "amount": round(float(amount), 2),
               "broker_ref": ref, "flags": [], "note": note, "evidence": ev}
    return _append(exp_dir, payload, recs, _now(now_utc))


def void_record(exp_dir, cfg: ExperimentConfig, target_hash: str, reason: str, evidence: dict,
                now_utc: Optional[datetime] = None) -> dict:
    """作废一条错录。原记录保留；重放时跳过；一个 hash 最多被作废一次；不能作废 CAPITAL 或 VOID。"""
    _require_real(cfg)
    if not str(reason).strip():
        raise ManualEntryError("作废必须写明原因")
    recs = _journal(exp_dir).records()
    target = next((r for r in recs if r["hash"] == target_hash), None)
    if target is None:
        raise ManualEntryError(f"找不到 hash 为 {target_hash[:12]}… 的记录")
    if target["type"] in ("CAPITAL", "VOID"):
        raise ManualEntryError(f"不能作废 {target['type']} 记录")
    if any(r["type"] == "VOID" and r["target_hash"] == target_hash for r in recs):
        raise ManualEntryError("该记录已被作废过一次")
    book = order_book(recs)
    if target["type"] == "BROKER_ORDER":
        o = book.get(target["broker_order_id"])
        if o and (o["fills"] or o["cancels"] or o["reject"] or o["expired"] or o["settlement"]):
            raise ManualEntryError("该委托下还有有效的成交/撤单/拒单/作废/交割记录，请先逐条作废")
    if target["type"] == "BROKER_FILL":
        o = book.get(target["broker_order_id"])
        if o and o["settlement"] is not None:
            raise ManualEntryError("该委托已有交割记录，作废成交前请先作废交割记录")
    payload = {"type": "VOID", "day": target["trade_date"], "trade_date": target["trade_date"], "target_hash": target_hash,
               "target_type": target["type"], "reason": str(reason), "flags": ["CORRECTION"],
               "evidence": _check_evidence(evidence)}
    return _append(exp_dir, payload, recs, _now(now_utc))
