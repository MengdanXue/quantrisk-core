"""
逐日运行与恢复

simulate()：只读行情存储，从实验起始日起逐个交易日确定性地计算三本模拟账（PAPER_STRICT、PAPER_TOUCH、BENCHMARK）。
每个交易日 D 的顺序：
  1. 除息日：按持仓入账现金分红；
  2. 撮合 D−1 收盘生成的限价单（只用 D 日的开盘价、最高/最低价判定）；
  3. 收盘估值；
  4. D 日收盘后：用截至 D 日的数据生成信号与交易建议（D+1 日有效）。
D 日收盘后的信息不会用于 D 日收盘前的成交。

前向实验另有成交时间约束：订单只按 D+1 开盘价撮合，必须赶上 D+1 开盘集合竞价。D 日决策所依赖的行情
（D 日及以前的 K 线与分红）首次获取时间晚于 D+1 日 09:15（北京时间）时，这笔单在现实中不可能下出，
撮合结果记为 LATE_DATA_NOT_EXECUTABLE（保留原因与时间），不补造成交。演练是历史回放，不适用。

无法解释的价格跳变（如份额拆分/折算）：信号从跳变日起重新计算，跳变后 2 × 慢线周期内不生成建议；
跳变当日若模拟账本仍有持仓，直接报错停止（数据源没有份额变动记录，不编造账本）。

sync()：先检查行情新鲜度（缺少最近一个已收盘交易日的数据就拒绝），再重算并与已写入的流水逐日比对：
已有记录必须与重算结果完全一致（否则抛 ReplayMismatchError，例如数据源事后新增了过去日期的分红），
只追加缺少的交易日。中断后再次运行即可恢复；同一份行情存储总能重现同样的账本。
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from ..strategies.backtest.ledger import Account
from .etf_rules import ETFRules, load_rules
from .experiment_config import ExperimentConfig
from .ledgers import PAPER_LEDGERS, Journal
from .market_data import CST, DataIntegrityError, ETFMarketStore
from .paper_execution import simulate_fill, touch_fill
from .trade_signal import propose, strategy_signals

INTEGRITY_LOOKBACK = 60          # 最近 60 个交易日内有完整性问题，就不生成建议
ORDER_SUBMIT_DEADLINE_CST = (9, 15)   # 开盘集合竞价开始接受申报的时刻；只按开盘价撮合，晚于此刻拿到行情就下不了单
LATE = "LATE_DATA_NOT_EXECUTABLE"
JUMP = "UNEXPLAINED_JUMP"


def warmup_bars(cfg: ExperimentConfig) -> int:
    """未建模价格跳变后，指标需要重新预热的交易日数：慢线周期的 2 倍。"""
    return 2 * int(cfg.strategy["params"]["slow_period"])


def signals_with_resets(bars: pd.DataFrame, divs: pd.DataFrame, cfg: ExperimentConfig, issues: List[dict]):
    """
    无法解释的价格跳变（如份额拆分/折算，数据源没有记录）会污染均线（EMA 记忆无限长）。
    因此信号按跳变日分段计算：跳变日 E 及以后的信号只用 E 及以后的数据；E 之前的信号不受影响。
    返回（信号序列, 跳变日列表）。
    """
    jumps = sorted({pd.Timestamp(i["date"]) for i in issues if i.get("kind") == JUMP and i["date"]})
    starts = [bars.index[0]] + [j for j in jumps if j in bars.index]
    parts = []
    for k, st in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else None
        seg = bars.loc[st:] if end is None else bars.loc[st:end - pd.Timedelta(days=1)]
        if len(seg):
            parts.append(strategy_signals(seg, divs, cfg.strategy))
    return pd.concat(parts), jumps


class ReplayMismatchError(RuntimeError):
    """已写入的流水与按同一份行情重算的结果不一致。"""


def _plain(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _cst(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return (t if t.tzinfo else t.tz_localize("UTC")).tz_convert(CST)


def data_available_at(bars: pd.DataFrame, divs: pd.DataFrame, d) -> pd.Timestamp:
    """D 日决策所依赖的行情（D 日及以前的 K 线与分红）全部到手的时刻（北京时间）：首次获取时间的最大值。"""
    d = pd.Timestamp(d)
    seen = list(bars.loc[bars.index <= d, "first_seen_utc"]) + list(divs.loc[divs.index <= d, "first_seen_utc"])
    return max(_cst(t) for t in seen)


def submit_deadline(fill_day) -> pd.Timestamp:
    h, m = ORDER_SUBMIT_DEADLINE_CST
    return pd.Timestamp(pd.Timestamp(fill_day).date()).tz_localize(CST) + pd.Timedelta(hours=h, minutes=m)


def signal_intent(signals: pd.Series, cal: pd.DatetimeIndex, day: pd.Timestamp, valid_days: int) -> Optional[dict]:
    """D 日有效的交易意图：最近 valid_days 个交易日（含 D）内最近一次非零信号。"""
    pos = cal.get_loc(day)
    window = cal[max(0, pos - valid_days + 1):pos + 1]
    s = signals.reindex(window).fillna(0)
    nz = s[s != 0]
    if nz.empty:
        return None
    d, v = nz.index[-1], int(nz.iloc[-1])
    return {"side": "BUY" if v > 0 else "SELL", "signal": v, "since": str(d.date()), "age": int(pos - cal.get_loc(d))}


def _blockers(cfg: ExperimentConfig, day, bar, issues: List[dict], cal) -> List[str]:
    out = []
    pos = cal.get_loc(day)
    jumps = [pd.Timestamp(i["date"]) for i in issues if i.get("kind") == JUMP and i["date"] and pd.Timestamp(i["date"]) <= day]
    if jumps:
        since = pos - cal.get_loc(max(jumps))
        if since < warmup_bars(cfg):
            out.append(f"{max(jumps).date()} 出现无法解释的价格跳变（可能是份额拆分/折算，未建模），"
                       f"指标重新预热中（{since}/{warmup_bars(cfg)} 个交易日）")
    recent = cal[max(0, pos - INTEGRITY_LOOKBACK + 1):pos + 1]
    bad = [i for i in issues if i["date"] and i.get("kind") != JUMP and pd.Timestamp(i["date"]) in recent]
    if bad:
        out.append(f"最近 {INTEGRITY_LOOKBACK} 个交易日行情完整性有问题：" + "；".join(f"{i['date']} {i['issue']}" for i in bad[:3]))
    if not cfg.rehearsal:
        seen = _cst(bar["first_seen_utc"])
        h, m = (int(x) for x in cfg.close_cutoff_cst.split(":"))
        cutoff = pd.Timestamp(day.date()).tz_localize(CST) + pd.Timedelta(hours=h, minutes=m)
        if seen < cutoff:
            out.append(f"{day.date()} 的 K 线获取于 {seen.isoformat()}，早于收盘截止时刻，可能不是完整 K 线")
    return out


def simulate(cfg: ExperimentConfig, rules: ETFRules, store: ETFMarketStore, through) -> Dict[str, List[dict]]:
    through = pd.Timestamp(through)
    bars = store.bars(through)
    divs = store.dividends(through)
    cal = store.calendar()
    cal = cal[cal <= through]
    start = pd.Timestamp(cfg.start_date)
    days = cal[cal >= start]
    if len(bars) == 0 or len(days) == 0:
        return {L: [] for L in PAPER_LEDGERS}
    issues = store.integrity_issues(rules, through)
    signals, jumps = signals_with_resets(bars, divs, cfg, issues)
    fee = cfg.fee_scenario
    code = cfg.etf
    pos = {d: i for i, d in enumerate(cal)}
    day_index_of = lambda d: pos[pd.Timestamp(d)]
    between = lambda a, b: pos[pd.Timestamp(b)] - pos[pd.Timestamp(a)]
    cash_div = divs["cash"].to_dict()

    state = {L: {"acct": Account(cfg.capital, settlement_lag=rules.settlement_lag), "pending": None,
                 "available": None} for L in PAPER_LEDGERS}
    out = {L: [] for L in PAPER_LEDGERS}
    for d in days:
        bar = bars.loc[d] if d in bars.index else None
        prior = bars.index[bars.index < d]
        prev_close = float(bars.loc[prior[-1], "close"]) if len(prior) else None
        ex_cash = float(cash_div[d]) if d in cash_div else None
        band = None
        if prev_close is not None:
            ref = rules.ex_dividend_reference(prev_close, ex_cash) if ex_cash else prev_close
            band = rules.limit_prices(ref)
        close = float(bar["close"]) if bar is not None else prev_close
        for L in PAPER_LEDGERS:
            st, acct, ev = state[L], state[L]["acct"], []
            if d in jumps and acct.quantity(code) > 0:
                raise DataIntegrityError(f"{L} 在 {d.date()} 持有 {acct.quantity(code)} 份，当日出现无法解释的价格跳变"
                                         "（可能是份额拆分/折算）。数据源没有份额变动记录，不能编造账本，需要人工核实后处理")
            if ex_cash and acct.quantity(code) > 0:
                before = acct.cash
                acct.cash_dividend(d, code, ex_cash)
                ev.append({"type": "DIVIDEND", "per_share": ex_cash, "quantity": acct.quantity(code),
                           "amount": round(acct.cash - before, 6), "cash_after": acct.cash})
            if st["pending"] is not None:
                order = st["pending"]
                timing = None
                if not cfg.rehearsal:
                    deadline = submit_deadline(d)
                    timing = {"decision_data_available_cst": st["available"].isoformat(),
                              "order_submit_deadline_cst": deadline.isoformat()}
                if timing and st["available"] > deadline:
                    res = {"order_id": order["order_id"], "side": order["side"], "qty": order["qty"],
                           "limit_price": order["limit_price"], "status": LATE, **timing,
                           "reason": f"决策所用行情获取于 {timing['decision_data_available_cst']}，晚于 {d.date()} "
                                     f"开盘集合竞价申报时刻 {timing['order_submit_deadline_cst']}：现实中无法按时下单，不补造成交"}
                else:
                    res = simulate_fill(order, bar, band, rules, fee, cfg.half_spread_ticks)
                    if timing:
                        res.update(timing)
                if L == "PAPER_TOUCH" and res["status"] == "UNCERTAIN_TOUCH":
                    res = touch_fill(res, order, fee)
                if res["status"] in ("STRICT_FILL", "TOUCH_FILL"):
                    if order["side"] == "BUY":
                        acct.buy(d, code, order["qty"], res["price"], res["fees"], order["order_id"])
                    else:
                        if acct.sellable(code, day_index_of(d), day_index_of) < order["qty"]:
                            res = {**res, "status": "EXPIRED", "reason": "T+1：可卖数量不足"}
                        else:
                            acct.sell(d, code, order["qty"], res["price"], res["fees"], order["order_id"], between)
                    res["cash_after"], res["holding_after"] = acct.cash, acct.quantity(code)
                ev.append({"type": "ORDER_RESULT", **res})
                st["pending"], st["available"] = None, None
            holding = acct.quantity(code)
            ev.append({"type": "DAY_CLOSE", "has_bar": bar is not None, "close": close, "cash": acct.cash,
                       "holding": holding, "market_value": holding * (close or 0.0),
                       "equity": acct.cash + holding * (close or 0.0)})
            if bar is not None:
                ev.append(_decide(L, cfg, rules, d, bar, signals, cal, issues, acct, code, fee, st))
                if st["pending"] is not None and not cfg.rehearsal:
                    st["available"] = data_available_at(bars, divs, d)
            out[L].append(_plain({"day": str(d.date()), "events": ev}))
    return out


def _decide(L, cfg, rules, d, bar, signals, cal, issues, acct, code, fee, st) -> dict:
    holding = acct.quantity(code)
    data = {"last_bar": str(d.date()), "bar_first_seen_utc": str(bar["first_seen_utc"]), "fetch_id": str(bar["fetch_id"])}
    if L == "BENCHMARK":
        intent = {"side": "BUY", "signal": None, "since": cfg.start_date, "age": None} if holding == 0 else None
        sig = None
    else:
        sig = int(signals.get(d, 0))
        intent = signal_intent(signals, cal, d, cfg.signal_valid_days)
        if intent and ((intent["side"] == "BUY" and holding > 0) or (intent["side"] == "SELL" and holding == 0)):
            intent = None
    rec = {"type": "DECISION", "signal": sig, "intent": intent, "data": data, "proposal": None}
    if intent:
        p = propose(d, intent["side"], float(bar["close"]), acct.cash, holding, rules, fee, cfg.friction_budget,
                    cfg.half_spread_ticks, cfg.limit_offset_ticks, order_id=f"{L}-{d:%Y%m%d}",
                    blockers=_blockers(cfg, d, bar, issues, cal))
        rec["proposal"] = p.as_dict()
        if p.action == "PROPOSE":
            st["pending"] = p.order
    return rec


def sync(exp_dir, cfg: ExperimentConfig, store: ETFMarketStore, now_utc: Optional[datetime] = None,
         through=None, repair: bool = False, append: bool = True) -> dict:
    now = now_utc or datetime.now(timezone.utc)
    rules = load_rules(cfg.etf)
    last_ok = store.check_fresh(now)                     # 行情过期即抛错，不生成任何建议
    end = last_ok if through is None else min(pd.Timestamp(through), last_ok)
    computed = simulate(cfg, rules, store, end)
    summary = {"through": str(end.date()), "appended": {}, "verified_existing": {}}
    for L in PAPER_LEDGERS:
        j = Journal(exp_dir, L)
        recs = j.records(repair=repair)
        if len(recs) > len(computed[L]):
            raise ReplayMismatchError(f"{L} 已有 {len(recs)} 个交易日的记录，多于重算结果 {len(computed[L])}")
        for i, r in enumerate(recs):
            c = computed[L][i]
            if r["day"] != c["day"] or json.dumps(r["events"], sort_keys=True) != json.dumps(c["events"], sort_keys=True):
                raise ReplayMismatchError(f"{L} {r['day']}：已写入的记录与按当前行情存储重算的结果不一致。"
                                          "可能是数据源事后新增或修订了过去日期的数据，或计算代码发生了变化；需要人工处理，不会覆盖历史记录")
        summary["verified_existing"][L] = len(recs)
        new = computed[L][len(recs):]
        if append:
            for item in new:
                j.append(item, "SIMULATED")
        summary["appended"][L] = len(new)
    return summary


def manual_suggestion(exp_dir, cfg: ExperimentConfig, store: ETFMarketStore, now_utc: Optional[datetime] = None) -> dict:
    """
    为 MANUAL_LIVE 生成最近一个已收盘交易日的交易建议（参考，不是订单），写入 suggestions.jsonl（不属于任何账本）。
    没有登记真实本金时不生成；行情过期时抛错。
    """
    from pathlib import Path
    from .manual_execution import replay_manual
    now = now_utc or datetime.now(timezone.utc)
    rules = load_rules(cfg.etf)
    day = store.check_fresh(now)
    cal = store.calendar()
    acct, _ = replay_manual(exp_dir, cfg, cal)
    if acct is None:
        return {"status": "NO_ACCOUNT", "message": "尚未登记真实本金：没有真实账户时不生成实盘交易建议"}
    bars = store.bars(day)
    issues = store.integrity_issues(rules, day)
    signals, _ = signals_with_resets(bars, store.dividends(day), cfg, issues)
    bar = bars.loc[day]
    holding = acct.quantity(cfg.etf)
    raw_intent = signal_intent(signals, cal[cal <= day], day, cfg.signal_valid_days)   # 持仓过滤前的原始意图
    intent = raw_intent
    if intent and ((intent["side"] == "BUY" and holding > 0) or (intent["side"] == "SELL" and holding == 0)):
        intent = None
    rec = {"suggestion_id": f"S-{day:%Y%m%d}", "decision_day": str(day.date()), "signal": int(signals.get(day, 0)),
           "raw_intent": raw_intent, "intent": intent, "holding_before": holding, "cash_before": round(acct.cash, 6),
           "data": {"last_bar": str(day.date()), "bar_first_seen_utc": str(bar["first_seen_utc"])},
           "generated_at_utc": now.isoformat(timespec="seconds"),
           "notice": "交易建议，不是订单：是否下单由本人在券商 App 中确认；本系统不会下单", "proposal": None}
    if intent:
        p = propose(day, intent["side"], float(bar["close"]), acct.cash, holding, rules, cfg.fee_scenario,
                    cfg.friction_budget, cfg.half_spread_ticks, cfg.limit_offset_ticks, order_id=rec["suggestion_id"],
                    blockers=_blockers(cfg, day, bar, issues, cal[cal <= day]))
        rec["proposal"] = p.as_dict()
    path = Path(exp_dir) / "suggestions.jsonl"
    if path.exists() and any(json.loads(x)["suggestion_id"] == rec["suggestion_id"] for x in path.read_text("utf-8").splitlines()):
        return {"status": "EXISTS", **rec}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_plain(rec), ensure_ascii=False) + "\n")
    return {"status": "WRITTEN", **rec}
