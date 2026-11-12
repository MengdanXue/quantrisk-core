"""
真实账户日报（MANUAL_LIVE，七项）——只读生成，不写任何账本

每个前向交易日一行（与 PAPER 观察表对齐），输出 observation/live_daily.csv|md：
1. PAPER 资金、持仓、净值、收益（PAPER_STRICT 的 DAY_CLOSE）；
2. MANUAL_LIVE 资金、持仓、净值、收益（由流水重放 + 当日收盘估值；交割单未到标 FEES_PENDING；与上次日报不同标 RESTATED）；
3. 双轨信号是否一致：PAPER 原始意图（在此现算，不写入 PAPER 流水）与 suggestions.jsonl 中建议的 raw_intent 比较；
4. 成交价与滑点差异：真实 VWAP 对 PAPER_STRICT 同日模拟成交价 / 决策日收盘 / 限价；
5. 实际手续费与模拟费用差异：交割单费用对情景估算 / PAPER 同日费用；
6. 数据质量与 SHA256：行情获取时间、新增 K 线、修订数、近 60 日完整性问题，配置/规则/选择记录/代码哈希与基线比对；
7. 是否满足下一笔真实交易条件：十项逐一判定（只对最新一行）。
没有登记本金时，所有真实数值列为 NO_ACCOUNT，不出现任何收益数字。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .etf_rules import explicit_fee_total, load_rules
from .experiment_config import ExperimentConfig
from .ledgers import PAPER_LEDGERS, Journal
from .manual_execution import daily_positions, manual_state
from .market_data import CST, StaleDataError
from .observation import check_baseline, run_log
from .runner import INTEGRITY_LOOKBACK, signal_intent, signals_with_resets, submit_deadline
from .trade_signal import propose

NO_ACCOUNT = "NO_ACCOUNT"


def _suggestions(exp_dir) -> Dict[str, dict]:
    p = Path(exp_dir) / "suggestions.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text("utf-8").splitlines():
        r = json.loads(line)
        out[r["decision_day"]] = r
    return out


def _paper_by_day(exp_dir) -> Dict[str, dict]:
    return {r["day"]: {e["type"]: e for e in r["events"]} for r in Journal(exp_dir, "PAPER_STRICT").records()}


def _cst(ts) -> Optional[pd.Timestamp]:
    if ts is None or ts == "None":
        return None
    t = pd.Timestamp(ts)
    return (t if t.tzinfo else t.tz_localize("UTC")).tz_convert(CST)


def paper_raw_intents(cfg: ExperimentConfig, store, through) -> Dict[str, Optional[dict]]:
    """PAPER 轨道持仓过滤前的原始意图，按与 runner.simulate 相同的方法现算（不写入流水）。"""
    rules = load_rules(cfg.etf)
    bars = store.bars(through)
    cal = store.calendar()
    cal = cal[cal <= pd.Timestamp(through)]
    if len(bars) == 0:
        return {}
    issues = store.integrity_issues(rules, through)
    signals, _ = signals_with_resets(bars, store.dividends(through), cfg, issues)
    out = {}
    for d in cal[cal >= pd.Timestamp(cfg.start_date)]:
        if d in bars.index:
            out[str(d.date())] = signal_intent(signals, cal, d, cfg.signal_valid_days)
    return out


def eligibility(exp_dir, cfg: ExperimentConfig, store, now_utc: Optional[datetime] = None) -> dict:
    """第 7 项：下一笔真实交易的十项条件，全部为真才 ELIGIBLE。"""
    now = now_utc or datetime.now(timezone.utc)
    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    checks: List[dict] = []
    add = lambda k, ok, why: checks.append({"check": k, "ok": bool(ok), "detail": why})
    add("1 本金已登记", st.capital is not None, "NO_ACCOUNT" if st.capital is None else f"{st.capital['amount']:.2f} 元")
    try:
        day = store.check_fresh(now)
        add("2 行情新鲜", True, f"最近已收盘交易日 {day.date()} 的 K 线已到")
    except StaleDataError as exc:
        day = None
        add("2 行情新鲜", False, str(exc))
    open_past = [o["order"]["broker_order_id"] for o in st.open_orders
                 if day is not None and pd.Timestamp(o["order"]["trade_date"]) < day]
    add("3 无 OPEN_PAST_DAY 委托", not open_past, f"待录终态的委托：{open_past}" if open_past else "无")
    unverified = [r["hash"][:8] for r in st.recs if "PRICE_UNVERIFIED" in (r.get("flags") or [])
                  and r["type"] in ("BROKER_ORDER", "BROKER_FILL") and pd.Timestamp(r["trade_date"]) not in store.bars().index]
    add("4 无待核对价格", not unverified, f"PRICE_UNVERIFIED 待核：{unverified}" if unverified else "无")
    sug = _suggestions(exp_dir).get(str(day.date())) if day is not None else None
    intent = (sug or {}).get("intent")
    add("5 有意图且与持仓相容", intent is not None,
        f"{intent['side']} 自 {intent['since']}" if intent else ("建议记录缺失" if sug is None else "无意图或与持仓不相容"))
    prop = (sug or {}).get("proposal") or {}
    order = prop.get("order") or {}
    add("6 建议为 PROPOSE", prop.get("action") == "PROPOSE", prop.get("action") or "—")
    ratio = order.get("est_total_ratio")
    add("7 预估摩擦 ≤ 预算", ratio is not None and ratio <= cfg.friction_budget + 1e-12,
        f"{ratio:.4%} vs {cfg.friction_budget:.2%}" if ratio is not None else "—")
    need = (order.get("notional_at_limit") or 0) + (order.get("est_explicit_total") or 0)
    cash = st.acct.cash if st.acct is not None else None
    add("8 现金足够", bool(order) and cash is not None and (order.get("side") == "SELL" or need <= cash + 1e-9),
        f"需 {need:.2f} / 有 {cash:.2f}" if order and cash is not None else "—")
    blockers = prop.get("reason") if prop.get("action") == "REJECT" and not order else None
    add("9 无阻断项", not blockers, blockers or "无")
    if day is not None:
        nxt = cal[cal > day]
        deadline = submit_deadline(nxt[0]) if len(nxt) else None
        ok10 = deadline is not None and pd.Timestamp(now).tz_convert(CST) < deadline
        add("10 早于次日 09:15", ok10, f"申报时刻 {deadline.isoformat() if deadline else '日历未覆盖次日'}")
    else:
        add("10 早于次日 09:15", False, "行情未就绪")
    first_fail = next((c for c in checks if not c["ok"]), None)
    pending = [o["order"]["broker_order_id"] for o in st.book.values() if o["fees_pending"]]
    return {"status": "ELIGIBLE" if first_fail is None else "NOT_ELIGIBLE", "first_failure": first_fail,
            "checks": checks, "fees_pending_orders": pending, "as_of_utc": now.isoformat(timespec="seconds"),
            "notice": "条件满足只表示系统建议可以参考；是否下单由本人在券商 App 中决定，本系统不下单"}


def live_daily(exp_dir, cfg: ExperimentConfig, store, selection_dir, root, previous_csv: Optional[Path] = None,
               now_utc: Optional[datetime] = None) -> (pd.DataFrame, dict):
    exp_dir = Path(exp_dir)
    cal = store.calendar()
    bars = store.bars()
    paper = _paper_by_day(exp_dir)
    if not paper:
        return pd.DataFrame(), {}
    through = max(paper)
    sugs = _suggestions(exp_dir)
    raw = paper_raw_intents(cfg, store, through)
    st = manual_state(exp_dir, cfg, cal)
    pos_df = daily_positions(exp_dir, cfg, store)
    pos = {r["date"]: r for r in pos_df.to_dict("records")} if len(pos_df) else {}
    prev = {}
    if previous_csv and Path(previous_csv).exists():
        old = pd.read_csv(previous_csv, dtype=str)
        prev = {r["date"]: r for r in old.to_dict("records")}
    rules = load_rules(cfg.etf)
    issues = store.integrity_issues(rules, through)
    runs = run_log(exp_dir)
    run_by_through = {}
    for r in runs:
        th = (r.get("sync") or {}).get("through")
        if th:
            run_by_through[th] = r
    orders_by_day: Dict[str, List[dict]] = {}
    for o in st.book.values():
        orders_by_day.setdefault(o["order"]["trade_date"], []).append(o)

    rows = []
    cal_list = list(cal)
    for day in sorted(paper):
        ev = paper[day]
        dec, res, close = ev.get("DECISION"), ev.get("ORDER_RESULT"), ev["DAY_CLOSE"]
        d = pd.Timestamp(day)
        row = {"date": day, "close": close["close"],
               # 1 PAPER
               "paper_cash": close["cash"], "paper_holding": close["holding"], "paper_equity": close["equity"],
               "paper_cum_return": close["equity"] / cfg.capital - 1}
        # 2 MANUAL_LIVE
        p = pos.get(day)
        if st.capital is None or p is None:
            row.update(live_status=NO_ACCOUNT if st.capital is None else "BEFORE_CAPITAL", live_cash=None, live_holding=None,
                       live_equity=None, live_cum_return=None, live_fees_cum=None, fees_pending=None, restated=None)
        else:
            old = prev.get(day)
            restated = bool(old and old.get("live_cash") not in (None, "", "nan") and abs(float(old["live_cash"]) - p["cash"]) > 0.005)
            row.update(live_status="FEES_PENDING" if p["fees_pending"] else "OK", live_cash=p["cash"], live_holding=p["holding"],
                       live_equity=p["equity"], live_cum_return=p["cum_return"], live_fees_cum=p["fees_cum"],
                       fees_pending=p["fees_pending"], restated=restated)
        # 3 信号一致
        pr = raw.get(day)
        sg = sugs.get(day)
        sr = sg.get("raw_intent") if sg else None
        if sg is None or "raw_intent" not in sg:
            consistency = "NA"
        else:
            consistency = "SAME" if (pr or {}).get("side") == (sr or {}).get("side") and (pr or {}).get("since") == (sr or {}).get("since") else "DIFF"
        row.update(paper_raw_intent=(pr or {}).get("side"), suggestion_raw_intent=(sr or {}).get("side"), signal_consistency=consistency,
                   suggestion_status=(NO_ACCOUNT if sg is None else ((sg.get("proposal") or {}).get("action") or "NO_INTENT")),
                   suggestion_reason=((sg or {}).get("proposal") or {}).get("reason") or None)
        # 4 / 5 真实成交 vs 模拟
        todays = orders_by_day.get(day, [])
        filled = [o for o in todays if o["F"]]
        F = sum(o["F"] for o in filled)
        vwap = (sum(o["vwap"] * o["F"] for o in filled) / F) if F else None
        paper_price = (res or {}).get("price")
        paper_fee = (res or {}).get("fee_total")
        pidx = cal_list.index(d) if d in cal else None
        dec_close = float(bars.loc[cal_list[pidx - 1], "close"]) if pidx and cal_list[pidx - 1] in bars.index else None
        limits = [o["order"]["limit_price"] for o in filled]
        live_fee = sum(o["settlement"]["fee_total"] for o in filled if o["settlement"]) if filled else None
        est_fee = explicit_fee_total(cfg.fee_scenario.breakdown(vwap, F)) if F else None
        row.update(live_orders=len(todays), live_order_status=";".join(f"{o['order']['broker_order_id']}:{o['status']}" for o in todays) or None,
                   live_flags=";".join(sorted({f for o in todays for f in (o["order"].get("flags") or []) +
                                               ((o["settlement"] or {}).get("flags") or [])})) or None,
                   live_filled_qty=F or None, live_vwap=vwap, paper_fill_status=(res or {}).get("status"), paper_fill_price=paper_price,
                   price_diff_live_minus_paper=(vwap - paper_price) if vwap is not None and paper_price is not None else None,
                   price_diff_ticks=round((vwap - paper_price) / rules.tick_size, 2) if vwap is not None and paper_price is not None else None,
                   price_diff_bp=round((vwap / paper_price - 1) * 1e4, 2) if vwap is not None and paper_price else None,
                   live_vs_decision_close=(vwap - dec_close) if vwap is not None and dec_close is not None else None,
                   live_vs_limit=(vwap - limits[0]) if vwap is not None and limits else None,
                   live_fee=live_fee, fee_pending_orders=";".join(o["order"]["broker_order_id"] for o in filled if o["fees_pending"]) or None,
                   scenario_est_fee=est_fee, paper_fee=paper_fee,
                   fee_diff_vs_scenario=(live_fee - est_fee) if live_fee is not None and est_fee is not None else None,
                   fee_diff_vs_paper=(live_fee - paper_fee) if live_fee is not None and paper_fee is not None else None)
        # 6 数据质量
        seen = _cst((dec or {}).get("data", {}).get("bar_first_seen_utc"))
        cutoff = d.tz_localize(CST) + pd.Timedelta(hours=15, minutes=30)
        rr = run_by_through.get(day, {})
        recent = [i for i in issues if i["date"] and pidx is not None
                  and pd.Timestamp(i["date"]) in cal_list[max(0, pidx - INTEGRITY_LOOKBACK + 1):pidx + 1]]
        row.update(bar_fetched_at_cst=seen.strftime("%Y-%m-%d %H:%M") if seen is not None else None,
                   fetch_delay_hours=round((seen - cutoff).total_seconds() / 3600, 2) if seen is not None else None,
                   new_bars=(rr.get("fetch") or {}).get("new_bars"), revisions=(rr.get("fetch") or {}).get("revisions"),
                   integrity_issues_60d=len(recent), run_status=rr.get("status"),
                   baseline_soft_changes=";".join(rr.get("baseline_soft_changes") or []) or None)
        rows.append(row)
    df = pd.DataFrame(rows)
    base = check_baseline(exp_dir, cfg, selection_dir, root)
    tails = {L: (Journal(exp_dir, L).records() or [{}])[-1].get("hash") for L in PAPER_LEDGERS + ("MANUAL_LIVE",)}
    summary = {"hashes": base["current"], "baseline": {k: base["baseline"][k] for k in base["current"]},
               "hard_changes": base["hard"], "soft_changes": base["soft"], "journal_tail_hashes": tails,
               "eligibility": eligibility(exp_dir, cfg, store, now_utc),
               "orders_total": len(st.book), "orders_open": len(st.open_orders),
               "corrections": sum(1 for r in st.recs if r["type"] == "VOID"),
               "flag_counts": pd.Series([f for r in st.recs for f in (r.get("flags") or [])]).value_counts().to_dict()}
    return df, summary


def write_live_daily(exp_dir, cfg: ExperimentConfig, store, selection_dir, root, out_dir,
                     now_utc: Optional[datetime] = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "live_daily.csv"
    df, summary = live_daily(exp_dir, cfg, store, selection_dir, root, previous_csv=csv_path, now_utc=now_utc)
    if df.empty:
        (out / "live_daily.md").write_text("# 真实账户日报\n\n尚无前向交易日记录。\n", encoding="utf-8")
        return out
    df.to_csv(csv_path, index=False)
    (out / "live_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    f = lambda v, fmt="{}": "—" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)
    money = lambda v: f(v, "{:.2f}")
    el = summary["eligibility"]
    no_acct = all(r["live_status"] == NO_ACCOUNT for r in df.to_dict("records"))
    lines = [f"# 真实账户日报（MANUAL_LIVE）：{cfg.experiment_id()}", "",
             f"生成 {pd.Timestamp(el['as_of_utc']).tz_convert(CST).isoformat(timespec='minutes')}（北京时间）。"
             "本系统不下单、不连接券商；真实记录全部来自人工录入的券商凭据。", ""]
    if no_acct:
        lines += ["**尚未登记真实本金（NO_ACCOUNT）**：以下真实列全部为空，不含任何真实收益数字。", ""]
    lines += ["## 第 6 项：数据质量与 SHA-256", "",
              "| 对象 | 当前 | 基线 | 一致 |", "|---|---|---|---|"]
    for k, v in summary["hashes"].items():
        b = summary["baseline"][k]
        lines.append(f"| {k} | `{v[:16]}…` | `{b[:16]}…` | {'是' if v == b else ('**否（硬变化）**' if k in summary['hard_changes'] else '否（软变化，只记录）')} |")
    lines += ["", "流水末端哈希：" + "；".join(f"{k} `{(v or '空')[:12]}`" for k, v in summary["journal_tail_hashes"].items()), "",
              f"委托 {summary['orders_total']} 个（开放 {summary['orders_open']}）；修正（VOID）{summary['corrections']} 次；"
              f"标记统计 {summary['flag_counts'] or '无'}", "",
              f"## 第 7 项：下一笔真实交易条件 → **{el['status']}**", "",
              "| 条件 | 结果 | 说明 |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'✓' if c['ok'] else '✗'} | {c['detail']} |" for c in el["checks"]]
    if el["fees_pending_orders"]:
        lines.append(f"\n交割单待录（不阻断）：{el['fees_pending_orders']}")
    lines += ["", el["notice"], "",
              "## 第 1–5 项：逐日记录", "",
              "| 日期 | 收盘 | PAPER 净值 | PAPER 收益 | 真实状态 | 真实现金 | 真实持仓 | 真实净值 | 真实收益 | 信号一致 | 建议 | 真实委托 | 真实 VWAP | PAPER 成交价 | 价差(tick) | 实收费用 | 情景估算 | PAPER 费用 | 行情获取 | 运行 |",
              "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in df.to_dict("records"):
        lines.append(f"| {r['date']} | {f(r['close'], '{:.3f}')} | {money(r['paper_equity'])} | {f(r['paper_cum_return'], '{:+.2%}')} | "
                     f"{r['live_status']}{' RESTATED' if r.get('restated') else ''} | {money(r['live_cash'])} | {f(r['live_holding'])} | "
                     f"{money(r['live_equity'])} | {f(r['live_cum_return'], '{:+.2%}')} | {r['signal_consistency']} | {r['suggestion_status']} | "
                     f"{f(r['live_order_status'])} | {f(r['live_vwap'], '{:.3f}')} | {f(r['paper_fill_price'], '{:.3f}')} | "
                     f"{f(r['price_diff_ticks'])} | {money(r['live_fee'])} | {money(r['scenario_est_fee'])} | {money(r['paper_fee'])} | "
                     f"{f(r['bar_fetched_at_cst'])} | {f(r['run_status'])} |")
    (out / "live_daily.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
