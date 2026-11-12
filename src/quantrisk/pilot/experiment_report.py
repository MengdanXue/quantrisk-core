"""
实验报告：只读流水与行情存储，不修改任何账本

输出（out_dir）：nav_daily.csv、orders_audit.csv、fills.csv、metrics.json、nav.png、report.md。
- 模拟账（PAPER_STRICT / PAPER_TOUCH / BENCHMARK）与真实账（MANUAL_LIVE）分开展示，标注来源；
- 没有任何人工录入的真实成交时，真实部分只显示“尚无真实交易数据”，不生成任何实盘收益；
- 历史回放演练（rehearsal）在报告开头标明：数据获取时间晚于决策日，不是前向实验，不能当作实盘或前向结果。
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .experiment_config import ExperimentConfig
from .ledgers import MANUAL, PAPER_LEDGERS, Journal
from .manual_execution import daily_positions, manual_state

LABEL = {"PAPER_STRICT": "严格模拟（只认开盘价满足限价）", "PAPER_TOUCH": "敏感性：盘中触价也按限价成交",
         "BENCHMARK": "基准：同一 ETF 买入持有（同样的成交规则与费用）", MANUAL: "真实成交（人工录入）"}


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def _nav(recs: List[dict], ledger: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        for e in r["events"]:
            if e["type"] == "DAY_CLOSE":
                rows.append({"date": r["day"], "ledger": ledger, "close": e["close"], "cash": e["cash"],
                             "holding": e["holding"], "equity": e["equity"]})
    return pd.DataFrame(rows)


def _orders(recs: List[dict], ledger: str) -> List[dict]:
    """逐笔审计：每个决策（信号 → 建议/拒绝）与次日结果配对。"""
    out, pending = [], None
    for r in recs:
        day_close = next((e["close"] for e in r["events"] if e["type"] == "DAY_CLOSE"), None)
        for e in r["events"]:
            if e["type"] == "ORDER_RESULT" and pending is not None:
                pending.update({"result_day": r["day"], "status": e["status"], "result_reason": e.get("reason", ""),
                                "fill_price": e.get("price"), "fee_total": e.get("fee_total"), "fees": e.get("fees"),
                                "spread_cost_vs_open": e.get("spread_cost_vs_open"),
                                "price_vs_signal_close": e.get("price_vs_signal_close"),
                                "cash_after": e.get("cash_after"), "holding_after": e.get("holding_after"),
                                "fill_basis": e.get("rule", ""), "next_day_bar": e.get("bar")})
                pending = None
            if e["type"] == "DECISION" and e.get("proposal"):
                p, o = e["proposal"], e["proposal"].get("order") or {}
                row = {"ledger": ledger, "mode": "PAPER", "decision_day": r["day"], "signal": e["signal"],
                       "intent_since": (e.get("intent") or {}).get("since"), "side": p["side"], "action": p["action"],
                       "reject_reason": p["reason"], "order_id": o.get("order_id"), "qty": o.get("qty"),
                       "signal_ref_price": o.get("ref_close", day_close), "planned_limit_price": o.get("limit_price"),
                       "est_explicit_total": o.get("est_explicit_total"), "est_explicit_ratio": o.get("est_explicit_ratio"),
                       "est_friction": o.get("est_friction"), "est_friction_ratio": o.get("est_friction_ratio"),
                       "est_total_ratio": o.get("est_total_ratio"), "cash_before": o.get("cash_before"),
                       "data_last_bar": e["data"]["last_bar"], "data_first_seen_utc": e["data"]["bar_first_seen_utc"],
                       "status": "REJECTED" if p["action"] == "REJECT" else "PENDING"}
                out.append(row)
                if p["action"] == "PROPOSE":
                    pending = row
    return out


def _reject_category(reason: str) -> str:
    for key, label in (("超过预算", "摩擦成本超预算"), ("不足一手", "资金不足一手"), ("价格跳变", "未建模价格跳变后预热中"),
                       ("完整性", "行情完整性问题"), ("早于收盘截止", "K 线可能未完成"), ("申报单位", "数量不符合申报单位")):
        if key in reason:
            return label
    return "其他：" + reason[:30]


def _execution_stats(recs: List[dict], orders: List[dict]) -> dict:
    signals = sum(1 for r in recs for e in r["events"] if e["type"] == "DECISION" and e.get("signal"))
    st = pd.Series([o["status"] for o in orders]).value_counts().to_dict() if orders else {}
    rejects = pd.Series([_reject_category(o["reject_reason"]) for o in orders if o["action"] == "REJECT"]).value_counts().to_dict()
    proposed = sum(1 for o in orders if o["action"] == "PROPOSE")
    filled = st.get("STRICT_FILL", 0) + st.get("TOUCH_FILL", 0)
    return {"nonzero_signals": signals, "decisions_with_intent": len(orders), "proposed_orders": proposed,
            "rejected": sum(1 for o in orders if o["action"] == "REJECT"), "reject_reasons": rejects,
            "results": st, "fill_rate_of_proposed": filled / proposed if proposed else None}


def build_report(exp_dir, cfg: ExperimentConfig, store, out_dir) -> Path:
    exp_dir, out = Path(exp_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    navs, orders, metrics = [], [], {}
    for L in PAPER_LEDGERS:
        recs = Journal(exp_dir, L).records()
        nav = _nav(recs, L)
        navs.append(nav)
        o = _orders(recs, L)
        orders += o
        fills = [x for x in o if x["status"] in ("STRICT_FILL", "TOUCH_FILL")]
        eq = nav["equity"] if len(nav) else pd.Series(dtype=float)
        metrics[L] = {"days": len(nav), "final_equity": float(eq.iloc[-1]) if len(eq) else cfg.capital,
                      "cumulative_return": float(eq.iloc[-1] / cfg.capital - 1) if len(eq) else 0.0,
                      "max_drawdown": _max_drawdown(eq), "fills": len(fills),
                      "explicit_fees": round(sum(x["fee_total"] or 0 for x in fills), 2),
                      "spread_cost_vs_open": round(sum(x["spread_cost_vs_open"] or 0 for x in fills), 4),
                      "avg_price_vs_signal_close": (sum(x["price_vs_signal_close"] for x in fills) / len(fills)) if fills else None,
                      "dividends": round(sum(e["amount"] for r in recs for e in r["events"] if e["type"] == "DIVIDEND"), 4),
                      "execution": _execution_stats(recs, o)}
    nav_all = pd.concat(navs) if navs else pd.DataFrame()

    cal = store.calendar()
    st = manual_state(exp_dir, cfg, cal)
    live_orders = [o for o in st.book.values() if o["F"]]
    if not live_orders:
        metrics[MANUAL] = {"status": "尚无真实交易数据", "capital_registered": st.capital is not None,
                           "orders_without_fill": len(st.book)}
        deviation = []
    else:
        deviation = []
        strict = {o["decision_day"]: o for o in orders if o["ledger"] == "PAPER_STRICT" and o["action"] == "PROPOSE"}
        for o in live_orders:
            od = o["order"]
            sid = od.get("suggestion_id") or ""
            key = f"{sid[2:6]}-{sid[6:8]}-{sid[8:10]}" if len(sid) == 10 and sid.startswith("S-") else None
            p = strict.get(key)
            fee = o["settlement"]["fee_total"] if o["settlement"] else None
            deviation.append({"broker_order_id": od["broker_order_id"], "trade_date": od["trade_date"], "side": od["side"],
                              "status": o["status"], "filled_qty": o["F"], "live_vwap": o["vwap"], "live_fees": fee,
                              "flags": sorted(set((od.get("flags") or []) + ((o["settlement"] or {}).get("flags") or []))),
                              "paper_status": p["status"] if p else "无对应模拟订单",
                              "paper_price": p["fill_price"] if p else None, "paper_fees": p["fee_total"] if p else None,
                              "price_diff_live_minus_paper": (o["vwap"] - p["fill_price"]) if p and p["fill_price"] else None})
        live_nav = daily_positions(exp_dir, cfg, store).assign(ledger=MANUAL)
        navs.append(live_nav[["date", "ledger", "close", "cash", "holding", "equity", "capital"]])
        nav_all = pd.concat(navs)
        eq = live_nav["equity"]
        metrics[MANUAL] = {"status": "有真实交易数据", "orders": len(st.book), "orders_with_fills": len(live_orders),
                           "fills": sum(len(o["fills"]) for o in live_orders),
                           "fees_pending_orders": [o["order"]["broker_order_id"] for o in live_orders if o["fees_pending"]],
                           "cumulative_return": float(eq.iloc[-1] / live_nav["capital"].iloc[0] - 1),
                           "max_drawdown": _max_drawdown(eq),
                           "explicit_fees": round(sum(o["settlement"]["fee_total"] for o in live_orders if o["settlement"]), 2),
                           "cash": st.acct.cash, "holding": st.acct.quantity(cfg.etf),
                           "corrections": sum(1 for r in st.recs if r["type"] == "VOID")}

    nav_all.to_csv(out / "nav_daily.csv", index=False)
    pd.DataFrame(orders).drop(columns=["fees", "next_day_bar"], errors="ignore").to_csv(out / "orders_audit.csv", index=False)
    pd.DataFrame([o for o in orders if o["status"] in ("STRICT_FILL", "TOUCH_FILL")]).to_csv(out / "fills.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    plotted = _plot(nav_all, cfg, out / "nav.png")
    md = _markdown(cfg, store, metrics, deviation, nav_all)
    md += ("\n净值曲线：nav.png\n" if plotted else "\n净值曲线：未安装 matplotlib，未生成图片；每日净值见 nav_daily.csv\n")
    (out / "report.md").write_text(md, encoding="utf-8")
    return out


def _plot(nav: pd.DataFrame, cfg: ExperimentConfig, path: Path) -> bool:
    """净值曲线图（matplotlib 不是本库的依赖：未安装时不画图，返回 False，净值数据见 nav_daily.csv）。"""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for L, g in nav.groupby("ledger") if len(nav) else []:
        ax.plot(pd.to_datetime(g["date"]), g["equity"], label=L)
    ax.axhline(cfg.capital, color="grey", lw=0.8, ls="--")
    ax.set_title(f"{cfg.experiment_id()} NAV (CNY){' - REHEARSAL' if cfg.rehearsal else ''}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _markdown(cfg: ExperimentConfig, store, metrics: Dict, deviation: List[dict], nav: pd.DataFrame) -> str:
    b = store.bars()
    lines = [f"# ETF Pilot 实验报告：{cfg.experiment_id()}", ""]
    if cfg.rehearsal:
        lines += ["> **历史回放演练**：行情的获取时间晚于决策日，这不是前向实验，也不是实盘结果；只用于检验流程与账本。", ""]
    lines += [f"- 标的：{cfg.etf}；配置哈希：`{cfg.config_hash()}`；起始决策日：{cfg.start_date}；本金上限：{cfg.capital:.0f} 元",
              f"- 策略：{cfg.strategy['name']} {cfg.strategy['params']}（登记后不调整；实验结果不作为策略有效性的证据）",
              f"- 佣金情景：{cfg.fee['name']} —— {cfg.fee['note']}",
              f"- 摩擦成本预算（单边）：{cfg.friction_budget:.2%}；估算半价差 {cfg.half_spread_ticks} 个报价单位；"
              f"限价偏移 {cfg.limit_offset_ticks} 个报价单位；信号有效期 {cfg.signal_valid_days} 个交易日",
              f"- 交易规则：{cfg.rules_file}（核实日期 {cfg.rules_verified_on}）；过户费未核实，按情景配置",
              f"- 行情：最新 K 线 {b.index.max().date() if len(b) else '无'}，首次获取于 "
              f"{b['first_seen_utc'].iloc[-1] if len(b) else '无'}（UTC）", "",
              "## 真实成交（MANUAL_LIVE）", ""]
    m = metrics[MANUAL]
    if m["status"] == "尚无真实交易数据":
        lines += ["**尚无真实交易数据。** 没有人工录入的券商成交记录，本报告不包含任何实盘收益。"
                  + (f"（已登记本金；无成交的委托 {m['orders_without_fill']} 个）" if m.get("capital_registered") else "（未登记本金）"), ""]
    else:
        pend = f"；交割单待录的委托 {m['fees_pending_orders']}（费用暂按 0 计，净值为临时值）" if m["fees_pending_orders"] else ""
        lines += [f"- 真实委托 {m['orders']} 个，其中有成交 {m['orders_with_fills']} 个（成交 {m['fills']} 笔），交割单显性费用合计 "
                  f"{m['explicit_fees']:.2f} 元{pend}；现金 {m['cash']:.2f} 元，持仓 {m['holding']} 份；"
                  f"累计收益 {m['cumulative_return']:+.2%}，最大回撤 {m['max_drawdown']:.2%}（真实资金，按收盘价估值）；修正 {m['corrections']} 次", "",
                  "| 券商委托编号 | 日期 | 方向 | 状态 | 成交量 | 真实 VWAP | 实收费用 | 标记 | 模拟结果 | 模拟成交价 | 模拟费用 | 价格偏差 |",
                  "|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
        for d in deviation:
            diff = d["price_diff_live_minus_paper"]
            diff_s = "—" if diff is None else f"{diff:+.3f}"
            fee_s = "待录" if d["live_fees"] is None else f"{d['live_fees']:.2f}"
            lines.append(f"| {d['broker_order_id']} | {d['trade_date']} | {d['side']} | {d['status']} | {d['filled_qty']} | {d['live_vwap']:.3f} | "
                         f"{fee_s} | {','.join(d['flags']) or '—'} | {d['paper_status']} | {d['paper_price'] or '—'} | "
                         f"{d['paper_fees'] if d['paper_fees'] is not None else '—'} | {diff_s} |")
        lines.append("")
    lines += ["## 模拟账本（PAPER，非真实成交）", "",
              "| 账本 | 交易日 | 期末权益 | 累计收益 | 最大回撤 | 成交 | 显性费用 | 价差成本（含在成交价中） | 分红 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L in PAPER_LEDGERS:
        x = metrics[L]
        lines.append(f"| {LABEL[L]} | {x['days']} | {x['final_equity']:.2f} | {x['cumulative_return']:+.2%} | "
                     f"{x['max_drawdown']:.2%} | {x['fills']} | {x['explicit_fees']:.2f} | {x['spread_cost_vs_open']:.3f} | {x['dividends']:.3f} |")
    lines += ["", "## 信号执行率", "",
              "| 账本 | 非零信号 | 有交易意图的决策 | 生成订单 | 拒绝 | 次日结果 | 订单成交率 | 拒绝原因 |",
              "|---|---:|---:|---:|---:|---|---:|---|"]
    for L in PAPER_LEDGERS:
        e = metrics[L]["execution"]
        rate = "—" if e["fill_rate_of_proposed"] is None else f"{e['fill_rate_of_proposed']:.0%}"
        lines.append(f"| {L} | {e['nonzero_signals']} | {e['decisions_with_intent']} | {e['proposed_orders']} | {e['rejected']} | "
                     f"{e['results']} | {rate} | {e['reject_reasons']} |")
    lines += ["", "说明：", "",
              "- 严格模拟只认开盘价满足限价的成交；盘中触价（UNCERTAIN_TOUCH）不计入严格账本，敏感性账本按限价计为成交；",
              "- 价差成本是成交价相对开盘价的偏离，已经包含在成交价里，不再另外计费；显性费用按佣金情景计算；",
              "- 基准与策略使用同样的成交规则、费用与预算；若预算拒绝所有买入，基准也保持现金；",
              "- 逐笔审计见 orders_audit.csv（含信号参考价、计划限价、估算费用与摩擦、实际结果、数据获取时间）。", ""]
    return "\n".join(lines)
