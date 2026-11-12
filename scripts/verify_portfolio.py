"""
组合回测独立核算（不导入 quantrisk，只读快照 CSV 与回测输出）。

逐项复核：
1. 每笔成交价 = 开盘价 ×(1±滑点)，限制在当日最高/最低价与涨跌停价之间；开盘即涨停不得买入、即跌停不得卖出；
2. 每笔费用 = 佣金 max(金额×佣金率, 最低佣金) + 印花税（卖出；2023-08-28 前 0.1%，之后 0.05%）
   + 过户费（2022-04-29 前 0.002%，之后 0.001%），各自四舍五入到分；
3. 数量符合申报单位（主板/创业板 100 股整数倍；科创板 ≥200 股），卖出日期晚于该股最近一次买入日期（T+1）；
4. 逐日重建：除权日开盘前按持股入账现金红利、送转股（不足 1 股舍去），再按成交更新现金与持仓，收盘按不复权收盘价估值；
5. 每个买单在信号日：预估金额 ≤ 权益 × 单票上限；当日新买单预估金额合计 ≤ max(权益 × 总仓位上限 − 持仓市值, 0)；
   持仓因价格变化超出上限（漂移）单独统计，不计为违规——系统规则是下单时约束、不强制减仓；
   成交时再按实际成交价含费用复核：以前一交易日收盘权益和收盘价为基准，单票与总仓位剩余额度不得被突破；
6. 每日现金不为负。

公司行为取自快照中的巨潮记录（同一除权日多条合并）。假设：组合内股票 2024 年均非风险警示股。

    python scripts/verify_portfolio.py <快照目录> <回测输出目录>
"""
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd


def cents(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def limit_pct(code):
    return 0.20 if code.startswith(("300", "301", "688", "689")) else 0.10


def lot_ok(code, qty, side, holding_before):
    if code.startswith(("688", "689")):
        return qty >= 200 or (side == "SELL" and qty == holding_before)
    return qty % 100 == 0 or (side == "SELL" and qty == holding_before)


def fees(price, qty, side, rate, min_comm, day):
    """公开规则：印花税卖出 0.1%（2023-08-28 起 0.05%）；过户费 0.002%（2022-04-29 起 0.001%）。仅支持 2015-08-01 之后。"""
    day = pd.Timestamp(day)
    if day < pd.Timestamp("2015-08-01"):
        raise ValueError("核算脚本未收录 2015-08-01 之前的费率")
    amount = price * qty
    stamp = (0.0005 if day >= pd.Timestamp("2023-08-28") else 0.001) if side == "SELL" else 0.0
    transfer = 0.00001 if day >= pd.Timestamp("2022-04-29") else 0.00002
    return cents(max(amount * rate, min_comm)) + cents(amount * stamp) + cents(amount * transfer)


def load_actions(snap, code):
    path = snap / f"{code}_dividends_cninfo.csv"
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        d = pd.to_datetime(r["除权日"], errors="coerce")
        if pd.isna(d):
            continue
        num = lambda v: 0.0 if pd.isna(v) else float(v)
        cash, bonus = out.get(d, (0.0, 0.0))
        out[d] = (cash + num(r["派息比例"]) / 10, bonus + (num(r["送股比例"]) + num(r["转增比例"])) / 10)
    return out


def verify(snapshot, report, capital, slippage, rate, min_comm, max_weight, max_gross):
    snap, rep = Path(snapshot), Path(report)
    fills = pd.read_csv(rep / "fills.csv", dtype={"symbol": str})
    orders = pd.read_csv(rep / "orders.csv", dtype={"symbol": str})
    system = pd.read_csv(rep / "equity_daily.csv", parse_dates=["date"]).set_index("date")
    sys_pos = pd.read_csv(rep / "positions_daily.csv", parse_dates=["date"], dtype={"symbol": str})
    codes = sorted(fills["symbol"].unique().tolist() + [c for c in sys_pos["symbol"].unique() if c not in set(fills["symbol"])])
    codes = sorted(set(codes))
    bars = {c: pd.read_csv(snap / f"{c}_daily_raw.csv", parse_dates=["date"]).set_index("date").sort_index()
            .loc[system.index[0]:system.index[-1]] for c in codes}
    actions = {c: load_actions(snap, c) for c in codes}
    days = system.index
    problems = []

    # 涨跌停参考价（前收盘，除权日用除权参考价）
    def ref_price(c, d):
        b = bars[c]
        prev = b.index[b.index < d]
        if len(prev) == 0:
            return None
        pc = b.loc[prev[-1], "close"]
        if d in actions[c]:
            cash, bonus = actions[c][d]
            pc = cents((pc - cash) / (1 + bonus))
        return pc

    # 1–3 逐笔成交
    last_buy = {}
    holding = {c: 0 for c in codes}
    for _, f in fills.sort_values(["date", "order_id"]).iterrows():
        c, d, side, qty = f["symbol"], pd.Timestamp(f["date"]), f["side"], int(f["quantity"])
        b = bars[c].loc[d]
        ref = ref_price(c, d)
        lo, up = (cents(ref * (1 - limit_pct(c))), cents(ref * (1 + limit_pct(c)))) if ref else (None, None)
        price = b["open"] * (1 + slippage) if side == "BUY" else b["open"] * (1 - slippage)
        price = min(max(price, b["low"]), b["high"])
        if up is not None:
            price = max(min(price, up), lo)
            if side == "BUY" and b["open"] >= up:
                problems.append(f"{d.date()} {c} 开盘涨停仍买入")
            if side == "SELL" and b["open"] <= lo:
                problems.append(f"{d.date()} {c} 开盘跌停仍卖出")
        if not math.isclose(price, f["price"], rel_tol=1e-12):
            problems.append(f"{d.date()} {c} 成交价 {f['price']} ≠ 独立计算 {price}")
        if abs(fees(f["price"], qty, side, rate, min_comm, d) - f["fee_total"]) > 1e-6:
            problems.append(f"{d.date()} {c} 费用 {f['fee_total']} ≠ 独立计算 {fees(f['price'], qty, side, rate, min_comm, d)}")
        if side == "SELL" and not (d > last_buy.get(c, pd.Timestamp.min)):
            problems.append(f"{d.date()} {c} 违反 T+1")
        if side == "BUY":
            last_buy[c] = d

    # 4 逐日重建
    cash, qty, last_close = capital, {c: 0 for c in codes}, {c: None for c in codes}
    fills_by_day = {d: g for d, g in fills.groupby("date")}
    rows = []
    for d in days:
        for c in codes:
            if d in actions[c] and qty[c]:
                dcash, bonus = actions[c][d]
                cash += qty[c] * dcash
                qty[c] += int(qty[c] * bonus)
        for _, f in fills_by_day.get(str(d.date()), pd.DataFrame()).iterrows():
            if not lot_ok(f["symbol"], int(f["quantity"]), f["side"], qty[f["symbol"]]):     # 持仓已含送转股
                problems.append(f"{d.date()} {f['symbol']} 数量 {int(f['quantity'])} 不符合申报单位")
            amount = f["quantity"] * f["price"]
            fee = fees(f["price"], int(f["quantity"]), f["side"], rate, min_comm, d)
            if f["side"] == "BUY":
                cash -= amount + fee
                qty[f["symbol"]] += int(f["quantity"])
            else:
                cash += amount - fee
                qty[f["symbol"]] -= int(f["quantity"])
        for c in codes:
            if d in bars[c].index:
                last_close[c] = bars[c].loc[d, "close"]
        mv = sum(qty[c] * last_close[c] for c in codes if qty[c])
        rows.append({"date": d, "cash_total": cash, "market_value": mv, "equity": cash + mv,
                     **{f"qty_{c}": qty[c] for c in codes}})
        if cash < -1e-6:
            problems.append(f"{d.date()} 现金为负 {cash:.2f}")
    ind = pd.DataFrame(rows).set_index("date")

    sys_total_cash = system["cash"] + system["frozen"]
    equity_diff = float((ind["equity"] - system["equity"]).abs().max())
    cash_diff = float((ind["cash_total"] - sys_total_cash).abs().max())
    pos_wide = sys_pos.pivot(index="date", columns="symbol", values="quantity")
    qty_diff = max(int((ind[f"qty_{c}"] - pos_wide[c]).abs().max()) for c in codes)

    # 5 下单时的风险预算：新买单只能使用剩余额度（上限 − 现有持仓，最低为 0）；
    #   已有持仓因价格变化超出上限属于“漂移”，系统不强制减仓，单独统计、不算违规
    for d, g in orders[orders["side"] == "BUY"].groupby("signal_date"):
        day = pd.Timestamp(d)
        eq, mv = ind.loc[day, "equity"], ind.loc[day, "market_value"]
        est = g["quantity"] * g["signal_close"] * (1 + slippage)
        for (_, o), e in zip(g.iterrows(), est):
            if e > max_weight * eq + 1e-6:
                problems.append(f"{d} {o['symbol']} 下单金额 {e:.0f} 超过单票上限 {max_weight * eq:.0f}")
        allowance = max(max_gross * eq - mv, 0.0)
        if est.sum() > allowance + 1e-6:
            problems.append(f"{d} 新买单预估金额 {est.sum():.0f} 超过总仓位剩余额度 {allowance:.0f}")
    # 5b 成交时的风险预算：以前一交易日收盘权益与收盘价为基准，按实际成交价含费用检查
    prev_eq = ind["equity"].shift(1)
    prev_close = {c: bars[c]["close"].reindex(days).ffill().shift(1) for c in codes}
    for d, g in fills.groupby("date"):
        day = pd.Timestamp(d)
        if pd.isna(prev_eq.get(day)):
            continue
        eq0 = prev_eq[day]
        held = {c: ind.loc[days[days.get_loc(day) - 1], f"qty_{c}"] for c in codes}
        for _, f in g[g["side"] == "SELL"].iterrows():
            held[f["symbol"]] -= int(f["quantity"])
        held_value = sum(held[c] * prev_close[c][day] for c in codes if held[c])
        bought = 0.0
        for _, f in g[g["side"] == "BUY"].sort_values("order_id").iterrows():
            cost = f["quantity"] * f["price"] + f["fee_total"]
            own = held[f["symbol"]] * prev_close[f["symbol"]][day] if held[f["symbol"]] else 0.0
            if cost > max_weight * eq0 - own + 1e-6:
                problems.append(f"{d} {f['symbol']} 成交金额 {cost:.0f} 超过单票剩余额度 {max_weight * eq0 - own:.0f}")
            if cost > max_gross * eq0 - held_value - bought + 1e-6:
                problems.append(f"{d} {f['symbol']} 成交金额 {cost:.0f} 超过总仓位剩余额度 {max_gross * eq0 - held_value - bought:.0f}")
            bought += cost
    weights = sys_pos.pivot(index="date", columns="symbol", values="weight")
    drift = {"max_single_weight": float(weights.max().max()),
             "days_single_weight_above_limit": int((weights > max_weight + 1e-9).any(axis=1).sum()),
             "max_gross_exposure": float((ind["market_value"] / ind["equity"]).max()),
             "days_gross_above_limit": int((ind["market_value"] / ind["equity"] > max_gross + 1e-9).sum())}

    return {"days": len(days), "fills_checked": len(fills), "max_abs_equity_diff": equity_diff,
            "max_abs_cash_diff": cash_diff, "max_abs_position_diff": qty_diff,
            "final_equity_independent": float(ind["equity"].iloc[-1]),
            "final_equity_system": float(system["equity"].iloc[-1]), "drift_after_entry": drift,
            "problems": problems}


if __name__ == "__main__":
    snapshot, report = sys.argv[1], sys.argv[2]
    params = json.loads((Path(report) / "run_manifest.json").read_text(encoding="utf-8"))["params"]
    result = verify(snapshot, report, params["initial_capital"], params["slippage"], params["commission_rate"],
                    params["min_commission"], params["max_weight"], params["max_gross"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
