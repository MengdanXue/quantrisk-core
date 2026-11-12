"""
独立核算：真实行情上的“买入并持有”。

不导入 quantrisk 的任何代码，只用 pandas 读取快照 CSV，按以下公开规则从头计算每日权益，
再与回测系统的逐日账本比较：
- 第 1 个交易日收盘发出买入信号；按当日收盘价 ×(1+滑点) 估算，取不超过 资金×仓位比例
  且含费用不超过可用资金的最大 100 股整数倍；
- 第 2 个交易日开盘成交：成交价 = 开盘价 ×(1+滑点)，并限制在当日最高/最低价之间；
- 佣金 max(成交额×佣金率, 最低佣金)，过户费 成交额×0.001%（2022-04-29 起），各自四舍五入到分；
- 除息日开盘前按持股数入账税前现金红利（每 10 股派息 ÷ 10）；
- 每日权益 = 现金 + 持股数 × 当日收盘价。

    python scripts/verify_buy_and_hold.py data/snapshots/ashare_2024 600519
"""
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd


def cents(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def buy_fees(amount, commission_rate, min_commission):
    return cents(max(amount * commission_rate, min_commission)) + cents(amount * 0.00001)


def independent_buy_and_hold(snapshot, symbol, capital=1_000_000.0, pct=0.95, slippage=0.0005,
                             commission_rate=0.00025, min_commission=5.0, start="2024-01-01", end="2024-12-31"):
    snap = Path(snapshot)
    bars = pd.read_csv(snap / f"{symbol}_daily_raw.csv", parse_dates=["date"]).set_index("date").sort_index()
    bars = bars.loc[start:end]
    div = pd.read_csv(snap / f"{symbol}_dividends.csv")
    div["除权除息日"] = pd.to_datetime(div["除权除息日"], errors="coerce")
    div = div[(div["进度"] == "实施") & div["除权除息日"].between(pd.Timestamp(start), pd.Timestamp(end))]
    cash_div = {row["除权除息日"]: row["派息"] / 10 for _, row in div.iterrows()}

    d0, d1 = bars.index[0], bars.index[1]
    est = bars.loc[d0, "close"] * (1 + slippage)
    qty = int(capital * pct / est // 100 * 100)
    while qty > 0 and qty * est + buy_fees(qty * est, commission_rate, min_commission) > capital:
        qty -= 100
    bar = bars.loc[d1]
    price = min(max(bar["open"] * (1 + slippage), bar["low"]), bar["high"])
    while qty > 0 and qty * price + buy_fees(qty * price, commission_rate, min_commission) > capital:
        qty -= 100
    fees = buy_fees(qty * price, commission_rate, min_commission)

    cash, held, rows = capital, 0, []
    for d, b in bars.iterrows():
        if d in cash_div and held:
            cash += held * cash_div[d]
        if d == d1:
            cash -= qty * price + fees
            held = qty
        rows.append({"date": d, "cash": cash, "position": held, "equity": cash + held * b["close"]})
    out = pd.DataFrame(rows).set_index("date")
    out.attrs.update(quantity=qty, fill_price=price, fees=fees, fill_date=d1)
    return out


if __name__ == "__main__":
    snapshot, symbol = sys.argv[1], sys.argv[2]
    res = independent_buy_and_hold(snapshot, symbol)
    a = res.attrs
    print(f"{symbol}: {a['fill_date'].date()} 买入 {a['quantity']} 股 @ {a['fill_price']:.4f}，费用 {a['fees']:.2f}")
    print(f"期末权益 {res['equity'].iloc[-1]:.2f}，总收益 {res['equity'].iloc[-1] / 1_000_000 - 1:+.4%}")
