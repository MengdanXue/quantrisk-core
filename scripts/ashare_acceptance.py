"""
真实 A 股行情回测验收：从快照离线运行，输出逐日账本、成交、订单与汇总。

    python scripts/ashare_acceptance.py [快照目录] [输出目录]
    默认：data/snapshots/ashare_2024 → reports/ashare_2024
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from quantrisk.data.ashare import load_dataset  # noqa: E402
from quantrisk.strategies import get_strategy  # noqa: E402
from quantrisk.strategies.backtest.daily import AShareMarket  # noqa: E402
from quantrisk.strategies.base import BaseStrategy, Signal  # noqa: E402
from scripts.verify_buy_and_hold import independent_buy_and_hold  # noqa: E402

CAPITAL = 1_000_000


class BuyAndHold(BaseStrategy):
    """第一个交易日收盘发出买入信号，之后一直持有。"""

    def default_params(self):
        return {"position_pct": 0.95}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        s.iloc[0] = Signal.BUY.value
        return s


def main(snapshot=ROOT / "data/snapshots/ashare_2024", out=ROOT / "reports/ashare_2024"):
    snapshot, out = Path(snapshot), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"snapshot": str(snapshot), "results": []}
    for symbol in ("600519", "000001"):
        ds = load_dataset(snapshot, symbol)
        summary.setdefault("manifest", ds.manifest)
        summary.setdefault("quality", {})[symbol] = ds.quality
        runs = {"buy_and_hold": BuyAndHold()}
        runs.update({n: get_strategy(n, {"position_pct": 0.9}) for n in ("ma_cross", "turtle", "rsi_reversion")})
        for name, strategy in runs.items():
            r = strategy.run(ds.signal_data, initial_capital=CAPITAL, market=AShareMarket(symbol), bars=ds.bars,
                             calendar=ds.calendar, corporate_actions=ds.corporate_actions, symbol=symbol)
            prefix = out / f"{symbol}_{name}"
            r.ledger.to_csv(f"{prefix}_ledger.csv")
            pd.DataFrame(r.fills).to_csv(f"{prefix}_fills.csv", index=False)
            pd.DataFrame(r.orders).to_csv(f"{prefix}_orders.csv", index=False)
            pd.DataFrame(r.journal).to_csv(f"{prefix}_journal.csv", index=False)
            row = {"symbol": symbol, "strategy": name, "final_equity": round(float(r.ledger["equity"].iloc[-1]), 2),
                   "total_return": r.metrics["total_return"], "max_drawdown": r.metrics["max_drawdown"],
                   "fills": len(r.fills), "round_trips": len(r.trades),
                   "rejected_or_expired": sum(o["status"] in ("REJECTED", "EXPIRED", "PARTIAL") for o in r.orders),
                   "fees": round(sum(f["fee_total"] for f in r.fills), 2),
                   "dividends": round(sum(j["amount"] for j in r.journal if j["type"] == "DIVIDEND"), 2)}
            if name == "buy_and_hold":
                ind = independent_buy_and_hold(snapshot, symbol, capital=CAPITAL)
                row["independent_final_equity"] = round(float(ind["equity"].iloc[-1]), 2)
                row["max_abs_daily_diff_vs_independent"] = float((r.ledger["equity"] - ind["equity"]).abs().max())
            summary["results"].append(row)

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    table = pd.DataFrame(summary["results"])
    print(table.to_string(index=False))
    print(f"\n输出目录: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
