"""
M2 组合回测验收：真实 A 股 2024 行情（6 只股票，共享现金账户），离线读取快照，输出审计报告并独立核算。

    python scripts/portfolio_acceptance.py
    输出：reports/portfolio_2024/<运行名>/ 以及 reports/portfolio_2024/verification.json

注意：两组运行用于验证回测的正确性与约束有效性，不构成任何策略有效性的证据。
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
from quantrisk.strategies.backtest.daily import RiskBudget  # noqa: E402
from quantrisk.strategies.backtest.portfolio_run import run_strategy_portfolio, write_report  # noqa: E402
from quantrisk.strategies.base import BaseStrategy, Signal  # noqa: E402
from scripts.verify_portfolio import verify  # noqa: E402

SNAPSHOT = ROOT / "data/snapshots/ashare_2024_portfolio"
OUT = ROOT / "reports/portfolio_2024"
SYMBOLS = ["600519", "000001", "601318", "300750", "300308", "688981"]
MARKET = {"slippage": 0.0005, "commission_rate": 0.00025, "min_commission": 5.0}
CAPITAL = 2_000_000


class BuyFirstDay(BaseStrategy):
    """第一个交易日收盘发出买入信号，之后一直持有。"""

    def default_params(self):
        return {"position_pct": 0.16}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        s.iloc[0] = Signal.BUY.value
        return s


RUNS = {
    "equal_weight_buy_and_hold": dict(factory=lambda s: BuyFirstDay(), strategy="BuyFirstDay", strategy_params={"position_pct": 0.16},
                                      budget=RiskBudget(max_weight=0.2, max_gross=0.95)),
    "ma_cross_budgeted": dict(factory=lambda s: get_strategy("ma_cross", {"position_pct": 0.3}), strategy="ma_cross",
                              strategy_params={"position_pct": 0.3}, budget=RiskBudget(max_weight=0.2, max_gross=0.6)),
    "ma_cross_budgeted_trim": dict(factory=lambda s: get_strategy("ma_cross", {"position_pct": 0.3}), strategy="ma_cross",
                                   strategy_params={"position_pct": 0.3},
                                   budget=RiskBudget(max_weight=0.2, max_gross=0.6, trim_to_limits=True)),
    "turtle_budgeted": dict(factory=lambda s: get_strategy("turtle", {"position_pct": 0.5}), strategy="turtle",
                            strategy_params={"position_pct": 0.5}, budget=RiskBudget(max_weight=0.25, max_gross=0.7)),
}


def main():
    datasets = [load_dataset(SNAPSHOT, s) for s in SYMBOLS]
    verification = {}
    for name, cfg in RUNS.items():
        out = run_strategy_portfolio(cfg["factory"], datasets, CAPITAL, budget=cfg["budget"], market_kwargs=MARKET)
        params = {"initial_capital": CAPITAL, "symbols": SYMBOLS, "strategy": cfg["strategy"],
                  "strategy_params": cfg["strategy_params"], "max_weight": cfg["budget"].max_weight,
                  "max_gross": cfg["budget"].max_gross, "trim_to_limits": cfg["budget"].trim_to_limits, "snapshot": str(SNAPSHOT.relative_to(ROOT)), **MARKET}
        d = write_report(out, OUT / name, params, datasets[0].manifest, "python scripts/portfolio_acceptance.py")
        v = verify(SNAPSHOT, d, CAPITAL, MARKET["slippage"], MARKET["commission_rate"], MARKET["min_commission"],
                   cfg["budget"].max_weight, cfg["budget"].max_gross)
        verification[name] = v
        ev = pd.DataFrame(out["risk_events"])
        pd.DataFrame(out["risk_log"]).to_csv(OUT / name / "risk_log.csv", index=False)
        ev.to_csv(OUT / name / "risk_events.csv", index=False)
        m = out["metrics"]
        print(f"{name:28s} 期末 {m['final_equity']:>14,.2f} 收益 {m['total_return']:+.2%} 回撤 {m['max_drawdown']:.2%} "
              f"换手 {m['turnover']:.2f} 平均仓位 {m['gross_exposure_mean']:.1%} 最大单票 {m['max_single_weight']:.1%} "
              f"成交 {m['fills']} 未全部成交 {m['orders_not_fully_filled']} | 独立核算权益差 {v['max_abs_equity_diff']:.2e} "
              f"持仓差 {v['max_abs_position_diff']} 问题 {len(v['problems'])} "
              f"单票超限天数 {v['drift_after_entry']['days_single_weight_above_limit']} "
              f"总仓位超限天数 {v['drift_after_entry']['days_gross_above_limit']} | 风险日志 {len(out['risk_log'])} "
              f"被动超限事件 {len(ev)}" + (f"（严重 {int((ev['severity'] == '严重').sum())}）" if len(ev) else ""))
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出目录: {OUT}")


if __name__ == "__main__":
    main()
