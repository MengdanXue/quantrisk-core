"""对比修复前后的回测收益计算，生成 docs/images/lookahead-before-after.png。

修复前的逻辑在 _BuggyReturns 里原样复现：成交当根K线用成交后的持仓回溯上一根收盘价计算收益。

    python scripts/lookahead_comparison.py [字体文件路径]
"""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantrisk.strategies import STRATEGY_REGISTRY
from quantrisk.strategies.base import Signal
from tests.conftest import random_walk

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

if len(sys.argv) > 1:
    font_manager.fontManager.addfont(sys.argv[1])
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=sys.argv[1]).get_name()
plt.rcParams["axes.unicode_minus"] = False

Base = STRATEGY_REGISTRY["volatility_breakout"]


class _BuggyReturns(Base):
    """修复前的收益计算（仅用于对比）。"""

    def _simulate_trading(self, data, signals, initial_capital, commission, slippage):
        import pandas as pd

        positions = pd.Series(0.0, index=data.index)
        returns = pd.Series(0.0, index=data.index)
        cash, position, entry_price = initial_capital, 0.0, 0.0
        for i, (date, row) in enumerate(data.iterrows()):
            price, signal = row["close"], signals.get(date, 0)
            if signal > 0 and position == 0:
                exec_price = price * (1 + slippage)
                size = self.calculate_position_size(Signal.BUY, exec_price, cash)
                cost = size * exec_price * (1 + commission)
                if cost <= cash and size > 0:
                    cash, position, entry_price = cash - cost, size, exec_price
            elif signal < 0 and position > 0:
                cash += position * price * (1 - slippage) * (1 - commission)
                position = 0
            equity = cash + position * price
            positions.iloc[i] = position
            if i > 0:
                prev_equity = cash + position * data.iloc[i - 1]["close"]  # 前视偏差所在
                returns.iloc[i] = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        return positions, returns, []


data = random_walk(n=750, seed=3)
params = {"position_pct": 0.95}
before = _BuggyReturns(params=params).run(data)
after = Base(params=params).run(data)
hold = data["close"] / data["close"].iloc[0] * 1_000_000

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
ax.plot(hold.index, hold / 1e6, color="#9aa5b1", lw=1.2, label="买入持有（随机游走）")
ax.plot(before.equity_curve.index, before.equity_curve / 1e6, color="#dc2626", lw=1.6,
        label=f"修复前：总收益 {before.metrics['total_return']:+.0%}，夏普 {before.metrics['sharpe_ratio']:.1f}")
ax.plot(after.equity_curve.index, after.equity_curve / 1e6, color="#2563eb", lw=1.6,
        label=f"修复后：总收益 {after.metrics['total_return']:+.0%}，夏普 {after.metrics['sharpe_ratio']:.1f}")
ax.set_title("同一策略、同一份随机游走数据：修复前后的权益曲线（波动率突破策略，95% 仓位）")
ax.set_ylabel("权益（百万）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "images" / "lookahead-before-after.png"
fig.savefig(out)
print(out, before.metrics["total_return"], after.metrics["total_return"])
