"""对比修复前后的回测收益计算，生成 docs/images/lookahead-before-after.png。

修复前的逻辑在 buggy_equity() 里原样复现：成交当根K线用成交后的持仓回溯上一根收盘价计算收益。

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


def buggy_equity(strategy, data, initial_capital=1_000_000, commission=0.001, slippage=0.0005):
    """原样复现修复前 BaseStrategy._simulate_trading 的收益计算（仅用于对比）：
    当根收盘成交，并用成交后的持仓回溯上一根收盘价计算当根收益。"""
    import pandas as pd

    signals = strategy.generate_signals(strategy.preprocess_data(data.copy()))
    returns = pd.Series(0.0, index=data.index)
    cash, position = initial_capital, 0.0
    for i, (date, row) in enumerate(data.iterrows()):
        price, signal = row["close"], signals.get(date, 0)
        if signal > 0 and position == 0:
            exec_price = price * (1 + slippage)
            size = strategy.calculate_position_size(Signal.BUY, exec_price, cash)
            cost = size * exec_price * (1 + commission)
            if cost <= cash and size > 0:
                cash, position = cash - cost, size
        elif signal < 0 and position > 0:
            cash += position * price * (1 - slippage) * (1 - commission)
            position = 0
        equity = cash + position * price
        if i > 0:
            prev_equity = cash + position * data.iloc[i - 1]["close"]  # 前视偏差所在
            returns.iloc[i] = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
    return (1 + returns).cumprod() * initial_capital


data = random_walk(n=750, seed=3)
params = {"position_pct": 0.95}
before_curve = buggy_equity(Base(params=params), data)
before_ret = before_curve.iloc[-1] / before_curve.iloc[0] - 1
after = Base(params=params).run(data)
hold = data["close"] / data["close"].iloc[0] * 1_000_000

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
ax.plot(hold.index, hold / 1e6, color="#9aa5b1", lw=1.2, label="买入持有（随机游走）")
ax.plot(before_curve.index, before_curve / 1e6, color="#dc2626", lw=1.6,
        label=f"修复前（当根收盘成交 + 回溯计算收益）：总收益 {before_ret:+.0%}")
ax.plot(after.equity_curve.index, after.equity_curve / 1e6, color="#2563eb", lw=1.6,
        label=f"现在（次日开盘成交，逐日账本）：总收益 {after.metrics['total_return']:+.0%}，夏普 {after.metrics['sharpe_ratio']:.1f}")
ax.set_title("同一策略、同一份随机游走数据：修复前后的权益曲线（波动率突破策略，95% 仓位）")
ax.set_ylabel("权益（百万）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "images" / "lookahead-before-after.png"
fig.savefig(out)
print(out, round(before_ret, 4), after.metrics["total_return"])
