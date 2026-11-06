import numpy as np
import pandas as pd
import pytest

from quantrisk.strategies import STRATEGY_REGISTRY, get_strategy
from quantrisk.strategies.base import BaseStrategy, Signal

from tests.conftest import random_walk

RUNNABLE = [name for name in STRATEGY_REGISTRY if name != "advanced_pairs"]


@pytest.mark.parametrize("name", RUNNABLE)
def test_strategy_runs(name, ohlcv):
    result = get_strategy(name).run(ohlcv)
    assert len(result.equity_curve) == len(ohlcv)
    assert np.isfinite(result.equity_curve).all()
    assert {"total_return", "sharpe_ratio", "max_drawdown"} <= set(result.metrics)


class _SameBarOracle(BaseStrategy):
    """偷看当根K线涨跌、并在当根收盘成交的策略：在收盘成交时它不应该占到任何便宜。"""

    def default_params(self):
        return {"position_pct": 1.0}

    def generate_signals(self, data):
        up = data["close"] > data["close"].shift(1)
        return pd.Series(np.where(up, Signal.BUY.value, Signal.SELL.value), index=data.index)


@pytest.mark.parametrize("seed", range(5))
def test_no_lookahead_when_trading_at_close(seed):
    data = random_walk(n=800, seed=seed)
    result = _SameBarOracle().run(data, commission=0.0, slippage=0.0)
    # 收盘才成交，不能吃到当根K线的涨幅；修复前这里会有数倍收益
    assert result.metrics["total_return"] < 0.5


def test_equity_tracks_buy_and_hold_when_always_long():
    data = random_walk(n=300, seed=7)

    class AlwaysLong(BaseStrategy):
        def default_params(self):
            return {"position_pct": 1.0}

        def generate_signals(self, d):
            return pd.Series(Signal.BUY.value, index=d.index)

    result = AlwaysLong().run(data, commission=0.0, slippage=0.0)
    position = result.positions.iloc[-1]
    expected = 1_000_000 - position * data["close"].iloc[0] + position * data["close"].iloc[-1]
    assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9)
