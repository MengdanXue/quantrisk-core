import numpy as np
import pandas as pd
import pytest

from quantrisk.strategies import STRATEGY_REGISTRY, get_strategy
from quantrisk.strategies.base import BaseStrategy, Signal

from tests.conftest import random_walk

RUNNABLE = list(STRATEGY_REGISTRY)


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


@pytest.mark.parametrize("name", RUNNABLE)
def test_truncation_invariance(name):
    """只用前 k 根K线的结果必须和全量数据的前 k 根完全一致，否则说明用到了未来数据。"""
    data = random_walk(n=600, seed=5)
    full = get_strategy(name).run(data)
    for k in (200, 350, 500):
        part = get_strategy(name).run(data.iloc[:k])
        pd.testing.assert_series_equal(full.signals.iloc[:k], part.signals, check_names=False)
        np.testing.assert_allclose(full.equity_curve.iloc[:k], part.equity_curve, rtol=1e-10)


def test_truncation_check_catches_peeking_strategy():
    """反例：用居中滚动均线（center=True，常见的隐蔽前视）的策略必须被上面的检查抓到。"""

    class Peeking(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.5}

        def generate_signals(self, d):
            centered = d["close"].rolling(21, center=True, min_periods=1).mean()
            return pd.Series(np.where(d["close"] > centered, Signal.BUY.value, Signal.SELL.value), index=d.index)

    data = random_walk(n=600, seed=5)
    full, part = Peeking().run(data), Peeking().run(data.iloc[:300])
    assert not full.signals.iloc[:300].equals(part.signals)


def test_round_trip_costs():
    """一次买入加一次卖出，在价格不变时权益只损失手续费和滑点。"""
    data = random_walk(n=10, seed=0)
    data[["open", "high", "low", "close"]] = 100.0

    class OneRoundTrip(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.5}

        def generate_signals(self, d):
            s = pd.Series(Signal.HOLD.value, index=d.index)
            s.iloc[2], s.iloc[6] = Signal.BUY.value, Signal.SELL.value
            return s

    commission, slippage, capital = 0.001, 0.0005, 1_000_000
    result = OneRoundTrip().run(data, initial_capital=capital, commission=commission, slippage=slippage)
    qty = result.trades[0]["quantity"]
    buy_cost = qty * 100 * (1 + slippage) * (1 + commission)
    sell_revenue = qty * 100 * (1 - slippage) * (1 - commission)
    assert result.equity_curve.iloc[-1] == pytest.approx(capital - buy_cost + sell_revenue, rel=1e-9)
    # 持仓期间价格不变，权益不应变化
    assert result.returns.iloc[3:6].abs().max() == pytest.approx(0, abs=1e-12)
