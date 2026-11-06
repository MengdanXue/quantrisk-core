import numpy as np
import pandas as pd
import pytest

from quantrisk.indicators.base.indicator_exceptions import MissingColumnError
from quantrisk.indicators.registry import indicator_registry


def test_registry_is_populated():
    assert indicator_registry.get_total_count() >= 380


def test_sma_matches_pandas(ohlcv):
    out = indicator_registry.create_indicator("SMA", period=20).calculate(ohlcv)
    expected = ohlcv["close"].rolling(20).mean()
    pd.testing.assert_series_equal(out.dropna(), expected.dropna(), check_names=False)


def test_ema_matches_pandas(ohlcv):
    out = indicator_registry.create_indicator("EMA").calculate(ohlcv)
    period = indicator_registry.get_metadata("EMA").default_params["period"]
    expected = ohlcv["close"].ewm(span=period, adjust=False).mean()
    np.testing.assert_allclose(out.iloc[period:].to_numpy(), expected.iloc[period:].to_numpy(), rtol=1e-6)


def test_rsi_is_bounded(ohlcv):
    out = indicator_registry.create_indicator("RSI").calculate(ohlcv).dropna()
    assert len(out) > 0
    assert out.between(0, 100).all()


def test_every_indicator_runs_or_reports_missing_input(ohlcv):
    """只给 OHLCV 时，每个指标要么算出结果，要么明确报缺少输入列。"""
    unexpected = {}
    for name in indicator_registry.list_all():
        try:
            indicator_registry.create_indicator(name).calculate(ohlcv)
        except (MissingColumnError, KeyError):
            continue
        except Exception as exc:  # pragma: no cover - 失败时打印明细
            unexpected[name] = f"{type(exc).__name__}: {exc}"
    assert not unexpected, unexpected
