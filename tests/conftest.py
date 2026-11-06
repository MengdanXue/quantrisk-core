import importlib
import logging
import pkgutil
import warnings

import numpy as np
import pandas as pd
import pytest

import quantrisk.indicators

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

for _mod in pkgutil.walk_packages(quantrisk.indicators.__path__, "quantrisk.indicators."):
    importlib.import_module(_mod.name)


def random_walk(n=500, seed=0, drift=0.0, vol=0.012):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[100.0, close[:-1]]
    spread = np.abs(rng.normal(0, vol, n)) * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


@pytest.fixture
def ohlcv():
    return random_walk()
