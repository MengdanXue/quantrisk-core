import importlib
import logging
import os
import pkgutil
import warnings
from pathlib import Path

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


# ---------- 真实行情快照依赖 ----------
# 快照只在本地（data/ 不进 git）。缺失时跳过并说明原因；设置 QRC_REQUIRE_SNAPSHOTS=1 时缺失即失败，
# 用于在完整数据环境中证明这些测试全部执行、没有被跳过。

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "data" / "snapshots"

SNAPSHOT_DOWNLOADS = {
    "ashare_2024": "python -c \"from quantrisk.data.ashare import download_snapshot as d; "
                   "d(['600519','000001'],'2024-01-01','2024-12-31','data/snapshots/ashare_2024')\"",
    "ashare_2024_portfolio": "python -c \"from quantrisk.data.ashare import download_snapshot as d; "
                             "d(['600519','000001','601318','300750','300308','688981'],'2024-01-01','2024-12-31',"
                             "'data/snapshots/ashare_2024_portfolio')\"",
    "research_sse50": "python -c \"from quantrisk.data.ashare import download_research_snapshot as d; "
                      "d('000016',['300104'],'data/snapshots/research_sse50')\"",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "snapshot(name): 需要本地真实行情快照 data/snapshots/<name>")


def pytest_runtest_setup(item):
    for mark in item.iter_markers(name="snapshot"):
        name = mark.args[0]
        if (SNAPSHOT_ROOT / name / "manifest.json").is_file():
            continue
        reason = f"缺少本地行情快照 data/snapshots/{name}（不随仓库分发）；获取方式：{SNAPSHOT_DOWNLOADS[name]}"
        if os.environ.get("QRC_REQUIRE_SNAPSHOTS") == "1":
            pytest.fail("QRC_REQUIRE_SNAPSHOTS=1 但" + reason, pytrace=False)
        pytest.skip(reason)
