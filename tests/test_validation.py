"""指标数值对照：与 TA-Lib 一致的项直接比对；口径不同的项对照其声明公式的独立实现。"""
import pandas as pd
import pytest

from quantrisk.indicators.registry import indicator_registry
from tests.conftest import random_walk
from tests.talib_cases import CASES, CONVENTION_CASES, compare

SEEDS = (0, 1, 2)
TOL = 1e-9


def _ours(name, col, data):
    out = indicator_registry.create_indicator(name).calculate(data)
    if col:
        return out[col]
    return out if isinstance(out, pd.Series) else out.iloc[:, 0]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name,col,ref,params", CASES, ids=[f"{c[0]}.{c[1] or ''}" for c in CASES])
def test_matches_talib(name, col, ref, params, seed):
    pytest.importorskip("talib")
    data = random_walk(n=1500, seed=seed)
    err, points = compare(_ours(name, col, data), ref(data), tail=500)
    assert points >= 450
    assert err < TOL, f"max relative error {err:.2e}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name,col,ref,note", CONVENTION_CASES, ids=[f"{c[0]}.{c[1] or ''}" for c in CONVENTION_CASES])
def test_matches_declared_convention(name, col, ref, note, seed):
    data = random_walk(n=1500, seed=seed)
    err, points = compare(_ours(name, col, data), ref(data), tail=500)
    assert points >= 450
    assert err < TOL, f"{note}: max relative error {err:.2e}"
