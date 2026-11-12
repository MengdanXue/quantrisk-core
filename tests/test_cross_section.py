"""横截面选股与换手约束、多重检验。除最后一个真实行情测试（需要本地研究快照）外均为合成数据。"""
import math

import numpy as np
import pandas as pd
import pytest

from quantrisk.data.ashare import AShareDataset
from quantrisk.data.universe import PointInTimeUniverse
from quantrisk.research.cross_section import (CrossSectionParams, cross_sectional_signals, factor_scores,
                                              run_cross_section)
from quantrisk.research.overfitting import (deflated_sharpe_ratio, expected_max_sharpe, pbo_cscv,
                                            per_period_sharpe, probabilistic_sharpe_ratio,
                                            sampling_sharpe_variance)
from quantrisk.strategies.backtest import RiskBudget
from tests.conftest import SNAPSHOT_ROOT

CAL = pd.bdate_range("2023-01-02", periods=260)
MARKET = {"slippage": 0.0005, "commission_rate": 0.00025, "min_commission": 5.0}


def make_ds(symbol, close, cal=CAL, drop=()):
    close = pd.Series(np.asarray(close, dtype=float), index=cal)
    open_ = close.shift(1).fillna(close.iloc[0])
    bars = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.004,
                         "low": np.minimum(open_, close) * 0.996, "close": close, "volume": 1e7})
    bars = bars.drop(index=list(drop))
    return AShareDataset(symbol, bars, bars.copy(), cal, [], {"usable": True})


def trending(n=12, seed=None, vol=0.0):
    """第 i 只股票的日漂移随 i 递增；vol>0 时叠加噪声。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = 0.0002 * i + (rng.normal(0, vol, len(CAL)) if vol else 0.0)
        out.append(make_ds(f"6000{i:02d}", 10 * np.exp(np.cumsum(np.full(len(CAL), 1.0) * r))))
    return out


def noisy(n=12, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    return [make_ds(f"6000{i:02d}", 10 * np.exp(np.cumsum(rng.normal(0, vol, len(CAL))))) for i in range(n)]


# ---------- 选股 ----------

def test_momentum_selects_top_n_by_past_return():
    p = CrossSectionParams(lookback=60, top_n=4, rebalance_days=20)
    _, log = cross_sectional_signals(trending(), p, trade_start=CAL[80])
    assert log[0]["formation"] and log[0]["target"] == ["600008", "600009", "600010", "600011"]
    assert all(r["target"] == log[0]["target"] for r in log)          # 排名不变就不换手


def test_low_vol_selects_least_volatile():
    rng = np.random.default_rng(1)
    dss = [make_ds(f"6000{i:02d}", 10 * np.exp(np.cumsum(rng.normal(0, 0.004 * (i + 1), len(CAL)))))
           for i in range(8)]
    _, log = cross_sectional_signals(dss, CrossSectionParams(factor="low_vol", lookback=60, top_n=3),
                                     trade_start=CAL[100])
    assert log[0]["target"] == ["600000", "600001", "600002"]


def test_scores_and_selection_ignore_future_data():
    dss = noisy(seed=3)
    p = CrossSectionParams(lookback=40, top_n=4, rebalance_days=10, buffer=1, max_turnover=0.25)
    cut = CAL[150]
    sig_a, log_a = cross_sectional_signals(dss, p, trade_start=CAL[60])
    changed, rng = [], np.random.default_rng(99)
    for ds in dss:
        b = ds.bars.copy()
        k = (b.index > cut).sum()
        b.loc[b.index > cut, ["open", "high", "low", "close"]] *= np.exp(np.cumsum(rng.normal(0, 0.05, k)))[:, None]
        changed.append(AShareDataset(ds.symbol, b, b.copy(), CAL, [], {"usable": True}))
    sig_b, log_b = cross_sectional_signals(changed, p, trade_start=CAL[60])
    assert [r for r in log_a if r["date"] <= str(cut.date())] == [r for r in log_b if r["date"] <= str(cut.date())]
    for s in sig_a:
        pd.testing.assert_series_equal(sig_a[s][:cut], sig_b[s][:cut])
    assert log_a != log_b                                              # 改动确实影响了之后的选择


def test_factor_value_on_day_t_uses_only_data_up_to_t():
    close = pd.DataFrame({"a": np.arange(1.0, 101.0)}, index=CAL[:100])
    s = factor_scores(close, CrossSectionParams(lookback=10, skip=5))
    assert s["a"].iloc[50] == pytest.approx(close["a"].iloc[45] / close["a"].iloc[35] - 1)
    assert s["a"].iloc[:15].isna().all()


# ---------- 换手约束 ----------

def _voluntary(log):
    return [len(r["dropped_voluntary"]) for r in log if not r["formation"]]


def test_buffer_reduces_voluntary_turnover():
    dss = noisy(seed=5)
    base = CrossSectionParams(lookback=20, top_n=4, rebalance_days=5)
    _, log0 = cross_sectional_signals(dss, base, trade_start=CAL[40])
    _, log3 = cross_sectional_signals(dss, CrossSectionParams(**{**base.__dict__, "buffer": 3}), trade_start=CAL[40])
    assert sum(_voluntary(log3)) < sum(_voluntary(log0))


def test_max_turnover_caps_voluntary_drops_and_defers_the_rest():
    dss = noisy(seed=5)
    p = CrossSectionParams(lookback=20, top_n=4, rebalance_days=5, max_turnover=0.25)     # 每次最多主动换出 1 只
    _, log = cross_sectional_signals(dss, p, trade_start=CAL[40])
    assert max(_voluntary(log)) == 1
    assert sum(len(r["deferred_by_turnover_limit"]) for r in log) > 0
    for r in log:
        # 延后换出的是排名较好的那些：被换出的排名不优于任何被延后的
        if r["dropped_voluntary"] and r["deferred_by_turnover_limit"]:
            assert min(r["ranks"][s] for s in r["dropped_voluntary"]) >= max(r["ranks"][s] for s in r["deferred_by_turnover_limit"])
    _, zero = cross_sectional_signals(dss, CrossSectionParams(**{**p.__dict__, "max_turnover": 0.0}), trade_start=CAL[40])
    assert all(r["target"] == zero[0]["target"] for r in zero)        # 上限为 0：建仓后不再主动换手


def test_forced_exit_bypasses_turnover_limit_and_is_recorded():
    dss = trending()
    leave = CAL[120]
    members = {ds.symbol: [(CAL[0], None)] for ds in dss}
    members["600011"] = [(CAL[0], leave)]                              # 排名第一的股票在 leave 调出股票池
    uni = PointInTimeUniverse(members, seasoning_days=0, trading_calendar=CAL)
    p = CrossSectionParams(lookback=60, top_n=4, rebalance_days=20, max_turnover=0.0)
    out = run_cross_section(dss, p, 1_000_000, RiskBudget(0.3, 0.95), MARKET, universe=uni, trade_start=CAL[80])
    log = out["rebalance_log"]
    after = next(r for r in log if r["date"] >= str(leave.date()))
    assert after["dropped_forced"] == ["600011"] and after["dropped_voluntary"] == []
    assert "600007" in after["added"]                                  # 空出的位置由下一名补上
    exits = [o for o in out["orders"] if o["symbol"] == "600011" and o["side"] == "SELL"]
    assert exits and exits[0]["signal"] == "UNIVERSE_EXIT"
    audit = out["turnover_audit"]
    assert audit["forced_drops_total"] == 1 and audit["voluntary_drops_total"] == 0 and audit["limit_violations"] == []


def test_suspended_member_on_rebalance_day_is_kept_not_dropped():
    dss = trending()
    p = CrossSectionParams(lookback=60, top_n=4, rebalance_days=20)
    _, base = cross_sectional_signals(dss, p, trade_start=CAL[80])
    day = pd.Timestamp(base[1]["date"])
    dss[11] = make_ds("600011", dss[11].bars["close"].reindex(CAL).values, drop=[day])
    _, log = cross_sectional_signals(dss, p, trade_start=CAL[80])
    r = next(x for x in log if x["date"] == str(day.date()))
    assert r["kept_without_rank"] == ["600011"] and "600011" in r["target"] and r["dropped_voluntary"] == []


# ---------- 与组合引擎的整合 ----------

def test_cross_section_runs_through_portfolio_engine_with_consistent_ledger():
    dss = noisy(seed=7, vol=0.015)
    p = CrossSectionParams(lookback=30, top_n=4, rebalance_days=10, buffer=1, max_turnover=0.5)
    budget = RiskBudget(max_weight=0.26, max_gross=0.95)
    out = run_cross_section(dss, p, 2_000_000, budget, MARKET, trade_start=CAL[40])
    led = out["ledger"]
    np.testing.assert_allclose(led["cash"] + led["frozen"] + led["market_value"], led["equity"], rtol=0, atol=1e-6)
    assert (led["cash"] >= -1e-9).all()
    audit = out["turnover_audit"]
    assert audit["limit_violations"] == [] and audit["unexplained_sell_orders"] == []
    assert audit["voluntary_drops_max"] <= 2
    targets = {r["date"]: set(r["target"]) for r in out["rebalance_log"]}
    for o in out["orders"]:                                            # 买单只会落在当时的目标内
        if o["side"] == "BUY":
            current = targets[max(d for d in targets if d <= o["signal_date"])]
            assert o["symbol"] in current
    assert out["metrics"]["fills"] > 0
    for f in out["fills"]:                                             # 引擎的成交规则照常生效
        assert f["quantity"] % 100 == 0


def test_cross_section_respects_hard_risk_budget():
    dss = trending()
    p = CrossSectionParams(lookback=60, top_n=2, rebalance_days=20)
    out = run_cross_section(dss, p, 1_000_000, RiskBudget(max_weight=0.2, max_gross=0.95), MARKET,
                            trade_start=CAL[80], gross_target=0.95)    # 目标 47.5%/只，被单票上限 20% 截断
    first_buys = [o for o in out["orders"] if o["side"] == "BUY"][:2]
    assert all("单票上限" in o["sizing_note"].split("：")[0] for o in first_buys)
    assert out["positions"]["weight"].groupby(level="date").max().iloc[1] <= 0.2 + 1e-9


def test_param_validation():
    with pytest.raises(ValueError):
        CrossSectionParams(factor="value")
    with pytest.raises(ValueError):
        CrossSectionParams(max_turnover=1.5)
    assert CrossSectionParams(top_n=10, max_turnover=0.25).max_voluntary_drops == 2


# ---------- 多重检验 ----------

def test_psr_matches_closed_form_for_normal_returns():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 2000)
    sr = per_period_sharpe(r)
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    z = sr * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5 * (1 + math.erf(z / math.sqrt(2))), rel=1e-12)


def test_psr_is_calibrated_and_sampling_variance_matches_simulation():
    sims = np.random.default_rng(0).normal(0, 0.01, (3000, 300))       # 真实夏普为 0
    psr = np.array([probabilistic_sharpe_ratio(x) for x in sims])
    assert 0.04 < (psr > 0.95).mean() < 0.065                           # 误报率约 5%
    sr = sims.mean(axis=1) / sims.std(axis=1, ddof=1)
    assert sr.var(ddof=1) == pytest.approx(sampling_sharpe_variance(300), rel=0.1)
    maxes = [sr[i:i + 30].max() for i in range(0, 3000, 30)]              # 100 组、每组 30 次独立试验
    assert np.mean(maxes) == pytest.approx(expected_max_sharpe(30, sr.var(ddof=1)), rel=0.1)


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.001) for n in (1, 2, 10, 100, 1000)]
    assert vals[0] == 0 and all(a < b for a, b in zip(vals, vals[1:]))


def test_best_of_many_noise_strategies_looks_significant_but_deflates():
    m = np.random.default_rng(0).normal(0, 0.01, (1000, 50))
    srs = [per_period_sharpe(m[:, j]) for j in range(50)]
    best = m[:, int(np.argmax(srs))]
    d = deflated_sharpe_ratio(best, srs)
    assert d["psr_vs_zero"] > 0.95          # 不考虑试验次数时“显著”
    assert d["dsr"] < 0.6                   # 扣除 50 次试验后不显著


def test_genuine_edge_survives_deflation():
    m = np.random.default_rng(0).normal(0, 0.01, (1000, 50))
    m[:, 0] += 0.002
    srs = [per_period_sharpe(m[:, j]) for j in range(50)]
    assert deflated_sharpe_ratio(m[:, 0], srs)["dsr"] > 0.99


def test_correlated_trials_understate_luck_unless_sampling_variance_is_used():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, 1000)
    m = common[:, None] + rng.normal(0, 0.002, (1000, 50))          # 50 个高度相关、真实夏普为 0 的配置
    srs = [per_period_sharpe(m[:, j]) for j in range(50)]
    best = m[:, int(np.argmax(srs))]
    cross = deflated_sharpe_ratio(best, srs)
    conservative = deflated_sharpe_ratio(best, srs, sr_variance=sampling_sharpe_variance(1000))
    assert cross["trial_sharpe_variance"] < sampling_sharpe_variance(1000) / 5
    assert conservative["expected_max_sharpe_per_period"] > cross["expected_max_sharpe_per_period"]
    assert conservative["dsr"] < cross["dsr"]


def test_pbo_is_near_half_for_noise_and_near_zero_for_a_real_edge():
    pbos = [pbo_cscv(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (800, 30))), n_blocks=10)["pbo"]
            for s in range(6)]
    assert 0.3 < np.mean(pbos) < 0.7
    m = np.random.default_rng(0).normal(0, 0.01, (800, 30))
    m[:, 0] += 0.002
    assert pbo_cscv(pd.DataFrame(m), n_blocks=10)["pbo"] < 0.05


def test_pbo_input_validation():
    with pytest.raises(ValueError):
        pbo_cscv(pd.DataFrame(np.zeros((100, 1))))
    with pytest.raises(ValueError):
        pbo_cscv(pd.DataFrame(np.zeros((100, 3))), n_blocks=7)
    bad = np.zeros((100, 3))
    bad[5, 1] = np.nan
    with pytest.raises(ValueError):
        pbo_cscv(pd.DataFrame(bad))


# ---------- 真实行情（需要本地研究快照） ----------

RESEARCH = SNAPSHOT_ROOT / "research_sse50"


@pytest.mark.snapshot("research_sse50")
def test_real_cross_section_reconciles_and_respects_turnover_limit(tmp_path):
    from quantrisk.data.ashare import load_dataset
    from scripts.verify_portfolio import verify
    uni = PointInTimeUniverse.from_research_snapshot(RESEARCH)
    dss = []
    for s in uni.all_symbols():
        try:
            ds = load_dataset(RESEARCH, s, "2022-01-01", "2023-12-31")
        except Exception:
            continue
        if ds.quality["usable"] and not ds.bars.empty and ds.bars.index[-1] >= pd.Timestamp("2023-01-03"):
            dss.append(ds)
    p = CrossSectionParams(lookback=120, top_n=8, rebalance_days=21, buffer=2, max_turnover=0.25)
    budget = RiskBudget(max_weight=0.12, max_gross=0.95)
    out = run_cross_section(dss, p, 10_000_000, budget, MARKET, universe=uni, trade_start="2023-01-03")
    audit = out["turnover_audit"]
    assert audit["limit_violations"] == [] and audit["unexplained_sell_orders"] == []
    assert audit["voluntary_drops_max"] <= 2 and audit["rebalances"] == 12
    out["ledger"].to_csv(tmp_path / "equity_daily.csv")
    out["positions"].to_csv(tmp_path / "positions_daily.csv")
    pd.DataFrame(out["orders"]).to_csv(tmp_path / "orders.csv", index=False)
    pd.DataFrame(out["fills"]).drop(columns=["fees"]).to_csv(tmp_path / "fills.csv", index=False)
    (tmp_path / "run_manifest.json").write_text('{"params": {"initial_capital": 10000000}}', encoding="utf-8")
    v = verify(RESEARCH, tmp_path, 10_000_000, MARKET["slippage"], MARKET["commission_rate"], MARKET["min_commission"],
               0.12, 0.95)
    assert v["problems"] == [] and v["max_abs_equity_diff"] < 1e-6 and v["fills_checked"] == len(out["fills"])
