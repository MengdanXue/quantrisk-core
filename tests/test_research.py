"""M3 第二、三阶段：数据质量与公司行为验证、按时点股票池、退市、Walk-Forward 时间隔离与可复现。"""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quantrisk.data.ashare import DataQualityError, load_dataset, point_in_time_usable
from quantrisk.data.universe import PointInTimeUniverse
from quantrisk.research.walk_forward import (Fold, Segment, StudyConfig, WalkForwardStudy, annual_folds,
                                             check_folds)
from quantrisk.strategies import get_strategy
from quantrisk.strategies.backtest import RiskBudget
from quantrisk.strategies.backtest.portfolio_run import run_strategy_portfolio
from quantrisk.strategies.base import BaseStrategy, Signal

RESEARCH = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "research_sse50"
needs_snapshot = pytest.mark.snapshot("research_sse50")
SMALL = ["600519", "601318", "600036", "600030", "601166", "600900"]
MARKET = {"slippage": 0.0005, "commission_rate": 0.00025, "min_commission": 5.0}


def mini_snapshot(dst: Path, symbols, mutate=None) -> Path:
    """复制研究快照中的部分文件到临时目录并重新生成哈希清单；mutate(name, df) 可修改某个文件。"""
    dst.mkdir(parents=True)
    src_manifest = json.loads((RESEARCH / "manifest.json").read_text(encoding="utf-8"))
    names = ["calendar.csv", "benchmark_index_daily.csv", "delisted_sh.csv", "delisted_sz.csv"]
    names += [f"{s}_{k}.csv" for s in symbols for k in ("daily_raw", "hfq_factor", "dividends", "dividends_cninfo")]
    files = {}
    for n in names:
        if mutate is not None:
            df = pd.read_csv(RESEARCH / n, dtype=str)
            df = mutate(n, df)
            df.to_csv(dst / n, index=False, lineterminator="\n")
        else:
            shutil.copy(RESEARCH / n, dst / n)
        files[n] = {**src_manifest["files"][n], "sha256": hashlib.sha256((dst / n).read_bytes()).hexdigest()}
    cons = pd.read_csv(RESEARCH / "universe_current_constituents.csv", dtype={"品种代码": str})
    cons[cons["品种代码"].isin(symbols)].to_csv(dst / "universe_current_constituents.csv", index=False)
    files["universe_current_constituents.csv"] = {"sha256": hashlib.sha256(
        (dst / "universe_current_constituents.csv").read_bytes()).hexdigest()}
    manifest = {**src_manifest, "symbols": symbols, "files": files}
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return dst


def small_study(snapshot, test_years=(2022,)):
    cfg = StudyConfig(snapshot=str(snapshot), strategy="ma_cross",
                      param_grid={"fast_period": [5, 20], "slow_period": [30, 60]}, test_years=list(test_years),
                      top_k=2, initial_capital=5_000_000, position_pct=0.2, max_weight=0.2)
    return WalkForwardStudy(cfg, lambda p: get_strategy("ma_cross", p))


# ---------- 公司行为与数据质量 ----------

@needs_snapshot
def test_corporate_action_validation_on_real_history():
    statuses = []
    for s in SMALL:
        q = load_dataset(RESEARCH, s, "2018-01-01", "2024-12-31").quality
        statuses += [v["status"] for v in q["corporate_action_validation"]]
    assert statuses.count("VERIFIED_TWO_SOURCES") >= 30
    assert "CONFLICT" not in statuses


@needs_snapshot
def test_unrecorded_adjustment_makes_data_unusable_and_backtest_refuses():
    ds = load_dataset(RESEARCH, "600030", "2021-06-01", "2022-06-30")       # 2022-01-27 因子跳变但无分红送转记录
    assert not ds.quality["usable"]
    assert any("UNRECORDED_ADJUSTMENT" in i for i in ds.quality["issues"])
    with pytest.raises(DataQualityError):
        run_strategy_portfolio(lambda s: get_strategy("ma_cross"), [ds], 1_000_000)
    # 不含该事件的区间仍可用
    assert load_dataset(RESEARCH, "600030", "2023-01-01", "2023-12-31").quality["usable"]


@needs_snapshot
def test_duplicate_corporate_action_rows_are_not_double_counted(tmp_path):
    def dup(name, df):
        return pd.concat([df, df.iloc[[0]]]) if name == "600519_dividends_cninfo.csv" else df
    snap = mini_snapshot(tmp_path / "s", ["600519"], mutate=dup)
    base = load_dataset(RESEARCH, "600519", "2018-01-01", "2024-12-31")
    test = load_dataset(snap, "600519", "2018-01-01", "2024-12-31")
    assert test.quality["cninfo_duplicate_rows_removed"] == 1
    assert [(a.ex_date, a.cash_per_share) for a in test.corporate_actions] == \
           [(a.ex_date, a.cash_per_share) for a in base.corporate_actions]


@needs_snapshot
def test_delisted_stock_fabricated_row_is_rejected():
    with pytest.raises(DataQualityError, match="不在交易日历"):
        load_dataset(RESEARCH, "300104", "2020-01-01", "2021-12-31")      # 新浪在 2021-06-27（周日）复制出一根K线


@needs_snapshot
def test_delisted_stock_handled_point_in_time():
    ds = load_dataset(RESEARCH, "300104", "2019-06-01", "2020-12-31")
    assert ds.quality["delist_date"] == "2020-07-21"
    assert ds.quality["data_end_status"].startswith("已退市")
    uni = PointInTimeUniverse({"300104": [(pd.Timestamp("2015-01-01"), None)]},
                              {"300104": pd.Timestamp("2010-08-12")}, {"300104": pd.Timestamp("2020-07-21")},
                              seasoning_days=0, trading_calendar=ds.calendar)
    assert uni.members(pd.Timestamp("2020-01-02")) == ["300104"]          # 退市前仍在历史股票池中
    elig = uni.eligibility("300104", ds.calendar)
    assert elig.loc["2020-07-20"] and not elig.loc["2020-07-21":].any()

    class BuyOnce(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.5}

        def generate_signals(self, d):
            s = pd.Series(0, index=d.index)
            s.iloc[0] = Signal.BUY.value
            return s

    out = run_strategy_portfolio(lambda s: BuyOnce(), [ds], 1_000_000, universe=uni, market_kwargs=MARKET)
    pos = out["positions"].xs("300104", level="symbol")
    after = pos.loc["2020-07-21":]
    assert (after["status"] == "数据已结束").all()
    assert (after["close"] == ds.bars["close"].iloc[-1]).all()            # 停在最后真实收盘价，不补造价格
    assert (after["quantity"] > 0).all()                                   # 无法卖出，损失如实留在账上
    assert all(pd.Timestamp(f["date"]) <= pd.Timestamp("2020-07-20") for f in out["fills"])   # 退市后没有任何成交


@needs_snapshot
def test_leaving_the_universe_triggers_exit_order_on_a_trading_stock():
    ds = load_dataset(RESEARCH, "600519", "2023-01-01", "2023-12-31")
    leave = pd.Timestamp("2023-06-30")
    uni = PointInTimeUniverse({"600519": [(pd.Timestamp("2005-07-01"), leave)]}, {}, {}, seasoning_days=0,
                              trading_calendar=ds.calendar)

    class BuyFirst(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.5}

        def generate_signals(self, d):
            s = pd.Series(0, index=d.index)
            s.iloc[0] = Signal.BUY.value
            return s

    out = run_strategy_portfolio(lambda s: BuyFirst(), [ds], 2_000_000, universe=uni, market_kwargs=MARKET)
    exits = [o for o in out["orders"] if o["signal"] == "UNIVERSE_EXIT"]
    assert exits[0]["signal_date"] == str(leave.date()) and exits[0]["status"] == "FILLED"
    assert out["positions"].xs("600519", level="symbol").loc["2023-07-04":, "quantity"].eq(0).all()


# ---------- 按时点股票池 ----------

@needs_snapshot
def test_point_in_time_universe_uses_inclusion_and_seasoning():
    uni = PointInTimeUniverse.from_research_snapshot(RESEARCH)
    cons = pd.read_csv(RESEARCH / "universe_current_constituents.csv", dtype={"品种代码": str}).set_index("品种代码")
    sym = cons["纳入日期"].astype(str).sort_values().index[-1]               # 最近纳入的股票
    joined = pd.Timestamp(cons.loc[sym, "纳入日期"])
    cal = uni.trading_calendar
    before = cal[cal < joined][-5:]
    assert not uni.eligibility(sym, before).any()
    assert any("幸存者偏差" in x for x in uni.limitations)


# ---------- Walk-Forward：时间隔离 ----------

@needs_snapshot
def test_folds_are_ordered_non_overlapping_with_embargo():
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(RESEARCH / "calendar.csv")["date"]))
    folds = annual_folds(cal, [2021, 2022, 2023, 2024], train_years=2, embargo_days=5)
    pos = {d: i for i, d in enumerate(cal)}
    for f in folds:
        assert f.train.end < f.validation.start < f.validation.end < f.test.start
        assert pos[f.validation.start] - pos[f.train.end] > 5 and pos[f.test.start] - pos[f.validation.end] > 5
    assert [f.test.start.year for f in folds] == [2021, 2022, 2023, 2024]
    bad = [Fold(folds[0].train, folds[0].validation, Segment(folds[0].validation.end, folds[0].test.end))]
    with pytest.raises(ValueError):
        check_folds(bad, cal, 5)


@needs_snapshot
def test_segment_data_never_extends_past_segment_end():
    study = small_study(RESEARCH)
    f = study.folds[0]
    for seg in (f.train, f.validation, f.test):
        for ds in study.datasets(seg):
            assert ds.bars.index.max() <= seg.end and ds.signal_data.index.max() <= seg.end


@needs_snapshot
def test_walk_forward_selection_and_test_ignore_future_data(tmp_path):
    cut = pd.Timestamp("2023-01-01")                                      # 测试年 2022 之后

    def scramble(name, df):
        if name.endswith("_daily_raw.csv"):
            later = pd.to_datetime(df["date"]) >= cut
            for c in ("open", "high", "low", "close"):
                df.loc[later, c] = (df.loc[later, c].astype(float) * 0.5).round(2).astype(str)
        if name == "benchmark_index_daily.csv":
            later = pd.to_datetime(df["date"]) >= cut
            df.loc[later, "close"] = "1"
        return df

    a = small_study(mini_snapshot(tmp_path / "a", SMALL)).run()
    b = small_study(mini_snapshot(tmp_path / "b", SMALL, mutate=scramble)).run()
    for fa, fb in zip(a["folds"], b["folds"]):
        assert fa["chosen_params"] == fb["chosen_params"]
        assert fa["train_candidates"] == fb["train_candidates"]
        assert fa["validation_candidates"] == fb["validation_candidates"]
        assert fa["test_performance"] == fb["test_performance"]


@needs_snapshot
def test_walk_forward_is_reproducible_and_registered(tmp_path):
    snap = mini_snapshot(tmp_path / "s", SMALL)
    registry = tmp_path / "registry.jsonl"
    r1 = small_study(snap).run(registry_path=registry)
    r2 = small_study(snap).run(registry_path=registry)
    assert r1["folds"] == r2["folds"] and r1["oos_performance"] == r2["oos_performance"]
    assert r1["registry"]["prior_runs_on_same_test_windows"] == 0
    assert r2["registry"]["prior_runs_on_same_test_windows"] == 1          # 同一测试区间被再次评估会被记录
    entries = [json.loads(x) for x in registry.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["snapshot_manifest_sha256"] == hashlib.sha256((snap / "manifest.json").read_bytes()).hexdigest()
    assert entries[0]["config_hash"] == entries[1]["config_hash"]


@needs_snapshot
def test_test_segment_trades_only_inside_test_window():
    study = small_study(RESEARCH)
    f = study.folds[0]
    out = study.run_segment(f.test, {"fast_period": 5, "slow_period": 30})
    assert out["ledger"].index.min() >= f.test.start and out["ledger"].index.max() <= f.test.end
    assert all(f.test.start <= pd.Timestamp(x["date"]) <= f.test.end for x in out["fills"])


# ---------- 按时点的数据质量判定（前视修复，2026-11-09） ----------
# 600030：2022-01-14 公告配股（10 配 1.5，14.43 元），2022-01-18 股权登记，01-19 至 01-26 停牌，01-27 除权复牌。
# 601012：2019-04-03 公告配股（10 配 3，4.65 元），2019-04-08 股权登记，04-17 除权复牌。
# 两者都不在分红记录中（UNRECORDED_ADJUSTMENT）；规则只使用除权日（价格实际跳变日），不使用公告日。

RIGHTS = [("600030", "2022-01-18", "2022-01-27", "2021-06-01", "2022-12-30"),
          ("601012", "2019-04-08", "2019-04-17", "2018-10-08", "2019-12-31")]


@needs_snapshot
@pytest.mark.parametrize("sym,last_before,ex_day,start,end", RIGHTS)
def test_rights_issue_is_unknown_before_ex_day_and_excluded_from_it(sym, last_before, ex_day, start, end):
    before = load_dataset(RESEARCH, sym, start, last_before)
    assert before.quality["usable"] and before.quality["issue_events"] == []      # 除权日之前，数据里没有任何迹象
    ds = load_dataset(RESEARCH, sym, start, end)
    assert ex_day in [e["date"] for e in ds.quality["issue_events"] if e["type"] == "UNRECORDED_ADJUSTMENT"]
    m = point_in_time_usable(ds, 120)["usable"]
    assert m[:last_before].all()                                                    # 事件前（含停牌期间）可用
    assert not m[ex_day] and not m[m.index > pd.Timestamp(ex_day)].iloc[:119].any()  # 除权日起 120 个交易日不可用
    after = m[m.index > pd.Timestamp(ex_day)]
    if len(after) > 119:
        assert after.iloc[119:].all()                                               # 窗口过后恢复


@needs_snapshot
@pytest.mark.parametrize("sym,last_before,ex_day,start,end", RIGHTS)
def test_rights_issue_mask_is_truncation_invariant(sym, last_before, ex_day, start, end):
    full = point_in_time_usable(load_dataset(RESEARCH, sym, start, end), 120)["usable"]
    cal = full.index
    i = cal.get_loc(pd.Timestamp(ex_day))
    for j in (i - 1, i, i + 1, i + 30):
        part = point_in_time_usable(load_dataset(RESEARCH, sym, start, str(cal[j].date())), 120)["usable"]
        pd.testing.assert_series_equal(part, full.loc[part.index])


@needs_snapshot
@pytest.mark.parametrize("sym,ex_day,resumed", [("600309", "2018-01-26", "2018-06-04"),
                                               ("603501", "2018-08-10", "2018-09-12")])
def test_dividend_during_suspension_no_longer_excludes_stock(sym, ex_day, resumed):
    q = load_dataset(RESEARCH, sym, "2017-06-01", "2018-12-28").quality
    v = next(x for x in q["corporate_action_validation"] if x["ex_date"] == ex_day)
    assert v["status"] == "VERIFIED_BY_FACTOR_NEXT_TRADED_DAY" and v["first_traded_day"] == resumed
    assert q["usable"] and q["issue_events"] == []


def _study_2022(mode):
    cfg = StudyConfig(snapshot=str(RESEARCH), strategy="ma_cross", param_grid={"fast_period": [5], "slow_period": [30]},
                      test_years=[2022], warmup_days=120, quality_mode=mode, market=MARKET)
    return WalkForwardStudy(cfg, lambda p: get_strategy("ma_cross", p))


@needs_snapshot
def test_600030_in_2022_test_year_point_in_time():
    study = _study_2022("point_in_time")
    seg = study.folds[0].test
    assert "600030" in [d.symbol for d in study.datasets(seg)]                      # 不再从段首整段排除
    cal = study.datasets(seg)[0].calendar
    elig = study.universe_for(seg).eligibility("600030", cal)
    assert elig["2022-01-04":"2022-01-18"].all()                                    # 事件前可参与选股
    assert not elig["2022-01-27"] and not elig["2022-01-28"]                        # 除权日起不可参与
    rec = next(e for e in study.excluded if e["symbol"] == "600030" and e["segment"] == seg.as_dict())
    assert rec["mode"] == "point_in_time" and rec["unusable_from"] == "2022-01-27" and rec["event_dates"] == ["2022-01-27"]

    class BuyFirst(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.15}

        def generate_signals(self, d):
            s = pd.Series(Signal.HOLD.value, index=d.index)
            s.loc[s.index >= seg.start] = Signal.BUY.value
            return s

    out = study.run_segment(seg, {}, strategy_factory=lambda s: BuyFirst())
    orders = [o for o in out["orders"] if o["symbol"] == "600030"]
    assert orders[0]["side"] == "BUY" and orders[0]["signal_date"] < "2022-01-27"
    exit_ = next(o for o in orders if o["side"] == "SELL")
    assert exit_["signal"] == "UNIVERSE_EXIT" and exit_["signal_date"] == "2022-01-27"  # 除权日收盘判定，次日卖出
    assert not [o for o in orders if o["side"] == "BUY" and "2022-01-27" <= o["signal_date"] < str(
        cal[cal.get_loc(pd.Timestamp("2022-01-27")) + 120].date())]                  # 不可用期间不买入
    assert out["data_quality_overrides"].keys() >= {"600030"}                        # 放行记录可审计


@needs_snapshot
def test_whole_segment_mode_reproduces_pre_fix_exclusion():
    study = _study_2022("whole_segment")
    seg = study.folds[0].test
    assert "600030" not in [d.symbol for d in study.datasets(seg)]
    assert study.universe_for(seg) is study.universe
