"""按时点的数据质量判定：事件之前可用、事件当日起不可用、污染窗口后恢复；停牌期间的分红不误报（合成快照）。"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quantrisk.data.ashare import CNINFO_COLUMNS, load_dataset, point_in_time_usable

CAL = pd.bdate_range("2023-01-02", periods=200)
SYM = "600000"


def write_snapshot(path, closes, hfq, dividends=(), drop=()):
    """closes：每个交易日的不复权收盘价；hfq：{日期: 后复权因子}；dividends：[(除权日, 每股现金)]；drop：无行情的交易日。"""
    path.mkdir(parents=True)
    close = pd.Series(closes, index=CAL)
    open_ = close.shift(1).fillna(close.iloc[0])
    bars = pd.DataFrame({"date": CAL, "open": open_.values, "high": np.maximum(open_, close).values * 1.002,
                         "low": np.minimum(open_, close).values * 0.998, "close": close.values,
                         "volume": 1e7, "amount": 1e8})
    bars = bars[~bars["date"].isin(list(drop))]
    files = {
        "calendar.csv": pd.DataFrame({"date": CAL.strftime("%Y-%m-%d")}),
        f"{SYM}_daily_raw.csv": bars.assign(date=bars["date"].dt.strftime("%Y-%m-%d")),
        f"{SYM}_hfq_factor.csv": pd.DataFrame({"date": [d for d in hfq], "hfq_factor": list(hfq.values())}),
        f"{SYM}_dividends.csv": pd.DataFrame(
            [{"公告日期": "2023-01-02", "送股": 0, "转增": 0, "派息": c * 10, "进度": "实施", "除权除息日": d}
             for d, c in dividends], columns=["公告日期", "送股", "转增", "派息", "进度", "除权除息日"]),
        f"{SYM}_dividends_cninfo.csv": pd.DataFrame(
            [{"实施方案公告日期": "2023-01-02", "派息比例": c * 10, "除权日": d} for d, c in dividends],
            columns=CNINFO_COLUMNS),
    }
    meta = {}
    for name, df in files.items():
        df.to_csv(path / name, index=False, lineterminator="\n")
        meta[name] = {"sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
                      **({"note": "完整历史（自上市起）"} if name.endswith("_daily_raw.csv") else {})}
    (path / "manifest.json").write_text(json.dumps({"source": "synthetic test", "akshare_version": "-",
                                                    "fetched_at": "-", "start": None, "end": None, "files": meta}),
                                        encoding="utf-8")
    return path


def flat(k_event=None, factor=1.08):
    """收盘价恒为 10；k_event 处出现一次未记录的复权跳变（如配股：价格按 factor 下跳，后复权因子同步上跳）。"""
    closes = np.full(len(CAL), 10.0)
    hfq = {"2023-01-02": 1.0}
    if k_event is not None:
        closes[k_event:] = 10.0 / factor
        hfq[str(CAL[k_event].date())] = factor
    return closes, hfq


K, W = 100, 30
DAY = str(CAL[K].date())


def test_event_unknown_before_it_happens(tmp_path):
    snap = write_snapshot(tmp_path / "s", *flat(K))
    before = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[K - 1].date()))
    assert before.quality["usable"] and before.quality["issue_events"] == []
    assert point_in_time_usable(before, W)["usable"].all()


def test_unusable_from_event_day_until_window_passes(tmp_path):
    snap = write_snapshot(tmp_path / "s", *flat(K))
    ds = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date()))
    assert not ds.quality["usable"]                                   # 整段判定仍然不可用（原有语义不变）
    assert [(e["date"], e["type"]) for e in ds.quality["issue_events"]] == [(DAY, "UNRECORDED_ADJUSTMENT")]
    m = point_in_time_usable(ds, W)["usable"]
    assert m.iloc[:K].all()                                           # 事件之前：可用
    assert not m.iloc[K]                                              # 事件当日收盘：不可用
    assert not m.iloc[K + 1:K + W].any()                              # 事件之后、回看窗口仍覆盖事件：不可用
    assert m.iloc[K + W:].all()                                       # 窗口过后：恢复


def test_knowledge_lag_delays_but_never_advances_exclusion(tmp_path):
    snap = write_snapshot(tmp_path / "s", *flat(K))
    ds = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date()))
    m1 = point_in_time_usable(ds, W, knowledge_lag_days=1)["usable"]
    assert m1.iloc[:K + 1].all() and not m1.iloc[K + 1] and m1.iloc[K + W:].all()


def test_mask_is_truncation_invariant(tmp_path):
    snap = write_snapshot(tmp_path / "s", *flat(K))
    full = point_in_time_usable(load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date())), W)["usable"]
    for end in (K - 1, K, K + 5, K + W + 3):
        part = point_in_time_usable(load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[end].date())), W)["usable"]
        pd.testing.assert_series_equal(part, full.loc[part.index])


def test_dividend_during_suspension_is_verified_on_resumption_day(tmp_path):
    closes = np.full(len(CAL), 10.0)
    gap = list(CAL[60:70])                                            # 停牌 10 个交易日，除息日在停牌期间
    ex = str(CAL[64].date())
    closes[70:] = 9.5
    hfq = {"2023-01-02": 1.0, str(CAL[70].date()): 10.0 / (10.0 - 0.5)}
    snap = write_snapshot(tmp_path / "s", closes, hfq, dividends=[(ex, 0.5)], drop=gap)
    q = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date())).quality
    v = next(x for x in q["corporate_action_validation"] if x["ex_date"] == ex)
    assert v["status"] == "VERIFIED_BY_FACTOR_NEXT_TRADED_DAY" and v["first_traded_day"] == str(CAL[70].date())
    assert q["usable"] and q["issue_events"] == []


def test_dividend_during_suspension_with_mismatching_factor_is_still_flagged(tmp_path):
    closes = np.full(len(CAL), 10.0)
    ex = str(CAL[64].date())
    closes[70:] = 8.0
    hfq = {"2023-01-02": 1.0, str(CAL[70].date()): 1.25}               # 远大于 0.5 元分红能解释的幅度（如同时配股）
    snap = write_snapshot(tmp_path / "s", closes, hfq, dividends=[(ex, 0.5)], drop=list(CAL[60:70]))
    q = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date())).quality
    assert next(x for x in q["corporate_action_validation"] if x["ex_date"] == ex)["status"] == "EX_DATE_NOT_TRADED"
    assert ("UNRECORDED_ADJUSTMENT", str(CAL[70].date())) in [(e["type"], e["date"]) for e in q["issue_events"]]


def test_data_interruption_is_permanent_from_first_missing_day(tmp_path):
    closes, hfq = flat()
    snap = write_snapshot(tmp_path / "s", closes, hfq, drop=list(CAL[150:]))
    ds = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date()))
    assert [e["type"] for e in ds.quality["issue_events"]] == ["DATA_INTERRUPTED"]
    m = point_in_time_usable(ds, W)["usable"]
    assert m.iloc[:150].all() and not m.iloc[150:].any()


def test_event_timing_order_before_event_fills_at_event_open_exit_after_close(tmp_path):
    """X-1 收盘的买单在 X 开盘照常成交（只用到 X-1 的信息）；X 收盘判定不可用，调出卖单在 X+1 开盘成交；X 收盘的买入信号不下单。"""
    from quantrisk.data.universe import PointInTimeUniverse
    from quantrisk.research.walk_forward import _PointInTimeQualityUniverse
    from quantrisk.strategies.backtest.portfolio_run import run_strategy_portfolio
    from quantrisk.strategies.base import BaseStrategy, Signal

    snap = write_snapshot(tmp_path / "s", *flat(K))
    ds = load_dataset(snap, SYM, str(CAL[0].date()), str(CAL[-1].date()))
    base = PointInTimeUniverse({SYM: [(CAL[0], None)]}, seasoning_days=0, trading_calendar=CAL)
    uni = _PointInTimeQualityUniverse(base, {SYM: point_in_time_usable(ds, W)["usable"]})

    class Script(BaseStrategy):
        def default_params(self):
            return {"position_pct": 0.5}

        def generate_signals(self, d):
            s = pd.Series(Signal.HOLD.value, index=d.index)
            s.loc[[CAL[K - 1], CAL[K]]] = Signal.BUY.value
            return s

    out = run_strategy_portfolio(lambda s: Script(), [ds], 1_000_000, universe=uni, allow_unusable=True)
    orders = {o["order_id"]: o for o in out["orders"]}
    fills = {f["order_id"]: f for f in out["fills"]}
    buy = next(o for o in orders.values() if o["side"] == "BUY")
    assert buy["signal_date"] == str(CAL[K - 1].date()) and fills[buy["order_id"]]["date"] == str(CAL[K].date())
    sell = next(o for o in orders.values() if o["side"] == "SELL")
    assert sell["signal"] == "UNIVERSE_EXIT" and sell["signal_date"] == str(CAL[K].date())
    assert fills[sell["order_id"]]["date"] == str(CAL[K + 1].date())
    assert len([o for o in orders.values() if o["side"] == "BUY"]) == 1          # X 收盘的买入信号没有生成订单


def test_point_in_time_is_the_default_quality_mode():
    from quantrisk.research.walk_forward import StudyConfig
    cfg = StudyConfig(snapshot="x", strategy="x", param_grid={}, test_years=[2024])
    assert cfg.quality_mode == "point_in_time" and cfg.knowledge_lag_days == 0
