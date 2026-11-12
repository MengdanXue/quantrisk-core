"""ETF Pilot：规则、费用、预算、限价成交、账本隔离、中断恢复、独立复算、行情过期与配置变更拦截（合成数据，不联网）。"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantrisk.pilot.etf_rules import SCENARIOS, FeeScenario, RuleNotVerifiedError, UnsupportedETFError, load_rules
from quantrisk.pilot.experiment_config import (ConfigChangedError, init_experiment, load_experiment, make_config,
                                               select_etf)
from quantrisk.pilot.experiment_report import build_report
from quantrisk.pilot.ledgers import CorruptJournalError, Journal, LedgerIsolationError
from quantrisk.pilot import manual_execution as mx
from quantrisk.pilot.manual_execution import ManualEntryError
from quantrisk.pilot.market_data import CST, ETFMarketStore, StaleDataError
from quantrisk.pilot.paper_execution import simulate_fill, touch_fill
from quantrisk.pilot.runner import ReplayMismatchError, manual_suggestion, simulate, sync
from quantrisk.pilot.trade_signal import adjusted_prices, propose, strategy_signals
from quantrisk.strategies import get_strategy
from scripts.verify_etf_ledger import verify as independent_verify

CODE = "sz159596"
R = load_rules(CODE)
CAL = pd.bdate_range("2025-01-02", periods=260)
EV = {"kind": "APP_SCREENSHOT", "ref": "app.png"}


def path_closes(n=len(CAL)):
    """先跌、再涨、再跌、再涨：保证 MA20/60 出现金叉和死叉。价格都在 0.001 元报价单位上。"""
    seg = [(0, 0.60), (70, 0.50), (150, 0.66), (210, 0.55), (n - 1, 0.62)]
    x = np.interp(np.arange(n), [a for a, _ in seg], [b for _, b in seg])
    x += 0.004 * np.sin(np.arange(n) / 3.0)
    return np.round(x, 3)


def make_bars(closes=None, gap_up=(), drop=()):
    c = path_closes() if closes is None else np.asarray(closes)
    o = np.r_[c[0], c[:-1]].copy()
    for i in gap_up:                                   # 开盘跳空高开 5 个报价单位，但盘中回落到前收盘以下
        o[i] = round(c[i - 1] + 0.005, 3)
    hi = np.round(np.maximum(o, c) + 0.003, 3)
    lo = np.round(np.minimum(o, c) - 0.003, 3)
    for i in gap_up:
        lo[i] = round(c[i - 1] - 0.002, 3)
    df = pd.DataFrame({"date": CAL.strftime("%Y-%m-%d"), "open": o, "high": hi, "low": lo, "close": c,
                       "volume": 10_000_000, "amount": (c * 10_000_000).round()})
    return df.drop(index=list(drop)).reset_index(drop=True)


class FakeFetcher:
    def __init__(self, bars, divs=None, cal=None):
        self._bars, self._cal = bars, cal or [str(d.date()) for d in pd.bdate_range(CAL[0], CAL[-1] + timedelta(days=30))]
        self._divs = divs if divs is not None else pd.DataFrame({"date": [], "cumulative": []})

    def source(self):
        return "fake"

    def bars(self, code):
        return self._bars.copy()

    def dividends(self, code):
        return self._divs.copy()

    def calendar(self):
        return list(self._cal)


def after_close(day, hour=16):
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=CST).astimezone(timezone.utc)


def build(tmp_path, scenario="B", bars=None, divs=None, start=None, last=None, name="t", **kw):
    bars = make_bars() if bars is None else bars
    last = pd.Timestamp(bars["date"].iloc[-1]) if last is None else pd.Timestamp(last)
    store = ETFMarketStore(tmp_path / "market", CODE, FakeFetcher(bars, divs))
    store.fetch(after_close(last))
    fee = scenario if isinstance(scenario, FeeScenario) else SCENARIOS[scenario]
    cfg = make_config(name, CODE, start or str(CAL[80].date()), fee, {"selected": CODE}, rehearsal=True, **kw)
    exp = init_experiment(tmp_path / "exps", cfg)
    return store, cfg, exp, last


# ---------- 规则 ----------

def test_rules_are_loaded_from_verified_szse_record():
    assert float(R.tick) == 0.001 and R.lot_min == 100 and R.lot_step == 100
    assert float(R.limit_pct) == 0.10 and R.settlement_lag == 1 and R.stamp_duty == 0.0
    assert R.verified_on == "2026-11-09" and not R.transfer_fee_verified


def test_unsupported_and_unverified_etfs_are_refused():
    with pytest.raises(UnsupportedETFError):
        load_rules("sh510300")                       # 不在白名单（价格也超出 100 元预算）
    with pytest.raises(UnsupportedETFError):
        load_rules("sh513100")                       # 跨境 ETF：第一阶段不支持
    with pytest.raises(RuleNotVerifiedError):
        load_rules("sh563500")                       # 白名单内，但上交所规则尚未核实


def test_tick_rounding_and_limit_prices():
    assert R.limit_prices(0.578) == (0.520, 0.636)
    assert R.limit_prices(0.005) == (0.004, 0.006)   # 与前收盘之差不足一个报价单位时按 ±1 个单位
    assert R.round_up(0.5781) == 0.579 and R.round_down(0.5789) == 0.578 and R.round_nearest(0.5785) == 0.579
    assert R.on_tick(0.578) and not R.on_tick(0.5785)
    assert R.ex_dividend_reference(0.600, 0.0123) == 0.588


def test_lot_rules():
    assert R.valid_buy_qty(100) and R.valid_buy_qty(300) and not R.valid_buy_qty(150) and not R.valid_buy_qty(0)
    assert R.max_buy_lots_qty(199) == 100 and R.max_buy_lots_qty(99) == 0
    assert R.valid_sell_qty(50, 50) and not R.valid_sell_qty(50, 150) and R.valid_sell_qty(100, 150)
    assert not R.valid_sell_qty(200, 150)


# ---------- 费用与预算 ----------

def test_fee_scenarios():
    a, b = SCENARIOS["A"].breakdown(0.578, 100), SCENARIOS["B"].breakdown(0.578, 100)
    assert a == {"commission": 5.0}                  # 最低 5 元
    assert b == {"commission": 0.0}                  # 57.8 × 0.0085% = 0.0049 元，四舍五入到分为 0
    assert SCENARIOS["B"].breakdown(5.0, 10000) == {"commission": 4.25}
    assert "stamp_duty" not in a and not SCENARIOS["A"].actual_broker and not SCENARIOS["B"].actual_broker


def test_budget_rejects_scenario_a_and_accepts_scenario_b_with_costs_recorded_separately():
    kw = dict(day=pd.Timestamp("2026-11-06"), side="BUY", close=0.578, cash=100.0, holding=0, rules=R, budget=0.005,
              half_spread_ticks=1, limit_offset_ticks=1, order_id="x")
    pa = propose(fee=SCENARIOS["A"], **kw)
    assert pa.action == "REJECT" and "超过预算" in pa.reason
    assert pa.order["est_explicit_total"] == 5.0 and pa.order["est_friction"] == pytest.approx(0.1)
    assert pa.order["est_explicit_ratio"] == pytest.approx(5.0 / 57.9) and pa.order["est_friction_ratio"] == pytest.approx(0.1 / 57.9)
    pb = propose(fee=SCENARIOS["B"], **kw)
    assert pb.action == "PROPOSE" and pb.order["qty"] == 100 and pb.order["limit_price"] == 0.579
    assert pb.order["est_total_ratio"] < 0.005 and pb.order["est_cash_after"] == pytest.approx(100 - 57.9)


def test_insufficient_cash_for_one_lot_is_rejected():
    p = propose(pd.Timestamp("2026-11-06"), "BUY", 0.95, 90.0, 0, R, SCENARIOS["B"], 0.005, 1, 1, "x")
    assert p.action == "REJECT" and "不足一手" in p.reason and p.order is None


def test_blockers_reject_order():
    p = propose(pd.Timestamp("2026-11-06"), "BUY", 0.578, 100.0, 0, R, SCENARIOS["B"], 0.005, 1, 1, "x",
                blockers=["行情过期"])
    assert p.action == "REJECT" and p.reason == "行情过期"


# ---------- 限价成交 ----------

ORDER = {"order_id": "o", "side": "BUY", "qty": 100, "limit_price": 0.579, "ref_close": 0.578}
BAND = R.limit_prices(0.578)


def bar(o, h, l, c=0.578):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def test_strict_fill_only_when_open_satisfies_limit_and_never_violates_it():
    f = simulate_fill(ORDER, bar(0.578, 0.590, 0.570), BAND, R, SCENARIOS["B"], 1)
    assert f["status"] == "STRICT_FILL" and f["price"] == 0.579           # 开盘 + 1 个单位 = 限价
    f = simulate_fill(ORDER, bar(0.579, 0.590, 0.570), BAND, R, SCENARIOS["B"], 1)
    assert f["status"] == "STRICT_FILL" and f["price"] == 0.579           # 开盘 + 半价差超过限价时截到限价
    assert f["spread_cost_vs_open"] == 0.0 and f["fee_total"] == 0.0
    t = simulate_fill(ORDER, bar(0.583, 0.590, 0.575), BAND, R, SCENARIOS["B"], 1)
    assert t["status"] == "UNCERTAIN_TOUCH" and "price" not in t           # 仅盘中触价：不确定，不计入严格账本
    tf = touch_fill(t, ORDER, SCENARIOS["B"])
    assert tf["status"] == "TOUCH_FILL" and tf["price"] == 0.579
    assert simulate_fill(ORDER, bar(0.583, 0.590, 0.581), BAND, R, SCENARIOS["B"], 1)["status"] == "EXPIRED"
    assert simulate_fill(ORDER, None, BAND, R, SCENARIOS["B"], 1)["status"] == "EXPIRED"


def test_fill_price_stays_inside_day_range():
    """回归：卖单限价 0.654、开盘 0.655 = 当日最低价时，开盘 − 半价差 = 0.654 低于最低价，必须取 0.655（2026-08-21 真实数据）。"""
    s = {"order_id": "s", "side": "SELL", "qty": 100, "limit_price": 0.654, "ref_close": 0.655}
    f = simulate_fill(s, bar(0.655, 0.661, 0.655, 0.658), R.limit_prices(0.655), R, SCENARIOS["B"], 1)
    assert f["status"] == "STRICT_FILL" and f["price"] == 0.655 and f["spread_cost_vs_open"] == 0.0
    b = {**ORDER, "limit_price": 0.585}
    f = simulate_fill(b, bar(0.580, 0.580, 0.575), BAND, R, SCENARIOS["B"], 1)
    assert f["status"] == "STRICT_FILL" and f["price"] == 0.580                # 开盘 = 最高价时不加价差


def test_open_at_limit_up_does_not_fill():
    o = {**ORDER, "limit_price": 0.636}
    assert simulate_fill(o, bar(0.636, 0.636, 0.636), BAND, R, SCENARIOS["B"], 1)["status"] == "EXPIRED"


def test_sell_side_is_symmetric():
    s = {"order_id": "s", "side": "SELL", "qty": 100, "limit_price": 0.577, "ref_close": 0.578}
    f = simulate_fill(s, bar(0.578, 0.580, 0.570), BAND, R, SCENARIOS["A"], 1)
    assert f["status"] == "STRICT_FILL" and f["price"] == 0.577 and f["fees"] == {"commission": 5.0}
    assert simulate_fill(s, bar(0.574, 0.578, 0.570), BAND, R, SCENARIOS["A"], 1)["status"] == "UNCERTAIN_TOUCH"
    assert simulate_fill(s, bar(0.520, 0.530, 0.520), BAND, R, SCENARIOS["A"], 1)["status"] == "EXPIRED"


def test_fill_price_never_violates_limit_randomised():
    rng = np.random.default_rng(0)
    for _ in range(500):
        side = rng.choice(["BUY", "SELL"])
        lim = round(0.5 + rng.integers(0, 200) / 1000, 3)
        o = round(lim + rng.integers(-5, 6) / 1000, 3)
        b = bar(o, round(max(o, lim) + 0.004, 3), round(min(o, lim) - 0.004, 3))
        r = simulate_fill({"order_id": "r", "side": side, "qty": 100, "limit_price": lim, "ref_close": lim}, b,
                          R.limit_prices(lim), R, SCENARIOS["B"], int(rng.integers(0, 3)))
        if r["status"] == "STRICT_FILL":
            assert R.on_tick(r["price"]) and (r["price"] <= lim if side == "BUY" else r["price"] >= lim)
            assert b["low"] <= r["price"] <= b["high"]


# ---------- 信号 ----------

def test_signals_reuse_ma_cross_and_ignore_future_data(tmp_path):
    store, cfg, exp, last = build(tmp_path)
    b, d = store.bars(), store.dividends()
    sig = strategy_signals(b, d, cfg.strategy)
    expected = get_strategy("ma_cross", {"fast_period": 20, "slow_period": 60}).generate_signals(adjusted_prices(b, d))
    pd.testing.assert_series_equal(sig, expected.astype(int))
    assert (sig > 0).any() and (sig < 0).any()
    cut = b.index[150]
    sig_cut = strategy_signals(b.loc[:cut], d, cfg.strategy)
    pd.testing.assert_series_equal(sig_cut, sig.loc[:cut])


def test_dividend_adjustment_uses_only_past_ex_dates():
    b = make_bars().assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")
    divs = pd.DataFrame({"cash": [0.02]}, index=pd.DatetimeIndex([b.index[100]]))
    adj = adjusted_prices(b, divs)
    assert (adj["close"].iloc[:100] == b["close"].iloc[:100]).all()
    assert adj["close"].iloc[100] == pytest.approx(b["close"].iloc[100] * b["close"].iloc[99] / (b["close"].iloc[99] - 0.02))


# ---------- 全流程、账本、恢复、复算 ----------

def test_full_run_scenario_b_trades_and_independent_recompute_passes(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B")
    s = sync(exp, cfg, store, now_utc=after_close(last))
    assert s["appended"]["PAPER_STRICT"] > 100
    v = independent_verify(exp, store.dir)
    assert v["problems"] == [] and v["checked"]["fills"] > 0
    out = build_report(exp, cfg, store, tmp_path / "rep")
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["PAPER_STRICT"]["fills"] >= 2 and m["BENCHMARK"]["fills"] == 1
    assert m["MANUAL_LIVE"]["status"] == "尚无真实交易数据"
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "尚无真实交易数据" in rep and "历史回放演练" in rep and (out / "nav_daily.csv").exists()
    try:
        import matplotlib  # noqa: F401
        assert (out / "nav.png").exists()
    except ImportError:
        assert "未安装 matplotlib" in rep


def test_scenario_a_rejects_every_buy_and_keeps_cash(tmp_path):
    store, cfg, exp, last = build(tmp_path, "A")
    sync(exp, cfg, store, now_utc=after_close(last))
    for L in ("PAPER_STRICT", "BENCHMARK"):
        recs = Journal(exp, L).records()
        props = [e["proposal"] for r in recs for e in r["events"] if e["type"] == "DECISION" and e["proposal"]]
        assert props and all(p["action"] == "REJECT" and "超过预算" in p["reason"] for p in props)
        assert all(e["cash"] == 100.0 for r in recs for e in r["events"] if e["type"] == "DAY_CLOSE")


def test_strict_and_touch_ledgers_differ_on_gap_up_days(tmp_path):
    ps, pc, pe, pl = build(tmp_path / "probe", "B")
    sync(pe, pc, ps, now_utc=after_close(pl))
    recs = Journal(pe, "PAPER_STRICT").records()
    buy_days = [r["day"] for r in recs for e in r["events"] if e["type"] == "DECISION" and e["proposal"]
                and e["proposal"]["action"] == "PROPOSE" and e["proposal"]["side"] == "BUY"]
    nxt = [CAL.get_loc(pd.Timestamp(d)) + 1 for d in buy_days[:1]]
    store, cfg, exp, last = build(tmp_path / "gap", "B", bars=make_bars(gap_up=nxt))
    sync(exp, cfg, store, now_utc=after_close(last))
    day = str(CAL[nxt[0]].date())
    res = {L: next(e for r in Journal(exp, L).records() if r["day"] == day for e in r["events"] if e["type"] == "ORDER_RESULT")
           for L in ("PAPER_STRICT", "PAPER_TOUCH")}
    assert res["PAPER_STRICT"]["status"] == "UNCERTAIN_TOUCH" and res["PAPER_TOUCH"]["status"] == "TOUCH_FILL"
    assert independent_verify(exp, store.dir)["problems"] == []


def test_resume_after_interruption_equals_one_shot(tmp_path):
    bars = make_bars()
    s1, cfg1, e1, last = build(tmp_path / "a", "B", bars=bars)
    sync(e1, cfg1, s1, now_utc=after_close(last))
    s2, cfg2, e2, _ = build(tmp_path / "b", "B", bars=bars)
    sync(e2, cfg2, s2, now_utc=after_close(last), through=CAL[150])            # 中途停下
    again = sync(e2, cfg2, s2, now_utc=after_close(last))                        # 恢复
    assert again["verified_existing"]["PAPER_STRICT"] > 0
    for L in ("PAPER_STRICT", "PAPER_TOUCH", "BENCHMARK"):
        a = [(r["day"], json.dumps(r["events"], sort_keys=True)) for r in Journal(e1, L).records()]
        b = [(r["day"], json.dumps(r["events"], sort_keys=True)) for r in Journal(e2, L).records()]
        assert a == b
    assert sync(e2, cfg2, s2, now_utc=after_close(last))["appended"]["PAPER_STRICT"] == 0


def test_partial_last_line_is_detected_and_repaired(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B")
    sync(exp, cfg, store, now_utc=after_close(last), through=CAL[120])
    p = Journal(exp, "PAPER_STRICT").path
    with open(p, "ab") as fh:
        fh.write(b'{"seq": 999, "trunc')                                         # 模拟写入中断
    with pytest.raises(CorruptJournalError):
        sync(exp, cfg, store, now_utc=after_close(last))
    sync(exp, cfg, store, now_utc=after_close(last), repair=True)
    assert p.with_suffix(".partial").exists() and independent_verify(exp, store.dir)["problems"] == []


def test_tampered_journal_is_detected(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B")
    sync(exp, cfg, store, now_utc=after_close(last), through=CAL[120])
    p = Journal(exp, "PAPER_STRICT").path
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace('"cash": 100.0', '"cash": 1000.0', 1)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorruptJournalError):
        Journal(exp, "PAPER_STRICT").records()
    assert any("哈希链" in x for x in independent_verify(exp, store.dir)["problems"])


def test_history_changed_by_data_source_is_not_overwritten(tmp_path):
    bars = make_bars()
    store, cfg, exp, last = build(tmp_path, "B", bars=bars)
    sync(exp, cfg, store, now_utc=after_close(last))
    divs = pd.DataFrame({"date": [str(CAL[200].date())], "cumulative": [0.03]})   # 数据源事后补了一笔过去的分红
    store.fetcher = FakeFetcher(bars, divs)
    store.fetch(after_close(last))
    with pytest.raises(ReplayMismatchError):
        sync(exp, cfg, store, now_utc=after_close(last))


# ---------- 隔离 ----------

def test_ledgers_cannot_be_mixed(tmp_path):
    with pytest.raises(LedgerIsolationError):
        Journal(tmp_path, "MANUAL_LIVE").append({"type": "BROKER_FILL", "day": "2026-01-05"}, "SIMULATED")
    with pytest.raises(LedgerIsolationError):
        Journal(tmp_path, "PAPER_STRICT").append({"day": "2026-01-05", "events": []}, "BROKER_CONFIRMED_MANUAL_ENTRY")
    good = Journal(tmp_path, "PAPER_STRICT")
    good.append({"day": "2026-01-05", "events": []}, "SIMULATED")
    good.path.rename(Journal(tmp_path, "MANUAL_LIVE").path)                       # 把模拟流水冒充成真实流水
    with pytest.raises(LedgerIsolationError):
        Journal(tmp_path, "MANUAL_LIVE").records()


def test_simulated_fee_experiment_refuses_every_manual_entry(tmp_path):
    """T25：actual_broker=false 的实验（A/B）上任何真实录入都拒绝，PAPER 流水不变。"""
    store, cfg, exp, last = build(tmp_path, "B")
    sync(exp, cfg, store, now_utc=after_close(last))
    paper_before = Journal(exp, "PAPER_STRICT").path.read_bytes()
    assert manual_suggestion(exp, cfg, store, now_utc=after_close(last))["status"] == "NO_ACCOUNT"
    d = str(CAL[200].date())
    calls = [lambda: mx.record_capital(exp, cfg, 100.0, d, EV),
             lambda: mx.record_order(exp, cfg, store, d, "09:30", "BUY", 100, 0.6, "O1", EV),
             lambda: mx.record_fill(exp, cfg, store, d, "09:31", "O1", 100, 0.6, "T1", EV),
             lambda: mx.record_cancel(exp, cfg, store, d, "09:32", "O1", 100, EV),
             lambda: mx.record_reject(exp, cfg, store, d, "09:32", "O1", "x", EV),
             lambda: mx.record_expired(exp, cfg, store, d, "O1", 100, EV),
             lambda: mx.record_settlement(exp, cfg, store, d, "O1", 100, 60.0, {"commission": 0.1}, "S1", EV),
             lambda: mx.record_dividend(exp, cfg, d, 0.5, EV),
             lambda: mx.void_record(exp, cfg, "0" * 64, "x", EV)]
    for call in calls:
        with pytest.raises(ManualEntryError, match="actual_broker=false"):
            call()
    assert not Journal(exp, "MANUAL_LIVE").path.exists()
    assert Journal(exp, "PAPER_STRICT").path.read_bytes() == paper_before


# ---------- 拦截：行情过期、未收盘 K 线、数据缺失、配置变更 ----------

def test_stale_data_blocks_run(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B")
    later = CAL[-1] + timedelta(days=7)
    with pytest.raises(StaleDataError):
        sync(exp, cfg, store, now_utc=after_close(later))
    assert not Journal(exp, "PAPER_STRICT").path.exists()
    with pytest.raises(StaleDataError):
        manual_suggestion(exp, cfg, store, now_utc=after_close(later))


def test_bar_fetched_before_close_cutoff_is_dropped(tmp_path):
    store = ETFMarketStore(tmp_path / "m", CODE, FakeFetcher(make_bars()))
    s = store.fetch(after_close(CAL[-1], hour=14))
    assert s["dropped_incomplete_bars"] == [str(CAL[-1].date())] and store.bars().index.max() == CAL[-2]
    s = store.fetch(after_close(CAL[-1], hour=16))
    assert s["new_bars"] == 1 and store.bars().index.max() == CAL[-1]


def test_revisions_are_logged_not_overwritten(tmp_path):
    bars = make_bars()
    store = ETFMarketStore(tmp_path / "m", CODE, FakeFetcher(bars))
    store.fetch(after_close(CAL[-1]))
    changed = bars.copy()
    changed.loc[10, "close"] = round(changed.loc[10, "close"] + 0.01, 3)
    store.fetcher = FakeFetcher(changed)
    assert store.fetch(after_close(CAL[-1]))["revisions"] == 1
    assert store.bars().loc[CAL[10], "close"] == bars.loc[10, "close"]
    assert (store.dir / "revisions.csv").exists()


def test_missing_bar_blocks_new_orders(tmp_path):
    gap = 150
    store, cfg, exp, last = build(tmp_path, "B", bars=make_bars(drop=[gap]))
    assert any("缺少行情" in i["issue"] for i in store.integrity_issues(R))
    recs = {r["day"]: r for r in simulate(cfg, R, store, last)["PAPER_STRICT"]}
    for d in CAL[gap + 1:gap + 30]:
        for e in recs[str(d.date())]["events"]:
            if e["type"] == "DECISION" and e["proposal"]:
                assert e["proposal"]["action"] == "REJECT" and "完整性" in e["proposal"]["reason"]


def test_unexplained_price_jump_is_flagged(tmp_path):
    c = path_closes()
    c[180:] = np.round(c[180:] / 2, 3)                                            # 未建模的份额折算
    store = ETFMarketStore(tmp_path / "m", CODE, FakeFetcher(make_bars(c)))
    store.fetch(after_close(CAL[-1]))
    assert any("无法由分红解释" in i["issue"] and i["date"] == str(CAL[180].date()) for i in store.integrity_issues(R))


def test_config_is_frozen(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B")
    load_experiment(exp)
    raw = (exp / "config.json").read_text(encoding="utf-8")
    (exp / "config.json").write_text(raw.replace('"friction_budget": 0.005', '"friction_budget": 0.02'), encoding="utf-8")
    with pytest.raises(ConfigChangedError):
        load_experiment(exp)
    with pytest.raises(ValueError):
        make_config("x", CODE, "2026-11-10", SCENARIOS["B"], {}, capital=150.0)
    with pytest.raises(FileExistsError):
        init_experiment(tmp_path / "exps", cfg)


# ---------- 选择规则 ----------

def test_select_etf_uses_real_20_day_turnover_and_price_cap():
    def h(close, amount, n=25):
        return pd.DataFrame({"date": pd.bdate_range("2026-10-01", periods=n), "close": close, "amount": amount})
    res = select_etf({"sz159596": h(0.578, 1.4e7), "sz159975": h(0.595, 4e6), "sh563500": h(0.549, 9e7),
                      "sz159602": h(0.932, 9e7), "sz159706": h(0.728, 9e8, n=10)}, "2026-11-09")
    assert res["selected"] == "sz159596"
    why = {c["code"]: c["reason"] for c in res["candidates"]}
    assert "未核实" in why["sh563500"] and "> 0.9" in why["sz159602"] and "不足 20 天" in why["sz159706"]


# ---------- 未建模的价格跳变（如份额拆分） ----------

def split_closes(k=200):
    c = path_closes()
    c[k:] = np.round(c[k:] / 2, 3)                                            # 约 1 拆 2，数据源没有拆分记录
    return c


def test_signals_restart_after_unexplained_jump(tmp_path):
    from quantrisk.pilot.runner import signals_with_resets
    store, cfg, exp, last = build(tmp_path, "B", bars=make_bars(split_closes()), start=str(CAL[60].date()))
    b, d = store.bars(), store.dividends()
    sig, jumps = signals_with_resets(b, d, cfg, store.integrity_issues(R))
    assert jumps == [CAL[200]]
    pd.testing.assert_series_equal(sig.loc[:CAL[199]], strategy_signals(b.loc[:CAL[199]], d, cfg.strategy))
    pd.testing.assert_series_equal(sig.loc[CAL[200]:], strategy_signals(b.loc[CAL[200]:], d, cfg.strategy))


def test_decisions_blocked_during_warmup_after_jump(tmp_path):
    store, cfg, exp, last = build(tmp_path, "B", bars=make_bars(split_closes(200)), start=str(CAL[205].date()))
    recs = simulate(cfg, R, store, last)["BENCHMARK"]
    props = [e["proposal"] for r in recs for e in r["events"] if e["type"] == "DECISION" and e["proposal"]]
    assert props and all(p["action"] == "REJECT" and "价格跳变" in p["reason"] for p in props)   # 预热 120 日内一律拒绝


def test_holding_across_unexplained_jump_stops_the_run(tmp_path):
    from quantrisk.pilot.market_data import DataIntegrityError
    store, cfg, exp, last = build(tmp_path, "B", bars=make_bars(split_closes(230)), start=str(CAL[80].date()))
    with pytest.raises(DataIntegrityError):
        simulate(cfg, R, store, last)                                         # 基准此时持仓，不能编造份额变动


def test_report_groups_reject_reasons(tmp_path):
    store, cfg, exp, last = build(tmp_path, "A")
    sync(exp, cfg, store, now_utc=after_close(last))
    m = json.loads((build_report(exp, cfg, store, tmp_path / "rep") / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["PAPER_STRICT"]["execution"]["reject_reasons"]) == {"摩擦成本超预算"}


# ---------- 前向观察 ----------

def forward_setup(tmp_path):
    """前向实验（rehearsal=False）：登记真实的选择记录文件，供基线检查使用。"""
    import hashlib
    sel_dir = tmp_path / "selection"
    sel_dir.mkdir()
    (sel_dir / "selection_x.json").write_text('{"selected": "sz159596"}', encoding="utf-8")
    sha = hashlib.sha256((sel_dir / "selection_x.json").read_bytes()).hexdigest()
    bars = make_bars()
    last = pd.Timestamp(bars["date"].iloc[-1])
    store = ETFMarketStore(tmp_path / "market", CODE, FakeFetcher(bars))
    store.fetch(after_close(last))
    cfg = make_config("fwd", CODE, str(CAL[200].date()), SCENARIOS["B"], {"file": "selection_x.json", "sha256": sha})
    exp = init_experiment(tmp_path / "exps", cfg)
    return store, cfg, exp, last, sel_dir


def test_observation_refuses_rehearsal(tmp_path):
    from quantrisk.pilot.observation import ObservationError, daily_log, ensure_baseline
    store, cfg, exp, last = build(tmp_path, "B")
    with pytest.raises(ObservationError):
        daily_log(exp, cfg)
    with pytest.raises(ObservationError):
        ensure_baseline(exp, cfg, tmp_path, tmp_path)


def test_daily_log_and_review(tmp_path):
    from pathlib import Path
    from quantrisk.pilot.observation import ObservationError, check_baseline, daily_log, log_run, review, run_log
    root = Path(__file__).resolve().parents[1]
    store, cfg, exp, last, sel = forward_setup(tmp_path)
    sync(exp, cfg, store, now_utc=after_close(last), through=CAL[210])
    base = check_baseline(exp, cfg, sel, root)
    assert base["hard"] == [] and base["soft"] == []
    df = daily_log(exp, cfg)
    assert len(df) == 11 and list(df["date"]) == [str(d.date()) for d in CAL[200:211]]
    assert df["fetch_delay_hours_after_1530"].min() > 0 and (df["cum_return"] == df["equity"] / 100 - 1).all()
    ver = independent_verify(exp, store.dir)
    with pytest.raises(ObservationError):
        review([(exp, cfg, ver, base)], store.dir, tmp_path / "r.md")              # 未满 20 个交易日
    log_run(exp, {"command": "daily", "status": "OK"})
    log_run(exp, {"command": "daily", "status": "STALE"})
    assert [r["status"] for r in run_log(exp)] == ["OK", "STALE"]
    sync(exp, cfg, store, now_utc=after_close(last))
    out = review([(exp, cfg, independent_verify(exp, store.dir), check_baseline(exp, cfg, sel, root))], store.dir,
                 tmp_path / "r.md")
    text = out.read_text(encoding="utf-8")
    assert "评价不以收益为主" in text and "尚无真实交易数据" in text and "{'OK': 1, 'STALE': 1}" in text


def test_baseline_detects_config_and_selection_changes(tmp_path):
    from pathlib import Path
    from quantrisk.pilot.observation import check_baseline
    root = Path(__file__).resolve().parents[1]
    store, cfg, exp, last, sel = forward_setup(tmp_path)
    check_baseline(exp, cfg, sel, root)                                           # 生成基线
    (sel / "selection_x.json").write_text('{"selected": "sz159975"}', encoding="utf-8")
    assert check_baseline(exp, cfg, sel, root)["hard"] == ["selection_sha256"]
    raw = (exp / "config.json").read_text(encoding="utf-8")
    (exp / "config.json").write_text(raw + " ", encoding="utf-8")
    assert "config_sha256" in check_baseline(exp, cfg, sel, root)["hard"]


# ---------- 前向实验的成交时间约束（行情获取时间 vs 下一交易日开盘） ----------

def at_cst(day, hour, minute=0):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST).astimezone(timezone.utc)


def staged_forward(tmp_path, stages, start_i=200):
    """前向实验；stages = [(K 线截至第几个交易日, 获取时刻 UTC)]，按顺序分批获取，模拟真实的行情到达时间。"""
    full = make_bars()
    for upto, when in stages:
        ETFMarketStore(tmp_path / "market", CODE, FakeFetcher(full.iloc[:upto + 1].copy())).fetch(when)
    store = ETFMarketStore(tmp_path / "market", CODE, FakeFetcher(full))
    cfg = make_config("fwd", CODE, str(CAL[start_i].date()), SCENARIOS["B"], {"selected": CODE})
    return store, cfg, init_experiment(tmp_path / "exps", cfg)


def results_on(exp, ledger, day):
    rec = next(r for r in Journal(exp, ledger).records() if r["day"] == str(day.date()))
    return [e for e in rec["events"] if e["type"] == "ORDER_RESULT"]


def test_order_fills_when_decision_data_arrived_after_close_same_day(tmp_path):
    s = 200
    store, cfg, exp = staged_forward(tmp_path, [(s, after_close(CAL[s])), (s + 2, after_close(CAL[s + 2]))])
    sync(exp, cfg, store, now_utc=after_close(CAL[s + 2]))
    (res,) = results_on(exp, "BENCHMARK", CAL[s + 1])          # 基准账首日买入，D+1 撮合
    assert res["status"] != "LATE_DATA_NOT_EXECUTABLE"
    assert res["decision_data_available_cst"].startswith(f"{CAL[s].date()}T16:00")
    assert res["order_submit_deadline_cst"].startswith(f"{CAL[s + 1].date()}T09:15")
    assert independent_verify(exp, store.dir)["problems"] == []


def test_order_fills_when_decision_data_arrived_next_morning_before_open(tmp_path):
    s = 200
    store, cfg, exp = staged_forward(tmp_path, [(s - 1, after_close(CAL[s - 1])), (s, at_cst(CAL[s + 1], 8, 30)),
                                                (s + 2, after_close(CAL[s + 2]))])
    sync(exp, cfg, store, now_utc=after_close(CAL[s + 2]))
    (res,) = results_on(exp, "BENCHMARK", CAL[s + 1])
    assert res["status"] != "LATE_DATA_NOT_EXECUTABLE"
    assert res["decision_data_available_cst"].startswith(f"{CAL[s + 1].date()}T08:30")
    assert independent_verify(exp, store.dir)["problems"] == []


def test_order_is_not_executable_when_decision_data_arrived_after_open(tmp_path):
    s = 200
    store, cfg, exp = staged_forward(tmp_path, [(s - 1, after_close(CAL[s - 1])), (s, at_cst(CAL[s + 1], 10, 0)),
                                                (s + 1, after_close(CAL[s + 1])), (s + 2, after_close(CAL[s + 2]))])
    sync(exp, cfg, store, now_utc=after_close(CAL[s + 2]))
    (res,) = results_on(exp, "BENCHMARK", CAL[s + 1])
    assert res["status"] == "LATE_DATA_NOT_EXECUTABLE" and "不补造成交" in res["reason"]
    assert "price" not in res and "fees" not in res
    close = next(e for e in Journal(exp, "BENCHMARK").records()[1]["events"] if e["type"] == "DAY_CLOSE")
    assert close["holding"] == 0 and close["cash"] == cfg.capital        # 没有补造的持仓
    # D+1 日收盘后行情按时到达：用 D+1 的数据重新决策，D+2 正常撮合
    (res2,) = results_on(exp, "BENCHMARK", CAL[s + 2])
    assert res2["status"] != "LATE_DATA_NOT_EXECUTABLE"
    assert independent_verify(exp, store.dir)["problems"] == []


def test_strategy_ledgers_get_no_fills_from_one_shot_catch_up(tmp_path):
    """A/B 策略账本同样受约束：所有行情在观察期结束后一次性获取时，任何账本都不得出现成交。"""
    store, cfg, exp = staged_forward(tmp_path, [(len(CAL) - 1, after_close(CAL[-1]))], start_i=80)
    sync(exp, cfg, store, now_utc=after_close(CAL[-1]))
    seen = {}
    for L in ("PAPER_STRICT", "PAPER_TOUCH", "BENCHMARK"):
        for r in Journal(exp, L).records()[:-1]:                          # 最后一天的订单还没到撮合日
            for e in r["events"]:
                if e["type"] == "ORDER_RESULT":
                    seen.setdefault(L, set()).add(e["status"])
    assert seen["PAPER_STRICT"] == seen["PAPER_TOUCH"] == seen["BENCHMARK"] == {"LATE_DATA_NOT_EXECUTABLE"}
    assert independent_verify(exp, store.dir)["problems"] == []
    # 同一份行情按演练回放会有成交，说明上面的结果来自时间约束而不是没有信号
    rehearsal = make_config("reh", CODE, str(CAL[80].date()), SCENARIOS["B"], {"selected": CODE}, rehearsal=True)
    sim = simulate(rehearsal, R, store, CAL[-1])
    assert any(e["type"] == "ORDER_RESULT" and e["status"] == "STRICT_FILL" for r in sim["PAPER_STRICT"] for e in r["events"])


def test_independent_verify_flags_fill_whose_data_arrived_after_open(tmp_path):
    s = 200
    store, cfg, exp = staged_forward(tmp_path, [(s, after_close(CAL[s])), (s + 2, after_close(CAL[s + 2]))])
    sync(exp, cfg, store, now_utc=after_close(CAL[s + 2]))
    bars = pd.read_csv(store.dir / "bars.csv", dtype=str)
    bars.loc[bars["date"] == str(CAL[s].date()), "first_seen_utc"] = at_cst(CAL[s + 1], 10).isoformat()
    bars.to_csv(store.dir / "bars.csv", index=False)
    problems = independent_verify(exp, store.dir)["problems"]
    assert any("晚于成交日 09:15" in p for p in problems)


# ---------- MANUAL_LIVE 真实轨道（双轨方案 §13 T1–T23；actual_broker=True 的测试费率，合成行情） ----------

REAL = FeeScenario("REAL_test", commission_rate=0.0001, min_commission=0.1, actual_broker=True)
LOW = np.round(path_closes() * 0.5, 3)          # 约 0.28 元：100 元本金可买 300 份，才能构造部分成交
DI = 200
REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def ds(i):
    return str(CAL[i].date())


class Live:
    """与真实目录同构：data/pilot/{market,selection,experiments}、reports/etf_pilot/<实验>/observation。"""

    def __init__(self, tmp_path, bars=None, capital=True, fee=REAL, start_i=DI - 2, rehearsal=False):
        import hashlib
        pilot = tmp_path / "data" / "pilot"
        self.sel = pilot / "selection"
        self.sel.mkdir(parents=True)
        (self.sel / "selection_x.json").write_text('{"selected": "sz159596"}', encoding="utf-8")
        sha = hashlib.sha256((self.sel / "selection_x.json").read_bytes()).hexdigest()
        self.full = make_bars(LOW)
        bars = self.full if bars is None else bars
        self.last = pd.Timestamp(bars["date"].iloc[-1])
        self.store = ETFMarketStore(pilot / "market", CODE, FakeFetcher(bars))
        self.store.fetch(after_close(self.last))
        self.cfg = make_config("live", CODE, ds(start_i), fee, {"file": "selection_x.json", "sha256": sha}, rehearsal=rehearsal)
        self.exp = init_experiment(pilot / "experiments", self.cfg)
        self.out = tmp_path / "reports" / "etf_pilot" / self.exp.name / "observation"
        if capital:
            mx.record_capital(self.exp, self.cfg, 100.0, ds(DI - 1), EV, now_utc=after_close(CAL[DI - 1]))

    def bar(self, i=DI):
        b = self.full.set_index(pd.to_datetime(self.full["date"])).loc[CAL[i]]
        return float(b["low"]), float(b["high"])

    def order(self, qty=300, oid="O1", side="BUY", i=DI, t="09:30:00", limit=None, **kw):
        lo, hi = self.bar(i)
        limit = limit if limit is not None else (hi if side == "BUY" else lo)
        return mx.record_order(self.exp, self.cfg, self.store, ds(i), t, side, qty, limit, oid, EV,
                               now_utc=after_close(CAL[i]), **kw)

    def fill(self, qty, tid, oid="O1", i=DI, t="09:31:00", price=None, **kw):
        price = price if price is not None else self.bar(i)[0]
        return mx.record_fill(self.exp, self.cfg, self.store, ds(i), t, oid, qty, price, tid, EV,
                              now_utc=after_close(CAL[i]), **kw)

    def cancel(self, qty, oid="O1", i=DI, t="09:40:00", **kw):
        return mx.record_cancel(self.exp, self.cfg, self.store, ds(i), t, oid, qty, EV, now_utc=after_close(CAL[i]), **kw)

    def expire(self, qty, oid="O1", i=DI, now=None):
        return mx.record_expired(self.exp, self.cfg, self.store, ds(i), oid, qty, EV, now_utc=now or after_close(CAL[i]))

    def settle(self, fees, oid="O1", i=DI, qty=None, amount=None, ref="ST1"):
        o = self.state().book[oid]
        qty = o["F"] if qty is None else qty
        amount = sum(x["qty"] * x["price"] for x in o["fills"]) if amount is None else amount
        return mx.record_settlement(self.exp, self.cfg, self.store, ds(i), oid, qty, amount, fees, ref, EV,
                                    now_utc=after_close(CAL[i]))

    def state(self):
        return mx.manual_state(self.exp, self.cfg, self.store.calendar())

    def lines(self):
        return len(Journal(self.exp, "MANUAL_LIVE").records())

    def suggest(self, sid, decision_i, side="BUY", qty=100, limit=None, action="PROPOSE", gen=None, raw="same", **order):
        intent = {"side": side, "since": ds(decision_i)}
        hi = self.bar(min(decision_i + 1, len(CAL) - 1))[1]
        o = {"side": side, "qty": qty, "limit_price": hi if limit is None else limit, **order} if action == "PROPOSE" else None
        rec = {"suggestion_id": sid, "decision_day": ds(decision_i), "raw_intent": intent if raw == "same" else raw,
               "intent": intent, "generated_at_utc": (gen or after_close(CAL[decision_i])).isoformat(),
               "proposal": {"action": action, "order": o, "reason": None if o else "预算不足"}}
        with (self.exp / "suggestions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def report(self, now=None):
        from quantrisk.pilot.live_report import write_live_daily
        write_live_daily(self.exp, self.cfg, self.store, self.sel, REPO, self.out, now_utc=now or after_close(self.last))
        return pd.read_csv(self.out / "live_daily.csv").set_index("date")

    def verify(self):
        return independent_verify(self.exp, self.store.dir)["problems"]


# 状态机 T1–T7

def test_t1_order_filled_in_full(tmp_path):
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(300)
    L.fill(300, "T1")
    st = L.state()
    o = st.book["O1"]
    assert (o["status"], o["terminal"], o["R"], o["fees_pending"]) == ("FILLED", True, 0, True)
    assert st.acct.cash == pytest.approx(100 - 300 * lo) and st.acct.quantity(CODE) == 300
    assert L.verify() == []


def test_t2_two_partial_fills_complete_the_order_with_correct_vwap(tmp_path):
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(300)
    L.fill(100, "T1")
    assert L.state().book["O1"]["status"] == "OPEN"
    L.fill(200, "T2", t="09:35:00", price=round(lo + 0.001, 3))
    o = L.state().book["O1"]
    assert o["status"] == "FILLED" and o["vwap"] == pytest.approx((100 * lo + 200 * (lo + 0.001)) / 300)


def test_t3_t5_partial_fill_closed_by_cancel_or_expiry(tmp_path):
    L = Live(tmp_path / "c")
    L.order(300)
    L.fill(100, "T1")
    L.cancel(200)
    o = L.state().book["O1"]
    assert (o["status"], o["F"], o["C"], o["R"]) == ("PARTIAL_CLOSED", 100, 200, 0)
    E = Live(tmp_path / "e")
    E.order(300)
    E.fill(100, "T1")
    E.expire(200)
    o = E.state().book["O1"]
    assert (o["status"], o["R"]) == ("PARTIAL_CLOSED", 0) and o["expired"]["expired_qty"] == 200
    assert L.verify() == [] and E.verify() == []


def test_t4_partial_cancel_keeps_order_open(tmp_path):
    L = Live(tmp_path)
    L.order(300)
    L.cancel(100)
    o = L.state().book["O1"]
    assert (o["status"], o["terminal"], o["R"]) == ("OPEN", False, 200)
    L.fill(200, "T1", t="09:45:00")
    o = L.state().book["O1"]
    assert (o["status"], o["R"]) == ("PARTIAL_CLOSED", 0)


def test_t6_whole_order_expired_cancelled_or_rejected(tmp_path):
    L = Live(tmp_path)
    L.order(300, "O1")
    L.expire(300, "O1")
    L.order(300, "O2", t="10:00:00")
    L.cancel(300, "O2", t="10:01:00")
    L.order(300, "O3", t="10:02:00")
    mx.record_reject(L.exp, L.cfg, L.store, ds(DI), "10:02:01", "O3", "资金不足", EV, now_utc=after_close(CAL[DI]))
    st = L.state()
    assert [st.book[k]["status"] for k in ("O1", "O2", "O3")] == ["EXPIRED", "CANCELLED", "REJECTED"]
    assert st.acct.cash == 100 and st.open_orders == [] and L.verify() == []


def test_t7_illegal_transitions_are_refused(tmp_path):
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(100, "O1")
    L.fill(100, "T1", "O1")
    with pytest.raises(ManualEntryError, match="终态"):
        L.fill(100, "T9", "O1", t="09:50:00")
    with pytest.raises(ManualEntryError, match="终态"):
        L.cancel(100, "O1")
    L.order(200, "O2", t="10:00:00")
    L.fill(100, "T2", "O2", t="10:01:00")
    with pytest.raises(ManualEntryError, match="已有成交"):
        mx.record_reject(L.exp, L.cfg, L.store, ds(DI), "10:02", "O2", "x", EV, now_utc=after_close(CAL[DI]))
    with pytest.raises(ManualEntryError, match="剩余未成交数量"):
        L.expire(50, "O2")
    with pytest.raises(ManualEntryError, match="剩余数量"):
        L.cancel(150, "O2", t="10:03:00")
    with pytest.raises(ManualEntryError, match="剩余数量"):
        L.fill(150, "T3", "O2", t="10:03:00")
    with pytest.raises(ManualEntryError, match="委托日期"):
        L.fill(100, "T3", "O2", i=DI + 1)
    with pytest.raises(ManualEntryError, match="早于委托时间"):
        L.fill(100, "T3", "O2", t="09:59:00")
    with pytest.raises(ManualEntryError, match="15:00"):
        L.expire(100, "O2", now=datetime(CAL[DI].year, CAL[DI].month, CAL[DI].day, 14, 0, tzinfo=CST))
    assert L.state().book["O2"]["status"] == "OPEN"


# 幂等与引用 T8–T10

def test_t8_identical_reentry_is_idempotent_and_conflicting_reentry_is_refused(tmp_path):
    L = Live(tmp_path)
    lo, _ = L.bar()
    first = L.order(100, "O1")
    n = L.lines()
    assert L.order(100, "O1")["hash"] == first["hash"] and L.lines() == n
    with pytest.raises(ManualEntryError, match="内容不同"):
        L.order(200, "O1")
    f = L.fill(100, "T1", "O1")
    n = L.lines()
    assert L.fill(100, "T1", "O1")["hash"] == f["hash"] and L.lines() == n           # 委托已 FILLED 仍可原样重录
    with pytest.raises(ManualEntryError, match="内容不同"):
        L.fill(100, "T1", "O1", price=round(lo + 0.001, 3))
    L.order(100, "O2", t="10:00:00")
    c = L.cancel(50, "O2", t="10:01:00", broker_cancel_id="C1")
    assert L.cancel(50, "O2", t="10:01:00", broker_cancel_id="C1")["hash"] == c["hash"]
    with pytest.raises(ManualEntryError, match="内容不同"):
        L.cancel(40, "O2", t="10:01:00", broker_cancel_id="C1")
    s = L.settle({"commission": 0.1}, "O1")
    assert L.settle({"commission": 0.1}, "O1")["hash"] == s["hash"]
    d = mx.record_dividend(L.exp, L.cfg, ds(DI + 2), 0.3, EV, broker_ref="D1", now_utc=after_close(CAL[DI + 2]))
    assert mx.record_dividend(L.exp, L.cfg, ds(DI + 2), 0.3, EV, broker_ref="D1", now_utc=after_close(CAL[DI + 2]))["hash"] == d["hash"]
    with pytest.raises(ManualEntryError, match="内容不同"):
        mx.record_dividend(L.exp, L.cfg, ds(DI + 2), 0.4, EV, broker_ref="D1")
    cap = L.state().capital
    assert mx.record_capital(L.exp, L.cfg, 100.0, ds(DI - 1), EV)["hash"] == cap["hash"]
    with pytest.raises(ManualEntryError, match="不允许追加"):
        mx.record_capital(L.exp, L.cfg, 50.0, ds(DI - 1), EV)
    assert L.verify() == []


def test_t9_dangling_reference_and_second_settlement_are_refused(tmp_path):
    L = Live(tmp_path)
    with pytest.raises(ManualEntryError, match="不存在"):
        L.fill(100, "T1", "NOPE")
    L.order(100)
    L.fill(100, "T1")
    L.settle({"commission": 0.1})
    with pytest.raises(ManualEntryError, match="已有交割记录"):
        L.settle({"commission": 0.2}, ref="ST2")


def test_t10_void_and_single_reentry(tmp_path):
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(100)
    f = L.fill(100, "T1")
    with pytest.raises(ManualEntryError, match="逐条作废"):
        mx.void_record(L.exp, L.cfg, L.state().book["O1"]["order"]["hash"], "x", EV)
    v = mx.void_record(L.exp, L.cfg, f["hash"], "成交价录错", EV, now_utc=after_close(CAL[DI]))
    st = L.state()
    assert st.book["O1"]["status"] == "SUBMITTED" and st.acct.quantity(CODE) == 0 and st.acct.cash == 100
    f2 = L.fill(100, "T1", price=round(lo + 0.001, 3))
    assert f2["reentry_of"] == f["hash"] and L.state().book["O1"]["status"] == "FILLED"
    with pytest.raises(ManualEntryError, match="作废过一次"):
        mx.void_record(L.exp, L.cfg, f["hash"], "again", EV)
    mx.void_record(L.exp, L.cfg, f2["hash"], "又录错", EV, now_utc=after_close(CAL[DI]))
    with pytest.raises(ManualEntryError, match="不允许再次重录"):
        L.fill(100, "T1")
    with pytest.raises(ManualEntryError, match="CAPITAL"):
        mx.void_record(L.exp, L.cfg, L.state().capital["hash"], "x", EV)
    with pytest.raises(ManualEntryError, match="VOID"):
        mx.void_record(L.exp, L.cfg, v["hash"], "x", EV)
    assert L.verify() == []


# 费用 T11–T14

def test_t11_settlement_fees_apportioned_by_quantity_with_remainder_on_last_fill(tmp_path):
    parts = mx.apportion_fees({"commission": 0.05, "transfer_fee": 0.01}, [{"qty": 100}, {"qty": 200}])
    assert parts == [{"commission": 0.02, "transfer_fee": 0.0}, {"commission": 0.03, "transfer_fee": 0.01}]
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(300)
    L.fill(100, "T1")
    L.fill(200, "T2", t="09:35:00")
    L.settle({"commission": 0.05, "transfer_fee": 0.01})
    assert L.state().acct.cash == pytest.approx(100 - 300 * lo - 0.06) and L.verify() == []


def test_t12_fees_pending_then_restated_and_fee_hint_not_booked(tmp_path):
    L = Live(tmp_path)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    lo, _ = L.bar()
    L.order(300)
    L.fill(300, "T1", fee_hint={"commission": 9.99})
    df = L.report()
    assert df.at[ds(DI), "live_status"] == "FEES_PENDING"
    assert df.at[ds(DI), "live_cash"] == pytest.approx(100 - 300 * lo)                # fee_hint 不入账
    s = L.settle({"commission": 0.05})
    assert "FEE_HINT_MISMATCH" in s["flags"]
    df = L.report()
    assert df.at[ds(DI), "live_status"] == "OK" and bool(df.at[ds(DI), "restated"])
    assert df.at[ds(DI), "live_cash"] == pytest.approx(100 - 300 * lo - 0.05)
    assert L.verify() == []


def test_t13_settlement_must_match_fills_and_has_no_stamp_duty(tmp_path):
    L = Live(tmp_path)
    L.order(100)
    L.fill(100, "T1")
    with pytest.raises(ManualEntryError, match="交割数量"):
        L.settle({"commission": 0.1}, qty=200)
    with pytest.raises(ManualEntryError, match="交割金额"):
        L.settle({"commission": 0.1}, amount=1.0)
    with pytest.raises(ManualEntryError, match="印花税"):
        L.settle({"commission": 0.1, "stamp_duty": 0.01})
    s = L.settle({"commission": 0.1, "transfer_fee": 0.01})
    assert s["warnings"] and "过户费" in s["warnings"][0]


def test_t14_scenario_difference_is_audit_only(tmp_path):
    from quantrisk.pilot.etf_rules import explicit_fee_total
    L = Live(tmp_path)
    lo, _ = L.bar()
    L.order(300)
    L.fill(300, "T1")
    s = L.settle({"commission": 0.5})
    est = explicit_fee_total(REAL.breakdown(lo, 300))
    assert s["audit"]["scenario_estimate_total"] == est and s["audit"]["diff_vs_scenario"] == round(0.5 - est, 2)
    assert L.state().acct.cash == pytest.approx(100 - 300 * lo - 0.5)                  # 账本只认交割单


# 标记 T15–T17

def test_t15_suggestion_flags_are_recorded_and_flagged_fills_still_booked(tmp_path):
    L = Live(tmp_path)
    lo, hi = L.bar()
    L.suggest("S-ok", DI - 1)
    L.suggest("S-stale", DI - 3, limit=hi)
    L.suggest("S-rej", DI - 1, action="REJECT")
    L.suggest("S-sell", DI - 1, side="SELL")
    cases = [(None, 100, hi, ["NO_SUGGESTION"]), ("S-ok", 100, hi, []), ("S-stale", 100, hi, ["SUGGESTION_STALE"]),
             ("S-rej", 100, hi, ["SUGGESTION_REJECTED"]), ("S-sell", 100, hi, ["SIDE_MISMATCH"]),
             ("S-ok", 100, round(hi - 0.001, 3), ["LIMIT_MISMATCH"])]
    for k, (sid, qty, lim, flags) in enumerate(cases):
        rec = L.order(qty, f"O{k}", t=f"10:0{k}:00", limit=lim, suggestion_id=sid)
        assert sorted(rec["flags"]) == flags, sid
        L.cancel(qty, f"O{k}", t=f"10:0{k}:30")
    rec = L.order(200, "OQ", t="11:00:00", suggestion_id="S-ok")
    assert rec["flags"] == ["QTY_MISMATCH"]
    L.fill(200, "TQ", "OQ", t="11:01:00")
    assert L.state().acct.quantity(CODE) == 200                                       # 带标记的成交照常入账
    assert L.verify() == []                                                           # 独立复算的标记与录入一致


def test_t16_order_before_suggestion_and_outside_limit_band_are_refused(tmp_path):
    L = Live(tmp_path)
    L.suggest("S-late", DI - 1, gen=after_close(CAL[DI]))
    with pytest.raises(ManualEntryError, match="ORDER_BEFORE_SUGGESTION"):
        L.order(100, suggestion_id="S-late")
    with pytest.raises(ManualEntryError, match="OUTSIDE_LIMIT_BAND"):
        L.order(100, limit=round(LOW[DI - 1] * 1.15, 3))
    assert L.lines() == 1


def test_t17_over_budget_both_sides_and_price_unverified_until_bar_arrives(tmp_path):
    from quantrisk.pilot.live_report import eligibility
    for fee, flagged in ((0.0, False), (1.0, True)):
        L = Live(tmp_path / f"b{fee}")
        L.order(300)
        L.fill(300, "T1")
        s = L.settle({"commission": fee})
        assert ("OVER_BUDGET_ACTUAL" in s["flags"]) is flagged
        assert (s["audit"]["actual_ratio"] > L.cfg.friction_budget) is flagged and L.verify() == []
    for price_off, ok in ((0.0, True), (-0.002, False)):
        L = Live(tmp_path / f"u{price_off}", bars=make_bars(LOW).iloc[:DI + 1])           # 行情只到 DI
        lo, hi = L.bar(DI + 1)
        L.order(100, i=DI + 1, limit=hi)
        f = L.fill(100, "T1", i=DI + 1, price=round(lo + price_off, 3))
        assert f["flags"] == ["PRICE_UNVERIFIED"]
        chk = lambda: next(c for c in eligibility(L.exp, L.cfg, L.store, after_close(CAL[DI + 1]))["checks"] if c["check"].startswith("4"))
        assert not chk()["ok"]
        L.store.fetcher = FakeFetcher(L.full)
        L.store.fetch(after_close(CAL[-1]))
        assert chk()["ok"]
        assert (not any("不在当日区间" in p for p in L.verify())) is ok


# 日报与复算 T18–T23

def test_t18_no_capital_report_has_no_live_numbers(tmp_path):
    L = Live(tmp_path, capital=False)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    df = L.report()
    assert (df["live_status"] == "NO_ACCOUNT").all()
    assert df[["live_cash", "live_equity", "live_cum_return"]].isna().all().all()
    assert "NO_ACCOUNT" in (L.out / "live_daily.md").read_text(encoding="utf-8")
    summary = json.loads((L.out / "live_summary.json").read_text(encoding="utf-8"))
    assert summary["eligibility"]["first_failure"]["check"].startswith("1")


def test_t19_signal_consistency_same_and_diff(tmp_path):
    from quantrisk.pilot.live_report import paper_raw_intents
    L = Live(tmp_path, capital=False)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    paper = paper_raw_intents(L.cfg, L.store, L.last)[ds(DI)]
    L.suggest("S-same", DI, raw=paper)
    assert L.report().at[ds(DI), "signal_consistency"] == "SAME" and L.verify() == []
    (L.exp / "suggestions.jsonl").unlink()
    other = "SELL" if (paper or {}).get("side") == "BUY" else "BUY"
    L.suggest("S-diff", DI, raw={"side": other, "since": ds(DI)})
    assert L.report().at[ds(DI), "signal_consistency"] == "DIFF"
    assert any("signal_consistency=DIFF" in p for p in L.verify())                    # daily 因此记 VERIFY_FAILED


def test_t20_price_and_fee_differences_match_hand_calculation(tmp_path):
    from quantrisk.pilot.etf_rules import explicit_fee_total
    L = Live(tmp_path)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    lo, hi = L.bar()
    L.order(300, limit=hi)
    L.fill(300, "T1", price=lo)
    L.settle({"commission": 0.05})
    r = L.report().loc[ds(DI)]
    est = explicit_fee_total(REAL.breakdown(lo, 300))
    assert r["live_vwap"] == pytest.approx(lo) and r["live_vs_limit"] == pytest.approx(lo - hi)
    assert r["live_vs_decision_close"] == pytest.approx(lo - LOW[DI - 1])
    assert r["live_fee"] == pytest.approx(0.05) and r["scenario_est_fee"] == pytest.approx(est)
    assert r["fee_diff_vs_scenario"] == pytest.approx(0.05 - est)
    assert L.verify() == []


def test_t21_eligibility_reports_first_failing_condition(tmp_path):
    from quantrisk.pilot.live_report import eligibility
    last_i = len(CAL) - 1
    nxt = CAL[-1] + timedelta(days=1)                                                 # 合成日历的下一交易日
    good = dict(est_total_ratio=0.001, notional_at_limit=30.0, est_explicit_total=0.1)

    def first(tag, capital=True, now=None, setup=None):
        L = Live(tmp_path / tag, capital=capital)
        if setup:
            setup(L)
        e = eligibility(L.exp, L.cfg, L.store, now or after_close(L.last))
        return e["status"], (e["first_failure"] or {}).get("check", "")[:2].strip()

    assert first("ok", setup=lambda L: L.suggest("S", last_i, **good)) == ("ELIGIBLE", "")
    assert first("c1", capital=False)[1] == "1"
    assert first("c2", now=after_close(CAL[-1] + timedelta(days=10)))[1] == "2"
    assert first("c3", setup=lambda L: (L.order(100, i=DI), L.suggest("S", last_i, **good)))[1] == "3"
    assert first("c5")[1] == "5"
    assert first("c6", setup=lambda L: L.suggest("S", last_i, action="REJECT"))[1] == "6"
    assert first("c7", setup=lambda L: L.suggest("S", last_i, **{**good, "est_total_ratio": 0.02}))[1] == "7"
    assert first("c8", setup=lambda L: L.suggest("S", last_i, **{**good, "notional_at_limit": 150.0}))[1] == "8"
    late = datetime(nxt.year, nxt.month, nxt.day, 9, 20, tzinfo=CST).astimezone(timezone.utc)
    assert first("c10", now=late, setup=lambda L: L.suggest("S", last_i, **good))[1] == "10"


def _verify_cli(exp, market, script=REPO / "scripts" / "verify_etf_ledger.py"):
    """以子进程运行独立复算脚本（不导入本库）：返回 (退出码, 解析后的 JSON 输出)。"""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(script), str(exp), str(market)], capture_output=True, text=True, timeout=120)
    return r.returncode, json.loads(r.stdout)


def _rechain(recs):
    """按流水哈希公式重建序号与哈希链：模拟“知道哈希公式的篡改者”，此时只有语义复算能发现问题。"""
    import hashlib
    prev = None
    for i, r in enumerate(recs):
        r["seq"], r["prev_hash"] = i, prev
        body = {k: v for k, v in r.items() if k != "hash"}
        r["hash"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        prev = r["hash"]
    return recs


def _forge(L, name, ledger, mutate, rechain=True):
    """复制实验目录，篡改副本的某本流水后以子进程复算；原实验不动。"""
    import shutil
    dst = L.exp.parent / f"{L.exp.name}_{name}"
    shutil.copytree(L.exp, dst)
    p = dst / "journals" / f"{ledger}.jsonl"
    recs = mutate([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()])
    recs = _rechain(recs) if rechain else recs
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs), encoding="utf-8")
    return _verify_cli(dst, L.store.dir)


def _child(template, **fields):
    keep = {k: template[k] for k in ("ledger", "source", "written_at_utc")}
    return {**keep, "flags": [], "note": "", "evidence": EV, **fields}


def test_t22_independent_verifier_subprocess_on_full_paper_and_manual_journals(tmp_path):
    L = Live(tmp_path, start_i=80, rehearsal=True)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    lo, hi = L.bar()
    p2 = round(lo + 0.001, 3)
    L.order(300, "O1")                                                              # 部分成交 + 撤单 + 交割
    L.fill(100, "T1", "O1")
    L.fill(100, "T2", "O1", t="09:35:00", price=p2)
    L.cancel(100, "O1", t="09:40:00")
    L.settle({"commission": 0.05}, "O1")
    slo, shi = L.bar(DI + 1)
    L.order(200, "O2", side="SELL", i=DI + 1, limit=slo)                             # 次日全部卖出
    L.fill(200, "T3", "O2", i=DI + 1, price=shi)
    L.settle({"commission": 0.02}, "O2", i=DI + 1, ref="ST2")
    assert L.state().book["O1"]["status"] == "PARTIAL_CLOSED" and L.state().book["O2"]["status"] == "FILLED"

    rc, out = _verify_cli(L.exp, L.store.dir)
    assert rc == 0 and out["problems"] == []
    c = out["checked"]
    assert c["fills"] > 0 and c["days"] > 0                                          # PAPER 确有模拟成交被逐笔复算
    assert c["manual_orders"] == 2 and c["manual_holding"] == 0 and c["manual_fee_total"] == pytest.approx(0.07)
    assert c["manual_cash"] == pytest.approx(100 - 100 * lo - 100 * p2 - 0.05 + 200 * shi - 0.02)

    def first(recs, **kv):
        return next(r for r in recs if all(r.get(k) == v for k, v in kv.items()))

    def paper_fill_fee(recs):                                                        # 模拟流水：改一笔成交费用
        for r in recs:
            for e in r["events"]:
                if e.get("type") == "ORDER_RESULT" and e.get("price") is not None:
                    e["fees"] = {k: round(v + 0.01, 2) for k, v in e["fees"].items()} or {"commission": 0.01}
                    return recs
        raise AssertionError("PAPER 没有成交")

    def manual_price(recs):                                                          # 真实流水：改成交金额
        first(recs, type="BROKER_FILL", broker_trade_id="T1")["price"] = p2
        return recs

    def reject_after_fill(recs):                                                     # 状态错误：有成交仍记拒单
        return recs + [_child(recs[0], type="BROKER_REJECT", day=ds(DI), trade_date=ds(DI), reject_time="15:01",
                              broker_order_id="O1", reason_text="伪造")]

    def wrong_expiry(recs):                                                          # 状态错误：作废数量 ≠ 剩余
        o = [r for r in recs if r["type"] != "BROKER_CANCEL"]
        return o + [_child(recs[0], type="BROKER_EXPIRED", day=ds(DI), trade_date=ds(DI), broker_order_id="O1", expired_qty=50)]

    def dup_trade(recs):                                                             # 重复成交编号
        return recs + [{k: v for k, v in first(recs, type="BROKER_FILL", broker_trade_id="T3").items()}]

    def overcancel(recs):                                                            # 成交 + 撤单 > 委托数量
        first(recs, type="BROKER_CANCEL")["cancelled_qty"] = 200
        return recs

    def settle_qty(recs):                                                            # 交割数量与成交不符
        first(recs, type="BROKER_SETTLEMENT", broker_order_id="O1")["settled_qty"] = 300
        return recs

    cases = [("paper_fee", "PAPER_STRICT", paper_fill_fee, "≠ 独立重算"),
             ("price", "MANUAL_LIVE", manual_price, "交割 O1：金额"),
             ("reject", "MANUAL_LIVE", reject_after_fill, "拒单却有成交或撤单"),
             ("expiry", "MANUAL_LIVE", wrong_expiry, "作废数量 50 ≠ 剩余 100"),
             ("dup", "MANUAL_LIVE", dup_trade, "自然键重复：('BROKER_FILL', 'T3')"),
             ("overcancel", "MANUAL_LIVE", overcancel, "超出 Q=300"),
             ("settle", "MANUAL_LIVE", settle_qty, "交割 O1：数量 300 ≠ 成交 200")]
    for name, ledger, mutate, expect in cases:
        rc, out = _forge(L, name, ledger, mutate)
        assert rc == 1 and any(expect in p for p in out["problems"]), (name, out["problems"])
    rc, out = _forge(L, "raw", "MANUAL_LIVE", manual_price, rechain=False)                # 不重建哈希链 → 链断
    assert rc == 1 and any("哈希链" in p for p in out["problems"])
    rc, _ = _verify_cli(L.exp, L.store.dir)
    assert rc == 0                                                                   # 原实验未受篡改影响

    B = Live(tmp_path / "sim", capital=False, fee=SCENARIOS["B"])                     # 模拟费率实验出现真实记录
    Journal(B.exp, "MANUAL_LIVE").append({"type": "CAPITAL", "day": ds(DI), "trade_date": ds(DI), "amount": 100.0,
                                          "evidence": EV}, mx.SOURCE)
    rc, out = _verify_cli(B.exp, B.store.dir)
    assert rc == 1 and any("actual_broker=false" in p for p in out["problems"])


def _real_fingerprint():
    """真实前向实验依赖的全部文件（源码、规则、策略、配置、基线、流水）的哈希；测试前后必须一致。"""
    import hashlib
    files = sorted((REPO / "src/quantrisk/pilot").glob("*.py")) + sorted((REPO / "src/quantrisk/pilot").glob("*.json"))
    files += sorted((REPO / "src/quantrisk/strategies").rglob("*.py")) + [REPO / "scripts/etf_pilot.py", REPO / "scripts/verify_etf_ledger.py"]
    exps = REPO / "data/pilot/experiments"
    if exps.exists():
        for d in sorted(exps.iterdir()):
            files += [d / "config.json", d / "config.sha256", d / "observation" / "baseline.json"] + sorted((d / "journals").glob("*.jsonl"))
    return {str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}


def test_t23_pilot_code_change_in_temp_copy_is_detected_and_recorded(tmp_path):
    import shutil
    from quantrisk.pilot.observation import check_baseline, log_run, run_log
    from quantrisk.pilot.live_report import write_live_daily
    real_before = _real_fingerprint()
    root = tmp_path / "root"                                                         # 临时副本：只复制哈希涉及的文件
    shutil.copytree(REPO / "src/quantrisk/pilot", root / "src/quantrisk/pilot", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    for n in ("etf_pilot.py", "verify_etf_ledger.py"):
        shutil.copy2(REPO / "scripts" / n, root / "scripts" / n)
    L = Live(root, capital=False)
    sync(L.exp, L.cfg, L.store, now_utc=after_close(L.last))
    base = check_baseline(L.exp, L.cfg, L.sel, root)
    assert base["hard"] == [] and base["soft"] == []
    script = root / "scripts" / "verify_etf_ledger.py"
    rc, out = _verify_cli(L.exp, L.store.dir, script)
    assert rc == 0 and out["checked"]["hashes"]["pilot_code_changed_since_baseline"] is False

    target = root / "src/quantrisk/pilot/paper_execution.py"                          # 只改副本
    target.write_text(target.read_text(encoding="utf-8") + "\n# T23: 仅测试副本中的改动\n", encoding="utf-8")
    rc, out = _verify_cli(L.exp, L.store.dir, script)
    h = out["checked"]["hashes"]
    assert h["pilot_code_changed_since_baseline"] is True and h["pilot_code_sha256"] != base["baseline"]["pilot_code_sha256"]
    assert rc == 0 and out["problems"] == []                                         # SOP：代码变化是软变化，记录不阻断
    now = check_baseline(L.exp, L.cfg, L.sel, root)
    assert now["soft"] == ["pilot_code_sha256"] and now["hard"] == []
    log_run(L.exp, {"command": "daily", "experiment": L.exp.name, "baseline_soft_changes": now["soft"], "status": "OK"})
    assert run_log(L.exp)[-1]["baseline_soft_changes"] == ["pilot_code_sha256"]       # 与 daily 相同的记录方式
    write_live_daily(L.exp, L.cfg, L.store, L.sel, root, L.out, now_utc=after_close(L.last))
    summary = json.loads((L.out / "live_summary.json").read_text(encoding="utf-8"))
    assert summary["soft_changes"] == ["pilot_code_sha256"] and summary["hard_changes"] == []
    assert "软变化，只记录" in (L.out / "live_daily.md").read_text(encoding="utf-8")   # 日报中可见，不被静默忽略
    assert _real_fingerprint() == real_before                                        # 真实源码、规则、策略、A/B 配置与流水不变


def test_t23_config_and_selection_changes_are_hard_changes(tmp_path):
    from quantrisk.pilot.observation import check_baseline
    L = Live(tmp_path)
    check_baseline(L.exp, L.cfg, L.sel, REPO)
    ver = independent_verify(L.exp, L.store.dir)
    assert ver["problems"] == [] and not ver["checked"]["hashes"]["pilot_code_changed_since_baseline"]
    (L.sel / "selection_x.json").write_text('{"selected": "sz159975"}', encoding="utf-8")
    assert any("基线硬变化" in p and "selection_sha256" in p for p in L.verify())
    (L.exp / "config.json").write_text((L.exp / "config.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("config_sha256" in p for p in L.verify())
