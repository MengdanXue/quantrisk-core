"""
M1 验收：基于真实 A 股行情快照（data/snapshots/ashare_2024，新浪财经，600519 / 000001，2024 全年）。

快照缺失时依赖快照的测试跳过并给出原因（QRC_REQUIRE_SNAPSHOTS=1 时改为失败）；重新下载：
    python -c "from quantrisk.data.ashare import download_snapshot as d; d(['600519','000001'],'2024-01-01','2024-12-31','data/snapshots/ashare_2024')"
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantrisk.data.ashare import DataUnavailableError, load_dataset
from quantrisk.strategies import get_strategy
from quantrisk.strategies.backtest.ashare_rules import ex_right_reference_price
from quantrisk.strategies.backtest.daily import AShareMarket, _Rules
from quantrisk.strategies.backtest.ledger import Account
from quantrisk.strategies.base import BaseStrategy, Signal
from scripts.verify_buy_and_hold import independent_buy_and_hold

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "ashare_2024"
needs_snapshot = pytest.mark.snapshot("ashare_2024")
SYMBOLS = ["600519", "000001"]


@pytest.fixture(scope="module", params=SYMBOLS)
def ds(request):
    return load_dataset(SNAPSHOT, request.param)


def run(strategy, ds, capital=1_000_000, bars=None, signal_data=None, **market_kw):
    return strategy.run(ds.signal_data if signal_data is None else signal_data, initial_capital=capital,
                        market=AShareMarket(ds.symbol, **market_kw), bars=ds.bars if bars is None else bars,
                        calendar=ds.calendar, corporate_actions=ds.corporate_actions, symbol=ds.symbol)


class Scripted(BaseStrategy):
    def __init__(self, script, pct=0.95):
        self.script = script
        super().__init__({"position_pct": pct})

    def default_params(self):
        return {"position_pct": 0.95}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        for date, v in self.script.items():
            s.loc[pd.Timestamp(date)] = v.value
        return s


# ---------- 数据 ----------

@needs_snapshot
def test_snapshot_is_real_and_complete(ds):
    assert ds.manifest["source"].startswith("sina")
    assert ds.quality["rows"] == ds.quality["calendar_days"] == 242
    assert ds.quality["invalid_bar_days"] == []


@needs_snapshot
def test_hfq_factor_matches_dividend_records(ds):
    checks = ds.quality["hfq_factor_check"]
    assert len(checks) == 2
    assert max(c["rel_diff"] for c in checks) < 1e-9


@needs_snapshot
def test_front_adjusted_prices_contain_future_information(ds):
    """前复权价 = 不复权价 ÷ 前复权因子，而前复权因子 = 最新后复权因子 ÷ 当日后复权因子：
    2024-01-02 的前复权价取决于快照下载时为止（远在 2024 年之后）的全部分红送转。"""
    qfq = pd.read_csv(SNAPSHOT / f"{ds.symbol}_qfq_factor.csv", parse_dates=["date"]).sort_values("date")
    hfq = pd.read_csv(SNAPSHOT / f"{ds.symbol}_hfq_factor.csv", parse_dates=["date"]).sort_values("date")
    day = pd.Timestamp("2024-01-02")
    q = float(qfq[qfq["date"] <= day]["qfq_factor"].iloc[-1])
    h_day = float(hfq[hfq["date"] <= day]["hfq_factor"].iloc[-1])
    h_latest = float(hfq["hfq_factor"].iloc[-1])
    assert hfq["date"].iloc[-1] > pd.Timestamp("2024-12-31")     # 用到了回测区间之后的公司行为
    assert q > 1.05                                                # 前复权价比当日真实成交价低 5% 以上
    assert q == pytest.approx(h_latest / h_day, rel=1e-9)


@needs_snapshot
def test_tampered_snapshot_is_rejected(tmp_path):
    copy = tmp_path / "snap"
    shutil.copytree(SNAPSHOT, copy)
    path = copy / "600519_daily_raw.csv"
    path.write_text(path.read_text().replace("1715.0", "1716.0", 1))
    with pytest.raises(DataUnavailableError, match="哈希不符"):
        load_dataset(copy, "600519")


def test_missing_snapshot_never_falls_back_to_synthetic(tmp_path):
    with pytest.raises(DataUnavailableError):
        load_dataset(tmp_path / "nope", "600519")


# ---------- 测试一：买入持有，独立核算 ----------

@needs_snapshot
def test_buy_and_hold_matches_independent_calculation(ds):
    first = ds.calendar[0]
    result = run(Scripted({first: Signal.BUY}), ds)
    independent = independent_buy_and_hold(SNAPSHOT, ds.symbol)
    fill = result.fills[0]
    assert fill["quantity"] == independent.attrs["quantity"]
    assert fill["price"] == pytest.approx(independent.attrs["fill_price"], rel=1e-12)
    assert fill["fee_total"] == pytest.approx(independent.attrs["fees"], abs=1e-9)
    np.testing.assert_allclose(result.ledger["equity"], independent["equity"], rtol=0, atol=1e-6)
    # 两次除息都已入账
    dividends = [j for j in result.journal if j["type"] == "DIVIDEND"]
    assert len(dividends) == 2


# ---------- 测试二：不可成交订单 ----------

def _limit_up_on(ds, day):
    i = ds.bars.index.get_loc(day)
    ref = ds.bars["close"].iloc[i - 1]
    for a in ds.corporate_actions:
        if a.ex_date == day:
            ref = ex_right_reference_price(ref, a.cash_per_share, a.bonus_per_share)
    return AShareMarket(ds.symbol).limits(day, ref)


@needs_snapshot
def test_buy_rejected_when_opening_at_limit_up():
    ds = load_dataset(SNAPSHOT, "000001")
    day = pd.Timestamp("2024-02-21")           # 真实行情：当日收盘封涨停
    assert ds.bars.loc[day, "close"] == _limit_up_on(ds, day)[1]
    # 真实当日开盘未涨停：次日开盘买单可以成交，且成交价不超过涨停价
    prev = ds.bars.index[ds.bars.index.get_loc(day) - 1]
    real = run(Scripted({prev: Signal.BUY}), ds)
    assert real.fills[0]["date"] == "2024-02-21"
    assert real.fills[0]["price"] <= _limit_up_on(ds, day)[1]
    # 把当日改成一字涨停：买单必须被拒绝
    bars = ds.bars.copy()
    up = _limit_up_on(ds, day)[1]
    bars.loc[day, ["open", "high", "low", "close"]] = up
    blocked = run(Scripted({prev: Signal.BUY}), ds, bars=bars)
    assert blocked.fills == []
    assert blocked.orders[0]["status"] == "REJECTED" and "涨停" in blocked.orders[0]["reason"]


@needs_snapshot
def test_sell_rejected_when_opening_at_limit_down(ds):
    buy_day, sell_signal = ds.calendar[0], ds.calendar[100]
    day = ds.calendar[101]
    i = ds.bars.index.get_loc(day)
    down = AShareMarket(ds.symbol).limits(day, ex_right_reference_price(
        ds.bars["close"].iloc[i - 1],
        *next(((a.cash_per_share, a.bonus_per_share) for a in ds.corporate_actions if a.ex_date == day), (0, 0))))[0]
    bars = ds.bars.copy()
    bars.loc[day, ["open", "high", "low", "close"]] = down
    result = run(Scripted({buy_day: Signal.BUY, sell_signal: Signal.SELL}), ds, bars=bars)
    sell = [o for o in result.orders if o["side"] == "SELL"][0]
    assert sell["status"] == "REJECTED" and "跌停" in sell["reason"]
    assert result.ledger.loc[day, "position"] > 0


@needs_snapshot
def test_suspended_day_rejects_order_and_marks_at_last_close(ds):
    day = ds.calendar[50]
    bars = ds.bars.drop(index=day)                    # 模拟当日停牌：没有任何成交
    prev = ds.calendar[49]
    result = run(Scripted({prev: Signal.BUY}), ds, bars=bars)
    assert result.orders[0]["status"] == "REJECTED" and "停牌" in result.orders[0]["reason"]
    assert bool(result.ledger.loc[day, "suspended"])
    assert result.ledger.loc[day, "close"] == ds.bars.loc[prev, "close"]


@needs_snapshot
def test_capital_below_one_lot_is_rejected():
    ds = load_dataset(SNAPSHOT, "600519")             # 约 1500 元/股，一手约 15 万元
    result = run(Scripted({ds.calendar[0]: Signal.BUY}), ds, capital=100_000)
    assert result.fills == []
    assert "不足一个申报单位" in result.orders[0]["reason"]


@needs_snapshot
def test_t_plus_one_blocks_same_day_sell():
    ds = load_dataset(SNAPSHOT, "600519")
    days = ds.calendar
    pos = {d: i for i, d in enumerate(days)}
    acct = Account(1_000_000)
    acct.buy(days[10], "600519", 100, 1700.0, {"commission": 42.5}, "B1")
    rules = _Rules(AShareMarket("600519"), acct, "600519", lambda d: pos[pd.Timestamp(d)])

    class _O:
        order_id, side, quantity = "S1", "SELL", 100

    class _Bar:
        open, high, low, close, timestamp = 1700.0, 1710.0, 1690.0, 1700.0, days[10]

    rules.set_day(days[10], 10, 1700.0)
    qty, reason = rules.check(_O, _Bar, 1700.0)
    assert qty == 0 and "T+1" in reason
    rules.set_day(days[11], 11, 1700.0)
    assert rules.check(_O, _Bar, 1700.0) == (100, "")


@needs_snapshot
def test_fills_never_violate_daily_range_or_limits(ds):
    for name in ["ma_cross", "turtle", "bollinger_reversion", "volatility_breakout"]:
        result = run(get_strategy(name, {"position_pct": 0.9}), ds)
        for f in result.fills:
            d = pd.Timestamp(f["date"])
            low, high = _limit_up_on(ds, d)
            bar = ds.bars.loc[d]
            assert bar["low"] - 1e-9 <= f["price"] <= bar["high"] + 1e-9
            assert low - 1e-9 <= f["price"] <= high + 1e-9
            assert f["quantity"] % 100 == 0


# ---------- 测试三：逐日资金一致性 ----------

@needs_snapshot
@pytest.mark.parametrize("name", ["ma_cross", "turtle", "rsi_reversion"])
def test_daily_ledger_reconciles(ds, name):
    capital = 1_000_000
    result = run(get_strategy(name, {"position_pct": 0.9}), ds, capital=capital)
    led = result.ledger
    np.testing.assert_allclose(led["equity"], led["cash"] + led["frozen"] + led["market_value"], rtol=1e-12)
    np.testing.assert_allclose(led["market_value"], led["position"] * led["close"], rtol=1e-12)
    assert (led["cash"] >= -1e-6).all()
    journal = pd.DataFrame(result.journal)
    assert journal["amount"].sum() == pytest.approx(led["cash"].iloc[-1], abs=1e-6)
    # 逐日：当日流水净额 = 当日（可用现金 + 冻结现金）的变化
    journal = journal[journal["date"].notna() & ~journal["type"].isin(["FREEZE", "UNFREEZE"])]
    by_day = journal.groupby("date")["amount"].sum()
    total_cash = led["cash"] + led["frozen"]
    change = total_cash.diff()
    change.iloc[0] = total_cash.iloc[0] - capital
    for d, delta in change.items():
        assert by_day.get(str(d.date()), 0.0) == pytest.approx(delta, abs=1e-6)
    # 已实现 + 未实现 = 总盈亏
    realized = sum(t["pnl"] for t in result.trades)
    account_lots = sum(f["quantity"] * f["price"] + f["fee_total"] for f in result.fills if f["side"] == "BUY")
    sold_cost = sum(t["entry_price"] * t["quantity"] for t in result.trades)
    open_cost = account_lots - sold_cost
    open_dividends = sum(j["amount"] for j in result.journal if j["type"] == "DIVIDEND") - sum(t["dividends"] for t in result.trades)
    unrealized = led["market_value"].iloc[-1] - open_cost + open_dividends
    assert realized + unrealized == pytest.approx(led["equity"].iloc[-1] - capital, abs=1e-4)


# ---------- 测试四：未来数据泄漏 ----------

@needs_snapshot
@pytest.mark.parametrize("name", ["ma_cross", "turtle", "macd", "bollinger_reversion"])
def test_changing_future_data_does_not_change_the_past(ds, name):
    k = 150
    cut = ds.calendar[k]
    base = run(get_strategy(name, {"position_pct": 0.9}), ds)
    bars = ds.bars.copy()
    signal = ds.signal_data.copy()
    bars.loc[bars.index >= cut, ["open", "high", "low", "close"]] *= 1.3
    signal.loc[signal.index >= cut, ["open", "high", "low", "close"]] *= 1.3
    changed = run(get_strategy(name, {"position_pct": 0.9}), ds, bars=bars, signal_data=signal)
    before = lambda r: r.ledger.loc[r.ledger.index < cut]
    pd.testing.assert_frame_equal(before(base), before(changed))
    pd.testing.assert_series_equal(base.signals[base.signals.index < cut], changed.signals[changed.signals.index < cut])
    assert [f for f in base.fills if f["date"] < str(cut.date())] == [f for f in changed.fills if f["date"] < str(cut.date())]


# ---------- 测试五：可复现 ----------

@needs_snapshot
def test_rerun_from_snapshot_is_identical():
    a = load_dataset(SNAPSHOT, "000001")
    b = load_dataset(SNAPSHOT, "000001")
    r1 = run(get_strategy("ma_cross", {"position_pct": 0.9}), a)
    r2 = run(get_strategy("ma_cross", {"position_pct": 0.9}), b)
    pd.testing.assert_frame_equal(r1.ledger, r2.ledger)
    assert r1.fills == r2.fills and r1.orders == r2.orders and r1.journal == r2.journal
    assert pd.util.hash_pandas_object(r1.ledger).sum() == pd.util.hash_pandas_object(r2.ledger).sum()
