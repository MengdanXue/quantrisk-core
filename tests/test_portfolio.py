"""M2：多证券组合回测（共享现金账户、风险预算、确定性分配、审计链）。"""
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantrisk.data.ashare import load_dataset
from quantrisk.strategies import get_strategy
from quantrisk.strategies.backtest import BacktestEngine, LegacyEngineError, PortfolioBacktest
from quantrisk.strategies.backtest.daily import AShareMarket, Instrument, RiskBudget, run_portfolio
from quantrisk.strategies.backtest.portfolio_run import run_strategy_portfolio, write_report
from quantrisk.strategies.base import BaseStrategy, Signal
from scripts.verify_portfolio import verify
from tests.conftest import random_walk

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "ashare_2024_portfolio"
needs_snapshot = pytest.mark.snapshot("ashare_2024_portfolio")
SYMBOLS = ["600519", "000001", "601318", "300750", "300308", "688981"]
MARKET = {"slippage": 0.0005, "commission_rate": 0.00025, "min_commission": 5.0}


class Scripted(BaseStrategy):
    def __init__(self, script=None, pct=0.5):
        self.script = script or {}
        super().__init__({"position_pct": pct})

    def default_params(self):
        return {"position_pct": 0.5}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        for k, v in self.script.items():
            if isinstance(k, int):
                s.iloc[k] = v.value
            else:
                s.loc[pd.Timestamp(k)] = v.value
        return s


def synth(symbol, seed, price=10.0, n=40):
    d = random_walk(n=n, seed=seed, vol=0.01)
    d[["open", "high", "low", "close"]] *= price / 100
    d["volume"] = 1e8
    return d


def inst(symbol, bars, script, pct=0.5):
    strat = Scripted(script, pct)
    return Instrument(symbol, strat, strat.generate_signals(bars), bars, AShareMarket(symbol))


def outcome(out):
    return (out["ledger"], out["positions"], out["orders"], out["fills"], out["journal"])


def assert_ledger_identity(out):
    led, pos = out["ledger"], out["positions"]
    mv = pos.groupby(level="date")["market_value"].sum()
    np.testing.assert_allclose(led["market_value"], mv, rtol=1e-12)
    np.testing.assert_allclose(led["equity"], led["cash"] + led["frozen"] + led["market_value"], rtol=1e-12)
    assert (led["cash"] >= -1e-9).all()
    journal = pd.DataFrame(out["journal"])
    assert journal["amount"].sum() == pytest.approx(led["cash"].iloc[-1], abs=1e-6)


# ---------- 资金竞争与确定性 ----------

def test_two_buys_compete_for_cash():
    a, b = synth("600000", 1, 10.0), synth("000002", 2, 20.0)
    out = run_portfolio([inst("600000", a, {5: Signal.BUY}, pct=0.8),
                         inst("000002", b, {5: Signal.BUY}, pct=0.8)], 100_000)
    o = {x["symbol"]: x for x in out["orders"]}
    # 同强度按代码升序分配：000002 先拿到 80%，600000 只能用剩余资金
    assert "策略仓位" in o["000002"]["sizing_note"].split("：")[0]
    assert "可用资金" in o["600000"]["sizing_note"].split("：")[0]
    assert out["ledger"]["frozen"].iloc[5] <= 100_000
    assert_ledger_identity(out)


def test_signal_strength_has_priority():
    a, b = synth("600000", 1, 10.0), synth("000002", 2, 20.0)
    out = run_portfolio([inst("600000", a, {5: Signal.STRONG_BUY}, pct=0.6),
                         inst("000002", b, {5: Signal.BUY}, pct=0.6)], 100_000)
    first = out["orders"][0]
    assert first["symbol"] == "600000" and first["signal"] == "STRONG_BUY"


def test_result_independent_of_input_order():
    bars = {s: synth(s, i, 10 + i) for i, s in enumerate(["600000", "000002", "300001", "601111"])}
    scripts = {"600000": {3: Signal.BUY, 20: Signal.SELL}, "000002": {3: Signal.BUY},
               "300001": {3: Signal.STRONG_BUY, 15: Signal.SELL, 25: Signal.BUY}, "601111": {10: Signal.BUY}}
    base = None
    for seed in range(4):
        order = list(bars)
        random.Random(seed).shuffle(order)
        out = run_portfolio([inst(s, bars[s], scripts[s], pct=0.4) for s in order], 300_000,
                            budget=RiskBudget(max_weight=0.35, max_gross=0.9))
        if base is None:
            base = outcome(out)
        else:
            pd.testing.assert_frame_equal(base[0], out["ledger"])
            pd.testing.assert_frame_equal(base[1], out["positions"])
            assert base[2:] == outcome(out)[2:]


# ---------- 风险预算 ----------

def test_max_weight_and_gross_enforced_at_order_time():
    bars = {s: synth(s, i, 10.0) for i, s in enumerate(["600000", "000002", "300001", "601111"])}
    out = run_portfolio([inst(s, b, {5: Signal.BUY}, pct=0.5) for s, b in bars.items()], 1_000_000,
                        budget=RiskBudget(max_weight=0.2, max_gross=0.5))
    equity = out["ledger"]["equity"].iloc[5]
    buys = [o for o in out["orders"] if o["side"] == "BUY"]
    est = [o["quantity"] * o["signal_close"] * 1.0005 for o in buys]
    assert max(est) <= 0.2 * equity + 1e-6
    assert sum(est) <= 0.5 * equity + 1e-6
    assert any("总仓位上限" in o["sizing_note"].split("：")[0] for o in buys)


def test_budget_rejects_leverage():
    with pytest.raises(ValueError):
        RiskBudget(max_gross=1.2)
    with pytest.raises(ValueError):
        RiskBudget(max_weight=0)


def test_trim_to_limits_sells_back_under_the_cap():
    up = synth("600000", 3, 10.0, n=60)
    up.iloc[10:, up.columns.get_indexer(["open", "high", "low", "close"])] *= np.linspace(1, 2.5, 50)[:, None]
    flat = synth("000002", 4, 10.0, n=60)
    budget = RiskBudget(max_weight=0.35, max_gross=0.9, trim_to_limits=True)
    out = run_portfolio([inst("600000", up, {2: Signal.BUY}, pct=0.3), inst("000002", flat, {2: Signal.BUY}, pct=0.3)],
                        1_000_000, budget=budget)
    trims = [o for o in out["orders"] if o["signal"] == "RISK_TRIM"]
    assert trims and all(o["symbol"] == "600000" for o in trims)
    led = out["ledger"]
    for o in trims:
        d = pd.Timestamp(o["signal_date"])
        pos = out["positions"].loc[(d, "600000")]
        after = (pos["market_value"] - o["quantity"] * o["signal_close"]) / led.loc[d, "equity"]
        assert after <= 0.35 + 1e-9                # 按信号日收盘价，减仓后不超过上限
        assert o["quantity"] % 100 == 0
    assert_ledger_identity(out)


# ---------- 冻结资金、部分成交、停牌 ----------

def test_partial_fill_and_suspension_keep_other_freezes_intact():
    a, b = synth("600000", 1, 10.0), synth("000002", 2, 10.0)
    a.iloc[6, a.columns.get_loc("volume")] = 20_000          # 10% 参与上限 → 最多 2000 股
    b = b.drop(index=b.index[6])                              # 000002 第 6 日停牌
    cal = a.index
    out = run_portfolio([inst("600000", a, {5: Signal.BUY}, pct=0.4), inst("000002", b, {5: Signal.BUY}, pct=0.4)],
                        1_000_000, calendar=cal)
    o = {x["symbol"]: x for x in out["orders"]}
    assert o["600000"]["status"] == "PARTIAL" and o["600000"]["filled_quantity"] == 2000
    assert o["000002"]["status"] == "REJECTED" and "停牌" in o["000002"]["reason"]
    assert out["ledger"]["frozen"].iloc[5] > 0 and out["ledger"]["frozen"].iloc[6] == 0
    assert out["positions"].loc[(cal[6], "000002"), "status"] == "停牌/无行情"
    assert_ledger_identity(out)


def test_data_ending_early_is_flagged_and_valued_at_last_close():
    a, b = synth("600000", 1, 10.0), synth("000002", 2, 10.0)
    b_short = b.iloc[:20]
    out = run_portfolio([inst("600000", a, {}), inst("000002", b_short, {5: Signal.BUY, 19: Signal.SELL})],
                        1_000_000, calendar=a.index)
    sell = [x for x in out["orders"] if x["side"] == "SELL"][0]
    assert sell["status"] == "REJECTED" and "数据已结束" in sell["reason"]
    pos = out["positions"].xs("000002", level="symbol")
    assert (pos["status"].iloc[20:] == "数据已结束").all()
    assert (pos["close"].iloc[20:] == b_short["close"].iloc[-1]).all()   # 不用未来或补造价格


def test_bar_outside_calendar_is_an_error():
    a = synth("600000", 1, 10.0)
    with pytest.raises(ValueError, match="交易日历之外"):
        run_portfolio([inst("600000", a, {})], 1_000_000, calendar=a.index[1:-1].append(pd.DatetimeIndex([a.index[-1]])).drop(a.index[10]))


def test_duplicate_symbol_rejected():
    a = synth("600000", 1, 10.0)
    with pytest.raises(ValueError, match="重复"):
        run_portfolio([inst("600000", a, {}), inst("600000", a, {})], 1_000_000)


def test_legacy_engines_are_disabled():
    with pytest.raises(LegacyEngineError):
        BacktestEngine()
    with pytest.raises(LegacyEngineError):
        PortfolioBacktest()


# ---------- 真实行情 ----------

@pytest.fixture(scope="module")
def datasets():
    return [load_dataset(SNAPSHOT, s) for s in SYMBOLS]


def run_real(datasets, factory, budget, capital=2_000_000):
    return run_strategy_portfolio(factory, datasets, capital, budget=budget, market_kwargs=MARKET)


@needs_snapshot
def test_corporate_actions_use_cninfo_and_catch_sina_omission(datasets):
    catl = next(d for d in datasets if d.symbol == "300750")
    assert catl.quality["corporate_action_source"] == "cninfo"
    rec = catl.quality["corporate_action_reconciliation"][0]
    assert rec["primary_cash"] == pytest.approx(5.028) and rec["secondary_cash"] == pytest.approx(2.011)
    assert not rec["match"]
    # 新浪后复权因子与巨潮记录一致（误差 < 1e-4），与新浪自己的分红表不一致
    assert catl.quality["hfq_factor_check"][0]["rel_diff"] < 1e-4


@needs_snapshot
def test_multi_stock_dividends_and_bonus_shares(datasets):
    out = run_real(datasets, lambda s: Scripted({datasets[0].calendar[0]: Signal.BUY}, pct=0.16),
                   RiskBudget(max_weight=0.2, max_gross=0.95))
    pos = out["positions"]
    before, after = pd.Timestamp("2024-06-05"), pd.Timestamp("2024-06-06")
    q0 = pos.loc[(before, "300308"), "quantity"]
    assert pos.loc[(after, "300308"), "quantity"] == q0 + int(q0 * 0.4)        # 10 转增 4
    div = [j for j in out["journal"] if j["type"] == "DIVIDEND"]
    catl = [j for j in div if j["symbol"] == "300750"][0]
    assert catl["amount"] == pytest.approx(pos.loc[(pd.Timestamp("2024-04-29"), "300750"), "quantity"] * 5.028)
    assert {j["symbol"] for j in div} == {"600519", "000001", "601318", "300750", "300308"}   # 688981 无分红
    assert_ledger_identity(out)


@needs_snapshot
def test_portfolio_suspension_and_limit_up_on_real_data(datasets):
    ds = {d.symbol: d for d in datasets}
    cal = ds["600519"].calendar
    day = cal[60]
    patched = []
    for d in datasets:
        bars = d.bars.copy()
        if d.symbol == "300750":
            bars = bars.drop(index=day)                                     # 构造停牌
        if d.symbol == "000001":
            prev_close = bars["close"].iloc[bars.index.get_loc(day) - 1]
            up = round(prev_close * 1.1 + 1e-9, 2)
            bars.loc[day, ["open", "high", "low", "close"]] = up            # 构造一字涨停
        patched.append(type(d)(d.symbol, bars, d.signal_data, d.calendar, d.corporate_actions, d.quality, d.manifest))
    out = run_real(patched, lambda s: Scripted({cal[59]: Signal.BUY}, pct=0.15), RiskBudget(max_weight=0.2, max_gross=0.95))
    o = {x["symbol"]: x for x in out["orders"]}
    assert o["300750"]["status"] == "REJECTED" and "停牌" in o["300750"]["reason"]
    assert o["000001"]["status"] == "REJECTED" and "涨停" in o["000001"]["reason"]
    assert o["600519"]["status"] == "FILLED"
    assert_ledger_identity(out)


@needs_snapshot
def test_no_sell_on_buy_day_across_real_runs(datasets):
    out = run_real(datasets, lambda s: get_strategy("ma_cross", {"position_pct": 0.3}),
                   RiskBudget(max_weight=0.2, max_gross=0.6, trim_to_limits=True))
    last_buy = {}
    for f in sorted(out["fills"], key=lambda f: (f["date"], f["order_id"])):
        if f["side"] == "SELL":
            assert f["date"] > last_buy[f["symbol"]]
        else:
            last_buy[f["symbol"]] = f["date"]


@needs_snapshot
@pytest.mark.parametrize("name", ["ma_cross", "turtle"])
def test_future_data_change_does_not_alter_past(datasets, name):
    cut = datasets[0].calendar[150]
    budget = RiskBudget(max_weight=0.25, max_gross=0.7, trim_to_limits=True)
    base = run_real(datasets, lambda s: get_strategy(name, {"position_pct": 0.4}), budget)
    changed = []
    for d in datasets:
        bars, sig = d.bars.copy(), d.signal_data.copy()
        if d.symbol in ("300308", "600519"):
            for frame in (bars, sig):
                frame.loc[frame.index >= cut, ["open", "high", "low", "close"]] *= 0.7
        changed.append(type(d)(d.symbol, bars, sig, d.calendar, d.corporate_actions, d.quality, d.manifest))
    alt = run_real(changed, lambda s: get_strategy(name, {"position_pct": 0.4}), budget)
    pd.testing.assert_frame_equal(base["ledger"][base["ledger"].index < cut], alt["ledger"][alt["ledger"].index < cut])
    before = lambda xs, key: [x for x in xs if x[key] < str(cut.date())]
    assert before(base["fills"], "date") == before(alt["fills"], "date")
    assert before(base["orders"], "signal_date") == before(alt["orders"], "signal_date")


@needs_snapshot
@pytest.mark.parametrize("name,budget", [
    ("ma_cross", RiskBudget(max_weight=0.2, max_gross=0.6)),
    ("ma_cross", RiskBudget(max_weight=0.2, max_gross=0.6, trim_to_limits=True)),
    ("turtle", RiskBudget(max_weight=0.25, max_gross=0.7)),
])
def test_real_portfolio_matches_independent_verification(datasets, tmp_path, name, budget):
    capital = 2_000_000
    out = run_real(datasets, lambda s: get_strategy(name, {"position_pct": 0.3}), budget, capital)
    write_report(out, tmp_path, {"initial_capital": capital, **MARKET}, datasets[0].manifest, "pytest")
    v = verify(SNAPSHOT, tmp_path, capital, MARKET["slippage"], MARKET["commission_rate"], MARKET["min_commission"],
               budget.max_weight, budget.max_gross)
    assert v["problems"] == []
    assert v["max_abs_equity_diff"] < 1e-6 and v["max_abs_cash_diff"] < 1e-6 and v["max_abs_position_diff"] == 0
    assert v["fills_checked"] == len(out["fills"]) > 10


@needs_snapshot
def test_snapshot_replay_is_identical():
    a = [load_dataset(SNAPSHOT, s) for s in SYMBOLS]
    b = [load_dataset(SNAPSHOT, s) for s in reversed(SYMBOLS)]
    budget = RiskBudget(max_weight=0.2, max_gross=0.6, trim_to_limits=True)
    r1 = run_real(a, lambda s: get_strategy("ma_cross", {"position_pct": 0.3}), budget)
    r2 = run_real(b, lambda s: get_strategy("ma_cross", {"position_pct": 0.3}), budget)
    pd.testing.assert_frame_equal(r1["ledger"], r2["ledger"])
    pd.testing.assert_frame_equal(r1["positions"], r2["positions"])
    assert r1["orders"] == r2["orders"] and r1["fills"] == r2["fills"] and r1["journal"] == r2["journal"]
    assert r1["metrics"] == r2["metrics"]
