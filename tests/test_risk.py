"""M3 第一阶段：硬性风险约束、主动/被动超限区分、减仓受交易规则约束。"""
import numpy as np
import pandas as pd
import pytest

from quantrisk.strategies.backtest.daily import AShareMarket, Instrument, RiskBudget, run_portfolio
from quantrisk.strategies.base import BaseStrategy, Signal
from tests.conftest import random_walk


class Scripted(BaseStrategy):
    def __init__(self, script=None, pct=0.5):
        self.script = script or {}
        super().__init__({"position_pct": pct})

    def default_params(self):
        return {"position_pct": 0.5}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        for k, v in self.script.items():
            s.iloc[k] = v.value
        return s


def synth(seed, price=10.0, n=60):
    d = random_walk(n=n, seed=seed, vol=0.005)
    d[["open", "high", "low", "close"]] *= price / 100
    d["volume"] = 1e9
    return d


def inst(symbol, bars, script, pct=0.5):
    st = Scripted(script, pct)
    return Instrument(symbol, st, st.generate_signals(bars), bars, AShareMarket(symbol))


def rally(bars, start, factor):
    out = bars.copy()
    cols = out.columns.get_indexer(["open", "high", "low", "close"])
    out.iloc[start:, cols] *= np.linspace(1, factor, len(out) - start)[:, None]
    return out


# ---------- 主动超限：下单与成交两道检查 ----------

def test_order_exceeding_single_name_limit_is_sized_down_and_logged():
    out = run_portfolio([inst("600000", synth(1), {5: Signal.BUY}, pct=0.6)], 1_000_000,
                        budget=RiskBudget(max_weight=0.25))
    order = out["orders"][0]
    equity = out["ledger"]["equity"].iloc[5]
    assert order["quantity"] * order["signal_close"] * 1.0005 <= 0.25 * equity
    log = out["risk_log"][0]
    assert log["type"] == "ORDER_SIZED_DOWN" and log["order_id"] == order["order_id"]


def test_order_rejected_when_gross_budget_is_full():
    a, b = synth(1), synth(2)
    out = run_portfolio([inst("600000", a, {3: Signal.BUY}, pct=0.5), inst("000002", b, {10: Signal.BUY}, pct=0.5)],
                        1_000_000, budget=RiskBudget(max_weight=0.5, max_gross=0.5))
    second = [o for o in out["orders"] if o["symbol"] == "000002"][0]
    assert second["status"] == "REJECTED" and "总仓位上限" in second["reason"]
    assert any(r["type"] == "ORDER_REJECTED" and r["symbol"] == "000002" for r in out["risk_log"])


def test_gap_up_at_open_is_shrunk_at_fill_time():
    bars = synth(3)
    cols = bars.columns.get_indexer(["open", "high", "low", "close"])
    bars.iloc[6, cols] = bars.iloc[5]["close"] * np.array([1.08, 1.09, 1.07, 1.08])   # 次日高开 8%
    out = run_portfolio([inst("600000", bars, {5: Signal.BUY}, pct=0.25)], 1_000_000,
                        budget=RiskBudget(max_weight=0.25))
    fill = out["fills"][0]
    prev_equity = out["ledger"]["equity"].iloc[5]
    assert fill["quantity"] * fill["price"] + fill["fee_total"] <= 0.25 * prev_equity + 1e-6
    assert any(r["type"] == "FILL_SHRUNK" for r in out["risk_log"])
    assert "风险预算缩单" in out["orders"][0]["reason"]


# ---------- 被动超限：记录，不自动减仓 ----------

def test_passive_breach_recorded_with_duration_and_severity_and_no_trim_by_default():
    bars = rally(synth(4), 10, 1.8)
    bars = pd.concat([bars, rally(bars.iloc[-15:].copy(), 0, 0.5).set_axis(
        pd.bdate_range(bars.index[-1] + pd.Timedelta(days=1), periods=15))])
    out = run_portfolio([inst("600000", bars, {2: Signal.BUY}, pct=0.3)], 1_000_000,
                        budget=RiskBudget(max_weight=0.3, max_gross=1.0))
    assert not [o for o in out["orders"] if o["signal"] == "RISK_TRIM"]           # 默认不减仓
    ev = max((e for e in out["risk_events"] if e["scope"] == "600000"), key=lambda e: e["trading_days"])
    assert ev["limit_type"] == "单票上限" and ev["trading_days"] > 5
    assert ev["severity"] == "严重" and ev["max_excess"] > 0.05
    assert ev["status"] == "已恢复" and ev["end"] is not None                       # 回落后事件结束
    w = out["positions"].xs("600000", level="symbol")["weight"]
    assert (w > 0.3).sum() == sum(e["trading_days"] for e in out["risk_events"] if e["scope"] == "600000")


@pytest.mark.parametrize("blocker", ["limit_down", "suspension"])
def test_trim_blocked_by_suspension_or_limit_down_is_recorded(blocker):
    bars = rally(synth(5), 5, 1.6)
    budget = RiskBudget(max_weight=0.3, trim_to_limits=True)
    base = run_portfolio([inst("600000", bars, {2: Signal.BUY}, pct=0.3)], 1_000_000, budget=budget)
    trim = next(o for o in base["orders"] if o["signal"] == "RISK_TRIM")
    k = bars.index.get_loc(pd.Timestamp(trim["signal_date"])) + 1          # 减仓单的撮合日
    blocked = bars.copy()
    if blocker == "limit_down":
        prev = blocked["close"].iloc[k - 1]
        blocked.iloc[k, blocked.columns.get_indexer(["open", "high", "low", "close"])] = round(prev * 0.9 + 1e-9, 2)
    else:
        blocked = blocked.drop(index=blocked.index[k])
    out = run_portfolio([inst("600000", blocked, {2: Signal.BUY}, pct=0.3)], 1_000_000, budget=budget,
                        calendar=bars.index)
    order = next(o for o in out["orders"] if o["order_id"] == trim["order_id"])
    assert order["status"] == "REJECTED"
    ev = next(e for e in out["risk_events"] if e["event_id"] == order["risk_event"])
    reasons = " ".join(b["reason"] for b in ev["trim_blocked"])
    assert ("跌停" if blocker == "limit_down" else "停牌") in reasons
