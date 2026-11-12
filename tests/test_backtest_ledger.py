"""M0：回测引擎与资金账本的正确性测试（对应 base.py 原有的 6 个问题）。"""
import numpy as np
import pandas as pd
import pytest

from quantrisk.strategies.backtest import execution as execution_module
from quantrisk.strategies.backtest.ledger import Account, LedgerError
from quantrisk.strategies.base import BaseStrategy, Signal
from tests.conftest import random_walk


class Scripted(BaseStrategy):
    """按日期序号给出固定信号的测试策略。"""

    def __init__(self, script, pct=0.5):
        self.script = script
        super().__init__({"position_pct": pct})

    def default_params(self):
        return {"position_pct": 0.5}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        for i, v in self.script.items():
            s.iloc[i] = v.value
        return s


@pytest.fixture
def data():
    return random_walk(n=60, seed=11)


# ---------- 问题 1：ffill().bfill() 的数据泄漏 ----------

def test_missing_rows_are_dropped_not_filled(data):
    data = data.copy()
    first_dates = data.index[:3]
    data.iloc[:3, data.columns.get_loc("close")] = np.nan      # 开头缺失：bfill 会用未来价格填充
    data.iloc[30, data.columns.get_loc("open")] = np.nan       # 中间缺失：ffill 会制造一根可成交的假K线
    result = Scripted({}).run(data)
    assert not set(first_dates) & set(result.signals.index)
    assert str(data.index[30]) in result.data_issues["dropped_signal_rows"]
    assert result.ledger.loc[data.index[30], "suspended"]          # 当日按无行情处理


def test_dropped_day_cannot_fill_orders(data):
    data = data.copy()
    data.iloc[11, data.columns.get_loc("open")] = np.nan
    result = Scripted({10: Signal.BUY}).run(data)
    order = result.orders[0]
    assert order["status"] == "REJECTED" and "无有效行情" in order["reason"]
    assert result.fills == []


# ---------- 问题 2：当根收盘成交 → T+1 开盘成交 ----------

def test_signal_fills_at_next_open(data):
    slippage = 0.0005
    result = Scripted({5: Signal.BUY, 20: Signal.SELL}).run(data, slippage=slippage)
    buy, sell = result.fills
    assert buy["date"] == str(data.index[6].date())
    assert sell["date"] == str(data.index[21].date())
    bar = data.iloc[6]
    expected = min(max(bar.open * (1 + slippage), bar.low), bar.high)
    assert buy["price"] == pytest.approx(expected)
    # 信号当日收盘后冻结资金，次日成交后释放
    assert result.ledger["frozen"].iloc[5] > 0
    assert result.ledger["frozen"].iloc[6] == 0


# ---------- 问题 3：首日/成交日成本进入收益序列 ----------

def test_returns_compound_to_final_equity_including_first_day(data):
    capital = 1_000_000
    result = Scripted({0: Signal.BUY, 30: Signal.SELL}).run(data, initial_capital=capital)
    assert result.returns.iloc[0] == pytest.approx(result.equity_curve.iloc[0] / capital - 1)
    assert capital * (1 + result.returns).prod() == pytest.approx(result.equity_curve.iloc[-1], rel=1e-12)
    assert result.metrics["total_return"] == pytest.approx(result.equity_curve.iloc[-1] / capital - 1, abs=1e-4)


# ---------- 问题 4：单笔盈亏完整扣除买卖费用 ----------

def test_trade_pnl_includes_buy_and_sell_fees(data):
    capital = 1_000_000
    result = Scripted({3: Signal.BUY, 25: Signal.SELL, 30: Signal.BUY, 50: Signal.SELL}).run(
        data, initial_capital=capital, commission=0.001)
    assert len(result.trades) == 2
    for trade, (buy, sell) in zip(result.trades, [result.fills[0:2], result.fills[2:4]]):
        expected = (sell["quantity"] * sell["price"] - sell["fee_total"]) - (buy["quantity"] * buy["price"] + buy["fee_total"])
        assert trade["pnl"] == pytest.approx(expected, rel=1e-12)
        assert trade["buy_fees"] == pytest.approx(buy["fee_total"])
    # 平仓后，全部交易盈亏之和 = 期末权益 − 初始资金
    assert sum(t["pnl"] for t in result.trades) == pytest.approx(result.equity_curve.iloc[-1] - capital, rel=1e-12)


# ---------- 问题 5：强信号的仓位 ----------

def test_strong_buy_scales_position_by_1_5(data):
    normal = Scripted({5: Signal.BUY}, pct=0.4).run(data, commission=0, slippage=0)
    strong = Scripted({5: Signal.STRONG_BUY}, pct=0.4).run(data, commission=0, slippage=0)
    assert strong.orders[0]["signal"] == "STRONG_BUY"
    assert strong.orders[0]["quantity"] == pytest.approx(normal.orders[0]["quantity"] * 1.5, abs=1)


def test_strong_buy_cannot_exceed_available_cash(data):
    result = Scripted({5: Signal.STRONG_BUY}, pct=0.9).run(data, initial_capital=100_000)   # 0.9 × 1.5 = 135%
    assert result.ledger["cash"].min() >= 0
    filled = result.fills[0]
    assert filled["quantity"] * filled["price"] + filled["fee_total"] <= 100_000 + 1e-6


# ---------- 问题 6：持有天数 ----------

def test_holding_days_natural_and_trading(data):
    result = Scripted({5: Signal.BUY, 20: Signal.SELL}).run(data)
    trade = result.trades[0]
    entry, exit_ = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
    assert trade["holding_days"] == (exit_ - entry).days > 1
    assert trade["holding_trading_days"] == data.index.get_loc(exit_) - data.index.get_loc(entry) == 15


# ---------- 资金账本 ----------

def test_daily_equity_identity_and_journal_reconciliation(data):
    result = Scripted({2: Signal.BUY, 15: Signal.SELL, 20: Signal.STRONG_BUY, 40: Signal.SELL}).run(data)
    led = result.ledger
    np.testing.assert_allclose(led["equity"], led["cash"] + led["frozen"] + led["position"] * led["close"], rtol=1e-12)
    journal = pd.DataFrame(result.journal)
    assert (journal["cash_after"] >= -1e-9).all()
    # 全部流水金额之和 = 期末可用现金；剔除冻结/解冻后 = 可用现金 + 冻结现金
    assert journal["amount"].sum() == pytest.approx(led["cash"].iloc[-1], rel=1e-12)
    moves = journal[~journal["type"].isin(["FREEZE", "UNFREEZE"])]["amount"].sum()
    assert moves == pytest.approx(led["cash"].iloc[-1] + led["frozen"].iloc[-1], rel=1e-12)
    # 每笔成交的资金变动都能在流水中找到
    for f in result.fills:
        rows = journal[journal["order_id"] == f["order_id"]]
        assert rows[rows["type"].isin(["BUY", "SELL"])]["amount"].abs().sum() == pytest.approx(f["quantity"] * f["price"])
        assert -rows[rows["type"].str.startswith("FEE_")]["amount"].sum() == pytest.approx(f["fee_total"])


def test_unrealized_plus_realized_equals_pnl(data):
    capital = 1_000_000
    result = Scripted({2: Signal.BUY, 15: Signal.SELL, 20: Signal.BUY}).run(data, initial_capital=capital)
    realized = sum(t["pnl"] for t in result.trades)
    led = result.ledger.iloc[-1]
    open_cost = sum(f["quantity"] * f["price"] + f["fee_total"] for f in result.fills[2:])
    unrealized = led["market_value"] - open_cost
    assert realized + unrealized == pytest.approx(led["equity"] - capital, rel=1e-10)


def test_account_rejects_negative_cash():
    acct = Account(1_000)
    with pytest.raises(LedgerError):
        acct.buy("2024-01-02", "X", 10, 100.0, {"commission": 5.0}, "O1")


def test_run_goes_through_execution_handler(data, monkeypatch):
    calls = []
    original = execution_module.ExecutionHandler.process_bar

    def spy(self, bar, volatility=None):
        calls.append(bar.timestamp)
        return original(self, bar, volatility)

    monkeypatch.setattr(execution_module.ExecutionHandler, "process_bar", spy)
    result = Scripted({5: Signal.BUY, 20: Signal.SELL}).run(data)
    assert [str(c.date()) for c in calls] == [f["date"] for f in result.fills]


def test_volume_cap_gives_partial_fill_and_expires_remainder(data):
    data = data.copy()
    data.iloc[6, data.columns.get_loc("volume")] = 1_000          # 成交量参与上限 10% → 最多 100 股
    result = Scripted({5: Signal.BUY}, pct=0.5).run(data)
    order, fill = result.orders[0], result.fills[0]
    assert fill["quantity"] == 100
    assert order["status"] == "PARTIAL" and order["filled_quantity"] == 100
    assert "剩余" in order["reason"] and "成交量参与上限" in order["reason"]
    assert result.ledger["frozen"].iloc[6] == 0                  # 未成交部分的冻结资金已释放
