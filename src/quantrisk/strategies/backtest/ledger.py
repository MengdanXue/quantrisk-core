"""
资金账本

账户总权益 = 可用现金 + 冻结现金 + 持仓市值

每一次资金或持仓变动都写入 journal，可逐笔追溯；任何操作导致可用现金为负都会立即抛错，
不允许把手续费算错的结果“吞掉”。持仓按买入批次（lot）记录，用于 T+1 可卖数量、
先进先出的成本结转、持有天数与分红归属。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

EPS = 1e-6


class LedgerError(RuntimeError):
    """账本一致性被破坏。"""


@dataclass
class Lot:
    quantity: int
    buy_date: pd.Timestamp
    cost: float                 # 该批次买入总成本（成交金额 + 买入费用）
    dividends: float = 0.0      # 该批次累计收到的现金分红
    buy_fees: float = 0.0


@dataclass
class Account:
    initial_cash: float
    settlement_lag: int = 1     # 1 = T+1；0 = 当日可卖
    cash: float = field(init=False)
    frozen: float = field(init=False, default=0.0)
    lots: Dict[str, List[Lot]] = field(init=False, default_factory=dict)
    journal: List[dict] = field(init=False, default_factory=list)
    closed_trades: List[dict] = field(init=False, default_factory=list)
    _frozen_by_order: Dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.cash = float(self.initial_cash)
        self._log(None, "DEPOSIT", amount=self.initial_cash, note="初始资金")

    # ---------- 查询 ----------
    def quantity(self, symbol: str) -> int:
        return sum(l.quantity for l in self.lots.get(symbol, []))

    def sellable(self, symbol: str, trade_day_index: int, day_index_of) -> int:
        """可卖数量：买入后经过 settlement_lag 个交易日的批次。day_index_of(date) 返回交易日序号。"""
        return sum(
            l.quantity for l in self.lots.get(symbol, [])
            if trade_day_index - day_index_of(l.buy_date) >= self.settlement_lag
        )

    def market_value(self, prices: Dict[str, float]) -> float:
        return sum(self.quantity(s) * prices[s] for s in self.lots if self.quantity(s) > 0)

    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.frozen + self.market_value(prices)

    # ---------- 资金冻结 ----------
    def freeze(self, date, order_id: str, amount: float):
        amount = float(amount)
        if amount > self.cash + EPS:
            raise LedgerError(f"{date} 冻结 {amount:.2f} 超过可用现金 {self.cash:.2f}")
        self.cash -= amount
        self.frozen += amount
        self._frozen_by_order[order_id] = self._frozen_by_order.get(order_id, 0.0) + amount
        self._log(date, "FREEZE", amount=-amount, order_id=order_id)

    def release(self, date, order_id: str) -> float:
        amount = self._frozen_by_order.pop(order_id, 0.0)
        if amount:
            self.frozen -= amount
            self.cash += amount
            self._log(date, "UNFREEZE", amount=amount, order_id=order_id)
        return amount

    def frozen_for(self, order_id: str) -> float:
        return self._frozen_by_order.get(order_id, 0.0)

    # ---------- 成交 ----------
    def buy(self, date, symbol: str, quantity: int, price: float, fees: Dict[str, float], order_id: str):
        gross = quantity * price
        total_fees = sum(fees.values())
        self.release(date, order_id)
        if gross + total_fees > self.cash + EPS:
            raise LedgerError(f"{date} 买入 {symbol} 需 {gross + total_fees:.2f}，可用现金仅 {self.cash:.2f}")
        self.cash -= gross
        self._log(date, "BUY", symbol=symbol, quantity=quantity, price=price, amount=-gross, order_id=order_id)
        for name, value in fees.items():
            self.cash -= value
            self._log(date, f"FEE_{name.upper()}", symbol=symbol, amount=-value, order_id=order_id)
        self.lots.setdefault(symbol, []).append(
            Lot(quantity=quantity, buy_date=pd.Timestamp(date), cost=gross + total_fees, buy_fees=total_fees))
        self._check(date)

    def sell(self, date, symbol: str, quantity: int, price: float, fees: Dict[str, float], order_id: str,
             trading_days_between=None) -> dict:
        """先进先出结转成本，返回本次卖出的交易记录（已实现盈亏含买卖全部费用与持有期分红）。"""
        if quantity > self.quantity(symbol):
            raise LedgerError(f"{date} 卖出 {quantity} 股 {symbol} 超过持仓 {self.quantity(symbol)}")
        gross = quantity * price
        total_fees = sum(fees.values())
        self.cash += gross
        self._log(date, "SELL", symbol=symbol, quantity=quantity, price=price, amount=gross, order_id=order_id)
        for name, value in fees.items():
            self.cash -= value
            self._log(date, f"FEE_{name.upper()}", symbol=symbol, amount=-value, order_id=order_id)

        remaining, cost, dividends, buy_fees, first_buy = quantity, 0.0, 0.0, 0.0, None
        lots = self.lots[symbol]
        while remaining > 0:
            lot = lots[0]
            take = min(remaining, lot.quantity)
            share = take / lot.quantity
            cost += lot.cost * share
            dividends += lot.dividends * share
            buy_fees += lot.buy_fees * share
            first_buy = first_buy or lot.buy_date
            lot.cost -= lot.cost * share
            lot.dividends -= lot.dividends * share
            lot.buy_fees -= lot.buy_fees * share
            lot.quantity -= take
            remaining -= take
            if lot.quantity == 0:
                lots.pop(0)
        proceeds = gross - total_fees
        sell_date = pd.Timestamp(date)
        trade = {
            "symbol": symbol,
            "direction": "long",
            "entry_date": str(first_buy.date()),
            "exit_date": str(sell_date.date()),
            "quantity": quantity,
            "entry_price": cost / quantity,          # 含买入费用的每股成本
            "exit_price": price,
            "buy_fees": buy_fees,
            "sell_fees": total_fees,
            "dividends": dividends,
            "pnl": proceeds - cost + dividends,
            "pnl_pct": (proceeds + dividends) / cost - 1,
            "holding_days": (sell_date - first_buy).days,                 # 自然日
            "holding_trading_days": trading_days_between(first_buy, sell_date) if trading_days_between else None,
        }
        self.closed_trades.append(trade)
        self._check(date)
        return trade

    # ---------- 公司行为 ----------
    def cash_dividend(self, date, symbol: str, per_share: float):
        for lot in self.lots.get(symbol, []):
            amount = lot.quantity * per_share
            lot.dividends += amount
            self.cash += amount
            self._log(date, "DIVIDEND", symbol=symbol, quantity=lot.quantity, price=per_share, amount=amount)

    def bonus_shares(self, date, symbol: str, per_share: float):
        for lot in self.lots.get(symbol, []):
            added = int(lot.quantity * per_share)     # 不足 1 股的部分不计（实际按交易所零碎股规则处理）
            lot.quantity += added
            self._log(date, "BONUS_SHARES", symbol=symbol, quantity=added, amount=0.0)

    # ---------- 内部 ----------
    def _check(self, date):
        if self.cash < -EPS:
            raise LedgerError(f"{date} 可用现金为负: {self.cash:.6f}")
        if abs(self.frozen - sum(self._frozen_by_order.values())) > EPS:
            raise LedgerError(f"{date} 冻结现金与订单冻结明细不一致")

    def _log(self, date, kind: str, amount: float = 0.0, symbol: Optional[str] = None,
             quantity: Optional[int] = None, price: Optional[float] = None, order_id: Optional[str] = None,
             note: str = ""):
        self.journal.append({
            "date": None if date is None else str(pd.Timestamp(date).date()),
            "type": kind, "symbol": symbol, "quantity": quantity, "price": price,
            "amount": float(amount), "cash_after": float(self.cash), "frozen_after": float(self.frozen),
            "order_id": order_id, "note": note,
        })
