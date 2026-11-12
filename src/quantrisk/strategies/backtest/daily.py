"""
日线回测主循环（BaseStrategy.run 的执行路径）

时序约定（保守假设，日线数据无法证明盘中排队能否成交）：
1. T 日收盘后，策略基于截至 T 日的数据给出信号，生成 DAY 市价单；买单按 T 日收盘价预估金额冻结资金。
2. T+1 交易日开盘，由 ExecutionHandler 以开盘价（加滑点、限制在当日最高/最低价与涨跌停价之内）撮合：
   - 停牌或无成交：订单作废；
   - 开盘价已在涨停价（买）/ 跌停价（卖）：视为无法成交，订单作废；
   - 数量受资金、申报单位、T+1 可卖数量、成交量参与上限约束，不足部分作废。
3. 除权除息日开盘前，按登记日持仓入账现金红利与送转股；涨跌停按除权除息参考价计算。
4. 每日收盘按收盘价（停牌日沿用最近收盘价）估值：权益 = 可用现金 + 冻结现金 + 持仓市值。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import itertools
import math

import numpy as np
import pandas as pd

from .ashare_rules import (Board, LotRule, classify_board, ex_right_reference_price, limit_prices, lot_rule,
                           price_limit_pct)
from .events import BarEvent, OrderEvent
from .execution import AShareFeeModel, ExecutionHandler, FixedSlippageModel, SimpleFeeModel
from .ledger import Account

OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class CorporateAction:
    ex_date: pd.Timestamp
    cash_per_share: float = 0.0     # 每股税前现金红利
    bonus_per_share: float = 0.0    # 每股送转股数（送股 + 转增）


class GenericMarket:
    """通用市场：整数股、无涨跌停、当日买入次日可卖，费用为单一比例佣金。适用于模拟数据。"""

    settlement_lag = 1
    fill_ratio_limit = 0.1

    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.fee_model = SimpleFeeModel(rate=commission)
        self.slippage_model = FixedSlippageModel(slippage)
        self.slippage = slippage

    def limits(self, trade_date, ref_price) -> Tuple[Optional[float], Optional[float]]:
        return None, None

    def lot_rule(self, trade_date) -> LotRule:
        return LotRule(min_qty=1, step=1)


class AShareMarket:
    """
    A 股市场规则：按板块与日期取涨跌幅、申报单位、税费；T+1。
    st_dates 为该股处于风险警示状态的交易日，需调用方按历史提供（本库不推断）。
    新股上市初期的涨跌幅例外未在引擎中启用（ashare_rules.price_limit_pct 已支持 listing_day 参数）。
    """

    settlement_lag = 1

    def __init__(self, symbol: str, commission_rate: float = 0.00025, min_commission: float = 5.0,
                 slippage: float = 0.0005, fill_ratio_limit: float = 0.1,
                 st_dates: Iterable = ()):
        self.symbol = symbol
        self.board: Board = classify_board(symbol)
        self.fee_model = AShareFeeModel(commission_rate, min_commission)
        self.slippage_model = FixedSlippageModel(slippage)
        self.slippage = slippage
        self.fill_ratio_limit = fill_ratio_limit
        self.st_dates = {pd.Timestamp(d).normalize() for d in st_dates}

    def limits(self, trade_date, ref_price):
        pct = price_limit_pct(self.board, trade_date, is_st=pd.Timestamp(trade_date).normalize() in self.st_dates)
        return limit_prices(ref_price, pct)

    def lot_rule(self, trade_date) -> LotRule:
        return lot_rule(self.board, trade_date)


class _Rules:
    """ExecutionHandler 的 market_rules 适配器，持有当日上下文（涨跌停价、账户、可卖数量）。"""

    def __init__(self, market, account: Account, symbol: str, day_index_of):
        self.market = market
        self.account = account
        self.symbol = symbol
        self.day_index_of = day_index_of
        self.day_index = 0
        self.lower = self.upper = None
        self.lot = market.lot_rule(pd.Timestamp("2024-01-02"))
        self.gate = None          # 组合风险闸门（run_portfolio 设置）

    def set_day(self, trade_date, day_index, ref_price):
        self.day_index = day_index
        self.lot = self.market.lot_rule(trade_date)
        self.lower, self.upper = (None, None) if ref_price is None else self.market.limits(trade_date, ref_price)

    def clamp_price(self, order, bar, price):
        price = min(max(price, bar.low), bar.high)
        if self.upper is not None:
            price = min(price, self.upper)
        if self.lower is not None:
            price = max(price, self.lower)
        return price

    def fees(self, price, qty, side, trade_date):
        return sum(self.market.fee_model.breakdown(price, qty, side, trade_date).values())

    def check(self, order, bar, price):
        if order.side == "BUY":
            if self.upper is not None and bar.open >= self.upper - 1e-9:
                return 0, f"开盘价 {bar.open} 已达涨停价 {self.upper}，保守假设无法买入"
            funds = self.account.cash + self.account.frozen_for(order.order_id)
            qty = self.lot.round_buy(order.quantity)
            while qty > 0 and qty * price + self.fees(price, qty, "BUY", bar.timestamp) > funds + 1e-9:
                qty = self.lot.round_buy(qty - self.lot.step)
            if qty <= 0:
                return 0, "可用资金不足一个申报单位"
            reason = f"按成交价 {price:.4f} 计算，可用资金只够 {qty} 股" if qty < order.quantity else ""
            if self.gate is not None:
                allowed = self.gate.allowed_buy_value(self.symbol)
                cost = lambda q: q * price + self.fees(price, q, "BUY", bar.timestamp)
                before = qty
                while qty > 0 and cost(qty) > allowed + 1e-9:
                    qty = self.lot.round_buy(qty - self.lot.step)
                if qty < before:
                    self.gate.log(bar.timestamp, self.symbol, order.order_id,
                                  "FILL_REJECTED" if qty <= 0 else "FILL_SHRUNK",
                                  requested=cost(before), allowed=allowed)
                    if qty <= 0:
                        return 0, f"风险预算：按成交价含费用 {cost(before):,.0f} 超过剩余额度 {max(allowed, 0):,.0f}"
                    reason = f"风险预算缩单：按成交价含费用剩余额度 {allowed:,.0f}，{before} 股缩为 {qty} 股"
            return qty, reason
        if self.lower is not None and bar.open <= self.lower + 1e-9:
            return 0, f"开盘价 {bar.open} 已达跌停价 {self.lower}，保守假设无法卖出"
        holding = self.account.quantity(self.symbol)
        sellable = self.account.sellable(self.symbol, self.day_index, self.day_index_of)
        if sellable <= 0:
            return 0, "T+1：没有可卖数量"
        qty = self.lot.valid_sell(min(int(order.quantity), sellable), holding)
        qty = min(qty, sellable)
        if qty <= 0:
            return 0, "可卖数量不满足申报单位"
        return qty, ""

    def round_quantity(self, order, qty):
        if order.side == "BUY":
            return self.lot.round_buy(qty)
        return self.lot.valid_sell(int(qty), self.account.quantity(self.symbol))


def _signal_value(v) -> int:
    if v is None:
        return 0
    if hasattr(v, "value"):
        v = v.value
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    return 0 if np.isnan(f) else int(f)


def valid_bar_mask(bars: pd.DataFrame) -> pd.Series:
    """有效K线：OHLC 均为正数且 low ≤ open/close ≤ high。无效K线不补值，按无行情处理。"""
    px = bars[OHLC]
    ok = px.notna().all(axis=1) & (px > 0).all(axis=1)
    ok &= (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9)
    ok &= (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9)
    return ok


@dataclass(frozen=True)
class RiskBudget:
    """
    组合风险预算。硬性检查两次，新订单不得主动造成超限：
    1. 下单时（T 日收盘）：开仓金额（含预估费用）≤ 权益 × max_weight；持仓市值 + 待成交买单冻结金额 ≤ 权益 × max_gross；
    2. 成交时（T+1 开盘）：按实际成交价与实际费用重新检查，以前一交易日收盘权益与持仓市值为基准，超出部分缩单或拒单。
    两者都必须在 (0, 1] 内：不允许杠杆；只做多，不允许卖空。
    价格波动造成的被动超限记为风险事件（开始/结束日期、持续交易日、最大超出幅度、严重程度）；
    严重程度按最大超出百分点划分：< severity_pp[0] 为“轻微”，< severity_pp[1] 为“中等”，其余为“严重”。
    trim_to_limits=False（默认）：价格变化导致的占比漂移不触发减仓，只在每日持仓表中如实记录；
    trim_to_limits=True：每个交易日收盘后，单票占比超过 max_weight 的减到上限以内，总仓位超过 max_gross 的
    按持仓市值比例分摊减仓；减仓卖单与普通卖单一样在 T+1 开盘撮合，受跌停、T+1、申报单位约束。
    """
    max_weight: float = 1.0
    max_gross: float = 1.0
    trim_to_limits: bool = False
    severity_pp: Tuple[float, float] = (0.02, 0.05)

    def __post_init__(self):
        for name in ("max_weight", "max_gross"):
            v = getattr(self, name)
            if not (0 < v <= 1):
                raise ValueError(f"{name} 必须在 (0, 1] 内（不允许杠杆），当前为 {v}")


class _RiskGate:
    """成交时的硬性风险检查与风险日志（主动超限的拦截记录）。"""

    def __init__(self, budget: "RiskBudget", account: Account, symbols: List[str]):
        self.budget, self.account, self.symbols = budget, account, symbols
        self.prev_equity = None
        self.prev_close: Dict[str, Optional[float]] = {}
        self.bought_qty: Dict[str, int] = {}
        self.bought_cost = 0.0
        self.log_rows: List[dict] = []

    def start_day(self, prev_equity, prev_close):
        self.prev_equity, self.prev_close = prev_equity, dict(prev_close)
        self.bought_qty, self.bought_cost = {}, 0.0

    def _held_prev_value(self, symbol=None):
        syms = [symbol] if symbol else self.symbols
        total = 0.0
        for s in syms:
            q = self.account.quantity(s) - self.bought_qty.get(s, 0)
            if q and self.prev_close.get(s) is not None:
                total += q * self.prev_close[s]
        return total

    def allowed_buy_value(self, symbol) -> float:
        if self.prev_equity is None:
            return float("inf")
        w = self.budget.max_weight * self.prev_equity - self._held_prev_value(symbol)
        g = self.budget.max_gross * self.prev_equity - self._held_prev_value() - self.bought_cost
        return min(w, g)

    def record_buy(self, symbol, qty, cost):
        self.bought_qty[symbol] = self.bought_qty.get(symbol, 0) + qty
        self.bought_cost += cost

    def log(self, date, symbol, order_id, kind, requested=None, allowed=None, binding=""):
        self.log_rows.append({"date": str(pd.Timestamp(date).date()), "symbol": symbol, "order_id": order_id,
                              "type": kind, "requested_value": requested, "allowed_value": allowed,
                              "binding": binding})


class _BreachTracker:
    """被动超限（价格波动造成）的风险事件。"""

    def __init__(self, budget: "RiskBudget"):
        self.budget = budget
        self.open: Dict[Tuple[str, str], dict] = {}
        self.events: List[dict] = []

    def _severity(self, excess):
        lo, hi = self.budget.severity_pp
        return "轻微" if excess < lo else "中等" if excess < hi else "严重"

    def observe(self, d, scope, kind, value, limit):
        key = (scope, kind)
        ev = self.open.get(key)
        if value > limit + 1e-9:
            if ev is None:
                ev = {"event_id": f"R{len(self.events) + 1:04d}", "scope": scope, "limit_type": kind,
                      "limit": limit, "start": str(d.date()), "end": None, "trading_days": 0,
                      "max_value": value, "max_excess": 0.0, "trim_orders": [], "trim_blocked": []}
                self.events.append(ev)
                self.open[key] = ev
            ev["trading_days"] += 1
            ev["max_value"] = max(ev["max_value"], value)
            ev["max_excess"] = ev["max_value"] - limit
            ev["severity"] = self._severity(ev["max_excess"])
        elif ev is not None:
            ev["end"] = str(d.date())
            del self.open[key]

    def event_for(self, scope, kind="单票上限"):
        return self.open.get((scope, kind)) or self.open.get(("组合", "总仓位上限"))

    def finish(self):
        for ev in self.open.values():
            ev["end"] = None
            ev["status"] = "回测结束时仍超限"
        for ev in self.events:
            ev.setdefault("status", "已恢复")
        return self.events


@dataclass
class Instrument:
    """组合中的一只证券：信号、实际成交价格、市场规则、公司行为，以及负责仓位计算的策略。"""
    symbol: str
    strategy: object
    signals: pd.Series
    bars: pd.DataFrame
    market: object
    corporate_actions: Tuple[CorporateAction, ...] = ()
    eligible: Optional[pd.Series] = None    # 按日期的股票池成员资格（按时点）；None 表示全程可交易


def run_daily(strategy, signals: pd.Series, bars: pd.DataFrame, market, initial_capital: float,
              symbol: str = "UNKNOWN", calendar: Optional[pd.DatetimeIndex] = None,
              corporate_actions: Iterable[CorporateAction] = ()) -> dict:
    """单只证券回测：即只含一只证券、无额外风险预算的组合回测，与多证券共用同一套撮合与账本。"""
    inst = Instrument(symbol, strategy, signals, bars, market, tuple(corporate_actions))
    out = run_portfolio([inst], initial_capital, calendar=calendar, budget=RiskBudget())
    pos = out["positions"].xs(symbol, level="symbol")
    ledger = out["ledger"].copy()
    ledger.insert(2, "position", pos["quantity"])
    ledger.insert(3, "close", pos["close"])
    ledger["suspended"] = pos["suspended"]
    out["ledger"] = ledger[["cash", "frozen", "position", "close", "market_value", "equity", "suspended"]]
    out["invalid_days"] = out["invalid_days"][symbol]
    return out


def run_portfolio(instruments: List[Instrument], initial_capital: float,
                  calendar: Optional[pd.DatetimeIndex] = None, budget: RiskBudget = RiskBudget()) -> dict:
    """
    多证券共享一个现金账户的日线回测。

    确定性规则（与输入顺序无关）：
    - 证券按代码排序处理；
    - T+1 开盘撮合时，先撮合全部卖单（按代码），再撮合买单（按订单号，即下单先后）；
    - T 日收盘生成买单时，按（信号强度降序，代码升序）依次分配资金与仓位预算，
      先下的单先冻结资金，后面的单只能使用剩余额度；当日卖单的回款不用于当日新买单的额度计算。
    """
    from ..base import Signal  # 避免循环导入

    insts = sorted(instruments, key=lambda x: x.symbol)
    symbols = [x.symbol for x in insts]
    if len(set(symbols)) != len(symbols):
        raise ValueError("组合中存在重复代码")

    good: Dict[str, pd.DataFrame] = {}
    invalid_days: Dict[str, List[str]] = {}
    for x in insts:
        b = x.bars.sort_index()
        mask = valid_bar_mask(b)
        if "volume" in b:
            mask &= b["volume"].fillna(0) > 0
        invalid_days[x.symbol] = [str(d.date()) for d in b.index[~mask]]
        good[x.symbol] = b[mask]
        x.bars = b
    if all(g.empty for g in good.values()):
        raise ValueError("没有可用的有效K线")

    first = min(x.bars.index[0] for x in insts)
    last = max(x.bars.index[-1] for x in insts)
    if calendar is None:
        days = pd.DatetimeIndex(sorted(set().union(*[set(x.bars.index) for x in insts])))
    else:
        days = pd.DatetimeIndex(calendar)
        days = days[(days >= first) & (days <= last)]
        for x in insts:
            missing = set(x.bars.index) - set(days)
            if missing:
                raise ValueError(f"{x.symbol} 行情中存在交易日历之外的日期: {sorted(missing)[:5]}")
    day_pos = {d: i for i, d in enumerate(days)}

    def day_index_of(d):
        return day_pos[pd.Timestamp(d)]

    def trading_days_between(a, b):
        return day_pos[pd.Timestamp(b)] - day_pos[pd.Timestamp(a)]

    account = Account(initial_capital, settlement_lag=max(x.market.settlement_lag for x in insts))
    rules: Dict[str, _Rules] = {}
    handlers: Dict[str, ExecutionHandler] = {}
    actions: Dict[str, Dict[pd.Timestamp, List[CorporateAction]]] = {}
    for x in insts:
        rules[x.symbol] = _Rules(x.market, account, x.symbol, day_index_of)
        handlers[x.symbol] = ExecutionHandler(
            slippage_model=x.market.slippage_model, commission_model=x.market.fee_model,
            fill_ratio_limit=x.market.fill_ratio_limit, allow_partial_fills=True, market_rules=rules[x.symbol])
        acts: Dict[pd.Timestamp, List[CorporateAction]] = {}
        for a in x.corporate_actions:
            acts.setdefault(pd.Timestamp(a.ex_date).normalize(), []).append(a)
        actions[x.symbol] = acts

    gate = _RiskGate(budget, account, symbols)
    tracker = _BreachTracker(budget)
    for r in rules.values():
        r.gate = gate
    prev_equity: Optional[float] = None

    ids = itertools.count(1)
    orders: Dict[str, dict] = {}
    fills: List[dict] = []
    rows, pos_rows = [], []
    last_close: Dict[str, Optional[float]] = {x.symbol: None for x in insts}
    first_bar = {x.symbol: x.bars.index[0] for x in insts}
    last_bar = {x.symbol: x.bars.index[-1] for x in insts}

    def pending_ids():
        return {o.order_id for h in handlers.values() for o in h.get_pending_orders()}

    def sync_logs(handler):
        for log in handler.order_log:
            if log.get("_seen"):
                continue
            log["_seen"] = True
            o = orders.get(log["order_id"])
            if o is None:
                continue
            if log["status"] in ("REJECTED", "EXPIRED"):
                if o["filled_quantity"]:
                    o.update(status="PARTIAL", reason=o["reason"] or log["reason"])
                else:
                    o.update(status=log["status"], reason=log["reason"])
            else:
                o.update(status=log["status"], reason=log["reason"])

    for i, d in enumerate(days):
        bars_today = {s: (good[s].loc[d] if d in good[s].index else None) for s in symbols}

        # 1. 公司行为（开盘前），并确定当日涨跌停参考价
        for s in symbols:
            ref_price = last_close[s]
            for a in actions[s].get(d, []):
                if a.cash_per_share:
                    account.cash_dividend(d, s, a.cash_per_share)
                if a.bonus_per_share:
                    account.bonus_shares(d, s, a.bonus_per_share)
                if ref_price is not None:
                    ref_price = ex_right_reference_price(ref_price, a.cash_per_share, a.bonus_per_share)
            if d >= first_bar[s]:                       # 首根K线之前（如尚未上市）不适用任何交易规则
                rules[s].set_day(d, i, ref_price)

        # 2. 撮合：先卖后买（买入的风险检查以前一交易日收盘权益与收盘价为基准）
        gate.start_day(prev_equity, last_close)
        def fill_symbol(s):
            handler, bar = handlers[s], bars_today[s]
            pending = handler.get_pending_orders(s)
            if not pending:
                return
            if bar is None:
                reason = "数据已结束（可能退市或数据缺失），无法成交" if d > last_bar[s] else "停牌或当日无有效行情"
                for o in pending:
                    handler.cancel_order(o.order_id)
                    orders[o.order_id].update(status="REJECTED", reason=reason)
                    if orders[o.order_id].get("risk_event"):
                        ev = next(e for e in tracker.events if e["event_id"] == orders[o.order_id]["risk_event"])
                        ev["trim_blocked"].append({"date": str(d.date()), "order_id": o.order_id, "reason": reason})
                return
            event = BarEvent(timestamp=d, symbol=s, open=float(bar["open"]), high=float(bar["high"]),
                             low=float(bar["low"]), close=float(bar["close"]),
                             volume=float(bar.get("volume", np.inf)))
            for f in handler.process_bar(event):
                fees = f.data["fees"]
                qty = int(round(f.quantity))
                if f.side == "BUY":
                    account.buy(d, s, qty, f.fill_price, fees, f.order_id)
                    gate.record_buy(s, qty, qty * f.fill_price + sum(fees.values()))
                else:
                    account.sell(d, s, qty, f.fill_price, fees, f.order_id, trading_days_between)
                fills.append({"date": str(d.date()), "symbol": s, "order_id": f.order_id, "side": f.side,
                              "quantity": qty, "price": f.fill_price, "open": event.open, "fees": fees,
                              "fee_total": sum(fees.values())})
                orders[f.order_id]["filled_quantity"] += qty
            sync_logs(handler)
            for o in pending:
                rec = orders[o.order_id]
                if rec.get("risk_event") and rec["status"] in ("REJECTED", "EXPIRED", "PARTIAL"):
                    ev = next(e for e in tracker.events if e["event_id"] == rec["risk_event"])
                    ev["trim_blocked"].append({"date": str(d.date()), "order_id": o.order_id,
                                               "reason": rec["reason"]})

        sell_syms = [s for s in symbols if any(o.side == "SELL" for o in handlers[s].get_pending_orders(s))]
        buy_syms = sorted((o.order_id, s) for s in symbols for o in handlers[s].get_pending_orders(s) if o.side == "BUY")
        for s in sell_syms:
            fill_symbol(s)
        for _, s in buy_syms:
            fill_symbol(s)
        live = pending_ids()
        for oid in list(account._frozen_by_order):
            if oid not in live:
                account.release(d, oid)

        # 3. 收盘估值（停牌或数据结束沿用最近收盘价）
        for s in symbols:
            if bars_today[s] is not None:
                last_close[s] = float(bars_today[s]["close"])
        mv = {s: account.quantity(s) * last_close[s] if (account.quantity(s) and last_close[s] is not None) else 0.0
              for s in symbols}
        equity_before_orders = account.cash + account.frozen + sum(mv.values())
        if equity_before_orders > 0:                     # 收盘时的被动超限（下单之前判定）
            for s in symbols:
                tracker.observe(d, s, "单票上限", mv[s] / equity_before_orders, budget.max_weight)
            tracker.observe(d, "组合", "总仓位上限", sum(mv.values()) / equity_before_orders, budget.max_gross)

        # 4. 收盘后生成订单
        sells, buys = [], []
        for x in insts:
            s = x.symbol
            if bars_today[s] is None or d not in x.signals.index or handlers[s].get_pending_orders(s):
                continue
            in_universe = True if x.eligible is None else bool(x.eligible.get(d, False))
            if not in_universe:
                if account.quantity(s) > 0:
                    sells.append((x, "UNIVERSE_EXIT"))         # 已不在股票池：清仓，不再买入
                continue
            v = _signal_value(x.signals.loc[d])
            if v < 0 and account.quantity(s) > 0:
                sells.append((x, v))
            elif v > 0 and account.quantity(s) == 0:
                buys.append((x, v))
        for x, v in sells:
            s = x.symbol
            oid = f"O{next(ids):05d}"
            orders[oid] = {"order_id": oid, "symbol": s, "signal_date": str(d.date()), "side": "SELL",
                           "signal": v if isinstance(v, str) else Signal(max(v, -2)).name,
                           "signal_close": last_close[s],
                           "quantity": account.quantity(s), "filled_quantity": 0, "status": "SUBMITTED",
                           "reason": "", "sizing_note": ""}
            handlers[s].submit_order(OrderEvent(timestamp=d, order_id=oid, symbol=s, side="SELL",
                                                quantity=account.quantity(s), order_type="MARKET",
                                                time_in_force="DAY"))
        if budget.trim_to_limits:
            selling = {x.symbol for x, _ in sells}
            trim_value: Dict[str, float] = {}
            for s in symbols:
                if s in selling or bars_today[s] is None or handlers[s].get_pending_orders(s) or not mv[s]:
                    continue
                excess = mv[s] - budget.max_weight * equity_before_orders
                if excess > 1e-9:
                    trim_value[s] = excess
            gross_excess = (sum(mv.values()) - sum(mv[s] for s in selling) - sum(trim_value.values())
                            - budget.max_gross * equity_before_orders)
            eligible = [s for s in symbols if s not in selling and bars_today[s] is not None
                        and not handlers[s].get_pending_orders(s) and mv[s]]
            if gross_excess > 1e-9 and eligible:
                base = sum(mv[s] - trim_value.get(s, 0.0) for s in eligible)
                for s in eligible:
                    trim_value[s] = trim_value.get(s, 0.0) + gross_excess * (mv[s] - trim_value.get(s, 0.0)) / base
            for s in sorted(trim_value):
                r, holding = rules[s], account.quantity(s)
                step = r.lot.step
                want = math.ceil(trim_value[s] / last_close[s] / step) * step     # 向上取整，保证减到上限以内
                qty = holding if want >= holding else r.lot.valid_sell(want, holding)
                oid = f"O{next(ids):05d}"
                record = {"order_id": oid, "symbol": s, "signal_date": str(d.date()), "side": "SELL",
                          "signal": "RISK_TRIM", "signal_close": last_close[s], "quantity": qty,
                          "filled_quantity": 0, "status": "SUBMITTED", "reason": "",
                          "sizing_note": f"风险预算减仓：需减持市值 {trim_value[s]:,.0f}",
                          "risk_event": (tracker.event_for(s) or {}).get("event_id")}
                if record["risk_event"]:
                    next(e for e in tracker.events if e["event_id"] == record["risk_event"])["trim_orders"].append(oid)
                orders[oid] = record
                if qty <= 0:
                    record.update(status="REJECTED", reason="减仓数量不满足申报单位")
                    continue
                handlers[s].submit_order(OrderEvent(timestamp=d, order_id=oid, symbol=s, side="SELL", quantity=qty,
                                                    order_type="MARKET", time_in_force="DAY"))
        for x, v in sorted(buys, key=lambda t: (-t[1], t[0].symbol)):
            s, r = x.symbol, rules[x.symbol]
            signal = Signal.STRONG_BUY if v >= 2 else Signal.BUY
            est_price = last_close[s] * (1 + x.market.slippage)
            committed = sum(mv.values()) + account.frozen
            caps = {
                "策略仓位": x.strategy.calculate_position_size(signal, est_price, equity_before_orders) * est_price,
                "单票上限": budget.max_weight * equity_before_orders,
                "总仓位上限": max(budget.max_gross * equity_before_orders - committed, 0.0),
                "可用资金": account.cash,
            }
            low = min(caps.values())
            binding = "、".join(k for k, val in caps.items() if val <= low + 1e-6)   # 并列时全部列出
            qty = r.lot.round_buy(min(caps.values()) / est_price)
            # 策略仓位是目标持仓金额（不含费用）；风险预算与可用资金是硬上限，预估金额含费用不得超过
            hard_limit = min(caps["单票上限"], caps["总仓位上限"], caps["可用资金"])
            while qty > 0 and qty * est_price + r.fees(est_price, qty, "BUY", d) > hard_limit + 1e-9:
                qty = r.lot.round_buy(qty - r.lot.step)
            risk_bound = caps["单票上限"] < caps["策略仓位"] or caps["总仓位上限"] < caps["策略仓位"]
            if risk_bound and min(caps["单票上限"], caps["总仓位上限"]) <= min(caps["策略仓位"], caps["可用资金"]):
                gate.log(d, s, None, "ORDER_REJECTED" if qty <= 0 else "ORDER_SIZED_DOWN",
                         requested=caps["策略仓位"], allowed=min(caps["单票上限"], caps["总仓位上限"]), binding=binding)
            oid = f"O{next(ids):05d}"
            record = {"order_id": oid, "symbol": s, "signal_date": str(d.date()), "side": "BUY",
                      "signal": signal.name, "signal_close": last_close[s], "quantity": qty, "filled_quantity": 0,
                      "status": "SUBMITTED", "reason": "",
                      "sizing_note": f"受“{binding}”约束：" + "，".join(f"{k} {val:,.0f}" for k, val in caps.items())}
            orders[oid] = record
            if gate.log_rows and gate.log_rows[-1]["order_id"] is None and gate.log_rows[-1]["symbol"] == s:
                gate.log_rows[-1]["order_id"] = oid
            if qty <= 0:
                record.update(status="REJECTED", reason=f"按“{binding}”计算不足一个申报单位")
            else:
                account.freeze(d, oid, qty * est_price + r.fees(est_price, qty, "BUY", d))
                handlers[s].submit_order(OrderEvent(timestamp=d, order_id=oid, symbol=s, side="BUY", quantity=qty,
                                                    order_type="MARKET", time_in_force="DAY"))

        total_mv = sum(mv.values())
        equity = account.cash + account.frozen + total_mv
        prev_equity = equity
        rows.append({"date": d, "cash": account.cash, "frozen": account.frozen, "market_value": total_mv,
                     "equity": equity, "gross_exposure": total_mv / equity if equity else 0.0,
                     "positions": sum(1 for s in symbols if account.quantity(s) > 0)})
        for s in symbols:
            pos_rows.append({"date": d, "symbol": s, "quantity": account.quantity(s), "close": last_close[s],
                             "market_value": mv[s], "weight": mv[s] / equity if equity else 0.0,
                             "suspended": bars_today[s] is None,
                             "status": ("未上市/无数据" if d < first_bar[s] else
                                        "数据已结束" if d > last_bar[s] else
                                        "停牌/无行情" if bars_today[s] is None else "正常")})

    ledger = pd.DataFrame(rows).set_index("date")
    positions = pd.DataFrame(pos_rows).set_index(["date", "symbol"])
    for o in orders.values():
        if o["status"] == "SUBMITTED":
            o["status"] = "OPEN_AT_END"
            o["reason"] = "回测结束时仍未到撮合日"
    return {
        "ledger": ledger,
        "positions": positions,
        "orders": list(orders.values()),
        "fills": fills,
        "journal": account.journal,
        "trades": account.closed_trades,
        "account": account,
        "invalid_days": invalid_days,
        "budget": budget,
        "risk_events": tracker.finish(),
        "risk_log": gate.log_rows,
        "open_lots": {s: [vars(l).copy() for l in lots] for s, lots in account.lots.items()},
    }
