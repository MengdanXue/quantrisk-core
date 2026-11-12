"""
信号与交易建议（信号和订单分开记录）

信号：现有 ma_cross 策略（参数在实验配置中登记），输入为按除息日调整后的价格：
每个除息日把之前的价格序列乘以 前收盘价 ÷（前收盘价 − 现金红利），只用当日及以前已发生的分红，
因此 D 日的信号只依赖 D 日及以前的数据。

交易建议（D 日收盘后生成，D+1 日有效的限价单）：
- 买入：限价 = D 日收盘价 + N 个报价单位（向上取整至报价单位，不超过 D+1 涨停价）；数量 = 资金可买的最大整手；
- 卖出：限价 = D 日收盘价 − N 个报价单位（不低于 D+1 跌停价）；数量 = 全部持仓；
- 显性费用（佣金等）与估算交易摩擦（半个买卖价差）分别计算与记录；两者之和占成交额超过预算即拒绝；
- 资金不足一手、行情完整性有问题、规则未核实时拒绝，并写明原因。
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..strategies import get_strategy
from .etf_rules import ETFRules, FeeScenario, explicit_fee_total


def adjusted_prices(bars: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """按已发生的分红前向累乘复权（截至当日），返回 open/high/low/close 调整后的价格。"""
    factor, value = [], 1.0
    cash = dividends["cash"].to_dict() if len(dividends) else {}
    prev_close = None
    for d, r in bars.iterrows():
        if d in cash and prev_close is not None and cash[d] > 0:
            value *= prev_close / (prev_close - cash[d])
        factor.append(value)
        prev_close = r["close"]
    f = pd.Series(factor, index=bars.index)
    out = bars[["open", "high", "low", "close"]].mul(f, axis=0)
    out["volume"] = bars["volume"]
    return out


def strategy_signals(bars: pd.DataFrame, dividends: pd.DataFrame, strategy_cfg: dict) -> pd.Series:
    strat = get_strategy(strategy_cfg["name"], dict(strategy_cfg["params"]))
    data = strat.preprocess_data(adjusted_prices(bars, dividends))
    return strat.generate_signals(data).astype(int)


@dataclass
class Proposal:
    action: str                   # PROPOSE / REJECT
    side: str
    reason: str = ""
    order: Optional[dict] = None

    def as_dict(self):
        return {"action": self.action, "side": self.side, "reason": self.reason, "order": self.order}


def propose(day: pd.Timestamp, side: str, close: float, cash: float, holding: int, rules: ETFRules,
            fee: FeeScenario, budget: float, half_spread_ticks: int, limit_offset_ticks: int,
            order_id: str, blockers=()) -> Proposal:
    if blockers:
        return Proposal("REJECT", side, "；".join(blockers))
    lo, up = rules.limit_prices(close)
    if side == "BUY":
        limit = min(rules.round_up(close + rules.ticks(limit_offset_ticks)), up)
        qty = 0
        for q in range(rules.max_buy_lots_qty(cash / limit), 0, -rules.lot_step):
            if q * limit + explicit_fee_total(fee.breakdown(limit, q)) <= cash + 1e-9:
                qty = q
                break
        if qty < rules.lot_min:
            need = rules.lot_min * limit + explicit_fee_total(fee.breakdown(limit, rules.lot_min))
            return Proposal("REJECT", side, f"可用资金 {cash:.2f} 元不足一手（{rules.lot_min} 份按限价 {limit:.3f} "
                                            f"含费用需 {need:.2f} 元）")
    else:
        limit = max(rules.round_down(close - rules.ticks(limit_offset_ticks)), lo)
        qty = holding
        if not rules.valid_sell_qty(qty, holding):
            return Proposal("REJECT", side, f"卖出数量 {qty} 不符合申报单位")
    notional = qty * limit
    fees = fee.breakdown(limit, qty)
    explicit = explicit_fee_total(fees)
    friction = round(qty * rules.ticks(half_spread_ticks), 6)
    order = {"order_id": order_id, "decision_day": str(day.date()), "side": side, "qty": qty,
             "ref_close": close, "limit_price": limit, "next_day_limit_band": [lo, up],
             "notional_at_limit": round(notional, 4), "est_explicit_fees": fees, "est_explicit_total": explicit,
             "est_explicit_ratio": explicit / notional, "est_friction": friction, "est_friction_ratio": friction / notional,
             "est_total_ratio": (explicit + friction) / notional, "budget": budget, "cash_before": round(cash, 6),
             "est_cash_after": round(cash - notional - explicit if side == "BUY" else cash + notional - explicit, 6),
             "holding_before": holding}
    if order["est_total_ratio"] > budget + 1e-12:
        return Proposal("REJECT", side, f"单边摩擦成本 {order['est_total_ratio']:.2%}（显性费用 {order['est_explicit_ratio']:.2%}"
                                        f" + 估算价差 {order['est_friction_ratio']:.2%}）超过预算 {budget:.2%}", order)
    return Proposal("PROPOSE", side, "", order)
