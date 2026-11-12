"""
PAPER 模式的限价单模拟成交（D 日收盘生成、D+1 日有效）

日线数据无法证明盘中排队能否成交，因此只认开盘价：
- STRICT_FILL：开盘价满足限价（买：开盘价 ≤ 限价；卖：开盘价 ≥ 限价），且开盘不在涨停（买）/ 跌停（卖）。
  成交价 = 开盘价 ± 半个估算价差，再限制在限价以内与当日最高/最低价之间，并落在报价单位上——
  成交价永远不违反限价，也不会落在当日实际成交区间之外；
- UNCERTAIN_TOUCH：开盘价不满足、但当日最低价（买）/ 最高价（卖）触及限价。是否成交无法确定，
  不计入严格账本；敏感性账本（PAPER_TOUCH）按限价成交；
- EXPIRED：当日无行情（停牌或数据缺失）、开盘即涨停（买）/ 跌停（卖）、或价格未触及限价。

价差与滑点只体现在成交价中（成交价相对开盘价的偏离单独统计），显性费用另行按情景计算——不重复计费。
"""

from typing import Optional

import pandas as pd

from .etf_rules import ETFRules, FeeScenario, explicit_fee_total


def simulate_fill(order: dict, bar: Optional[pd.Series], band, rules: ETFRules, fee: FeeScenario,
                  half_spread_ticks: int) -> dict:
    side, limit, qty = order["side"], order["limit_price"], order["qty"]
    base = {"order_id": order["order_id"], "side": side, "qty": qty, "limit_price": limit}
    if bar is None:
        return {**base, "status": "EXPIRED", "reason": "当日无行情（停牌或数据缺失），订单作废"}
    lo, up = band
    o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
    basis = {"bar": {"open": o, "high": h, "low": l, "close": float(bar["close"])}, "limit_band": [lo, up]}
    hs = rules.ticks(half_spread_ticks)
    if side == "BUY":
        if o >= up - 1e-9:
            return {**base, **basis, "status": "EXPIRED", "reason": f"开盘价 {o} 已达涨停价 {up}，保守假设无法买入"}
        if o <= limit + 1e-9:
            price = min(limit, rules.round_up(o + hs), up, h)     # 不超过限价、涨停价与当日最高价
            rule = "开盘价 ≤ 限价：按开盘价 + 半个价差成交（不超过限价）"
        elif l <= limit + 1e-9:
            return {**base, **basis, "status": "UNCERTAIN_TOUCH", "touch_price": limit,
                    "reason": f"开盘价 {o} 高于限价 {limit}，盘中最低价 {l} 触及限价：是否成交无法确定"}
        else:
            return {**base, **basis, "status": "EXPIRED", "reason": f"最低价 {l} 未触及限价 {limit}"}
    else:
        if o <= lo + 1e-9:
            return {**base, **basis, "status": "EXPIRED", "reason": f"开盘价 {o} 已达跌停价 {lo}，保守假设无法卖出"}
        if o >= limit - 1e-9:
            price = max(limit, rules.round_down(o - hs), lo, l)   # 不低于限价、跌停价与当日最低价
            rule = "开盘价 ≥ 限价：按开盘价 − 半个价差成交（不低于限价）"
        elif h >= limit - 1e-9:
            return {**base, **basis, "status": "UNCERTAIN_TOUCH", "touch_price": limit,
                    "reason": f"开盘价 {o} 低于限价 {limit}，盘中最高价 {h} 触及限价：是否成交无法确定"}
        else:
            return {**base, **basis, "status": "EXPIRED", "reason": f"最高价 {h} 未触及限价 {limit}"}
    assert rules.on_tick(price) and l - 1e-9 <= price <= h + 1e-9
    assert price <= limit + 1e-9 if side == "BUY" else price >= limit - 1e-9
    fees = fee.breakdown(price, qty)
    return {**base, **basis, "status": "STRICT_FILL", "rule": rule, "price": price, "fees": fees,
            "fee_total": explicit_fee_total(fees), "spread_cost_vs_open": round(abs(price - o) * qty, 6),
            "price_vs_signal_close": round(price - order["ref_close"], 6)}


def touch_fill(result: dict, order: dict, fee: FeeScenario) -> dict:
    """敏感性账本：把 UNCERTAIN_TOUCH 按限价视为成交（价差成本按限价相对开盘价计）。"""
    price = result["touch_price"]
    fees = fee.breakdown(price, order["qty"])
    return {**result, "status": "TOUCH_FILL", "rule": "盘中触价，按限价成交（仅敏感性分析）", "price": price,
            "fees": fees, "fee_total": explicit_fee_total(fees),
            "spread_cost_vs_open": round(abs(price - result["bar"]["open"]) * order["qty"], 6),
            "price_vs_signal_close": round(price - order["ref_close"], 6)}
