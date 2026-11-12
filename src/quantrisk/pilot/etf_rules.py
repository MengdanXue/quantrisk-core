"""
ETF 交易规则与费用（第一阶段：深交所上市的境内普通股票型宽基 ETF）

规则只从核实记录（rules_szse_etf_20261109.json，附交易所原文条款与核实日期）加载；
白名单之外的代码、白名单内但交易所规则未核实的代码、任一必需规则未核实，一律抛错，不做猜测。
价格一律用 Decimal 按 0.001 元报价单位处理，避免浮点误差。
"""

import json
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Optional, Tuple

RULES_DIR = Path(__file__).resolve().parent
RULE_FILES = {"sz": "rules_szse_etf_20261109.json"}
REQUIRED_RULES = ("tick_size", "lot", "price_limit_pct", "limit_price_formula", "ex_dividend",
                  "settlement_t_plus_1", "stamp_duty")
CATEGORY = "DOMESTIC_EQUITY_BROAD"          # 境内普通股票型宽基；跨境、债券、黄金、货币、商品等一律不支持

# 2026-11-09 审计过的白名单（ETF_PILOT_AUDIT.md 第 3 节）：只收录普通股票型宽基，不含主题、ESG、指数增强
WHITELIST: Dict[str, dict] = {
    "sh560050": {"name": "中国A50ETF汇添富", "index": "MSCI中国A50互联互通"},
    "sz159602": {"name": "中国A50ETF南方", "index": "MSCI中国A50互联互通"},
    "sz159596": {"name": "A50ETF华宝", "index": "中证A50"},
    "sh563750": {"name": "中证500ETF汇添富", "index": "中证500"},
    "sh563500": {"name": "A500ETF华宝", "index": "中证A500"},
    "sz159975": {"name": "深100ETF招商", "index": "深证100"},
    "sz159706": {"name": "深证100ETF华安", "index": "深证100"},
    "sh510370": {"name": "沪深300ETF兴业", "index": "沪深300"},
    "sz159716": {"name": "深证100ETF华宝", "index": "深证100"},
    "sh510670": {"name": "上证180ETF华夏", "index": "上证180"},
}
WHITELIST_AUDIT_DATE = "2026-11-09"


class UnsupportedETFError(ValueError):
    """代码不在白名单内，或类别不受支持。"""


class RuleNotVerifiedError(ValueError):
    """所需交易规则没有经过核实，拒绝使用。"""


@dataclass(frozen=True)
class ETFRules:
    code: str
    exchange: str
    tick: Decimal
    lot_min: int
    lot_step: int
    limit_pct: Decimal
    settlement_lag: int
    stamp_duty: float
    exchange_handling_fee_rate: float
    transfer_fee_verified: bool
    verified_on: str
    rule_file: str

    # ---------- 价格 ----------
    def _q(self, value, rounding) -> Decimal:
        return (Decimal(str(value)) / self.tick).quantize(Decimal(1), rounding=rounding) * self.tick

    def round_nearest(self, value) -> float:
        return float(self._q(value, ROUND_HALF_UP))

    def round_up(self, value) -> float:
        return float(self._q(value, ROUND_CEILING))

    def round_down(self, value) -> float:
        return float(self._q(value, ROUND_FLOOR))

    def on_tick(self, price) -> bool:
        d = Decimal(str(price))
        return d == self._q(d, ROUND_HALF_UP)

    def ticks(self, n: int) -> float:
        return float(self.tick * n)

    def limit_prices(self, ref_price) -> Tuple[float, float]:
        """(跌停价, 涨停价)：前收盘价（或除息参考价）×（1±比例），四舍五入至报价单位；与基准之差不足一个报价单位时按基准 ± 1 个单位。"""
        ref = Decimal(str(ref_price))
        lo = self._q(ref * (1 - self.limit_pct), ROUND_HALF_UP)
        up = self._q(ref * (1 + self.limit_pct), ROUND_HALF_UP)
        if up - ref < self.tick:
            up = ref + self.tick
        if ref - lo < self.tick:
            lo = ref - self.tick
        return float(lo), float(up)

    def ex_dividend_reference(self, prev_close, cash_per_share) -> float:
        """除息参考价 = 前收盘价 − 现金红利（ETF 无送转、配股），四舍五入至报价单位。"""
        return self.round_nearest(Decimal(str(prev_close)) - Decimal(str(cash_per_share)))

    # ---------- 数量 ----------
    def max_buy_lots_qty(self, qty: float) -> int:
        q = int(qty // self.lot_step * self.lot_step)
        return q if q >= self.lot_min else 0

    def valid_buy_qty(self, qty: int) -> bool:
        return qty >= self.lot_min and qty % self.lot_step == 0

    def valid_sell_qty(self, qty: int, holding: int) -> bool:
        """卖出：100 份整数倍；余额不足 100 份的部分须一次性卖出（卖出全部持仓总是合法）。"""
        if qty <= 0 or qty > holding:
            return False
        return qty == holding or qty % self.lot_step == 0


def load_rules(code: str) -> ETFRules:
    if code not in WHITELIST:
        raise UnsupportedETFError(f"{code} 不在已审计的普通股票型宽基 ETF 白名单内（审计日期 {WHITELIST_AUDIT_DATE}）")
    exchange = code[:2]
    if exchange not in RULE_FILES:
        raise RuleNotVerifiedError(f"{code}：{exchange} 交易所的 ETF 交易规则尚未核实，拒绝生成交易建议")
    path = RULES_DIR / RULE_FILES[exchange]
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data["rules"]
    missing = [k for k in REQUIRED_RULES if rules.get(k, {}).get("status") != "VERIFIED"]
    if missing:
        raise RuleNotVerifiedError(f"{code}：以下规则未核实 {missing}")
    lot = rules["lot"]
    return ETFRules(code=code, exchange=exchange, tick=Decimal(str(rules["tick_size"]["value"])),
                    lot_min=int(lot["min_qty"]), lot_step=int(lot["step"]),
                    limit_pct=Decimal(str(rules["price_limit_pct"]["value"])),
                    settlement_lag=int(rules["settlement_t_plus_1"]["value"]),
                    stamp_duty=float(rules["stamp_duty"]["value"]),
                    exchange_handling_fee_rate=float(rules["exchange_handling_fee"]["value"]),
                    transfer_fee_verified=rules.get("transfer_fee", {}).get("status") == "VERIFIED",
                    verified_on=data["verified_on"], rule_file=path.name)


# ---------------- 费用 ----------------

def _cents(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class FeeScenario:
    """
    显性费用：佣金（max(成交额 × 费率, 最低佣金)）、过户费、其他费用，各项按分四舍五入；ETF 无印花税。
    actual_broker=False 表示这是假设的情景，不是某家券商的真实报价。
    """
    name: str
    commission_rate: float
    min_commission: float
    transfer_fee_rate: float = 0.0
    other_fee_rate: float = 0.0
    actual_broker: bool = False
    note: str = ""

    def breakdown(self, price: float, qty: int) -> Dict[str, float]:
        amount = Decimal(str(price)) * qty
        out = {"commission": _cents(max(amount * Decimal(str(self.commission_rate)),
                                       Decimal(str(self.min_commission))))}
        if self.transfer_fee_rate:
            out["transfer_fee"] = _cents(amount * Decimal(str(self.transfer_fee_rate)))
        if self.other_fee_rate:
            out["other_fee"] = _cents(amount * Decimal(str(self.other_fee_rate)))
        return out

    def as_dict(self) -> dict:
        return asdict(self)


SCENARIOS: Dict[str, FeeScenario] = {
    "A": FeeScenario("A_min5", commission_rate=0.00025, min_commission=5.0,
                     note="情景 A：最低佣金 5 元（深交所投教指南：佣金“起点5元”）；费率 0.025% 为占位，最低佣金时不起作用"),
    "B": FeeScenario("B_0.85bp_nomin", commission_rate=0.000085, min_commission=0.0,
                     note="情景 B：万分之 0.85、无最低收费；仅作理论敏感性测试，不代表任何券商的真实报价"),
}


def fee_scenario(key_or_dict) -> FeeScenario:
    if isinstance(key_or_dict, dict):
        return FeeScenario(**key_or_dict)
    if key_or_dict not in SCENARIOS:
        raise ValueError(f"未知的佣金情景 {key_or_dict}，可选 {sorted(SCENARIOS)}")
    return SCENARIOS[key_or_dict]


def explicit_fee_total(fees: Dict[str, float]) -> float:
    return round(sum(fees.values()), 2)


def limit_band(rules: ETFRules, prev_close: float, ex_cash: Optional[float] = None) -> Tuple[float, float]:
    ref = rules.ex_dividend_reference(prev_close, ex_cash) if ex_cash else prev_close
    return rules.limit_prices(ref)
