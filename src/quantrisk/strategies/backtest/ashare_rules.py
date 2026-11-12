"""
A 股交易规则（带生效日期的版本表）

只收录本模块能说明来源的规则；超出已收录时间范围的日期一律抛出 UnsupportedRuleError，
不做猜测。规则来源：
- 涨跌幅：沪深交易所交易规则；创业板改革（2020-08-24 起 20%）；科创板（2019-07-22 开市，20%）；
  北交所（2021-11-15 开市，30%）；沪深北交易所 2026 年修订交易规则（2026-07-06 起主板风险警示股票 5% → 10%）。
- 印花税：2008-09-19 起单边对卖方征收 0.1%；2023-08-28 起减半为 0.05%。
- 过户费：2015-08-01 起沪深统一按成交金额 0.002%；2022-04-29 起降为 0.001%。
- 申报数量：主板、创业板买入为 100 股整数倍；科创板单笔不低于 200 股、以 1 股递增；
  北交所单笔不低于 100 股、以 1 股递增；卖出余额不足最低申报数量时须一次性申报卖出。

未建模（调用方须知）：新股上市初期的特殊涨跌幅（见 price_limit_pct 的 listing_day 参数）、
风险警示状态的历史判定（需调用方提供 is_st）、退市整理期、盘后固定价格交易、北交所以外的大宗交易。
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Optional

import pandas as pd


class UnsupportedRuleError(ValueError):
    """日期或证券类别超出本模块已收录的规则范围。"""


class Board(str, Enum):
    SH_MAIN = "sh_main"      # 上交所主板
    SZ_MAIN = "sz_main"      # 深交所主板（含原中小板）
    CHINEXT = "chinext"      # 创业板
    STAR = "star"            # 科创板
    BSE = "bse"              # 北交所


def classify_board(symbol: str) -> Board:
    """按 6 位证券代码判断板块（仅 A 股股票）。"""
    code = str(symbol).strip()[-6:]
    if not (len(code) == 6 and code.isdigit()):
        raise UnsupportedRuleError(f"无法识别的证券代码: {symbol}")
    if code.startswith(("688", "689")):
        return Board.STAR
    if code.startswith(("600", "601", "603", "605")):
        return Board.SH_MAIN
    if code.startswith(("300", "301")):
        return Board.CHINEXT
    if code.startswith(("000", "001", "002", "003")):
        return Board.SZ_MAIN
    if code.startswith(("43", "83", "87", "88", "92")):
        return Board.BSE
    raise UnsupportedRuleError(f"不是本模块支持的 A 股股票代码: {symbol}")


def _d(value) -> date:
    return pd.Timestamp(value).date()


PRICE_LIMIT_START = date(1996, 12, 16)
CHINEXT_REFORM = date(2020, 8, 24)
STAR_OPEN = date(2019, 7, 22)
BSE_OPEN = date(2021, 11, 15)
MAIN_BOARD_REGISTRATION = date(2023, 4, 10)
MAIN_ST_10PCT = date(2026, 7, 6)


def price_limit_pct(board: Board, trade_date, is_st: bool = False,
                    listing_day: Optional[int] = None) -> Optional[float]:
    """
    当日涨跌幅限制比例；返回 None 表示当日无涨跌幅限制。

    listing_day: 上市后的第几个交易日（从 1 开始）；None 表示上市已满 5 个交易日。
    """
    d = _d(trade_date)
    if d < PRICE_LIMIT_START:
        raise UnsupportedRuleError(f"{d} 早于 1996-12-16 涨跌幅制度，未收录")

    if listing_day is not None and listing_day <= 5:
        if board in (Board.STAR, Board.CHINEXT) and (board == Board.STAR or d >= CHINEXT_REFORM):
            return None
        if board in (Board.SH_MAIN, Board.SZ_MAIN) and d >= MAIN_BOARD_REGISTRATION:
            return None
        if board == Board.BSE and listing_day == 1:
            return None
        if listing_day == 1:
            raise UnsupportedRuleError(f"{board.value} 在 {d} 的新股首日涨跌幅规则未收录")

    if board == Board.STAR:
        if d < STAR_OPEN:
            raise UnsupportedRuleError(f"科创板 {d} 尚未开市")
        return 0.20
    if board == Board.BSE:
        if d < BSE_OPEN:
            raise UnsupportedRuleError(f"北交所 {d} 尚未开市")
        return 0.30
    if board == Board.CHINEXT:
        if d >= CHINEXT_REFORM:
            return 0.20
        return 0.05 if is_st else 0.10
    # 沪深主板
    if is_st:
        return 0.10 if d >= MAIN_ST_10PCT else 0.05
    return 0.10


def round_price(value: float) -> float:
    """按 0.01 元四舍五入。"""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def limit_prices(ref_price: float, pct: Optional[float]):
    """由前收盘（或除权除息参考价）计算 (跌停价, 涨停价)；pct 为 None 时返回 (None, None)。"""
    if pct is None:
        return None, None
    return round_price(ref_price * (1 - pct)), round_price(ref_price * (1 + pct))


def ex_right_reference_price(prev_close: float, cash_per_share: float = 0.0,
                             bonus_per_share: float = 0.0) -> float:
    """除权（息）参考价 =（前收盘 − 每股现金红利）÷（1 + 每股送转股比例）。"""
    return round_price((prev_close - cash_per_share) / (1 + bonus_per_share))


@dataclass(frozen=True)
class LotRule:
    min_qty: int      # 单笔最低申报数量
    step: int         # 递增单位

    def round_buy(self, qty: float) -> int:
        q = int(qty // self.step * self.step)
        return q if q >= self.min_qty else 0

    def valid_sell(self, qty: int, holding: int) -> int:
        """把卖出数量调整为合法申报数量；余额不足最低申报数量时只能一次性卖出全部。"""
        qty = int(min(qty, holding))
        if qty <= 0:
            return 0
        if qty == holding:
            return qty          # 全部卖出总是合法（零股部分随同一次申报）
        q = qty // self.step * self.step
        return q if q >= self.min_qty else 0


def lot_rule(board: Board, trade_date) -> LotRule:
    d = _d(trade_date)
    if board == Board.STAR:
        if d < STAR_OPEN:
            raise UnsupportedRuleError(f"科创板 {d} 尚未开市")
        return LotRule(min_qty=200, step=1)
    if board == Board.BSE:
        if d < BSE_OPEN:
            raise UnsupportedRuleError(f"北交所 {d} 尚未开市")
        return LotRule(min_qty=100, step=1)
    return LotRule(min_qty=100, step=100)


STAMP_DUTY_SELL_ONLY = date(2008, 9, 19)
STAMP_DUTY_HALVED = date(2023, 8, 28)
TRANSFER_FEE_UNIFIED = date(2015, 8, 1)
TRANSFER_FEE_HALVED = date(2022, 4, 29)


def stamp_duty_rate(trade_date, side: str) -> float:
    d = _d(trade_date)
    if d < STAMP_DUTY_SELL_ONLY:
        raise UnsupportedRuleError(f"{d} 早于 2008-09-19 单边征收印花税，未收录")
    if side.upper() != "SELL":
        return 0.0
    return 0.0005 if d >= STAMP_DUTY_HALVED else 0.001


def transfer_fee_rate(trade_date) -> float:
    d = _d(trade_date)
    if d < TRANSFER_FEE_UNIFIED:
        raise UnsupportedRuleError(f"{d} 早于 2015-08-01 过户费统一按金额收取，未收录")
    return 0.00001 if d >= TRANSFER_FEE_HALVED else 0.00002


# T+1：当日买入的股票，下一交易日起才能卖出
SETTLEMENT_LAG_DAYS = 1
