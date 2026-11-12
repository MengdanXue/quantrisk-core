"""测试六：A 股交易规则的历史版本与板块例外。"""
import pytest

from quantrisk.strategies.backtest.ashare_rules import (Board, UnsupportedRuleError, classify_board,
                                                        ex_right_reference_price, limit_prices, lot_rule,
                                                        price_limit_pct, stamp_duty_rate, transfer_fee_rate)
from quantrisk.strategies.backtest.execution import AShareFeeModel


@pytest.mark.parametrize("code,board", [
    ("600519", Board.SH_MAIN), ("601318", Board.SH_MAIN), ("000001", Board.SZ_MAIN), ("002594", Board.SZ_MAIN),
    ("300750", Board.CHINEXT), ("688981", Board.STAR), ("830799", Board.BSE), ("920001", Board.BSE),
])
def test_board_classification(code, board):
    assert classify_board(code) == board


def test_unknown_code_rejected():
    with pytest.raises(UnsupportedRuleError):
        classify_board("510300")   # ETF 不在本模块的股票规则范围内


def test_main_board_st_limit_switch_2026_07_06():
    assert price_limit_pct(Board.SH_MAIN, "2026-07-03", is_st=True) == 0.05
    assert price_limit_pct(Board.SH_MAIN, "2026-07-06", is_st=True) == 0.10
    assert price_limit_pct(Board.SZ_MAIN, "2026-07-03", is_st=True) == 0.05
    assert price_limit_pct(Board.SZ_MAIN, "2026-07-06", is_st=True) == 0.10
    assert price_limit_pct(Board.SH_MAIN, "2026-07-03") == 0.10       # 非 ST 不受影响


def test_chinext_reform_2020_08_24():
    assert price_limit_pct(Board.CHINEXT, "2020-08-21") == 0.10
    assert price_limit_pct(Board.CHINEXT, "2020-08-21", is_st=True) == 0.05
    assert price_limit_pct(Board.CHINEXT, "2020-08-24") == 0.20
    assert price_limit_pct(Board.CHINEXT, "2020-08-24", is_st=True) == 0.20
    assert price_limit_pct(Board.CHINEXT, "2026-07-06", is_st=True) == 0.20   # 2026 修订只涉及主板


def test_star_and_bse_limits_and_opening_dates():
    assert price_limit_pct(Board.STAR, "2024-01-02") == 0.20
    assert price_limit_pct(Board.BSE, "2024-01-02") == 0.30
    with pytest.raises(UnsupportedRuleError):
        price_limit_pct(Board.STAR, "2019-07-19")
    with pytest.raises(UnsupportedRuleError):
        price_limit_pct(Board.BSE, "2021-11-12")
    with pytest.raises(UnsupportedRuleError):
        price_limit_pct(Board.SH_MAIN, "1996-12-13")


def test_new_listing_windows():
    assert price_limit_pct(Board.STAR, "2024-01-02", listing_day=3) is None
    assert price_limit_pct(Board.SH_MAIN, "2023-04-10", listing_day=1) is None      # 全面注册制后主板前 5 日不设限
    assert price_limit_pct(Board.SH_MAIN, "2023-04-10", listing_day=6) == 0.10
    with pytest.raises(UnsupportedRuleError):
        price_limit_pct(Board.SH_MAIN, "2022-06-01", listing_day=1)                   # 旧首日 44% 规则未收录


def test_limit_price_rounding_half_up():
    assert limit_prices(10.05, 0.10) == (9.05, 11.06)     # 11.055 → 11.06，9.045 → 9.05
    assert limit_prices(1.0, None) == (None, None)


def test_ex_right_reference_price():
    assert ex_right_reference_price(10.00, cash_per_share=0.5) == 9.50
    assert ex_right_reference_price(10.00, cash_per_share=0.0, bonus_per_share=1.0) == 5.00
    assert ex_right_reference_price(10.00, cash_per_share=0.2, bonus_per_share=0.5) == 6.53


def test_lot_rules_by_board():
    main = lot_rule(Board.SH_MAIN, "2024-01-02")
    assert main.round_buy(250) == 200 and main.round_buy(99) == 0
    assert main.valid_sell(150, holding=250) == 100
    assert main.valid_sell(250, holding=250) == 250       # 含零股的全部卖出
    assert main.valid_sell(50, holding=50) == 50          # 余额不足 100 股须一次卖出
    star = lot_rule(Board.STAR, "2024-01-02")
    assert star.round_buy(199) == 0 and star.round_buy(201) == 201
    assert star.valid_sell(150, holding=300) == 0         # 科创板单笔卖出不低于 200 股（非清仓）
    bse = lot_rule(Board.BSE, "2024-01-02")
    assert bse.round_buy(101) == 101


def test_stamp_duty_and_transfer_fee_versions():
    assert stamp_duty_rate("2023-08-25", "SELL") == 0.001
    assert stamp_duty_rate("2023-08-28", "SELL") == 0.0005
    assert stamp_duty_rate("2023-08-28", "BUY") == 0.0
    assert transfer_fee_rate("2022-04-28") == 0.00002
    assert transfer_fee_rate("2022-04-29") == 0.00001
    with pytest.raises(UnsupportedRuleError):
        stamp_duty_rate("2008-09-18", "SELL")
    with pytest.raises(UnsupportedRuleError):
        transfer_fee_rate("2015-07-31")


def test_ashare_fee_breakdown():
    fees = AShareFeeModel(commission_rate=0.00025, min_commission=5.0)
    sell = fees.breakdown(10.0, 1000, "SELL", "2024-03-01")      # 成交额 1 万
    assert sell == {"commission": 5.0, "stamp_duty": 5.0, "transfer_fee": 0.1}
    big = fees.breakdown(100.0, 10000, "BUY", "2023-08-25")      # 成交额 100 万
    assert big == {"commission": 250.0, "stamp_duty": 0.0, "transfer_fee": 10.0}
    with pytest.raises(ValueError):
        fees.calculate_commission(10.0, 100, "BUY")               # 没有日期无法确定税费版本


def test_ashare_market_applies_st_dates_with_version_switch():
    from quantrisk.strategies.backtest.daily import AShareMarket
    m = AShareMarket("600000", st_dates=["2026-07-03", "2026-07-06"])
    assert m.limits("2026-07-03", 10.0) == (9.5, 10.5)       # 主板 ST：5%
    assert m.limits("2026-07-06", 10.0) == (9.0, 11.0)       # 2026-07-06 起：10%
    assert m.limits("2026-07-07", 10.0) == (9.0, 11.0)       # 非 ST 日：10%
    assert AShareMarket("300750", st_dates=["2026-07-03"]).limits("2026-07-03", 10.0) == (8.0, 12.0)
