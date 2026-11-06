"""
跨品种分析指标模块
包含15个跨市场分析指标
"""

from .intermarket_indicators import (
    # 货币与商品关系
    DollarCommodityRelation, GoldEquityRatio, CopperGoldRatio,
    OilEquityCorrelation, CurrencyCarryTrade,
    # 债券与股票关系
    BondEquityCorrelation, YieldCurveSlope, RealYield,
    CreditSpread, TIPSSpread,
    # 风险情绪指标
    RiskOnOffIndicator, SafeHavenDemand, LiquidityStress,
    GlobalRiskAppetite, MarketRegimeIndicator
)

__all__ = [
    'DollarCommodityRelation', 'GoldEquityRatio', 'CopperGoldRatio',
    'OilEquityCorrelation', 'CurrencyCarryTrade',
    'BondEquityCorrelation', 'YieldCurveSlope', 'RealYield',
    'CreditSpread', 'TIPSSpread',
    'RiskOnOffIndicator', 'SafeHavenDemand', 'LiquidityStress',
    'GlobalRiskAppetite', 'MarketRegimeIndicator'
]
