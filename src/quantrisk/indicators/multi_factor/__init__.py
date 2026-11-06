"""
多因子指标模块

包含价值、质量、成长、动量、波动、规模、流动性和组合因子等25个多因子指标
"""

from .factor_indicators import (
    # 价值因子 (5)
    PBFactor, PEFactor, PSFactor, PCFFactor, EVEBITDAFactor,

    # 质量因子 (5)
    ROEFactor, ROAFactor, GrossProfitMarginFactor, NetProfitMarginFactor, AssetTurnoverFactor,

    # 成长因子 (3)
    RevenueGrowthFactor, ProfitGrowthFactor, EPSGrowthFactor,

    # 动量因子 (3)
    PriceMomentumFactor, ReturnMomentumFactor, RelativeStrengthFactor,

    # 波动因子 (3)
    BetaFactor, IdiosyncraticVolatilityFactor, DownsideVolatilityFactor,

    # 规模因子 (2)
    MarketCapFactor, FloatMarketCapFactor,

    # 流动性因子 (2)
    TurnoverRateFactor, AmihudIlliquidityFactor,

    # 组合因子 (2)
    MultiFactorScore, FactorRotationSignal
)

__all__ = [
    # Value Factors (5)
    'PBFactor', 'PEFactor', 'PSFactor', 'PCFFactor', 'EVEBITDAFactor',

    # Quality Factors (5)
    'ROEFactor', 'ROAFactor', 'GrossProfitMarginFactor',
    'NetProfitMarginFactor', 'AssetTurnoverFactor',

    # Growth Factors (3)
    'RevenueGrowthFactor', 'ProfitGrowthFactor', 'EPSGrowthFactor',

    # Momentum Factors (3)
    'PriceMomentumFactor', 'ReturnMomentumFactor', 'RelativeStrengthFactor',

    # Volatility Factors (3)
    'BetaFactor', 'IdiosyncraticVolatilityFactor', 'DownsideVolatilityFactor',

    # Size Factors (2)
    'MarketCapFactor', 'FloatMarketCapFactor',

    # Liquidity Factors (2)
    'TurnoverRateFactor', 'AmihudIlliquidityFactor',

    # Composite Factors (2)
    'MultiFactorScore', 'FactorRotationSignal',
]
