"""
跨市场分析指标模块

包含20个跨市场分析指标：
- 相关性指标 (5个)
- 套利指标 (5个)
- 市场间传导 (5个)
- 相对强弱 (5个)
"""

from .cross_market_indicators import (
    # 相关性指标 (5个)
    RollingCorrelation,
    DynamicCorrelation,
    CointegrationTest,
    GrangerCausality,
    LeadLagRelation,

    # 套利指标 (5个)
    PairSpread,
    ZScore,
    HedgeRatio,
    MeanReversionSpeed,
    ArbitrageSignal,

    # 市场间传导 (5个)
    SpilloverIndex,
    ContagionRisk,
    SystemicRisk,
    MarketIntegration,
    InformationFlow,

    # 相对强弱 (5个)
    RelativeStrength,
    SectorRotation,
    StyleMomentum,
    AssetAllocationSignal,
    RiskParityWeight,
)

__all__ = [
    # 相关性指标 (5个)
    'RollingCorrelation',
    'DynamicCorrelation',
    'CointegrationTest',
    'GrangerCausality',
    'LeadLagRelation',

    # 套利指标 (5个)
    'PairSpread',
    'ZScore',
    'HedgeRatio',
    'MeanReversionSpeed',
    'ArbitrageSignal',

    # 市场间传导 (5个)
    'SpilloverIndex',
    'ContagionRisk',
    'SystemicRisk',
    'MarketIntegration',
    'InformationFlow',

    # 相对强弱 (5个)
    'RelativeStrength',
    'SectorRotation',
    'StyleMomentum',
    'AssetAllocationSignal',
    'RiskParityWeight',
]
