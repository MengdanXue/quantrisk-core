"""
动量类指标模块

包含振荡器、MACD系列、价格动量等70+个动量指标
"""

# 振荡器指标
from .oscillators import (
    RSI, StochasticOscillator, StochasticRSI, CCI,
    WilliamsR, UltimateOscillator, AwesomeOscillator,
    AcceleratorOscillator, PrettyGoodOscillator, TSI,
    ConnorsRSI, LaguerreRSI, SMI, RVI
)

# MACD系列
from .macd_series import (
    MACD, MACDHistogram, MACDZeroLag, MACDLeader,
    MACDBB, PPOMACD
)

# 价格动量指标
from .price_momentum import (
    Momentum, ROC, APO, CMO, InertiaIndicator,
    ForecastOscillator, LinearRegressionSlope, RSquared
)

__all__ = [
    # Oscillators (14)
    'RSI', 'StochasticOscillator', 'StochasticRSI', 'CCI',
    'WilliamsR', 'UltimateOscillator', 'AwesomeOscillator',
    'AcceleratorOscillator', 'PrettyGoodOscillator', 'TSI',
    'ConnorsRSI', 'LaguerreRSI', 'SMI', 'RVI',

    # MACD Series (6)
    'MACD', 'MACDHistogram', 'MACDZeroLag', 'MACDLeader',
    'MACDBB', 'PPOMACD',

    # Price Momentum (8)
    'Momentum', 'ROC', 'APO', 'CMO', 'InertiaIndicator',
    'ForecastOscillator', 'LinearRegressionSlope', 'RSquared',
]
