"""
波动率类指标模块

包含布林带系列、ATR系列、历史波动率等25个波动率指标
"""

# 布林带系列
from .bollinger_series import (
    BollingerBands, BollingerPercentB, BollingerBandwidth, BollingerSqueeze,
    BollingerTrend, DoubleBollingerBands, KeltnerBollingerSqueeze, BollingerMomentum
)

# ATR系列
from .atr_series import (
    ATR, TrueRange, ATRPercent, ATRTrailingStop,
    ATRChannel, NormalizedATR, ATRRatio
)

# 历史波动率
from .historical_volatility import (
    HistoricalVolatility, ParkinsonVolatility, GarmanKlassVolatility,
    RogersSatchellVolatility, YangZhangVolatility, RealizedVolatility,
    VolatilityCone, VolatilityRatio, EWMA_Volatility, GARCH_Volatility
)

__all__ = [
    # Bollinger Series (8)
    'BollingerBands', 'BollingerPercentB', 'BollingerBandwidth', 'BollingerSqueeze',
    'BollingerTrend', 'DoubleBollingerBands', 'KeltnerBollingerSqueeze', 'BollingerMomentum',

    # ATR Series (7)
    'ATR', 'TrueRange', 'ATRPercent', 'ATRTrailingStop',
    'ATRChannel', 'NormalizedATR', 'ATRRatio',

    # Historical Volatility (10)
    'HistoricalVolatility', 'ParkinsonVolatility', 'GarmanKlassVolatility',
    'RogersSatchellVolatility', 'YangZhangVolatility', 'RealizedVolatility',
    'VolatilityCone', 'VolatilityRatio', 'EWMA_Volatility', 'GARCH_Volatility',
]
