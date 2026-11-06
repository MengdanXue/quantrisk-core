"""
趋势类指标模块

包含移动平均线、趋势强度、支撑阻力等80+个趋势指标
"""

# 移动平均线系列
from .moving_averages import (
    SMA, EMA, WMA, DEMA, TEMA,
    KAMA, HMA, ALMA, T3, TRIMA,
    ZLEMA, SMMA, LSMA, VWMA, FRAMA,
    VIDYA, JMA, EVWMA, McGinleyDynamic, SWMA
)

# 趋势强度指标
from .trend_strength import (
    ADX, Aroon, AroonOscillator, DI_Plus, DI_Minus,
    DMI, VortexIndicator, MassIndex, DPO, KST,
    TRIX, Supertrend, ParabolicSAR, ChandelierExit
)

# 趋势通道指标
from .trend_channels import (
    DonchianChannel, KeltnerChannel, EnvelopeChannel,
    RegressionChannel, LinearRegressionLine
)

__all__ = [
    # Moving Averages (20)
    'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA',
    'KAMA', 'HMA', 'ALMA', 'T3', 'TRIMA',
    'ZLEMA', 'SMMA', 'LSMA', 'VWMA', 'FRAMA',
    'VIDYA', 'JMA', 'EVWMA', 'McGinleyDynamic', 'SWMA',

    # Trend Strength (14)
    'ADX', 'Aroon', 'AroonOscillator', 'DI_Plus', 'DI_Minus',
    'DMI', 'VortexIndicator', 'MassIndex', 'DPO', 'KST',
    'TRIX', 'Supertrend', 'ParabolicSAR', 'ChandelierExit',

    # Trend Channels (5)
    'DonchianChannel', 'KeltnerChannel', 'EnvelopeChannel',
    'RegressionChannel', 'LinearRegressionLine',
]
