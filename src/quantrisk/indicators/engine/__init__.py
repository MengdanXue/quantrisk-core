"""
指标计算引擎模块

提供批量计算、流水线处理、预设指标组合和Numba加速
"""

from .batch_calculator import (
    BatchCalculator,
    IndicatorConfig,
    CalculationResult,
    IndicatorPipeline,
    IndicatorPresets,
    batch_calculator
)

from .numba_core import (
    calculate_rsi_fast,
    calculate_ema_fast,
    calculate_atr_fast,
    calculate_bollinger_fast,
    calculate_macd_fast,
    rolling_mean_fast,
    rolling_std_fast,
    NUMBA_AVAILABLE,
)

from .v2_adapter import (
    V2IndicatorAdapter,
    V2CompatibilityLayer,
    create_v2_adapter,
    create_compatibility_layer,
    v2_adapter,
    v2_compat,
)

__all__ = [
    # 批量计算器
    'BatchCalculator',
    'IndicatorConfig',
    'CalculationResult',
    'IndicatorPipeline',
    'IndicatorPresets',
    'batch_calculator',
    # Numba加速函数
    'calculate_rsi_fast',
    'calculate_ema_fast',
    'calculate_atr_fast',
    'calculate_bollinger_fast',
    'calculate_macd_fast',
    'rolling_mean_fast',
    'rolling_std_fast',
    'NUMBA_AVAILABLE',
    # V2适配器
    'V2IndicatorAdapter',
    'V2CompatibilityLayer',
    'create_v2_adapter',
    'create_compatibility_layer',
    'v2_adapter',
    'v2_compat',
]
