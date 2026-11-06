"""
高级统计指标模块

包含20个高级统计分析指标
"""

from .advanced_statistics import (
    RollingRegression, KalmanFilter, CopulaCorrelation, EntropyIndicator,
    MutualInformation, FractalDimension, WaveletDecomposition, PeakValleyDetection,
    BootstrapConfidence, RobustStatistics, QuantileRegression, ExponentialSmoothing,
    CointegrationTest, ChangePointDetection, RegimeSwitching, OutlierDetection,
    SerialDependence, NonlinearDynamics
)

__all__ = [
    # Advanced Statistics (20)
    'RollingRegression', 'KalmanFilter', 'CopulaCorrelation', 'EntropyIndicator',
    'MutualInformation', 'FractalDimension', 'WaveletDecomposition', 'PeakValleyDetection',
    'BootstrapConfidence', 'RobustStatistics', 'QuantileRegression', 'ExponentialSmoothing',
    'CointegrationTest', 'ChangePointDetection', 'RegimeSwitching', 'OutlierDetection',
    'SerialDependence', 'NonlinearDynamics',
]
