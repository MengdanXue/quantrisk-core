"""
统计类指标模块

包含20个统计学相关指标
"""

from .statistical_indicators import (
    ZScore, PercentRank, StandardDeviation, Variance,
    Skewness, Kurtosis, Correlation, Beta,
    Sharpe, Sortino, Calmar, MaxDrawdown,
    VaR, CVaR, InformationRatio, OmegaRatio,
    TailRatio, HurstExponent, AutoCorrelation
)

__all__ = [
    # Statistical Indicators (20)
    'ZScore', 'PercentRank', 'StandardDeviation', 'Variance',
    'Skewness', 'Kurtosis', 'Correlation', 'Beta',
    'Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown',
    'VaR', 'CVaR', 'InformationRatio', 'OmegaRatio',
    'TailRatio', 'HurstExponent', 'AutoCorrelation',
]
