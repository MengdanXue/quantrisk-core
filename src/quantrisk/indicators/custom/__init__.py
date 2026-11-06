"""
Custom Composite Indicators Module

This module provides 25 advanced custom composite indicators for technical analysis,
including trend quality, multi-timeframe, market state, trading signals, and risk-adjusted metrics.

Copyright (c) 2025 TraderX. All rights reserved.
Licensed under the MIT License.
"""

from .custom_indicators import (
    # Trend Quality Indicators
    TrendStrengthScore,
    TrendPersistence,
    TrendReversalProbability,

    # Multi-Timeframe Indicators
    MTFTrendConsistency,
    MTFMomentum,
    MTFSupportResistance,

    # Market State Indicators
    MarketStateClassifier,
    VolatilityState,
    TrendOscillatorDetector,

    # Trading Signal Indicators
    ComprehensiveBuySellSignal,
    SignalStrength,
    SignalReliability,

    # Risk-Adjusted Indicators
    RiskAdjustedMomentum,
    VolatilityAdjustedRSI,
    ATRNormalizedPrice,

    # Relative Strength Indicators
    SectorRelativeStrength,
    StockVsMarket,
    RelativeMomentumRank,

    # Extremes Detection
    PriceExtremes,
    VolumeExtremes,
    VolatilityExtremes,

    # Composite Scores
    TechnicalScore,
    MomentumScore,
    RiskScore,
    MarketConditionIndex,
)

__all__ = [
    # Trend Quality
    'TrendStrengthScore',
    'TrendPersistence',
    'TrendReversalProbability',

    # Multi-Timeframe
    'MTFTrendConsistency',
    'MTFMomentum',
    'MTFSupportResistance',

    # Market State
    'MarketStateClassifier',
    'VolatilityState',
    'TrendOscillatorDetector',

    # Trading Signals
    'ComprehensiveBuySellSignal',
    'SignalStrength',
    'SignalReliability',

    # Risk-Adjusted
    'RiskAdjustedMomentum',
    'VolatilityAdjustedRSI',
    'ATRNormalizedPrice',

    # Relative Strength
    'SectorRelativeStrength',
    'StockVsMarket',
    'RelativeMomentumRank',

    # Extremes
    'PriceExtremes',
    'VolumeExtremes',
    'VolatilityExtremes',

    # Scores
    'TechnicalScore',
    'MomentumScore',
    'RiskScore',
    'MarketConditionIndex',
]
