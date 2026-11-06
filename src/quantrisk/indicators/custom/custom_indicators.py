"""
Custom Composite Indicators

This module implements 25 advanced custom composite indicators for technical analysis,
combining multiple classic indicators and price action analysis.

Copyright (c) 2025 TraderX. All rights reserved.
Licensed under the MIT License.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


class _MetadataDrivenIndicator(BaseIndicator):
    """Adapter for composite indicators whose contract lives in metadata."""

    def _validate_params(self):
        return None

    @property
    def required_columns(self) -> List[str]:
        return list(self.get_metadata().required_columns)


# ============================================================================
# TREND QUALITY INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class TrendStrengthScore(_MetadataDrivenIndicator):
    """
    Trend Strength Score - Composite indicator measuring trend strength
    Combines ADX, price position vs MA, and trend consistency
    Output: Score 0-100, higher means stronger trend
    """

    _metadata = IndicatorMetadata(
        name="TrendStrengthScore",
        display_name="Trend Strength Score",
        category="custom",
        description="Composite trend strength score combining ADX, price position, and trend consistency",
        parameters={
            'adx_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 200},
            'lookback': {'type': 'int', 'default': 10, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Trend Strength Score"""
        adx_period = params.get('adx_period', 14)
        ma_period = params.get('ma_period', 20)
        lookback = params.get('lookback', 10)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=adx_period).mean()

        # Normalize ADX to 0-100 scale (ADX rarely exceeds 50)
        adx_score = np.minimum(adx * 2, 100)

        # Calculate price position vs MA (0-100)
        ma = close.rolling(window=ma_period).mean()
        price_deviation = ((close - ma) / ma * 100).abs()
        price_score = np.minimum(price_deviation * 10, 100)

        # Calculate trend consistency (percentage of closes moving in same direction)
        close_diff = close.diff()
        trend_direction = np.sign(close_diff)
        consistency = trend_direction.rolling(window=lookback).apply(
            lambda x: abs(x.sum()) / len(x) * 100
        )

        # Composite score (weighted average)
        score = (adx_score * 0.5 + price_score * 0.2 + consistency * 0.3)

        return score.fillna(0)


@indicator_registry.register('custom')
class TrendPersistence(_MetadataDrivenIndicator):
    """
    Trend Persistence - Measures how long the current trend has persisted
    Output: Number of consecutive periods in same trend direction
    """

    _metadata = IndicatorMetadata(
        name="TrendPersistence",
        display_name="Trend Persistence",
        category="custom",
        description="Counts consecutive periods in the same trend direction",
        parameters={
            'ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 100}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Trend Persistence"""
        ma_period = params.get('ma_period', 20)
        close = data['close']

        # Calculate MA and determine trend
        ma = close.rolling(window=ma_period).mean()
        trend = (close > ma).astype(int)
        trend[close < ma] = -1

        # Count consecutive same values
        persistence = pd.Series(0, index=data.index, dtype=float)

        current_count = 0
        current_trend = 0

        for i in range(len(trend)):
            if pd.isna(trend.iloc[i]):
                persistence.iloc[i] = 0
                continue

            if trend.iloc[i] == current_trend and current_trend != 0:
                current_count += 1
            else:
                current_trend = trend.iloc[i]
                current_count = 1

            persistence.iloc[i] = current_count if current_trend == 1 else -current_count

        return persistence


@indicator_registry.register('custom')
class TrendReversalProbability(_MetadataDrivenIndicator):
    """
    Trend Reversal Probability - Estimates probability of trend reversal
    Combines price extremes, RSI divergence, and trend exhaustion signals
    Output: Probability 0-100 (higher = higher reversal probability)
    """

    _metadata = IndicatorMetadata(
        name="TrendReversalProbability",
        display_name="Trend Reversal Probability",
        category="custom",
        description="Estimates probability of trend reversal based on multiple factors",
        parameters={
            'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'lookback': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'bb_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Trend Reversal Probability"""
        rsi_period = params.get('rsi_period', 14)
        lookback = params.get('lookback', 20)
        bb_period = params.get('bb_period', 20)

        close = data['close']
        high = data['high']
        low = data['low']

        # RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # RSI extreme score (0-100)
        rsi_extreme = pd.Series(0, index=data.index, dtype=float)
        rsi_extreme[rsi > 70] = (rsi - 70) * 3.33  # Scale 70-100 to 0-100
        rsi_extreme[rsi < 30] = (30 - rsi) * 3.33  # Scale 0-30 to 100-0

        # Price extreme score (distance from recent high/low)
        rolling_high = high.rolling(window=lookback).max()
        rolling_low = low.rolling(window=lookback).min()

        range_size = rolling_high - rolling_low
        upper_distance = (rolling_high - close) / range_size * 100
        lower_distance = (close - rolling_low) / range_size * 100

        price_extreme = pd.Series(0, index=data.index, dtype=float)
        price_extreme = np.maximum(100 - upper_distance * 100, 100 - lower_distance * 100)

        # Bollinger Band squeeze
        ma = close.rolling(window=bb_period).mean()
        std = close.rolling(window=bb_period).std()
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std

        bb_extreme = pd.Series(0, index=data.index, dtype=float)
        bb_extreme[close > bb_upper] = ((close - bb_upper) / std * 50).clip(0, 100)
        bb_extreme[close < bb_lower] = ((bb_lower - close) / std * 50).clip(0, 100)

        # Composite probability
        probability = (rsi_extreme * 0.4 + price_extreme * 0.3 + bb_extreme * 0.3)

        return probability.fillna(0).clip(0, 100)


# ============================================================================
# MULTI-TIMEFRAME INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class MTFTrendConsistency(_MetadataDrivenIndicator):
    """
    Multi-Timeframe Trend Consistency - Measures trend alignment across timeframes
    Simulates multiple timeframes using different MA periods
    Output: -100 to 100 (-100 = all bearish, 100 = all bullish)
    """

    _metadata = IndicatorMetadata(
        name="MTFTrendConsistency",
        display_name="MTF Trend Consistency",
        category="custom",
        description="Measures trend alignment across multiple simulated timeframes",
        parameters={
            'fast_period': {'type': 'int', 'default': 10, 'min': 5, 'max': 50},
            'medium_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'slow_period': {'type': 'int', 'default': 50, 'min': 20, 'max': 200},
            'very_slow_period': {'type': 'int', 'default': 100, 'min': 50, 'max': 300}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate MTF Trend Consistency"""
        fast = params.get('fast_period', 10)
        medium = params.get('medium_period', 20)
        slow = params.get('slow_period', 50)
        very_slow = params.get('very_slow_period', 100)

        close = data['close']

        # Calculate MAs for different "timeframes"
        ma_fast = close.rolling(window=fast).mean()
        ma_medium = close.rolling(window=medium).mean()
        ma_slow = close.rolling(window=slow).mean()
        ma_very_slow = close.rolling(window=very_slow).mean()

        # Determine trend for each timeframe
        trends = pd.DataFrame({
            'fast': (close > ma_fast).astype(int) * 2 - 1,
            'medium': (close > ma_medium).astype(int) * 2 - 1,
            'slow': (close > ma_slow).astype(int) * 2 - 1,
            'very_slow': (close > ma_very_slow).astype(int) * 2 - 1
        })

        # Calculate consistency (-100 to 100)
        consistency = trends.mean(axis=1) * 100

        return consistency.fillna(0)


@indicator_registry.register('custom')
class MTFMomentum(_MetadataDrivenIndicator):
    """
    Multi-Timeframe Momentum - Composite momentum across multiple timeframes
    Output: Normalized momentum score -100 to 100
    """

    _metadata = IndicatorMetadata(
        name="MTFMomentum",
        display_name="MTF Momentum",
        category="custom",
        description="Composite momentum score across multiple timeframes",
        parameters={
            'short_period': {'type': 'int', 'default': 5, 'min': 3, 'max': 20},
            'medium_period': {'type': 'int', 'default': 14, 'min': 10, 'max': 50},
            'long_period': {'type': 'int', 'default': 28, 'min': 20, 'max': 100}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate MTF Momentum"""
        short = params.get('short_period', 5)
        medium = params.get('medium_period', 14)
        long = params.get('long_period', 28)

        close = data['close']

        # Calculate ROC for each timeframe
        roc_short = (close / close.shift(short) - 1) * 100
        roc_medium = (close / close.shift(medium) - 1) * 100
        roc_long = (close / close.shift(long) - 1) * 100

        # Normalize each to -100 to 100 scale (assuming max 20% move)
        def normalize_roc(roc):
            return (roc / 20 * 100).clip(-100, 100)

        # Weighted composite (short-term weighted less)
        momentum = (
            normalize_roc(roc_short) * 0.2 +
            normalize_roc(roc_medium) * 0.3 +
            normalize_roc(roc_long) * 0.5
        )

        return momentum.fillna(0)


@indicator_registry.register('custom')
class MTFSupportResistance(_MetadataDrivenIndicator):
    """
    Multi-Timeframe Support/Resistance - Distance to nearest support/resistance
    Output: Negative = near support, Positive = near resistance, 0 = neutral
    """

    _metadata = IndicatorMetadata(
        name="MTFSupportResistance",
        display_name="MTF Support/Resistance",
        category="custom",
        description="Distance to nearest support/resistance across timeframes",
        parameters={
            'short_lookback': {'type': 'int', 'default': 10, 'min': 5, 'max': 30},
            'medium_lookback': {'type': 'int', 'default': 20, 'min': 10, 'max': 60},
            'long_lookback': {'type': 'int', 'default': 50, 'min': 20, 'max': 100}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate MTF Support/Resistance"""
        short = params.get('short_lookback', 10)
        medium = params.get('medium_lookback', 20)
        long = params.get('long_lookback', 50)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate support/resistance levels for each timeframe
        def calc_sr_distance(lookback):
            resistance = high.rolling(window=lookback).max()
            support = low.rolling(window=lookback).min()

            dist_to_resistance = (resistance - close) / close * 100
            dist_to_support = (close - support) / close * 100

            # Positive if closer to resistance, negative if closer to support
            return dist_to_resistance - dist_to_support

        sr_short = calc_sr_distance(short)
        sr_medium = calc_sr_distance(medium)
        sr_long = calc_sr_distance(long)

        # Weighted average
        mtf_sr = (sr_short * 0.2 + sr_medium * 0.3 + sr_long * 0.5)

        return mtf_sr.fillna(0).clip(-100, 100)


# ============================================================================
# MARKET STATE INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class MarketStateClassifier(_MetadataDrivenIndicator):
    """
    Market State Classifier - Classifies market into states
    Output: 1=Strong Uptrend, 2=Weak Uptrend, 3=Ranging, 4=Weak Downtrend, 5=Strong Downtrend
    """

    _metadata = IndicatorMetadata(
        name="MarketStateClassifier",
        display_name="Market State Classifier",
        category="custom",
        description="Classifies market state based on trend and volatility",
        parameters={
            'adx_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 100},
            'adx_threshold': {'type': 'float', 'default': 25.0, 'min': 10.0, 'max': 50.0}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Market State"""
        adx_period = params.get('adx_period', 14)
        ma_period = params.get('ma_period', 20)
        adx_threshold = params.get('adx_threshold', 25.0)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=adx_period).mean()

        # Calculate MA
        ma = close.rolling(window=ma_period).mean()

        # Classify states
        state = pd.Series(3, index=data.index, dtype=float)  # Default: Ranging

        # Strong trends (ADX > threshold)
        strong_trend = adx > adx_threshold
        state.loc[strong_trend & (close > ma) & (plus_di > minus_di)] = 1  # Strong Uptrend
        state.loc[strong_trend & (close < ma) & (minus_di > plus_di)] = 5  # Strong Downtrend

        # Weak trends (ADX <= threshold but price vs MA shows direction)
        weak_trend = adx <= adx_threshold
        state.loc[weak_trend & (close > ma)] = 2  # Weak Uptrend
        state.loc[weak_trend & (close < ma)] = 4  # Weak Downtrend

        return state.fillna(3)


@indicator_registry.register('custom')
class VolatilityState(_MetadataDrivenIndicator):
    """
    Volatility State - Classifies volatility regime
    Output: 1=Low Vol, 2=Normal Vol, 3=High Vol, 4=Extreme Vol
    """

    _metadata = IndicatorMetadata(
        name="VolatilityState",
        display_name="Volatility State",
        category="custom",
        description="Classifies current volatility regime",
        parameters={
            'atr_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'lookback': {'type': 'int', 'default': 100, 'min': 50, 'max': 500}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Volatility State"""
        atr_period = params.get('atr_period', 14)
        lookback = params.get('lookback', 100)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=atr_period).mean()

        # Calculate ATR percentile rank over lookback period
        atr_percentile = atr.rolling(window=lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Classify volatility states
        state = pd.Series(2, index=data.index, dtype=float)  # Default: Normal
        state.loc[atr_percentile < 25] = 1  # Low Vol
        state.loc[atr_percentile > 75] = 3  # High Vol
        state.loc[atr_percentile > 90] = 4  # Extreme Vol

        return state.fillna(2)


@indicator_registry.register('custom')
class TrendOscillatorDetector(_MetadataDrivenIndicator):
    """
    Trend/Oscillator Detector - Determines if market is trending or oscillating
    Output: -100 to 100 (-100 = strong oscillation, 100 = strong trend)
    """

    _metadata = IndicatorMetadata(
        name="TrendOscillatorDetector",
        display_name="Trend/Oscillator Detector",
        category="custom",
        description="Determines if market is trending or oscillating",
        parameters={
            'adx_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'bb_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Trend/Oscillator Score"""
        adx_period = params.get('adx_period', 14)
        bb_period = params.get('bb_period', 20)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ADX (trend strength)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=adx_period).mean()

        # Calculate Bollinger Band width (oscillation indicator)
        ma = close.rolling(window=bb_period).mean()
        std = close.rolling(window=bb_period).std()
        bb_width = (std * 4 / ma * 100)  # Percentage width

        # Normalize BB width (assume 10% is very wide)
        bb_width_norm = (bb_width / 10 * 100).clip(0, 100)

        # Combine: High ADX + Wide BB = Trending, Low ADX + Narrow BB = Oscillating
        trend_score = (adx * 2).clip(0, 100)  # ADX to 0-100
        osc_score = 100 - bb_width_norm  # Narrow bands = oscillating

        # Composite: positive = trending, negative = oscillating
        detector = trend_score - osc_score

        return detector.fillna(0).clip(-100, 100)


# ============================================================================
# TRADING SIGNAL INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class ComprehensiveBuySellSignal(_MetadataDrivenIndicator):
    """
    Comprehensive Buy/Sell Signal - Composite signal from multiple indicators
    Output: -100 to 100 (-100 = strong sell, 100 = strong buy, 0 = neutral)
    """

    _metadata = IndicatorMetadata(
        name="ComprehensiveBuySellSignal",
        display_name="Comprehensive Buy/Sell Signal",
        category="custom",
        description="Composite buy/sell signal from multiple technical indicators",
        parameters={
            'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'macd_fast': {'type': 'int', 'default': 12, 'min': 5, 'max': 50},
            'macd_slow': {'type': 'int', 'default': 26, 'min': 10, 'max': 100},
            'macd_signal': {'type': 'int', 'default': 9, 'min': 3, 'max': 30}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close', 'volume']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Comprehensive Signal"""
        rsi_period = params.get('rsi_period', 14)
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)

        close = data['close']
        volume = data['volume']

        # RSI Signal
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        rsi_signal = pd.Series(0, index=data.index, dtype=float)
        rsi_signal.loc[rsi < 30] = (30 - rsi) * 3.33  # Oversold = buy signal
        rsi_signal.loc[rsi > 70] = -(rsi - 70) * 3.33  # Overbought = sell signal

        # MACD Signal
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # Normalize MACD histogram
        macd_std = macd_hist.rolling(window=50).std()
        macd_signal = (macd_hist / macd_std * 50).clip(-100, 100)

        # Moving Average Signal
        ma_20 = close.rolling(window=20).mean()
        ma_50 = close.rolling(window=50).mean()
        ma_signal = pd.Series(0, index=data.index, dtype=float)
        ma_signal.loc[(close > ma_20) & (ma_20 > ma_50)] = 50  # Bullish alignment
        ma_signal.loc[(close < ma_20) & (ma_20 < ma_50)] = -50  # Bearish alignment

        # Volume Signal
        vol_ma = volume.rolling(window=20).mean()
        vol_ratio = volume / vol_ma
        vol_signal = (vol_ratio - 1) * 50  # High volume = stronger signal

        # Composite Signal
        composite = (
            rsi_signal.fillna(0) * 0.25 +
            macd_signal.fillna(0) * 0.30 +
            ma_signal.fillna(0) * 0.30 +
            vol_signal.fillna(0) * 0.15
        )

        return composite.clip(-100, 100)


@indicator_registry.register('custom')
class SignalStrength(_MetadataDrivenIndicator):
    """
    Signal Strength - Measures the strength/conviction of current signals
    Output: 0-100 (higher = stronger conviction)
    """

    _metadata = IndicatorMetadata(
        name="SignalStrength",
        display_name="Signal Strength",
        category="custom",
        description="Measures strength and conviction of trading signals",
        parameters={
            'lookback': {'type': 'int', 'default': 14, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close', 'volume']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Signal Strength"""
        lookback = params.get('lookback', 14)

        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # Price momentum strength
        roc = (close / close.shift(lookback) - 1).abs() * 100
        momentum_strength = (roc / 10 * 100).clip(0, 100)  # Assume 10% move is strong

        # Volume strength
        vol_ma = volume.rolling(window=lookback).mean()
        vol_strength = ((volume / vol_ma - 1) * 100).clip(0, 100)

        # Volatility strength (higher volatility = stronger signals)
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=lookback).mean()
        atr_pct = atr / close * 100
        volatility_strength = (atr_pct / 5 * 100).clip(0, 100)  # 5% ATR is strong

        # Trend consistency strength
        close_diff = close.diff()
        trend_direction = np.sign(close_diff)
        consistency = trend_direction.rolling(window=lookback).apply(
            lambda x: abs(x.sum()) / len(x) * 100
        )

        # Composite strength
        strength = (
            momentum_strength * 0.30 +
            vol_strength * 0.20 +
            volatility_strength * 0.20 +
            consistency * 0.30
        )

        return strength.fillna(0).clip(0, 100)


@indicator_registry.register('custom')
class SignalReliability(_MetadataDrivenIndicator):
    """
    Signal Reliability - Estimates reliability of current signals
    Based on historical success rate and market conditions
    Output: 0-100 (higher = more reliable)
    """

    _metadata = IndicatorMetadata(
        name="SignalReliability",
        display_name="Signal Reliability",
        category="custom",
        description="Estimates reliability of trading signals based on market conditions",
        parameters={
            'lookback': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'threshold': {'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 5.0}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Signal Reliability"""
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 2.0)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ADX (trending markets more reliable)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()

        trend_reliability = (adx * 2).clip(0, 100)

        # Price stability (lower noise = more reliable)
        returns = close.pct_change()
        rolling_std = returns.rolling(window=lookback).std()
        rolling_mean = returns.rolling(window=lookback).mean().abs()

        noise_ratio = rolling_std / (rolling_mean + 1e-10)
        stability = (1 / (1 + noise_ratio / threshold) * 100).clip(0, 100)

        # Pattern consistency
        ma_20 = close.rolling(window=20).mean()
        ma_50 = close.rolling(window=50).mean()

        pattern_score = pd.Series(100, index=data.index, dtype=float)
        pattern_score.loc[(close > ma_20) & (ma_20 < ma_50)] = 50  # Mixed signals
        pattern_score.loc[(close < ma_20) & (ma_20 > ma_50)] = 50  # Mixed signals

        # Composite reliability
        reliability = (
            trend_reliability * 0.4 +
            stability * 0.3 +
            pattern_score * 0.3
        )

        return reliability.fillna(50).clip(0, 100)


# ============================================================================
# RISK-ADJUSTED INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class RiskAdjustedMomentum(_MetadataDrivenIndicator):
    """
    Risk-Adjusted Momentum - Momentum adjusted for volatility (Sharpe-like)
    Output: Risk-adjusted return score
    """

    _metadata = IndicatorMetadata(
        name="RiskAdjustedMomentum",
        display_name="Risk-Adjusted Momentum",
        category="custom",
        description="Momentum adjusted for volatility risk",
        parameters={
            'period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'risk_free_rate': {'type': 'float', 'default': 0.02, 'min': 0.0, 'max': 0.1}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Risk-Adjusted Momentum"""
        period = params.get('period', 20)
        risk_free_rate = params.get('risk_free_rate', 0.02)

        close = data['close']

        # Calculate returns
        returns = close.pct_change()

        # Rolling mean return (annualized)
        mean_return = returns.rolling(window=period).mean() * 252

        # Rolling volatility (annualized)
        volatility = returns.rolling(window=period).std() * np.sqrt(252)

        # Risk-adjusted momentum (Sharpe ratio)
        risk_adj_momentum = (mean_return - risk_free_rate) / (volatility + 1e-10)

        # Scale to more readable range
        scaled = risk_adj_momentum * 20

        return scaled.fillna(0).clip(-100, 100)


@indicator_registry.register('custom')
class VolatilityAdjustedRSI(_MetadataDrivenIndicator):
    """
    Volatility-Adjusted RSI - RSI normalized by current volatility
    High volatility moves require larger price changes to register
    Output: Adjusted RSI 0-100
    """

    _metadata = IndicatorMetadata(
        name="VolatilityAdjustedRSI",
        display_name="Volatility-Adjusted RSI",
        category="custom",
        description="RSI adjusted for current volatility levels",
        parameters={
            'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'vol_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Volatility-Adjusted RSI"""
        rsi_period = params.get('rsi_period', 14)
        vol_period = params.get('vol_period', 20)

        close = data['close']

        # Calculate standard RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate volatility
        returns = close.pct_change()
        volatility = returns.rolling(window=vol_period).std()
        avg_volatility = volatility.rolling(window=100).mean()

        # Adjust RSI based on volatility
        vol_ratio = volatility / (avg_volatility + 1e-10)

        # In high vol, RSI extremes mean less (compress toward 50)
        # In low vol, RSI extremes mean more (expand from 50)
        adjusted_rsi = 50 + (rsi - 50) / vol_ratio

        return adjusted_rsi.fillna(50).clip(0, 100)


@indicator_registry.register('custom')
class ATRNormalizedPrice(_MetadataDrivenIndicator):
    """
    ATR-Normalized Price - Price movement normalized by ATR
    Shows how many ATRs the price has moved
    Output: Price deviation in ATR units
    """

    _metadata = IndicatorMetadata(
        name="ATRNormalizedPrice",
        display_name="ATR-Normalized Price",
        category="custom",
        description="Price movement normalized by Average True Range",
        parameters={
            'atr_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'ma_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 100}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate ATR-Normalized Price"""
        atr_period = params.get('atr_period', 14)
        ma_period = params.get('ma_period', 20)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=atr_period).mean()

        # Calculate price deviation from MA
        ma = close.rolling(window=ma_period).mean()
        price_deviation = close - ma

        # Normalize by ATR
        atr_normalized = price_deviation / (atr + 1e-10)

        return atr_normalized.fillna(0).clip(-10, 10)


# ============================================================================
# RELATIVE STRENGTH INDICATORS (3)
# ============================================================================

@indicator_registry.register('custom')
class SectorRelativeStrength(_MetadataDrivenIndicator):
    """
    Sector Relative Strength - Strength vs sector average
    Note: Requires sector data, here we simulate with market comparison
    Output: Relative strength score -100 to 100
    """

    _metadata = IndicatorMetadata(
        name="SectorRelativeStrength",
        display_name="Sector Relative Strength",
        category="custom",
        description="Relative strength vs sector/market benchmark",
        parameters={
            'period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Sector Relative Strength"""
        period = params.get('period', 20)
        close = data['close']

        # Calculate relative performance
        roc = (close / close.shift(period) - 1) * 100

        # Normalize to percentile rank (simulating vs sector)
        percentile = roc.rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Convert to -100 to 100 scale (50th percentile = 0)
        relative_strength = (percentile - 50) * 2

        return relative_strength.fillna(0).clip(-100, 100)


@indicator_registry.register('custom')
class StockVsMarket(_MetadataDrivenIndicator):
    """
    Stock vs Market - Performance relative to market
    Calculates beta-adjusted relative performance
    Output: Relative performance score
    """

    _metadata = IndicatorMetadata(
        name="StockVsMarket",
        display_name="Stock vs Market",
        category="custom",
        description="Stock performance relative to market benchmark",
        parameters={
            'period': {'type': 'int', 'default': 60, 'min': 20, 'max': 250}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Stock vs Market"""
        period = params.get('period', 60)
        close = data['close']

        # Calculate rolling returns
        returns = close.pct_change()
        rolling_return = (close / close.shift(period) - 1) * 100

        # Simulate market returns (using MA as proxy)
        market_proxy = close.rolling(window=period).mean()
        market_return = (market_proxy / market_proxy.shift(period) - 1) * 100

        # Relative performance
        relative_perf = rolling_return - market_return

        # Normalize
        std = relative_perf.rolling(window=period).std()
        normalized = relative_perf / (std + 1e-10)

        return normalized.fillna(0).clip(-10, 10)


@indicator_registry.register('custom')
class RelativeMomentumRank(_MetadataDrivenIndicator):
    """
    Relative Momentum Rank - Percentile rank of momentum
    Output: 0-100 percentile rank
    """

    _metadata = IndicatorMetadata(
        name="RelativeMomentumRank",
        display_name="Relative Momentum Rank",
        category="custom",
        description="Percentile rank of momentum over time",
        parameters={
            'momentum_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'rank_period': {'type': 'int', 'default': 100, 'min': 50, 'max': 500}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Relative Momentum Rank"""
        momentum_period = params.get('momentum_period', 14)
        rank_period = params.get('rank_period', 100)

        close = data['close']

        # Calculate momentum (ROC)
        momentum = (close / close.shift(momentum_period) - 1) * 100

        # Calculate percentile rank
        rank = momentum.rolling(window=rank_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        return rank.fillna(50)


# ============================================================================
# EXTREMES DETECTION (3)
# ============================================================================

@indicator_registry.register('custom')
class PriceExtremes(_MetadataDrivenIndicator):
    """
    Price Extremes - Detects price extremes using statistical methods
    Output: Positive = upper extreme, Negative = lower extreme, 0 = normal
    """

    _metadata = IndicatorMetadata(
        name="PriceExtremes",
        display_name="Price Extremes",
        category="custom",
        description="Detects statistical price extremes using Z-score and percentiles",
        parameters={
            'period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'std_threshold': {'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 4.0}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Price Extremes"""
        period = params.get('period', 20)
        std_threshold = params.get('std_threshold', 2.0)

        close = data['close']

        # Calculate Z-score
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        z_score = (close - ma) / (std + 1e-10)

        # Detect extremes
        extremes = pd.Series(0, index=data.index, dtype=float)
        extremes.loc[z_score > std_threshold] = (z_score - std_threshold) * 50
        extremes.loc[z_score < -std_threshold] = (z_score + std_threshold) * 50

        return extremes.clip(-100, 100)


@indicator_registry.register('custom')
class VolumeExtremes(_MetadataDrivenIndicator):
    """
    Volume Extremes - Detects extreme volume conditions
    Output: 0-100 (higher = more extreme volume)
    """

    _metadata = IndicatorMetadata(
        name="VolumeExtremes",
        display_name="Volume Extremes",
        category="custom",
        description="Detects extreme volume conditions using statistical analysis",
        parameters={
            'period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'threshold': {'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 5.0}
        },
        output_type='line',
        required_inputs=['volume']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Volume Extremes"""
        period = params.get('period', 20)
        threshold = params.get('threshold', 2.0)

        volume = data['volume']

        # Calculate volume Z-score
        vol_ma = volume.rolling(window=period).mean()
        vol_std = volume.rolling(window=period).std()
        vol_z_score = (volume - vol_ma) / (vol_std + 1e-10)

        # Convert to 0-100 scale
        extreme_score = pd.Series(0, index=data.index, dtype=float)
        extreme_score.loc[vol_z_score > threshold] = ((vol_z_score - threshold) * 20).clip(0, 100)

        return extreme_score.fillna(0)


@indicator_registry.register('custom')
class VolatilityExtremes(_MetadataDrivenIndicator):
    """
    Volatility Extremes - Detects extreme volatility conditions
    Output: 0-100 (higher = more extreme volatility)
    """

    _metadata = IndicatorMetadata(
        name="VolatilityExtremes",
        display_name="Volatility Extremes",
        category="custom",
        description="Detects extreme volatility conditions",
        parameters={
            'atr_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'lookback': {'type': 'int', 'default': 100, 'min': 50, 'max': 500}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Volatility Extremes"""
        atr_period = params.get('atr_period', 14)
        lookback = params.get('lookback', 100)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=atr_period).mean()

        # Calculate ATR percentile
        atr_percentile = atr.rolling(window=lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Extreme score (emphasize upper extremes)
        extreme_score = pd.Series(0, index=data.index, dtype=float)
        extreme_score.loc[atr_percentile > 75] = (atr_percentile - 75) * 4
        extreme_score.loc[atr_percentile > 90] = 100

        return extreme_score.fillna(0).clip(0, 100)


# ============================================================================
# COMPOSITE SCORES (4)
# ============================================================================

@indicator_registry.register('custom')
class TechnicalScore(_MetadataDrivenIndicator):
    """
    Technical Score - Comprehensive technical analysis score
    Combines trend, momentum, and support/resistance analysis
    Output: -100 to 100 (-100 = very bearish, 100 = very bullish)
    """

    _metadata = IndicatorMetadata(
        name="TechnicalScore",
        display_name="Technical Score",
        category="custom",
        description="Comprehensive technical analysis score combining multiple factors",
        parameters={
            'ma_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close', 'volume']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Technical Score"""
        ma_period = params.get('ma_period', 20)
        rsi_period = params.get('rsi_period', 14)

        high = data['high']
        low = data['low']
        close = data['close']

        # Trend Score (MA alignment)
        ma_20 = close.rolling(window=20).mean()
        ma_50 = close.rolling(window=50).mean()
        ma_100 = close.rolling(window=100).mean()

        trend_score = pd.Series(0, index=data.index, dtype=float)
        # All bullish alignment
        trend_score.loc[(close > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_100)] = 100
        # All bearish alignment
        trend_score.loc[(close < ma_20) & (ma_20 < ma_50) & (ma_50 < ma_100)] = -100
        # Mixed signals
        trend_score.loc[(close > ma_20) & (ma_20 > ma_50) & (ma_50 < ma_100)] = 50
        trend_score.loc[(close < ma_20) & (ma_20 < ma_50) & (ma_50 > ma_100)] = -50

        # Momentum Score (RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        momentum_score = (rsi - 50) * 2  # Convert 0-100 to -100 to 100

        # Support/Resistance Score
        rolling_high = high.rolling(window=ma_period).max()
        rolling_low = low.rolling(window=ma_period).min()
        position = (close - rolling_low) / (rolling_high - rolling_low + 1e-10) * 200 - 100

        # Composite Technical Score
        technical_score = (
            trend_score * 0.40 +
            momentum_score * 0.35 +
            position * 0.25
        )

        return technical_score.fillna(0).clip(-100, 100)


@indicator_registry.register('custom')
class MomentumScore(_MetadataDrivenIndicator):
    """
    Momentum Score - Multi-timeframe momentum composite
    Output: -100 to 100 (-100 = strong bearish momentum, 100 = strong bullish momentum)
    """

    _metadata = IndicatorMetadata(
        name="MomentumScore",
        display_name="Momentum Score",
        category="custom",
        description="Multi-timeframe momentum composite score",
        parameters={
            'short_period': {'type': 'int', 'default': 5, 'min': 3, 'max': 20},
            'medium_period': {'type': 'int', 'default': 14, 'min': 10, 'max': 50},
            'long_period': {'type': 'int', 'default': 28, 'min': 20, 'max': 100}
        },
        output_type='line',
        required_inputs=['close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Momentum Score"""
        short = params.get('short_period', 5)
        medium = params.get('medium_period', 14)
        long = params.get('long_period', 28)

        close = data['close']

        # Calculate ROC for multiple timeframes
        roc_short = (close / close.shift(short) - 1) * 100
        roc_medium = (close / close.shift(medium) - 1) * 100
        roc_long = (close / close.shift(long) - 1) * 100

        # Normalize each (assume 10% move = score of 100)
        def normalize_roc(roc, max_move=10):
            return (roc / max_move * 100).clip(-100, 100)

        short_score = normalize_roc(roc_short, 5)  # Short-term more volatile
        medium_score = normalize_roc(roc_medium, 10)
        long_score = normalize_roc(roc_long, 15)

        # Weighted composite
        momentum_score = (
            short_score * 0.25 +
            medium_score * 0.35 +
            long_score * 0.40
        )

        return momentum_score.fillna(0).clip(-100, 100)


@indicator_registry.register('custom')
class RiskScore(_MetadataDrivenIndicator):
    """
    Risk Score - Comprehensive risk assessment
    Combines volatility, drawdown, and trend instability
    Output: 0-100 (higher = higher risk)
    """

    _metadata = IndicatorMetadata(
        name="RiskScore",
        display_name="Risk Score",
        category="custom",
        description="Comprehensive risk assessment combining multiple risk factors",
        parameters={
            'atr_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'lookback': {'type': 'int', 'default': 50, 'min': 20, 'max': 200}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Risk Score"""
        atr_period = params.get('atr_period', 14)
        lookback = params.get('lookback', 50)

        high = data['high']
        low = data['low']
        close = data['close']

        # Volatility Risk (ATR-based)
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=atr_period).mean()
        atr_pct = atr / close * 100

        # Normalize ATR (5% = high risk)
        volatility_risk = (atr_pct / 5 * 100).clip(0, 100)

        # Drawdown Risk
        rolling_max = close.rolling(window=lookback, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max * 100
        drawdown_risk = abs(drawdown).clip(0, 100)

        # Trend Instability Risk (frequent direction changes)
        close_diff = close.diff()
        direction_changes = (close_diff.shift(1) * close_diff < 0).astype(int)
        change_rate = direction_changes.rolling(window=20).sum() / 20 * 100

        # Composite Risk Score
        risk_score = (
            volatility_risk * 0.40 +
            drawdown_risk * 0.35 +
            change_rate * 0.25
        )

        return risk_score.fillna(0).clip(0, 100)


@indicator_registry.register('custom')
class MarketConditionIndex(_MetadataDrivenIndicator):
    """
    Market Condition Index - Overall market condition assessment
    Combines trend, volatility, and momentum into single index
    Output: 1-5 (1=Very Bearish, 2=Bearish, 3=Neutral, 4=Bullish, 5=Very Bullish)
    """

    _metadata = IndicatorMetadata(
        name="MarketConditionIndex",
        display_name="Market Condition Index",
        category="custom",
        description="Overall market condition index from 1 (very bearish) to 5 (very bullish)",
        parameters={
            'ma_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'adx_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50}
        },
        output_type='line',
        required_inputs=['high', 'low', 'close']
    )

    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        """Calculate Market Condition Index"""
        ma_period = params.get('ma_period', 20)
        adx_period = params.get('adx_period', 14)
        rsi_period = params.get('rsi_period', 14)

        high = data['high']
        low = data['low']
        close = data['close']

        # Trend Component
        ma = close.rolling(window=ma_period).mean()
        trend_signal = (close > ma).astype(int) * 2 - 1  # 1 or -1

        # Strength Component (ADX)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=adx_period).mean()

        # Momentum Component (RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Composite Index
        # Strong trends with high ADX get more extreme ratings
        condition = pd.Series(3, index=data.index, dtype=float)  # Default: Neutral

        strong_trend = adx > 25

        # Very Bullish: Uptrend + Strong ADX + RSI > 50
        condition.loc[(trend_signal == 1) & strong_trend & (rsi > 60)] = 5
        # Bullish: Uptrend + (Strong ADX OR RSI > 50)
        condition.loc[(trend_signal == 1) & ((strong_trend) | (rsi > 50)) & (condition == 3)] = 4

        # Very Bearish: Downtrend + Strong ADX + RSI < 50
        condition.loc[(trend_signal == -1) & strong_trend & (rsi < 40)] = 1
        # Bearish: Downtrend + (Strong ADX OR RSI < 50)
        condition.loc[(trend_signal == -1) & ((strong_trend) | (rsi < 50)) & (condition == 3)] = 2

        return condition.fillna(3)
