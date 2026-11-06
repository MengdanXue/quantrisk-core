"""
均值回归策略模板
Mean Reversion Strategy Templates

包含经典均值回归策略：
- 布林带回归
- RSI均值回归

作者: TraderX Team
创建: 2025-12-23
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base import (
    BaseStrategy, StrategyContext, Signal, SignalType,
    sma, rsi, bollinger_bands, atr
)


class BollingerRevertStrategy(BaseStrategy):
    """
    布林带均值回归策略

    当价格触及布林带边界时入场，回归中轨时出场

    参数:
        period: 布林带周期（默认20）
        std_dev: 标准差倍数（默认2.0）
        entry_std: 入场标准差（默认2.0）
        exit_std: 出场标准差（默认0.5）
        use_rsi_filter: 是否使用RSI过滤（默认True）
        rsi_period: RSI周期（默认14）
        rsi_oversold: RSI超卖阈值（默认30）
        rsi_overbought: RSI超买阈值（默认70）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'entry_std': 2.0,
            'exit_std': 0.5,
            'use_rsi_filter': True,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        return max(self.params['period'], self.params['rsi_period']) + 5

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        close = history['close']

        # 计算布林带
        upper, middle, lower = bollinger_bands(
            close,
            self.params['period'],
            self.params['std_dev']
        )

        # 计算入场和出场带
        std = close.rolling(window=self.params['period']).std()
        entry_upper = middle + self.params['entry_std'] * std
        entry_lower = middle - self.params['entry_std'] * std
        exit_upper = middle + self.params['exit_std'] * std
        exit_lower = middle - self.params['exit_std'] * std

        current_price = close.iloc[-1]
        current_middle = middle.iloc[-1]
        current_entry_upper = entry_upper.iloc[-1]
        current_entry_lower = entry_lower.iloc[-1]

        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # RSI过滤
        rsi_filter_long = True
        rsi_filter_short = True

        if self.params['use_rsi_filter']:
            rsi_value = rsi(close, self.params['rsi_period']).iloc[-1]
            rsi_filter_long = rsi_value < self.params['rsi_oversold']
            rsi_filter_short = rsi_value > self.params['rsi_overbought']

        # 入场信号
        if not context.has_position(symbol):
            if current_price < current_entry_lower and rsi_filter_long:
                # 价格触及下轨，做多
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, (current_entry_lower - current_price) / std.iloc[-1]),
                    stop_loss=current_price - 1.5 * std.iloc[-1],
                    take_profit=current_middle,
                    metadata={
                        'entry_type': 'lower_band',
                        'distance_to_mean': current_middle - current_price
                    }
                ))

            elif current_price > current_entry_upper and rsi_filter_short:
                # 价格触及上轨，做空
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, (current_price - current_entry_upper) / std.iloc[-1]),
                    stop_loss=current_price + 1.5 * std.iloc[-1],
                    take_profit=current_middle,
                    metadata={
                        'entry_type': 'upper_band',
                        'distance_to_mean': current_price - current_middle
                    }
                ))

        # 出场信号
        else:
            if context.is_long(symbol):
                # 价格回归中轨附近
                if current_price >= exit_lower.iloc[-1]:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        strength=1.0,
                        metadata={'exit_type': 'mean_reversion'}
                    ))

            elif context.is_short(symbol):
                if current_price <= exit_upper.iloc[-1]:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_SHORT,
                        strength=1.0,
                        metadata={'exit_type': 'mean_reversion'}
                    ))

        return signals


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI均值回归策略

    利用RSI超买超卖进行均值回归交易

    参数:
        rsi_period: RSI周期（默认14）
        oversold_threshold: 超卖阈值（默认30）
        overbought_threshold: 超买阈值（默认70）
        exit_threshold: 出场阈值（默认50）
        use_divergence: 是否使用背离确认（默认True）
        atr_period: ATR周期（默认14）
        atr_stop_multiplier: ATR止损倍数（默认1.5）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'exit_threshold': 50,
            'use_divergence': True,
            'atr_period': 14,
            'atr_stop_multiplier': 1.5,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        return max(self.params['rsi_period'], self.params['atr_period']) + 20

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        close = history['close']
        high = history['high']
        low = history['low']

        # 计算RSI
        rsi_values = rsi(close, self.params['rsi_period'])
        current_rsi = rsi_values.iloc[-1]

        # 计算ATR
        atr_value = atr(high, low, close, self.params['atr_period']).iloc[-1]

        current_price = close.iloc[-1]
        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 背离检测
        bullish_divergence = False
        bearish_divergence = False

        if self.params['use_divergence']:
            bullish_divergence = self._detect_bullish_divergence(close, rsi_values)
            bearish_divergence = self._detect_bearish_divergence(close, rsi_values)

        # 入场信号
        if not context.has_position(symbol):
            # RSI超卖 + 可能的底背离
            if current_rsi < self.params['oversold_threshold']:
                strength = (self.params['oversold_threshold'] - current_rsi) / self.params['oversold_threshold']
                if bullish_divergence:
                    strength = min(1.0, strength * 1.5)  # 背离增强信号

                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    stop_loss=current_price - self.params['atr_stop_multiplier'] * atr_value,
                    metadata={
                        'rsi': current_rsi,
                        'bullish_divergence': bullish_divergence
                    }
                ))

            # RSI超买 + 可能的顶背离
            elif current_rsi > self.params['overbought_threshold']:
                strength = (current_rsi - self.params['overbought_threshold']) / (100 - self.params['overbought_threshold'])
                if bearish_divergence:
                    strength = min(1.0, strength * 1.5)

                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=strength,
                    stop_loss=current_price + self.params['atr_stop_multiplier'] * atr_value,
                    metadata={
                        'rsi': current_rsi,
                        'bearish_divergence': bearish_divergence
                    }
                ))

        # 出场信号
        else:
            if context.is_long(symbol):
                # RSI回到中性区域
                if current_rsi >= self.params['exit_threshold']:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        strength=1.0,
                        metadata={'exit_rsi': current_rsi}
                    ))

            elif context.is_short(symbol):
                if current_rsi <= self.params['exit_threshold']:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_SHORT,
                        strength=1.0,
                        metadata={'exit_rsi': current_rsi}
                    ))

        return signals

    def _detect_bullish_divergence(
        self,
        price: pd.Series,
        rsi_values: pd.Series,
        lookback: int = 10
    ) -> bool:
        """
        检测RSI底背离

        价格创新低，但RSI没有创新低
        """
        if len(price) < lookback + 5:
            return False

        # 最近的价格低点
        recent_price = price.iloc[-lookback:]
        recent_rsi = rsi_values.iloc[-lookback:]

        price_low_idx = recent_price.idxmin()
        rsi_at_price_low = rsi_values.loc[price_low_idx]

        # 当前是否接近新低
        is_near_low = price.iloc[-1] <= recent_price.min() * 1.01

        # RSI是否高于之前低点时的RSI
        is_rsi_higher = rsi_values.iloc[-1] > rsi_at_price_low

        return is_near_low and is_rsi_higher

    def _detect_bearish_divergence(
        self,
        price: pd.Series,
        rsi_values: pd.Series,
        lookback: int = 10
    ) -> bool:
        """
        检测RSI顶背离

        价格创新高，但RSI没有创新高
        """
        if len(price) < lookback + 5:
            return False

        recent_price = price.iloc[-lookback:]
        recent_rsi = rsi_values.iloc[-lookback:]

        price_high_idx = recent_price.idxmax()
        rsi_at_price_high = rsi_values.loc[price_high_idx]

        is_near_high = price.iloc[-1] >= recent_price.max() * 0.99
        is_rsi_lower = rsi_values.iloc[-1] < rsi_at_price_high

        return is_near_high and is_rsi_lower


# 注册策略
MEAN_REVERSION_STRATEGIES = {
    'bollinger_revert': BollingerRevertStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy,
}


def get_mean_reversion_strategy(
    name: str,
    params: Optional[Dict[str, Any]] = None
) -> BaseStrategy:
    """获取均值回归策略"""
    if name not in MEAN_REVERSION_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    return MEAN_REVERSION_STRATEGIES[name](params)
