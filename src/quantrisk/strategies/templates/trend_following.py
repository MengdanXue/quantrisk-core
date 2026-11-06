"""
趋势跟踪策略模板
Trend Following Strategy Templates

包含经典趋势跟踪策略：
- 双均线策略
- 海龟交易策略
- 唐奇安通道突破

作者: TraderX Team
创建: 2025-12-23
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base import (
    BaseStrategy, StrategyContext, Signal, SignalType,
    crossover, crossunder, sma, ema, atr, highest, lowest
)


class DualMAStrategy(BaseStrategy):
    """
    双均线策略

    经典的趋势跟踪策略，使用快慢均线交叉产生信号

    参数:
        fast_period: 快线周期（默认10）
        slow_period: 慢线周期（默认30）
        ma_type: 均线类型 'sma' 或 'ema'（默认'ema'）
        use_atr_stop: 是否使用ATR止损（默认True）
        atr_multiplier: ATR止损倍数（默认2.0）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'ema',
            'use_atr_stop': True,
            'atr_multiplier': 2.0,
            'atr_period': 14,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        return max(self.params['slow_period'], self.params['atr_period']) + 10

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        close = history['close']

        # 计算均线
        ma_func = ema if self.params['ma_type'] == 'ema' else sma
        fast_ma = ma_func(close, self.params['fast_period'])
        slow_ma = ma_func(close, self.params['slow_period'])

        # 计算ATR用于止损
        stop_loss = None
        if self.params['use_atr_stop']:
            atr_value = atr(
                history['high'], history['low'], close,
                self.params['atr_period']
            ).iloc[-1]
            stop_loss = close.iloc[-1] - self.params['atr_multiplier'] * atr_value

        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 生成信号
        if crossover(fast_ma, slow_ma):
            # 金叉做多
            if context.is_short(symbol):
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_SHORT,
                    strength=1.0,
                    metadata={'reason': 'ma_crossover'}
                ))

            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=1.0,
                stop_loss=stop_loss,
                metadata={'fast_ma': fast_ma.iloc[-1], 'slow_ma': slow_ma.iloc[-1]}
            ))

        elif crossunder(fast_ma, slow_ma):
            # 死叉做空
            if context.is_long(symbol):
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    strength=1.0,
                    metadata={'reason': 'ma_crossunder'}
                ))

            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strength=1.0,
                stop_loss=close.iloc[-1] + self.params['atr_multiplier'] * atr_value if stop_loss else None,
                metadata={'fast_ma': fast_ma.iloc[-1], 'slow_ma': slow_ma.iloc[-1]}
            ))

        return signals


class TurtleStrategy(BaseStrategy):
    """
    海龟交易策略

    传奇的趋势跟踪策略，使用唐奇安通道突破

    参数:
        entry_period: 入场突破周期（默认20）
        exit_period: 出场突破周期（默认10）
        atr_period: ATR周期（默认20）
        risk_per_trade: 单笔风险比例（默认0.01即1%）
        max_units: 最大加仓单位（默认4）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'entry_period': 20,
            'exit_period': 10,
            'atr_period': 20,
            'risk_per_trade': 0.01,
            'max_units': 4,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        return max(self.params['entry_period'], self.params['atr_period']) + 5

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        high = history['high']
        low = history['low']
        close = history['close']

        # 计算唐奇安通道
        entry_high = highest(high.iloc[:-1], self.params['entry_period'])
        entry_low = lowest(low.iloc[:-1], self.params['entry_period'])
        exit_high = highest(high.iloc[:-1], self.params['exit_period'])
        exit_low = lowest(low.iloc[:-1], self.params['exit_period'])

        # 计算ATR和N值
        atr_value = atr(high, low, close, self.params['atr_period']).iloc[-1]

        current_price = close.iloc[-1]
        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 获取当前持仓单位
        current_units = self.get_state(f'{symbol}_units', 0)
        last_entry_price = self.get_state(f'{symbol}_last_entry', 0)

        # 入场信号
        if not context.has_position(symbol):
            if current_price > entry_high:
                # 突破做多
                stop_loss = current_price - 2 * atr_value
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=1.0,
                    stop_loss=stop_loss,
                    metadata={
                        'entry_type': 'breakout_high',
                        'entry_high': entry_high,
                        'atr': atr_value,
                        'unit': 1
                    }
                ))
                self.set_state(f'{symbol}_units', 1)
                self.set_state(f'{symbol}_last_entry', current_price)

            elif current_price < entry_low:
                # 突破做空
                stop_loss = current_price + 2 * atr_value
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=1.0,
                    stop_loss=stop_loss,
                    metadata={
                        'entry_type': 'breakout_low',
                        'entry_low': entry_low,
                        'atr': atr_value,
                        'unit': 1
                    }
                ))
                self.set_state(f'{symbol}_units', 1)
                self.set_state(f'{symbol}_last_entry', current_price)

        # 加仓信号（金字塔加仓）
        elif current_units < self.params['max_units']:
            if context.is_long(symbol):
                # 多头加仓条件：价格上涨0.5N
                if current_price >= last_entry_price + 0.5 * atr_value:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=0.5,  # 减少仓位
                        stop_loss=current_price - 2 * atr_value,
                        metadata={'entry_type': 'pyramid', 'unit': current_units + 1}
                    ))
                    self.set_state(f'{symbol}_units', current_units + 1)
                    self.set_state(f'{symbol}_last_entry', current_price)

            elif context.is_short(symbol):
                # 空头加仓条件：价格下跌0.5N
                if current_price <= last_entry_price - 0.5 * atr_value:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        strength=0.5,
                        stop_loss=current_price + 2 * atr_value,
                        metadata={'entry_type': 'pyramid', 'unit': current_units + 1}
                    ))
                    self.set_state(f'{symbol}_units', current_units + 1)
                    self.set_state(f'{symbol}_last_entry', current_price)

        # 出场信号
        if context.is_long(symbol) and current_price < exit_low:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={'exit_type': 'donchian_exit', 'exit_low': exit_low}
            ))
            self.set_state(f'{symbol}_units', 0)

        elif context.is_short(symbol) and current_price > exit_high:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_SHORT,
                strength=1.0,
                metadata={'exit_type': 'donchian_exit', 'exit_high': exit_high}
            ))
            self.set_state(f'{symbol}_units', 0)

        return signals

    def calculate_position_size(
        self,
        context: StrategyContext,
        symbol: str,
        signal_strength: float = 1.0
    ) -> float:
        """
        海龟仓位计算

        Unit = (账户 * 风险比例) / (N * 合约乘数)
        """
        history = context.history
        if history is None or len(history) < self.params['atr_period']:
            return super().calculate_position_size(context, symbol, signal_strength)

        # 计算N值
        n_value = atr(
            history['high'], history['low'], history['close'],
            self.params['atr_period']
        ).iloc[-1]

        if n_value == 0:
            return 0

        # 单位价值 = 1% 账户 / N
        dollar_volatility = n_value  # 假设合约乘数为1
        position_size = (context.equity * self.params['risk_per_trade']) / dollar_volatility

        return position_size * signal_strength


class DonchianBreakoutStrategy(BaseStrategy):
    """
    唐奇安通道突破策略

    简化版的通道突破策略

    参数:
        channel_period: 通道周期（默认20）
        atr_period: ATR周期（默认14）
        atr_stop_multiplier: ATR止损倍数（默认2.0）
        use_filter: 是否使用趋势过滤（默认True）
        filter_period: 过滤周期（默认50）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'channel_period': 20,
            'atr_period': 14,
            'atr_stop_multiplier': 2.0,
            'use_filter': True,
            'filter_period': 50,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        periods = [self.params['channel_period'], self.params['atr_period']]
        if self.params['use_filter']:
            periods.append(self.params['filter_period'])
        return max(periods) + 5

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        high = history['high']
        low = history['low']
        close = history['close']

        # 唐奇安通道（不包括当前bar）
        channel_high = highest(high.iloc[:-1], self.params['channel_period'])
        channel_low = lowest(low.iloc[:-1], self.params['channel_period'])

        # ATR
        atr_value = atr(high, low, close, self.params['atr_period']).iloc[-1]

        current_price = close.iloc[-1]
        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 趋势过滤
        trend_filter = True
        if self.params['use_filter']:
            ma_filter = sma(close, self.params['filter_period']).iloc[-1]
            trend_filter = current_price > ma_filter  # 只做多头趋势

        # 入场信号
        if not context.has_position(symbol):
            if current_price > channel_high and trend_filter:
                # 突破做多
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=1.0,
                    stop_loss=current_price - self.params['atr_stop_multiplier'] * atr_value,
                    take_profit=current_price + 3 * self.params['atr_stop_multiplier'] * atr_value,
                    metadata={
                        'channel_high': channel_high,
                        'atr': atr_value
                    }
                ))

            elif current_price < channel_low and not trend_filter:
                # 突破做空（反趋势）
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=1.0,
                    stop_loss=current_price + self.params['atr_stop_multiplier'] * atr_value,
                    take_profit=current_price - 3 * self.params['atr_stop_multiplier'] * atr_value,
                    metadata={
                        'channel_low': channel_low,
                        'atr': atr_value
                    }
                ))

        # 出场信号（通道中轨）
        channel_mid = (channel_high + channel_low) / 2

        if context.is_long(symbol) and current_price < channel_mid:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={'exit_type': 'channel_mid'}
            ))

        elif context.is_short(symbol) and current_price > channel_mid:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_SHORT,
                strength=1.0,
                metadata={'exit_type': 'channel_mid'}
            ))

        return signals


# 注册策略
TREND_FOLLOWING_STRATEGIES = {
    'dual_ma': DualMAStrategy,
    'turtle': TurtleStrategy,
    'donchian_breakout': DonchianBreakoutStrategy,
}


def get_trend_strategy(name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """
    获取趋势跟踪策略

    Args:
        name: 策略名称
        params: 策略参数

    Returns:
        策略实例
    """
    if name not in TREND_FOLLOWING_STRATEGIES:
        raise ValueError(f"Unknown trend strategy: {name}. Available: {list(TREND_FOLLOWING_STRATEGIES.keys())}")

    return TREND_FOLLOWING_STRATEGIES[name](params)
