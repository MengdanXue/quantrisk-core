"""
趋势跟踪策略模块

包含多种经典趋势跟踪策略：
- MovingAverageCrossStrategy: 双均线交叉策略
- TurtleTradingStrategy: 海龟交易法
- MACDStrategy: MACD策略
- TripleMAStrategy: 三均线策略
- DonchianChannelStrategy: 唐奇安通道策略
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class MovingAverageCrossStrategy(BaseStrategy):
    """
    双均线交叉策略

    策略逻辑：
    - 快线上穿慢线时买入（金叉）
    - 快线下穿慢线时卖出（死叉）
    - 可选择SMA或EMA

    适用场景：
    - 趋势市场
    - 中长期交易
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 10,        # 快线周期
            'slow_period': 30,        # 慢线周期
            'ma_type': 'ema',         # 均线类型：'sma' 或 'ema'
            'signal_period': 5,       # 信号确认周期
            'use_price_filter': True, # 使用价格过滤
            'position_pct': 0.1,
        }

    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """计算均线"""
        if self.params['ma_type'] == 'ema':
            return series.ewm(span=period, adjust=False).mean()
        else:
            return series.rolling(period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成均线交叉信号"""
        fast_ma = self._calculate_ma(data['close'], self.params['fast_period'])
        slow_ma = self._calculate_ma(data['close'], self.params['slow_period'])

        # 计算均线差值
        ma_diff = fast_ma - slow_ma
        ma_diff_prev = ma_diff.shift(1)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 金叉：快线从下方穿越慢线
        golden_cross = (ma_diff > 0) & (ma_diff_prev <= 0)

        # 死叉：快线从上方穿越慢线
        death_cross = (ma_diff < 0) & (ma_diff_prev >= 0)

        # 价格过滤：价格需在均线上方/下方
        if self.params['use_price_filter']:
            price_above = data['close'] > slow_ma
            price_below = data['close'] < slow_ma
            golden_cross = golden_cross & price_above
            death_cross = death_cross & price_below

        signals[golden_cross] = Signal.BUY.value
        signals[death_cross] = Signal.SELL.value

        # 强信号：均线差距持续扩大
        ma_diff_expanding = ma_diff.diff(self.params['signal_period']) > 0
        strong_buy = golden_cross & ma_diff_expanding
        signals[strong_buy] = Signal.STRONG_BUY.value

        ma_diff_contracting = ma_diff.diff(self.params['signal_period']) < 0
        strong_sell = death_cross & ma_diff_contracting
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class TurtleTradingStrategy(BaseStrategy):
    """
    海龟交易法

    策略逻辑：
    - 系统1（短期）：20日突破入场，10日突破出场
    - 系统2（长期）：55日突破入场，20日突破出场
    - 使用ATR进行仓位管理和止损

    适用场景：
    - 趋势明显的市场
    - 商品期货
    - 外汇

    经典量化策略，由Richard Dennis于1983年提出
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'system': 1,              # 使用系统1或系统2
            'entry_period': 20,       # 入场突破周期（系统1=20，系统2=55）
            'exit_period': 10,        # 出场突破周期（系统1=10，系统2=20）
            'atr_period': 20,         # ATR周期
            'atr_stop_multiplier': 2, # ATR止损倍数
            'unit_risk_pct': 0.01,    # 单位风险（账户的1%）
            'max_units': 4,           # 最大加仓次数
            'position_pct': 0.1,
        }

    def validate_params(self) -> None:
        if self.params['system'] == 2:
            self.params['entry_period'] = 55
            self.params['exit_period'] = 20

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """计算ATR"""
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.params['atr_period']).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成海龟信号"""
        entry_period = self.params['entry_period']
        exit_period = self.params['exit_period']

        # 计算通道
        entry_high = data['high'].rolling(entry_period).max().shift(1)
        entry_low = data['low'].rolling(entry_period).min().shift(1)
        exit_high = data['high'].rolling(exit_period).max().shift(1)
        exit_low = data['low'].rolling(exit_period).min().shift(1)

        # 计算ATR
        atr = self._calculate_atr(data)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 突破最高点买入
        breakout_up = data['close'] > entry_high
        signals[breakout_up] = Signal.BUY.value

        # 强势突破（突破幅度超过1个ATR）
        strong_breakout_up = breakout_up & ((data['close'] - entry_high) > atr)
        signals[strong_breakout_up] = Signal.STRONG_BUY.value

        # 跌破最低点卖出
        breakout_down = data['close'] < exit_low
        signals[breakout_down] = Signal.SELL.value

        # 强势下跌
        strong_breakout_down = breakout_down & ((exit_low - data['close']) > atr)
        signals[strong_breakout_down] = Signal.STRONG_SELL.value

        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD策略

    策略逻辑：
    - MACD线上穿信号线时买入
    - MACD线下穿信号线时卖出
    - 可选择使用零轴过滤

    适用场景：
    - 趋势确认
    - 动量交易
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 12,        # 快线EMA周期
            'slow_period': 26,        # 慢线EMA周期
            'signal_period': 9,       # 信号线周期
            'use_zero_line': True,    # 使用零轴过滤
            'histogram_confirm': True, # 使用柱状图确认
            'position_pct': 0.1,
        }

    def _calculate_macd(self, data: pd.DataFrame) -> tuple:
        """计算MACD"""
        fast_ema = data['close'].ewm(span=self.params['fast_period'], adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.params['slow_period'], adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params['signal_period'], adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成MACD信号"""
        macd_line, signal_line, histogram = self._calculate_macd(data)

        # 金叉死叉
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 基本信号
        buy_signal = macd_cross_up
        sell_signal = macd_cross_down

        # 零轴过滤
        if self.params['use_zero_line']:
            buy_signal = buy_signal & (macd_line > 0)
            sell_signal = sell_signal & (macd_line < 0)

        # 柱状图确认
        if self.params['histogram_confirm']:
            hist_increasing = histogram > histogram.shift(1)
            hist_decreasing = histogram < histogram.shift(1)
            buy_signal = buy_signal & hist_increasing
            sell_signal = sell_signal & hist_decreasing

        signals[buy_signal] = Signal.BUY.value
        signals[sell_signal] = Signal.SELL.value

        # 强信号：MACD在零轴附近穿越
        strong_buy = buy_signal & (macd_line.shift(1) < 0) & (macd_line > 0)
        strong_sell = sell_signal & (macd_line.shift(1) > 0) & (macd_line < 0)

        signals[strong_buy] = Signal.STRONG_BUY.value
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class TripleMAStrategy(BaseStrategy):
    """
    三均线策略

    策略逻辑：
    - 使用短、中、长三条均线
    - 短线在中线上方，中线在长线上方 -> 多头排列，买入
    - 短线在中线下方，中线在长线下方 -> 空头排列，卖出

    适用场景：
    - 强趋势确认
    - 过滤震荡市
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'short_period': 5,        # 短期均线
            'medium_period': 20,      # 中期均线
            'long_period': 60,        # 长期均线
            'ma_type': 'ema',
            'alignment_confirm': 2,   # 排列确认天数
            'position_pct': 0.1,
        }

    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.params['ma_type'] == 'ema':
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成三均线信号"""
        short_ma = self._calculate_ma(data['close'], self.params['short_period'])
        medium_ma = self._calculate_ma(data['close'], self.params['medium_period'])
        long_ma = self._calculate_ma(data['close'], self.params['long_period'])

        # 多头排列：短 > 中 > 长
        bullish_alignment = (short_ma > medium_ma) & (medium_ma > long_ma)

        # 空头排列：短 < 中 < 长
        bearish_alignment = (short_ma < medium_ma) & (medium_ma < long_ma)

        # 确认天数
        confirm = self.params['alignment_confirm']
        bullish_confirmed = bullish_alignment.rolling(confirm).sum() == confirm
        bearish_confirmed = bearish_alignment.rolling(confirm).sum() == confirm

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 多头排列开始
        bullish_start = bullish_confirmed & (~bullish_confirmed.shift(1).fillna(False))
        signals[bullish_start] = Signal.BUY.value

        # 强势多头：价格也在所有均线上方
        price_above_all = data['close'] > long_ma
        strong_bullish = bullish_start & price_above_all
        signals[strong_bullish] = Signal.STRONG_BUY.value

        # 空头排列开始
        bearish_start = bearish_confirmed & (~bearish_confirmed.shift(1).fillna(False))
        signals[bearish_start] = Signal.SELL.value

        # 强势空头
        price_below_all = data['close'] < long_ma
        strong_bearish = bearish_start & price_below_all
        signals[strong_bearish] = Signal.STRONG_SELL.value

        return signals


class DonchianChannelStrategy(BaseStrategy):
    """
    唐奇安通道策略

    策略逻辑：
    - 价格突破N日最高价时买入
    - 价格跌破N日最低价时卖出
    - 可选择不同的入场和出场周期

    适用场景：
    - 趋势突破
    - 商品期货

    这是海龟交易法的核心组件
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'entry_period': 20,       # 入场通道周期
            'exit_period': 10,        # 出场通道周期
            'use_middle': False,      # 使用中轨
            'atr_filter': True,       # ATR过滤
            'atr_period': 14,
            'min_atr_multiple': 0.5,  # 最小突破幅度（ATR倍数）
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成唐奇安通道信号"""
        entry_period = self.params['entry_period']
        exit_period = self.params['exit_period']

        # 计算通道
        upper = data['high'].rolling(entry_period).max().shift(1)
        lower = data['low'].rolling(entry_period).min().shift(1)
        middle = (upper + lower) / 2

        exit_upper = data['high'].rolling(exit_period).max().shift(1)
        exit_lower = data['low'].rolling(exit_period).min().shift(1)

        # ATR过滤
        if self.params['atr_filter']:
            tr = pd.concat([
                data['high'] - data['low'],
                (data['high'] - data['close'].shift(1)).abs(),
                (data['low'] - data['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.params['atr_period']).mean()
            min_breakout = atr * self.params['min_atr_multiple']
        else:
            min_breakout = 0

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 突破上轨买入
        breakout_up = (data['close'] > upper) & ((data['close'] - upper) > min_breakout)
        signals[breakout_up] = Signal.BUY.value

        # 强势突破
        strong_breakout_up = breakout_up & (data['close'] > upper * 1.02)
        signals[strong_breakout_up] = Signal.STRONG_BUY.value

        # 跌破下轨卖出
        breakout_down = (data['close'] < exit_lower) & ((exit_lower - data['close']) > min_breakout)
        signals[breakout_down] = Signal.SELL.value

        # 强势下跌
        strong_breakout_down = breakout_down & (data['close'] < exit_lower * 0.98)
        signals[strong_breakout_down] = Signal.STRONG_SELL.value

        return signals


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend策略

    策略逻辑：
    - 基于ATR计算动态止损线
    - 价格在SuperTrend上方时做多
    - 价格在SuperTrend下方时做空

    适用场景：
    - 趋势跟踪
    - 动态止损
    """

    category = "trend_following"

    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 10,         # ATR周期
            'multiplier': 3.0,        # ATR倍数
            'position_pct': 0.1,
        }

    def _calculate_supertrend(self, data: pd.DataFrame) -> tuple:
        """计算SuperTrend"""
        atr_period = self.params['atr_period']
        multiplier = self.params['multiplier']

        # 计算ATR
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # 计算基本上下轨
        hl2 = (data['high'] + data['low']) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        # 计算最终上下轨
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        for i in range(1, len(data)):
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or data['close'].iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]

            if basic_lower.iloc[i] > final_lower.iloc[i-1] or data['close'].iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]

        # 确定趋势方向
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(1, index=data.index)  # 1=上涨, -1=下跌

        for i in range(1, len(data)):
            if data['close'].iloc[i] > final_upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif data['close'].iloc[i] < final_lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]

        return supertrend, direction

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成SuperTrend信号"""
        supertrend, direction = self._calculate_supertrend(data)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 趋势转向上
        trend_up = (direction == 1) & (direction.shift(1) == -1)
        signals[trend_up] = Signal.BUY.value

        # 趋势转向下
        trend_down = (direction == -1) & (direction.shift(1) == 1)
        signals[trend_down] = Signal.SELL.value

        # 强信号：趋势持续
        strong_up = trend_up & (data['close'] > supertrend * 1.02)
        strong_down = trend_down & (data['close'] < supertrend * 0.98)

        signals[strong_up] = Signal.STRONG_BUY.value
        signals[strong_down] = Signal.STRONG_SELL.value

        return signals
