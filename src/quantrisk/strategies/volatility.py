"""
波动率策略模块

包含多种基于波动率的交易策略：
- VolatilityBreakoutStrategy: 波动率突破策略
- ATRChannelStrategy: ATR通道策略
- VolatilityContractionStrategy: 波动率收缩策略
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    波动率突破策略

    策略逻辑：
    - 使用前一日的波动率（范围）设定今日突破阈值
    - 价格突破开盘价+K*前日波动率时买入
    - 价格跌破开盘价-K*前日波动率时卖出

    适用场景：
    - 日内交易
    - 突破行情

    经典策略，由Larry Williams提出
    """

    category = "volatility"

    def default_params(self) -> Dict[str, Any]:
        return {
            'k_multiplier': 0.5,      # 突破系数K
            'use_atr': True,          # 使用ATR而非简单范围
            'atr_period': 14,         # ATR周期
            'exit_at_close': True,    # 收盘平仓
            'stop_loss_k': 1.0,       # 止损K值
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成波动率突破信号"""
        k = self.params['k_multiplier']

        # 计算波动率
        if self.params['use_atr']:
            tr = pd.concat([
                data['high'] - data['low'],
                (data['high'] - data['close'].shift(1)).abs(),
                (data['low'] - data['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            volatility = tr.rolling(self.params['atr_period']).mean()
        else:
            volatility = (data['high'] - data['low']).shift(1)

        # 计算突破阈值（使用前一日收盘价作为基准）
        base_price = data['close'].shift(1)
        upper_threshold = base_price + k * volatility
        lower_threshold = base_price - k * volatility

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 向上突破
        breakout_up = data['high'] > upper_threshold
        signals[breakout_up] = Signal.BUY.value

        # 强势突破（突破幅度超过2倍K）
        strong_breakout_up = data['high'] > (base_price + 2 * k * volatility)
        signals[strong_breakout_up] = Signal.STRONG_BUY.value

        # 向下突破
        breakout_down = data['low'] < lower_threshold
        signals[breakout_down] = Signal.SELL.value

        # 强势下跌
        strong_breakout_down = data['low'] < (base_price - 2 * k * volatility)
        signals[strong_breakout_down] = Signal.STRONG_SELL.value

        return signals


class ATRChannelStrategy(BaseStrategy):
    """
    ATR通道策略

    策略逻辑：
    - 使用ATR构建动态通道
    - 价格突破通道上轨买入
    - 价格跌破通道下轨卖出
    - 通道基于移动平均线

    适用场景：
    - 趋势跟踪
    - 动态止损
    """

    category = "volatility"

    def default_params(self) -> Dict[str, Any]:
        return {
            'ma_period': 20,          # 均线周期
            'atr_period': 14,         # ATR周期
            'atr_multiplier': 2.0,    # ATR倍数
            'ma_type': 'ema',         # 均线类型
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成ATR通道信号"""
        # 计算均线
        if self.params['ma_type'] == 'ema':
            ma = data['close'].ewm(span=self.params['ma_period'], adjust=False).mean()
        else:
            ma = data['close'].rolling(self.params['ma_period']).mean()

        # 计算ATR
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.params['atr_period']).mean()

        # 计算通道
        multiplier = self.params['atr_multiplier']
        upper_channel = ma + multiplier * atr
        lower_channel = ma - multiplier * atr

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 突破上轨
        breakout_up = (data['close'] > upper_channel) & (data['close'].shift(1) <= upper_channel.shift(1))
        signals[breakout_up] = Signal.BUY.value

        # 跌破下轨
        breakout_down = (data['close'] < lower_channel) & (data['close'].shift(1) >= lower_channel.shift(1))
        signals[breakout_down] = Signal.SELL.value

        # 强信号：持续在通道外
        strong_buy = (data['close'] > upper_channel) & (data['close'].shift(1) > upper_channel.shift(1))
        strong_sell = (data['close'] < lower_channel) & (data['close'].shift(1) < lower_channel.shift(1))

        signals[strong_buy] = Signal.STRONG_BUY.value
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class VolatilityContractionStrategy(BaseStrategy):
    """
    波动率收缩策略

    策略逻辑：
    - 监测波动率收缩（布林带收窄）
    - 波动率收缩后等待方向性突破
    - 向上突破买入，向下突破卖出

    适用场景：
    - 盘整后的突破
    - 大行情前的布局

    基于波动率周期性原理
    """

    category = "volatility"

    def default_params(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,          # 布林带周期
            'bb_std': 2.0,            # 布林带标准差
            'squeeze_threshold': 0.05, # 收缩阈值（带宽百分比）
            'squeeze_duration': 5,    # 收缩持续天数
            'breakout_confirm': 2,    # 突破确认天数
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成波动率收缩信号"""
        period = self.params['bb_period']
        std_dev = self.params['bb_std']

        # 计算布林带
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std

        # 计算带宽百分比
        bandwidth = (upper - lower) / sma

        # 检测收缩
        squeeze_threshold = self.params['squeeze_threshold']
        squeeze = bandwidth < squeeze_threshold

        # 收缩持续
        squeeze_duration = self.params['squeeze_duration']
        sustained_squeeze = squeeze.rolling(squeeze_duration).sum() >= squeeze_duration

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 收缩后向上突破
        breakout_up = sustained_squeeze.shift(1) & (data['close'] > upper)
        signals[breakout_up] = Signal.BUY.value

        # 强势突破（收缩越久，突破越有效）
        long_squeeze = squeeze.rolling(squeeze_duration * 2).sum() >= squeeze_duration * 2
        strong_breakout_up = long_squeeze.shift(1) & (data['close'] > upper)
        signals[strong_breakout_up] = Signal.STRONG_BUY.value

        # 收缩后向下突破
        breakout_down = sustained_squeeze.shift(1) & (data['close'] < lower)
        signals[breakout_down] = Signal.SELL.value

        # 强势下跌
        strong_breakout_down = long_squeeze.shift(1) & (data['close'] < lower)
        signals[strong_breakout_down] = Signal.STRONG_SELL.value

        return signals


class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner通道策略

    策略逻辑：
    - 使用EMA作为中轨
    - 使用ATR构建上下轨
    - 突破通道时入场

    适用场景：
    - 趋势跟踪
    - 与布林带配合使用
    """

    category = "volatility"

    def default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 20,         # EMA周期
            'atr_period': 10,         # ATR周期
            'atr_multiplier': 2.0,    # ATR倍数
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成Keltner通道信号"""
        # 计算EMA
        ema = data['close'].ewm(span=self.params['ema_period'], adjust=False).mean()

        # 计算ATR
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.params['atr_period']).mean()

        # 计算通道
        multiplier = self.params['atr_multiplier']
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 突破上轨买入
        signals[data['close'] > upper] = Signal.BUY.value

        # 跌破下轨卖出
        signals[data['close'] < lower] = Signal.SELL.value

        # 强信号
        signals[data['close'] > upper * 1.02] = Signal.STRONG_BUY.value
        signals[data['close'] < lower * 0.98] = Signal.STRONG_SELL.value

        return signals


class HistoricalVolatilityStrategy(BaseStrategy):
    """
    历史波动率策略

    策略逻辑：
    - 监测历史波动率的相对水平
    - 波动率处于历史低位时买入波动率（买入期权或波动率产品）
    - 波动率处于历史高位时卖出波动率

    适用场景：
    - VIX相关产品
    - 期权策略
    """

    category = "volatility"

    def default_params(self) -> Dict[str, Any]:
        return {
            'vol_period': 20,         # 波动率计算周期
            'lookback': 252,          # 历史回看周期
            'low_percentile': 20,     # 低波动率分位数
            'high_percentile': 80,    # 高波动率分位数
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成历史波动率信号"""
        vol_period = self.params['vol_period']
        lookback = self.params['lookback']

        # 计算历史波动率
        returns = data['close'].pct_change()
        hist_vol = returns.rolling(vol_period).std() * np.sqrt(252)

        # 计算波动率分位数
        vol_percentile = hist_vol.rolling(lookback).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
            if x.max() != x.min() else 50
        )

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        low_pct = self.params['low_percentile']
        high_pct = self.params['high_percentile']

        # 波动率低位 -> 买入（预期波动率上升）
        signals[vol_percentile < low_pct] = Signal.BUY.value
        signals[vol_percentile < low_pct / 2] = Signal.STRONG_BUY.value

        # 波动率高位 -> 卖出（预期波动率下降）
        signals[vol_percentile > high_pct] = Signal.SELL.value
        signals[vol_percentile > (100 + high_pct) / 2] = Signal.STRONG_SELL.value

        return signals
