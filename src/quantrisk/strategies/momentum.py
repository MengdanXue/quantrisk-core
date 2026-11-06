"""
动量策略模块

包含多种基于动量效应的交易策略：
- MomentumStrategy: 基础价格动量策略
- RelativeStrengthStrategy: 相对强度策略 (RSI)
- BreakoutStrategy: 突破策略
- DualMomentumStrategy: 双动量策略 (绝对动量 + 相对动量)
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    基础价格动量策略

    策略逻辑：
    - 当过去N天的收益率超过阈值时买入
    - 当过去N天的收益率低于负阈值时卖出
    - 使用动量的持续性来捕捉趋势

    适用场景：
    - 趋势明显的市场
    - 中长期持仓
    """

    category = "momentum"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 20,           # 回看周期（天）
            'threshold': 0.05,        # 买入阈值（5%涨幅）
            'exit_threshold': -0.03,  # 卖出阈值（-3%跌幅）
            'holding_period': 5,      # 最小持仓周期
            'position_pct': 0.1,      # 仓位比例
            'use_volume': False,      # 是否使用成交量确认
            'volume_ma': 20,          # 成交量均线周期
        }

    def validate_params(self) -> None:
        if self.params['lookback'] < 1:
            raise ValueError("lookback must be >= 1")
        if self.params['threshold'] <= 0:
            raise ValueError("threshold must be > 0")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成动量信号"""
        lookback = self.params['lookback']
        threshold = self.params['threshold']
        exit_threshold = self.params['exit_threshold']

        # 计算N天收益率
        returns = data['close'].pct_change(lookback)

        # 计算动量得分
        momentum_score = returns / returns.rolling(lookback * 2).std()

        # 初始化信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 生成买入信号
        buy_condition = returns > threshold
        if self.params['use_volume'] and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.params['volume_ma']).mean()
            buy_condition = buy_condition & (data['volume'] > volume_ma)

        signals[buy_condition] = Signal.BUY.value

        # 强烈买入（动量特别强）
        strong_buy = buy_condition & (momentum_score > 2)
        signals[strong_buy] = Signal.STRONG_BUY.value

        # 生成卖出信号
        sell_condition = returns < exit_threshold
        signals[sell_condition] = Signal.SELL.value

        # 强烈卖出
        strong_sell = sell_condition & (momentum_score < -2)
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class RelativeStrengthStrategy(BaseStrategy):
    """
    相对强度策略 (RSI)

    策略逻辑：
    - RSI超卖（<30）时买入
    - RSI超买（>70）时卖出
    - 可配置RSI阈值和周期

    适用场景：
    - 震荡市场
    - 短期交易
    """

    category = "momentum"

    def default_params(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,         # RSI周期
            'oversold': 30,           # 超卖阈值
            'overbought': 70,         # 超买阈值
            'extreme_oversold': 20,   # 极度超卖
            'extreme_overbought': 80, # 极度超买
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成RSI信号"""
        period = self.params['rsi_period']

        # 计算RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 生成信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 超卖买入
        signals[rsi < self.params['oversold']] = Signal.BUY.value
        signals[rsi < self.params['extreme_oversold']] = Signal.STRONG_BUY.value

        # 超买卖出
        signals[rsi > self.params['overbought']] = Signal.SELL.value
        signals[rsi > self.params['extreme_overbought']] = Signal.STRONG_SELL.value

        return signals


class BreakoutStrategy(BaseStrategy):
    """
    突破策略

    策略逻辑：
    - 价格突破N日最高价时买入
    - 价格跌破N日最低价时卖出
    - 可选择使用ATR过滤假突破

    适用场景：
    - 趋势启动阶段
    - 波动较大的市场
    """

    category = "momentum"

    def default_params(self) -> Dict[str, Any]:
        return {
            'breakout_period': 20,    # 突破周期
            'exit_period': 10,        # 退出周期
            'atr_period': 14,         # ATR周期
            'atr_filter': 1.0,        # ATR过滤倍数（0=不过滤）
            'volume_confirm': True,   # 成交量确认
            'volume_multiplier': 1.5, # 成交量放大倍数
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成突破信号"""
        bp = self.params['breakout_period']
        ep = self.params['exit_period']

        # 计算通道
        high_channel = data['high'].rolling(bp).max().shift(1)
        low_channel = data['low'].rolling(bp).min().shift(1)
        exit_low = data['low'].rolling(ep).min().shift(1)
        exit_high = data['high'].rolling(ep).max().shift(1)

        # 计算ATR用于过滤
        if self.params['atr_filter'] > 0:
            tr = pd.concat([
                data['high'] - data['low'],
                (data['high'] - data['close'].shift(1)).abs(),
                (data['low'] - data['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.params['atr_period']).mean()
            atr_threshold = atr * self.params['atr_filter']
        else:
            atr_threshold = 0

        # 生成信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 突破买入
        breakout_up = (data['close'] > high_channel) & \
                      (data['close'] - high_channel > atr_threshold)

        # 成交量确认
        if self.params['volume_confirm'] and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            volume_surge = data['volume'] > volume_ma * self.params['volume_multiplier']
            breakout_up = breakout_up & volume_surge

        signals[breakout_up] = Signal.BUY.value

        # 突破卖出
        breakout_down = data['close'] < exit_low
        signals[breakout_down] = Signal.SELL.value

        return signals


class DualMomentumStrategy(BaseStrategy):
    """
    双动量策略

    策略逻辑：
    - 绝对动量：资产过去收益为正
    - 相对动量：资产表现优于基准
    - 两者同时满足时才买入

    适用场景：
    - 资产轮动
    - 风险控制型策略

    参考：Gary Antonacci的双动量理论
    """

    category = "momentum"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 12 * 21,      # 12个月（约252个交易日）
            'short_lookback': 21,     # 1个月
            'rf_rate': 0.02,          # 无风险利率（年化）
            'benchmark_return': 0.0,  # 基准收益率（可选）
            'rebalance_freq': 21,     # 再平衡频率（天）
            'position_pct': 0.2,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成双动量信号"""
        lookback = self.params['lookback']
        short_lookback = self.params['short_lookback']
        daily_rf = self.params['rf_rate'] / 252

        # 计算绝对动量（过去N天收益）
        absolute_momentum = data['close'].pct_change(lookback)

        # 计算短期动量
        short_momentum = data['close'].pct_change(short_lookback)

        # 计算超额收益（相对于无风险利率）
        excess_return = absolute_momentum - (daily_rf * lookback)

        # 生成信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 双动量确认：绝对动量为正 + 超额收益为正
        buy_condition = (absolute_momentum > 0) & (excess_return > 0)

        # 短期动量加强信号
        strong_buy = buy_condition & (short_momentum > 0)

        signals[buy_condition] = Signal.BUY.value
        signals[strong_buy] = Signal.STRONG_BUY.value

        # 卖出条件：绝对动量转负
        sell_condition = absolute_momentum < 0
        signals[sell_condition] = Signal.SELL.value

        # 强烈卖出：短期动量也转负
        strong_sell = sell_condition & (short_momentum < 0)
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class ROCMomentumStrategy(BaseStrategy):
    """
    变动率动量策略 (Rate of Change)

    策略逻辑：
    - 基于多周期ROC综合判断
    - 短、中、长期ROC加权平均
    - 可配置各周期权重

    适用场景：
    - 多时间框架分析
    - 捕捉不同级别的趋势
    """

    category = "momentum"

    def default_params(self) -> Dict[str, Any]:
        return {
            'short_period': 5,        # 短期周期
            'medium_period': 20,      # 中期周期
            'long_period': 60,        # 长期周期
            'short_weight': 0.5,      # 短期权重
            'medium_weight': 0.3,     # 中期权重
            'long_weight': 0.2,       # 长期权重
            'buy_threshold': 0.02,    # 买入阈值
            'sell_threshold': -0.02,  # 卖出阈值
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成ROC动量信号"""
        # 计算多周期ROC
        roc_short = data['close'].pct_change(self.params['short_period'])
        roc_medium = data['close'].pct_change(self.params['medium_period'])
        roc_long = data['close'].pct_change(self.params['long_period'])

        # 加权平均
        weighted_roc = (
            roc_short * self.params['short_weight'] +
            roc_medium * self.params['medium_weight'] +
            roc_long * self.params['long_weight']
        )

        # 生成信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        buy_threshold = self.params['buy_threshold']
        sell_threshold = self.params['sell_threshold']

        # 买入
        signals[weighted_roc > buy_threshold] = Signal.BUY.value
        signals[weighted_roc > buy_threshold * 2] = Signal.STRONG_BUY.value

        # 卖出
        signals[weighted_roc < sell_threshold] = Signal.SELL.value
        signals[weighted_roc < sell_threshold * 2] = Signal.STRONG_SELL.value

        return signals
