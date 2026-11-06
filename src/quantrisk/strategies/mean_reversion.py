"""
均值回归策略模块

包含多种基于均值回归效应的交易策略：
- BollingerReversionStrategy: 布林带均值回归
- RSIReversionStrategy: RSI均值回归
- PairsTradingStrategy: 配对交易
- ZScoreReversionStrategy: Z-Score均值回归
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class BollingerReversionStrategy(BaseStrategy):
    """
    布林带均值回归策略

    策略逻辑：
    - 价格触及下轨时买入（超卖）
    - 价格触及上轨时卖出（超买）
    - 价格回归中轨时平仓

    适用场景：
    - 震荡市场
    - 区间交易
    """

    category = "mean_reversion"

    def default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,             # 布林带周期
            'std_dev': 2.0,           # 标准差倍数
            'entry_std': 2.0,         # 入场标准差
            'exit_std': 0.5,          # 出场标准差（回归到中轨附近）
            'extreme_std': 2.5,       # 极端标准差（强信号）
            'use_atr_filter': True,   # 使用ATR过滤
            'atr_period': 14,
            'min_atr_pct': 0.01,      # 最小ATR百分比
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成布林带信号"""
        period = self.params['period']
        std_dev = self.params['std_dev']

        # 计算布林带
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()

        upper = sma + std_dev * std
        lower = sma - std_dev * std
        extreme_upper = sma + self.params['extreme_std'] * std
        extreme_lower = sma - self.params['extreme_std'] * std

        # 计算位置百分比 (0=下轨, 0.5=中轨, 1=上轨)
        bb_pct = (data['close'] - lower) / (upper - lower)

        # ATR过滤（只在足够波动时交易）
        if self.params['use_atr_filter']:
            tr = pd.concat([
                data['high'] - data['low'],
                (data['high'] - data['close'].shift(1)).abs(),
                (data['low'] - data['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.params['atr_period']).mean()
            atr_pct = atr / data['close']
            valid_volatility = atr_pct > self.params['min_atr_pct']
        else:
            valid_volatility = True

        # 生成信号
        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 触及下轨买入
        buy_condition = (data['close'] <= lower) & valid_volatility
        signals[buy_condition] = Signal.BUY.value

        # 极度超卖强烈买入
        strong_buy = (data['close'] <= extreme_lower) & valid_volatility
        signals[strong_buy] = Signal.STRONG_BUY.value

        # 触及上轨卖出
        sell_condition = (data['close'] >= upper) & valid_volatility
        signals[sell_condition] = Signal.SELL.value

        # 极度超买强烈卖出
        strong_sell = (data['close'] >= extreme_upper) & valid_volatility
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class RSIReversionStrategy(BaseStrategy):
    """
    RSI均值回归策略

    策略逻辑：
    - RSI超卖后开始回升时买入
    - RSI超买后开始回落时卖出
    - 使用RSI变化方向确认反转

    适用场景：
    - 超跌反弹
    - 高抛低吸
    """

    category = "mean_reversion"

    def default_params(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'confirm_bars': 1,        # 确认K线数
            'position_pct': 0.1,
        }

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI"""
        period = self.params['rsi_period']
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成RSI回归信号"""
        rsi = self._calculate_rsi(data)
        rsi_change = rsi.diff(self.params['confirm_bars'])

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 超卖且RSI开始回升 -> 买入
        buy_condition = (rsi < self.params['oversold']) & (rsi_change > 0)
        signals[buy_condition] = Signal.BUY.value

        # 极度超卖且回升 -> 强烈买入
        strong_buy = (rsi < self.params['extreme_oversold']) & (rsi_change > 0)
        signals[strong_buy] = Signal.STRONG_BUY.value

        # 超买且RSI开始回落 -> 卖出
        sell_condition = (rsi > self.params['overbought']) & (rsi_change < 0)
        signals[sell_condition] = Signal.SELL.value

        # 极度超买且回落 -> 强烈卖出
        strong_sell = (rsi > self.params['extreme_overbought']) & (rsi_change < 0)
        signals[strong_sell] = Signal.STRONG_SELL.value

        return signals


class PairsTradingStrategy(BaseStrategy):
    """
    配对交易策略

    策略逻辑：
    - 找到两个高度相关的资产
    - 计算价差的Z-Score
    - Z-Score偏离均值时交易，回归时平仓

    适用场景：
    - 统计套利
    - 市场中性策略

    注意：此策略需要两个资产的数据
    """

    category = "mean_reversion"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 60,           # 回看周期（计算均值和标准差）
            'entry_z': 2.0,           # 入场Z-Score阈值
            'exit_z': 0.5,            # 出场Z-Score阈值
            'stop_z': 3.5,            # 止损Z-Score
            'spread_type': 'ratio',   # 价差类型：'ratio' 或 'diff'
            'hedge_ratio': 1.0,       # 对冲比率
            'position_pct': 0.1,
        }

    def calculate_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算价差和Z-Score

        Returns:
            (spread, zscore, hedge_ratio)
        """
        lookback = self.params['lookback']

        if self.params['spread_type'] == 'ratio':
            spread = asset1 / asset2
        else:
            spread = asset1 - asset2 * self.params['hedge_ratio']

        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        zscore = (spread - spread_mean) / spread_std

        return spread, zscore, spread_mean

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成配对交易信号

        data需要包含两列：'asset1' 和 'asset2' 的收盘价
        或者 'close' 和 'close2'
        """
        # 获取两个资产的价格
        if 'asset1' in data.columns and 'asset2' in data.columns:
            asset1 = data['asset1']
            asset2 = data['asset2']
        elif 'close' in data.columns and 'close2' in data.columns:
            asset1 = data['close']
            asset2 = data['close2']
        else:
            # 单资产模式：使用价格与均线的偏离
            asset1 = data['close']
            asset2 = data['close'].rolling(self.params['lookback']).mean()

        # 计算Z-Score
        spread, zscore, _ = self.calculate_spread(asset1, asset2)

        entry_z = self.params['entry_z']
        exit_z = self.params['exit_z']
        stop_z = self.params['stop_z']

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 价差过高 -> 做空价差（卖asset1，买asset2）
        signals[zscore > entry_z] = Signal.SELL.value
        signals[zscore > stop_z] = Signal.STRONG_SELL.value

        # 价差过低 -> 做多价差（买asset1，卖asset2）
        signals[zscore < -entry_z] = Signal.BUY.value
        signals[zscore < -stop_z] = Signal.STRONG_BUY.value

        # 回归中性 -> 平仓（这里用HOLD表示）
        signals[(zscore > -exit_z) & (zscore < exit_z)] = Signal.HOLD.value

        return signals


class ZScoreReversionStrategy(BaseStrategy):
    """
    Z-Score均值回归策略

    策略逻辑：
    - 计算价格相对于历史均值的Z-Score
    - Z-Score极端偏离时反向交易
    - 适用于均值回归的标的

    适用场景：
    - 商品期货
    - ETF
    - 汇率
    """

    category = "mean_reversion"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 60,           # 回看周期
            'entry_z': 2.0,           # 入场Z-Score
            'exit_z': 0.0,            # 出场Z-Score
            'stop_loss_z': 3.0,       # 止损Z-Score
            'use_log_price': True,    # 使用对数价格
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成Z-Score信号"""
        lookback = self.params['lookback']

        # 使用对数价格或普通价格
        if self.params['use_log_price']:
            price = np.log(data['close'])
        else:
            price = data['close']

        # 计算Z-Score
        rolling_mean = price.rolling(lookback).mean()
        rolling_std = price.rolling(lookback).std()
        zscore = (price - rolling_mean) / rolling_std

        entry_z = self.params['entry_z']
        exit_z = self.params['exit_z']
        stop_z = self.params['stop_loss_z']

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 价格过低 -> 买入
        signals[zscore < -entry_z] = Signal.BUY.value
        signals[zscore < -stop_z] = Signal.STRONG_BUY.value  # 止损级别反而是强信号（均值回归）

        # 价格过高 -> 卖出
        signals[zscore > entry_z] = Signal.SELL.value
        signals[zscore > stop_z] = Signal.STRONG_SELL.value

        return signals


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    """
    Ornstein-Uhlenbeck均值回归策略

    策略逻辑：
    - 基于OU过程的统计模型
    - 估计均值回归速度和长期均值
    - 价格偏离长期均值时交易

    适用场景：
    - 利率产品
    - 商品价差
    - 统计套利

    数学基础：
    dX = θ(μ - X)dt + σdW
    其中θ是回归速度，μ是长期均值
    """

    category = "mean_reversion"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 252,          # 估计参数的回看周期
            'entry_std': 1.5,         # 入场标准差
            'exit_std': 0.5,          # 出场标准差
            'min_half_life': 5,       # 最小半衰期（天）
            'max_half_life': 60,      # 最大半衰期（天）
            'position_pct': 0.1,
        }

    def estimate_ou_params(self, prices: pd.Series) -> Tuple[float, float, float]:
        """
        估计OU过程参数

        Returns:
            (theta, mu, sigma) - 回归速度、长期均值、波动率
        """
        log_prices = np.log(prices)
        delta_t = 1.0 / 252  # 假设日频数据

        # 使用OLS估计参数
        y = log_prices.diff().dropna()
        x = log_prices.shift(1).dropna()

        # 对齐数据
        y = y.iloc[1:]
        x = x.iloc[:-1]

        if len(x) < 10:
            return 0, prices.mean(), prices.std()

        # 回归: dy = a + b*x + epsilon
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            a, b = coeffs
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"OLS regression failed in OU parameter estimation: {e}")
            return 0, prices.mean(), prices.std()

        # 转换为OU参数
        theta = -b / delta_t
        mu = a / (-b) if b != 0 else np.log(prices.mean())
        sigma = y.std() / np.sqrt(delta_t)

        return theta, np.exp(mu), sigma

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成OU回归信号"""
        lookback = self.params['lookback']
        prices = data['close']

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        for i in range(lookback, len(data)):
            window = prices.iloc[i-lookback:i]

            theta, mu, sigma = self.estimate_ou_params(window)

            # 检查半衰期是否在合理范围
            if theta > 0:
                half_life = np.log(2) / theta
                if half_life < self.params['min_half_life'] or half_life > self.params['max_half_life']:
                    continue

            # 计算当前价格偏离
            current_price = prices.iloc[i]
            deviation = (current_price - mu) / sigma

            entry_std = self.params['entry_std']
            exit_std = self.params['exit_std']

            # 价格过低 -> 买入
            if deviation < -entry_std:
                signals.iloc[i] = Signal.BUY.value
            elif deviation < -entry_std * 1.5:
                signals.iloc[i] = Signal.STRONG_BUY.value
            # 价格过高 -> 卖出
            elif deviation > entry_std:
                signals.iloc[i] = Signal.SELL.value
            elif deviation > entry_std * 1.5:
                signals.iloc[i] = Signal.STRONG_SELL.value

        return signals
