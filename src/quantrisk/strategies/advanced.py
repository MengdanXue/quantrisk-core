"""
高级策略模块

包含多种高级和复合交易策略：
- GridTradingStrategy: 网格交易策略
- MartingaleStrategy: 马丁格尔策略
- TimeSeriesMomentumStrategy: 时间序列动量策略
- AdaptiveMomentumStrategy: 自适应动量策略
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class GridTradingStrategy(BaseStrategy):
    """
    网格交易策略

    策略逻辑：
    - 在预设价格区间内设置多个网格
    - 价格下跌到网格线时买入
    - 价格上涨到网格线时卖出
    - 适合震荡市场

    适用场景：
    - 区间震荡市场
    - 稳定币/外汇
    - 低波动资产
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            'grid_count': 10,         # 网格数量
            'grid_range_pct': 0.20,   # 网格范围（上下各10%）
            'use_dynamic_grid': True, # 使用动态网格
            'lookback': 60,           # 动态网格回看周期
            'position_per_grid': 0.1, # 每格仓位
            'position_pct': 0.1,
        }

    def _calculate_grid_levels(self, data: pd.DataFrame) -> tuple:
        """计算网格价格水平"""
        if self.params['use_dynamic_grid']:
            # 动态网格：基于历史高低点
            lookback = self.params['lookback']
            high = data['high'].rolling(lookback).max()
            low = data['low'].rolling(lookback).min()
        else:
            # 静态网格：基于当前价格
            current_price = data['close']
            range_pct = self.params['grid_range_pct']
            high = current_price * (1 + range_pct)
            low = current_price * (1 - range_pct)

        return high, low

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成网格交易信号"""
        grid_count = self.params['grid_count']
        high, low = self._calculate_grid_levels(data)

        # 计算网格间距
        grid_size = (high - low) / grid_count

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            current_high = high.iloc[i]
            current_low = low.iloc[i]
            current_grid_size = grid_size.iloc[i]

            if pd.isna(current_grid_size) or current_grid_size <= 0:
                continue

            # 计算价格在哪个网格
            current_grid = int((current_price - current_low) / current_grid_size)
            prev_grid = int((prev_price - current_low) / current_grid_size)

            # 价格下穿网格线 -> 买入
            if current_grid < prev_grid:
                signals.iloc[i] = Signal.BUY.value
                # 下穿多个网格 -> 强烈买入
                if prev_grid - current_grid >= 2:
                    signals.iloc[i] = Signal.STRONG_BUY.value

            # 价格上穿网格线 -> 卖出
            elif current_grid > prev_grid:
                signals.iloc[i] = Signal.SELL.value
                # 上穿多个网格 -> 强烈卖出
                if current_grid - prev_grid >= 2:
                    signals.iloc[i] = Signal.STRONG_SELL.value

        return signals


class MartingaleStrategy(BaseStrategy):
    """
    马丁格尔策略

    策略逻辑：
    - 亏损后加倍仓位
    - 盈利后恢复初始仓位
    - 高风险策略，需严格控制

    警告：此策略风险极高，可能导致巨大亏损！
    仅供学习研究，不建议实盘使用

    适用场景：
    - 均值回归性强的标的
    - 严格控制的资金管理
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            'trigger_threshold': -0.02,  # 触发加仓的亏损阈值
            'max_multiplier': 8,         # 最大加仓倍数
            'reset_on_profit': True,     # 盈利后重置
            'profit_threshold': 0.01,    # 盈利阈值
            'stop_loss_pct': 0.10,       # 总止损比例
            'position_pct': 0.05,        # 初始仓位（保守）
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成马丁格尔信号"""
        trigger = self.params['trigger_threshold']
        max_mult = self.params['max_multiplier']
        profit_threshold = self.params['profit_threshold']

        # 计算累计收益
        returns = data['close'].pct_change()
        cum_returns = (1 + returns).cumprod() - 1

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 跟踪当前倍数
        current_multiplier = 1
        entry_price = None
        position_return = 0

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]

            if entry_price is not None:
                position_return = (current_price - entry_price) / entry_price

            # 亏损达到阈值 -> 加仓买入
            if position_return < trigger and current_multiplier < max_mult:
                signals.iloc[i] = Signal.BUY.value
                current_multiplier = min(current_multiplier * 2, max_mult)
                # 强信号表示加大仓位
                if current_multiplier >= 4:
                    signals.iloc[i] = Signal.STRONG_BUY.value

            # 盈利达到阈值 -> 平仓
            elif position_return > profit_threshold:
                signals.iloc[i] = Signal.SELL.value
                current_multiplier = 1
                entry_price = None

            # 总止损
            elif position_return < -self.params['stop_loss_pct']:
                signals.iloc[i] = Signal.STRONG_SELL.value
                current_multiplier = 1
                entry_price = None

            # 初始入场
            if entry_price is None and signals.iloc[i] == Signal.HOLD.value:
                # 简单的均值回归入场条件
                sma = data['close'].rolling(20).mean()
                if current_price < sma.iloc[i] * 0.98:
                    signals.iloc[i] = Signal.BUY.value
                    entry_price = current_price

        return signals


class TimeSeriesMomentumStrategy(BaseStrategy):
    """
    时间序列动量策略 (TSMOM)

    策略逻辑：
    - 根据资产自身的历史收益决定多空
    - 过去收益为正则做多，为负则做空
    - 使用波动率调整仓位

    适用场景：
    - 期货市场
    - 全球宏观策略
    - CTA策略

    学术基础：Moskowitz, Ooi, Pedersen (2012)
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 252,          # 回看周期（12个月）
            'vol_lookback': 60,       # 波动率估计周期
            'vol_target': 0.15,       # 目标波动率（15%）
            'vol_scale': True,        # 波动率调整
            'skip_recent': 21,        # 跳过最近N天（避免短期反转）
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成时间序列动量信号"""
        lookback = self.params['lookback']
        skip = self.params['skip_recent']

        # 计算过去收益（跳过最近N天）
        past_return = data['close'].shift(skip).pct_change(lookback - skip)

        # 计算波动率
        vol_lookback = self.params['vol_lookback']
        returns = data['close'].pct_change()
        volatility = returns.rolling(vol_lookback).std() * np.sqrt(252)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 过去收益为正 -> 做多
        signals[past_return > 0] = Signal.BUY.value

        # 过去收益为负 -> 做空
        signals[past_return < 0] = Signal.SELL.value

        # 强信号：收益极端
        return_zscore = past_return / past_return.rolling(lookback).std()
        signals[return_zscore > 1.5] = Signal.STRONG_BUY.value
        signals[return_zscore < -1.5] = Signal.STRONG_SELL.value

        return signals


class AdaptiveMomentumStrategy(BaseStrategy):
    """
    自适应动量策略

    策略逻辑：
    - 根据市场状态动态调整策略参数
    - 高波动时使用短期动量
    - 低波动时使用长期动量
    - 趋势强时加大仓位，震荡时减小仓位

    适用场景：
    - 多变的市场环境
    - 需要自动适应的策略
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            'short_momentum': 10,     # 短期动量周期
            'long_momentum': 60,      # 长期动量周期
            'vol_period': 20,         # 波动率周期
            'vol_threshold': 0.02,    # 波动率阈值
            'trend_period': 50,       # 趋势判断周期
            'position_pct': 0.1,
        }

    def _calculate_market_state(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市场状态"""
        vol_period = self.params['vol_period']
        trend_period = self.params['trend_period']

        returns = data['close'].pct_change()

        # 波动率状态
        volatility = returns.rolling(vol_period).std()
        vol_ma = volatility.rolling(vol_period * 2).mean()
        high_vol = volatility > vol_ma

        # 趋势强度（使用ADX简化版）
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        smoothed_tr = tr.rolling(trend_period).mean()
        smoothed_plus = plus_dm.rolling(trend_period).mean()
        smoothed_minus = minus_dm.rolling(trend_period).mean()

        plus_di = 100 * smoothed_plus / smoothed_tr
        minus_di = 100 * smoothed_minus / smoothed_tr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(trend_period).mean()

        trending = adx > 25

        return pd.DataFrame({
            'high_vol': high_vol,
            'trending': trending,
            'volatility': volatility,
            'adx': adx
        })

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成自适应动量信号"""
        market_state = self._calculate_market_state(data)

        short_period = self.params['short_momentum']
        long_period = self.params['long_momentum']

        # 计算短期和长期动量
        short_momentum = data['close'].pct_change(short_period)
        long_momentum = data['close'].pct_change(long_period)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        for i in range(max(short_period, long_period), len(data)):
            is_high_vol = market_state['high_vol'].iloc[i]
            is_trending = market_state['trending'].iloc[i]

            # 选择动量周期
            if is_high_vol:
                momentum = short_momentum.iloc[i]
            else:
                momentum = long_momentum.iloc[i]

            # 生成信号
            if momentum > 0:
                if is_trending:
                    signals.iloc[i] = Signal.STRONG_BUY.value
                else:
                    signals.iloc[i] = Signal.BUY.value
            elif momentum < 0:
                if is_trending:
                    signals.iloc[i] = Signal.STRONG_SELL.value
                else:
                    signals.iloc[i] = Signal.SELL.value

        return signals


class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略

    策略逻辑：
    - 综合多个因子打分
    - 因子包括：动量、价值、质量、波动率
    - 根据综合得分生成信号

    适用场景：
    - 股票选股
    - 因子投资
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            # 动量因子
            'momentum_period': 60,
            'momentum_weight': 0.3,
            # RSI因子
            'rsi_period': 14,
            'rsi_weight': 0.2,
            # 波动率因子
            'vol_period': 20,
            'vol_weight': 0.2,
            # 趋势因子
            'trend_period': 50,
            'trend_weight': 0.3,
            # 阈值
            'buy_threshold': 0.6,
            'sell_threshold': 0.4,
            'position_pct': 0.1,
        }

    def _calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算各个因子"""
        factors = pd.DataFrame(index=data.index)

        # 动量因子（收益率排名）
        mom_period = self.params['momentum_period']
        momentum = data['close'].pct_change(mom_period)
        factors['momentum'] = momentum.rank(pct=True)

        # RSI因子
        rsi_period = self.params['rsi_period']
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # RSI反转因子（低RSI看多）
        factors['rsi'] = 1 - (rsi / 100)

        # 波动率因子（低波动率优先）
        vol_period = self.params['vol_period']
        returns = data['close'].pct_change()
        volatility = returns.rolling(vol_period).std()
        factors['volatility'] = 1 - volatility.rank(pct=True)

        # 趋势因子
        trend_period = self.params['trend_period']
        sma = data['close'].rolling(trend_period).mean()
        factors['trend'] = (data['close'] / sma - 1).rank(pct=True)

        return factors

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成多因子信号"""
        factors = self._calculate_factors(data)

        # 加权平均得分
        weights = {
            'momentum': self.params['momentum_weight'],
            'rsi': self.params['rsi_weight'],
            'volatility': self.params['vol_weight'],
            'trend': self.params['trend_weight'],
        }

        composite_score = pd.Series(0.0, index=data.index)
        for factor, weight in weights.items():
            composite_score += factors[factor].fillna(0.5) * weight

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        buy_threshold = self.params['buy_threshold']
        sell_threshold = self.params['sell_threshold']

        # 高分买入
        signals[composite_score > buy_threshold] = Signal.BUY.value
        signals[composite_score > (1 + buy_threshold) / 2] = Signal.STRONG_BUY.value

        # 低分卖出
        signals[composite_score < sell_threshold] = Signal.SELL.value
        signals[composite_score < sell_threshold / 2] = Signal.STRONG_SELL.value

        return signals


class AntiMartingaleStrategy(BaseStrategy):
    """
    反马丁格尔策略（金字塔加仓）

    策略逻辑：
    - 盈利时加仓，亏损时减仓
    - 让利润奔跑，截断亏损
    - 比马丁格尔策略更安全

    适用场景：
    - 趋势市场
    - 顺势加仓
    """

    category = "advanced"

    def default_params(self) -> Dict[str, Any]:
        return {
            'profit_threshold': 0.02,  # 盈利加仓阈值
            'loss_threshold': -0.01,   # 亏损减仓阈值
            'max_position': 4,         # 最大仓位倍数
            'add_size': 0.5,           # 每次加仓比例
            'position_pct': 0.1,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成反马丁格尔信号"""
        profit_threshold = self.params['profit_threshold']
        loss_threshold = self.params['loss_threshold']

        # 计算短期收益
        returns = data['close'].pct_change(5)

        signals = pd.Series(Signal.HOLD.value, index=data.index)

        # 初始入场条件（简单均线策略）
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(30).mean()

        # 趋势向上时入场
        uptrend = sma_short > sma_long
        downtrend = sma_short < sma_long

        # 盈利加仓
        profit_add = uptrend & (returns > profit_threshold)
        signals[profit_add] = Signal.BUY.value

        # 持续盈利强烈加仓
        strong_profit = profit_add & (returns > profit_threshold * 2)
        signals[strong_profit] = Signal.STRONG_BUY.value

        # 亏损减仓/止损
        loss_cut = returns < loss_threshold
        signals[loss_cut] = Signal.SELL.value

        # 趋势反转止损
        trend_reverse = downtrend & (sma_short.shift(1) > sma_long.shift(1))
        signals[trend_reverse] = Signal.STRONG_SELL.value

        return signals
