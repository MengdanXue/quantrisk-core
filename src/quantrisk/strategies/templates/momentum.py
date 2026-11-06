"""
动量策略模板
Momentum Strategy Templates

包含经典动量策略：
- 动量轮动策略
- 相对强度策略

作者: TraderX Team
创建: 2025-12-23
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base import (
    BaseStrategy, StrategyContext, Signal, SignalType,
    sma, ema, rsi
)


class MomentumRotationStrategy(BaseStrategy):
    """
    动量轮动策略

    在多个资产中选择动量最强的进行投资

    参数:
        momentum_period: 动量计算周期（默认20）
        top_n: 选择前N个（默认3）
        rebalance_period: 再平衡周期（默认20）
        use_volatility_weight: 是否使用波动率加权（默认True）
        min_momentum: 最低动量阈值（默认0，负动量也买入）
        use_filter: 是否使用趋势过滤（默认True）
        filter_period: 过滤周期（默认200）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'momentum_period': 20,
            'top_n': 3,
            'rebalance_period': 20,
            'use_volatility_weight': True,
            'min_momentum': 0,
            'use_filter': True,
            'filter_period': 200,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        return max(
            self.params['momentum_period'],
            self.params['filter_period'] if self.params['use_filter'] else 0
        ) + 10

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        多资产动量轮动

        注意：此策略需要在多资产环境下使用
        """
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        # 检查是否到再平衡日
        bars_since_rebalance = self.get_state('bars_since_rebalance', self.params['rebalance_period'])
        if bars_since_rebalance < self.params['rebalance_period']:
            self.set_state('bars_since_rebalance', bars_since_rebalance + 1)
            return signals

        self.set_state('bars_since_rebalance', 0)

        # 获取所有标的的动量
        close = history['close']
        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 计算动量（收益率）
        momentum = self._calculate_momentum(close)

        # 趋势过滤
        if self.params['use_filter']:
            ma = sma(close, self.params['filter_period'])
            if close.iloc[-1] < ma.iloc[-1]:
                # 价格低于长期均线，不做多
                if context.has_position(symbol):
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.FLAT,
                        metadata={'reason': 'trend_filter'}
                    ))
                return signals

        # 动量阈值过滤
        if momentum < self.params['min_momentum']:
            if context.has_position(symbol):
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.FLAT,
                    metadata={'reason': 'low_momentum', 'momentum': momentum}
                ))
            return signals

        # 计算目标权重
        target_weight = self._calculate_target_weight(close, momentum)

        # 生成信号
        if target_weight > 0:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=1.0,
                target_weight=target_weight,
                metadata={'momentum': momentum}
            ))
        elif context.has_position(symbol):
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.FLAT,
                metadata={'reason': 'no_allocation'}
            ))

        return signals

    def _calculate_momentum(self, close: pd.Series) -> float:
        """计算动量（收益率）"""
        period = self.params['momentum_period']
        if len(close) < period:
            return 0

        return (close.iloc[-1] / close.iloc[-period] - 1)

    def _calculate_target_weight(
        self,
        close: pd.Series,
        momentum: float
    ) -> float:
        """计算目标权重"""
        if not self.params['use_volatility_weight']:
            return 1.0 / self.params['top_n']

        # 使用波动率倒数加权
        volatility = close.pct_change().tail(20).std()
        if volatility == 0:
            return 1.0 / self.params['top_n']

        # 简化的风险平价权重
        inv_vol = 1.0 / volatility
        return min(inv_vol / 10, 0.3)  # 限制最大权重


class RelativeStrengthStrategy(BaseStrategy):
    """
    相对强度策略

    买入相对基准表现强势的资产

    参数:
        rs_period: 相对强度计算周期（默认20）
        entry_threshold: 入场RS阈值（默认1.05）
        exit_threshold: 出场RS阈值（默认0.95）
        use_momentum_confirm: 是否使用动量确认（默认True）
        momentum_period: 动量周期（默认10）
        use_volatility_filter: 是否使用波动率过滤（默认True）
        max_volatility: 最大波动率阈值（默认0.3）
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'rs_period': 20,
            'entry_threshold': 1.05,
            'exit_threshold': 0.95,
            'use_momentum_confirm': True,
            'momentum_period': 10,
            'use_volatility_filter': True,
            'max_volatility': 0.3,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)

        # 存储基准数据
        self._benchmark_data: Optional[pd.Series] = None

    @property
    def required_history(self) -> int:
        return max(self.params['rs_period'], self.params['momentum_period']) + 10

    def set_benchmark(self, benchmark_close: pd.Series):
        """设置基准价格序列"""
        self._benchmark_data = benchmark_close

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        signals = []
        history = context.history

        if history is None or len(history) < self.required_history:
            return signals

        close = history['close']
        symbol = context.current_bar.name if hasattr(context.current_bar, 'name') else 'default'

        # 如果没有基准，使用市场均线作为简化基准
        if self._benchmark_data is None:
            benchmark = sma(close, self.params['rs_period'])
        else:
            benchmark = self._benchmark_data

        # 计算相对强度
        rs = self._calculate_relative_strength(close, benchmark)

        if rs is None:
            return signals

        # 动量确认
        momentum_ok = True
        if self.params['use_momentum_confirm']:
            momentum = (close.iloc[-1] / close.iloc[-self.params['momentum_period']] - 1)
            momentum_ok = momentum > 0

        # 波动率过滤
        volatility_ok = True
        if self.params['use_volatility_filter']:
            volatility = close.pct_change().tail(20).std() * np.sqrt(252)
            volatility_ok = volatility < self.params['max_volatility']

        # 入场信号
        if not context.has_position(symbol):
            if rs > self.params['entry_threshold'] and momentum_ok and volatility_ok:
                strength = min(1.0, (rs - 1) * 2)  # RS越高强度越大
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    metadata={
                        'rs': rs,
                        'momentum_ok': momentum_ok,
                        'volatility_ok': volatility_ok
                    }
                ))

        # 出场信号
        elif context.is_long(symbol):
            if rs < self.params['exit_threshold']:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    strength=1.0,
                    metadata={'rs': rs, 'reason': 'rs_weakness'}
                ))

        return signals

    def _calculate_relative_strength(
        self,
        asset: pd.Series,
        benchmark: pd.Series
    ) -> Optional[float]:
        """
        计算相对强度

        RS = (资产当前/资产N日前) / (基准当前/基准N日前)
        """
        period = self.params['rs_period']

        if len(asset) < period or len(benchmark) < period:
            return None

        # 对齐索引
        try:
            asset_return = asset.iloc[-1] / asset.iloc[-period]
            benchmark_return = benchmark.iloc[-1] / benchmark.iloc[-period]

            if benchmark_return == 0:
                return None

            return asset_return / benchmark_return

        except Exception:
            return None


# 注册策略
MOMENTUM_STRATEGIES = {
    'momentum_rotation': MomentumRotationStrategy,
    'relative_strength': RelativeStrengthStrategy,
}


def get_momentum_strategy(
    name: str,
    params: Optional[Dict[str, Any]] = None
) -> BaseStrategy:
    """获取动量策略"""
    if name not in MOMENTUM_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    return MOMENTUM_STRATEGIES[name](params)
