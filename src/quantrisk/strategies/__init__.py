"""
TraderX 量化策略模块

提供多种量化交易策略的实现，包括：
- 动量策略 (Momentum)
- 均值回归策略 (Mean Reversion)
- 趋势跟踪策略 (Trend Following)
- 多因子策略 (Multi-Factor)
- 波动率策略 (Volatility)
- 统计套利策略 (Statistical Arbitrage)
"""

from .base import BaseStrategy, Signal, StrategyResult
from .momentum import (
    MomentumStrategy,
    RelativeStrengthStrategy,
    BreakoutStrategy,
    DualMomentumStrategy,
)
from .mean_reversion import (
    BollingerReversionStrategy,
    RSIReversionStrategy,
    PairsTradingStrategy,
    ZScoreReversionStrategy,
)
from .trend_following import (
    MovingAverageCrossStrategy,
    TurtleTradingStrategy,
    MACDStrategy,
    TripleMAStrategy,
    DonchianChannelStrategy,
)
from .volatility import (
    VolatilityBreakoutStrategy,
    ATRChannelStrategy,
    VolatilityContractionStrategy,
)
from .advanced import (
    GridTradingStrategy,
    MartingaleStrategy,
    TimeSeriesMomentumStrategy,
    AdaptiveMomentumStrategy,
)
from .pairs_trading_advanced import (
    AdvancedPairsTradingStrategy,
    PairsTradingConfig,
    CointegrationTest,
    CointegrationResult,
    HalfLife,
    SpreadCalculator,
    SpreadType,
    PairSelection,
    KalmanHedgeRatio,
    PairSignal,
    PairPosition,
    SignalType,
)

# 策略注册表
STRATEGY_REGISTRY = {
    # 动量策略
    'momentum': MomentumStrategy,
    'relative_strength': RelativeStrengthStrategy,
    'breakout': BreakoutStrategy,
    'dual_momentum': DualMomentumStrategy,

    # 均值回归
    'bollinger_reversion': BollingerReversionStrategy,
    'rsi_reversion': RSIReversionStrategy,
    'pairs_trading': PairsTradingStrategy,
    'zscore_reversion': ZScoreReversionStrategy,

    # 趋势跟踪
    'ma_cross': MovingAverageCrossStrategy,
    'turtle': TurtleTradingStrategy,
    'macd': MACDStrategy,
    'triple_ma': TripleMAStrategy,
    'donchian': DonchianChannelStrategy,

    # 波动率
    'volatility_breakout': VolatilityBreakoutStrategy,
    'atr_channel': ATRChannelStrategy,
    'volatility_contraction': VolatilityContractionStrategy,

    # 高级策略
    'grid': GridTradingStrategy,
    'martingale': MartingaleStrategy,
    'tsmom': TimeSeriesMomentumStrategy,
    'adaptive_momentum': AdaptiveMomentumStrategy,
    # AdvancedPairsTradingStrategy 是双标的配对框架（接口为 initialize(price_a, price_b)），
    # 不符合单标的 BaseStrategy 接口，不放入注册表，直接导入使用
}


def get_strategy(name: str, params: dict = None) -> BaseStrategy:
    """
    根据名称获取策略实例

    Args:
        name: 策略名称
        params: 策略参数

    Returns:
        策略实例
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")

    strategy_class = STRATEGY_REGISTRY[name]
    return strategy_class(params=params)


def list_strategies() -> list:
    """列出所有可用策略"""
    return [
        {
            'name': name,
            'class': cls.__name__,
            'description': cls.__doc__.strip().split('\n')[0] if cls.__doc__ else '',
            'category': cls.category if hasattr(cls, 'category') else 'other',
            'default_params': cls({}).default_params() if hasattr(cls, 'default_params') else {},
        }
        for name, cls in STRATEGY_REGISTRY.items()
    ]


__all__ = [
    'BaseStrategy',
    'Signal',
    'StrategyResult',
    'get_strategy',
    'list_strategies',
    'STRATEGY_REGISTRY',
    # 动量
    'MomentumStrategy',
    'RelativeStrengthStrategy',
    'BreakoutStrategy',
    'DualMomentumStrategy',
    # 均值回归
    'BollingerReversionStrategy',
    'RSIReversionStrategy',
    'PairsTradingStrategy',
    'ZScoreReversionStrategy',
    # 趋势跟踪
    'MovingAverageCrossStrategy',
    'TurtleTradingStrategy',
    'MACDStrategy',
    'TripleMAStrategy',
    'DonchianChannelStrategy',
    # 波动率
    'VolatilityBreakoutStrategy',
    'ATRChannelStrategy',
    'VolatilityContractionStrategy',
    # 高级
    'GridTradingStrategy',
    'MartingaleStrategy',
    'TimeSeriesMomentumStrategy',
    'AdaptiveMomentumStrategy',
    # 高级配对交易
    'AdvancedPairsTradingStrategy',
    'PairsTradingConfig',
    'CointegrationTest',
    'CointegrationResult',
    'HalfLife',
    'SpreadCalculator',
    'SpreadType',
    'PairSelection',
    'KalmanHedgeRatio',
    'PairSignal',
    'PairPosition',
    'SignalType',
]