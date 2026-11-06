"""
策略模板库
Strategy Templates

提供经典交易策略的模板实现：
- 趋势跟踪策略
- 均值回归策略
- 动量策略
- 多因子策略
- 配对交易策略

作者: TraderX Team
创建: 2025-12-23
"""

from .base import BaseStrategy, StrategyContext
from .trend_following import (
    DualMAStrategy,
    TurtleStrategy,
    DonchianBreakoutStrategy,
)
from .mean_reversion import (
    BollingerRevertStrategy,
    RSIMeanReversionStrategy,
)
from .momentum import (
    MomentumRotationStrategy,
    RelativeStrengthStrategy,
)

__all__ = [
    # Base
    'BaseStrategy',
    'StrategyContext',
    # Trend Following
    'DualMAStrategy',
    'TurtleStrategy',
    'DonchianBreakoutStrategy',
    # Mean Reversion
    'BollingerRevertStrategy',
    'RSIMeanReversionStrategy',
    # Momentum
    'MomentumRotationStrategy',
    'RelativeStrengthStrategy',
]
