"""
策略基类
Base Strategy Class

定义策略的标准接口和生命周期
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

import pandas as pd
import numpy as np


class SignalType(str, Enum):
    """信号类型"""
    LONG = "long"           # 做多
    SHORT = "short"         # 做空
    EXIT_LONG = "exit_long"   # 平多
    EXIT_SHORT = "exit_short" # 平空
    FLAT = "flat"           # 清仓
    HOLD = "hold"           # 持有


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: float = 1.0       # 信号强度 0-1
    target_weight: float = 0.0  # 目标权重
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Signal({self.symbol}, {self.signal_type.value}, strength={self.strength:.2f})"


@dataclass
class StrategyContext:
    """
    策略上下文

    提供策略运行所需的环境信息
    """
    # 时间
    current_time: datetime = None
    bar_index: int = 0

    # 数据
    current_bar: Optional[pd.Series] = None
    history: Optional[pd.DataFrame] = None

    # 持仓
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    equity: float = 0.0

    # 配置
    initial_capital: float = 1000000.0
    max_position_size: float = 0.2

    # 风险
    margin_used: float = 0.0
    margin_ratio: float = 0.0

    def get_position(self, symbol: str) -> float:
        """获取持仓"""
        return self.positions.get(symbol, 0)

    def has_position(self, symbol: str) -> bool:
        """是否有持仓"""
        return symbol in self.positions and self.positions[symbol] != 0

    def is_long(self, symbol: str) -> bool:
        """是否多头"""
        return self.positions.get(symbol, 0) > 0

    def is_short(self, symbol: str) -> bool:
        """是否空头"""
        return self.positions.get(symbol, 0) < 0

    def get_available_capital(self) -> float:
        """获取可用资金"""
        return self.cash


class BaseStrategy(ABC):
    """
    策略基类

    所有策略都应该继承此类并实现必要的方法
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略

        Args:
            params: 策略参数
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self._is_initialized = False
        self._signals: List[Signal] = []

        # 策略状态
        self._state: Dict[str, Any] = {}

    @property
    @abstractmethod
    def required_history(self) -> int:
        """
        需要的历史数据长度

        策略在初始化时需要的最小历史数据bar数
        """
        pass

    @abstractmethod
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成交易信号

        Args:
            context: 策略上下文

        Returns:
            信号列表
        """
        pass

    def initialize(self, context: StrategyContext):
        """
        初始化策略

        在回测开始时调用一次
        """
        self._is_initialized = True
        self.on_start(context)

    def on_start(self, context: StrategyContext):
        """策略启动回调（可选覆盖）"""
        pass

    def on_end(self, context: StrategyContext):
        """策略结束回调（可选覆盖）"""
        pass

    def on_bar(self, context: StrategyContext) -> List[Signal]:
        """
        处理每个bar

        Args:
            context: 策略上下文

        Returns:
            信号列表
        """
        if not self._is_initialized:
            self.initialize(context)

        # 检查历史数据是否足够
        if context.history is None or len(context.history) < self.required_history:
            return []

        return self.generate_signals(context)

    def on_fill(self, symbol: str, quantity: float, price: float, side: str):
        """订单成交回调（可选覆盖）"""
        pass

    def calculate_position_size(
        self,
        context: StrategyContext,
        symbol: str,
        signal_strength: float = 1.0
    ) -> float:
        """
        计算仓位大小

        Args:
            context: 策略上下文
            symbol: 标的
            signal_strength: 信号强度

        Returns:
            目标仓位（金额）
        """
        max_position = context.equity * context.max_position_size
        return max_position * signal_strength

    def get_param(self, key: str, default: Any = None) -> Any:
        """获取策略参数"""
        return self.params.get(key, default)

    def set_state(self, key: str, value: Any):
        """设置状态"""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self._state.get(key, default)

    def __repr__(self):
        return f"{self.name}(params={self.params})"


# 辅助函数
def crossover(series1: pd.Series, series2: pd.Series) -> bool:
    """
    检测上穿

    Args:
        series1: 快线
        series2: 慢线

    Returns:
        是否发生上穿
    """
    if len(series1) < 2 or len(series2) < 2:
        return False

    return (series1.iloc[-2] <= series2.iloc[-2] and
            series1.iloc[-1] > series2.iloc[-1])


def crossunder(series1: pd.Series, series2: pd.Series) -> bool:
    """
    检测下穿

    Args:
        series1: 快线
        series2: 慢线

    Returns:
        是否发生下穿
    """
    if len(series1) < 2 or len(series2) < 2:
        return False

    return (series1.iloc[-2] >= series2.iloc[-2] and
            series1.iloc[-1] < series2.iloc[-1])


def highest(series: pd.Series, period: int) -> float:
    """获取最高值"""
    return series.tail(period).max()


def lowest(series: pd.Series, period: int) -> float:
    """获取最低值"""
    return series.tail(period).min()


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple:
    """布林带"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower
