"""
回测事件系统
Backtest Event System

实现事件驱动回测架构的核心组件：
- 事件类型定义
- 事件队列
- 事件处理器

作者: TraderX Team
创建: 2025-12-23
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import heapq
import pandas as pd


class EventType(str, Enum):
    """事件类型"""
    # 市场数据事件
    BAR = "bar"                     # K线数据事件
    TICK = "tick"                   # Tick数据事件
    QUOTE = "quote"                 # 报价事件

    # 信号事件
    SIGNAL = "signal"               # 策略信号

    # 订单事件
    ORDER = "order"                 # 订单创建
    ORDER_UPDATE = "order_update"   # 订单状态更新

    # 成交事件
    FILL = "fill"                   # 订单成交

    # 持仓事件
    POSITION = "position"           # 持仓变化

    # 风控事件
    RISK = "risk"                   # 风险检查
    MARGIN_CALL = "margin_call"     # 保证金预警
    LIQUIDATION = "liquidation"     # 强平

    # 系统事件
    START = "start"                 # 回测开始
    END = "end"                     # 回测结束
    REBALANCE = "rebalance"         # 再平衡触发
    DIVIDEND = "dividend"           # 分红
    SPLIT = "split"                 # 拆股


@dataclass(order=True)
class Event:
    """
    事件基类

    所有事件都可以按时间戳排序，用于事件队列
    """
    timestamp: datetime = field(compare=True, default=None)
    event_type: EventType = field(compare=False, default=None)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    priority: int = field(default=0, compare=True)  # 优先级（同一时间戳时使用）

    def __post_init__(self):
        # 确保priority参与排序（负数表示高优先级）
        object.__setattr__(self, 'priority', -self.priority if self.priority else 0)
        # 默认时间戳
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.now())


@dataclass
class BarEvent(Event):
    """K线事件"""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0  # 持仓量（期货）

    def __post_init__(self):
        self.event_type = EventType.BAR
        self.data = {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'open_interest': self.open_interest,
        }


@dataclass
class SignalEvent(Event):
    """信号事件"""
    symbol: str = ""
    signal_type: str = ""       # 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'FLAT'
    strength: float = 1.0       # 信号强度 0-1
    target_weight: float = 0.0  # 目标权重
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        self.data = {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'target_weight': self.target_weight,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata,
        }


@dataclass
class OrderEvent(Event):
    """订单事件"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""              # 'BUY', 'SELL'
    quantity: float = 0.0
    order_type: str = "MARKET"  # 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # 'GTC', 'DAY', 'IOC', 'FOK'
    status: str = "PENDING"     # 'PENDING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'

    def __post_init__(self):
        self.event_type = EventType.ORDER
        self.data = {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'status': self.status,
        }


@dataclass
class FillEvent(Event):
    """成交事件"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exchange: str = ""

    def __post_init__(self):
        self.event_type = EventType.FILL
        self.data = {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'exchange': self.exchange,
        }


@dataclass
class RiskEvent(Event):
    """风控事件"""
    risk_type: str = ""         # 'MARGIN', 'POSITION_LIMIT', 'VOLATILITY', 'DRAWDOWN'
    level: str = "WARNING"      # 'INFO', 'WARNING', 'CRITICAL'
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action_required: bool = False

    def __post_init__(self):
        self.event_type = EventType.RISK
        self.data = {
            'risk_type': self.risk_type,
            'level': self.level,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'action_required': self.action_required,
        }


class EventQueue:
    """
    事件队列

    使用优先队列实现，按时间戳排序
    """

    def __init__(self):
        self._queue: List[Event] = []
        self._counter = 0  # 用于处理同时间戳事件的FIFO

    def put(self, event: Event):
        """
        添加事件到队列

        Args:
            event: 事件对象
        """
        # 使用计数器确保同时间戳的事件按插入顺序处理
        self._counter += 1
        heapq.heappush(self._queue, (event.timestamp, event.priority, self._counter, event))

    def get(self) -> Optional[Event]:
        """
        获取下一个事件

        Returns:
            Event或None
        """
        if self._queue:
            _, _, _, event = heapq.heappop(self._queue)
            return event
        return None

    def peek(self) -> Optional[Event]:
        """
        查看下一个事件但不移除

        Returns:
            Event或None
        """
        if self._queue:
            return self._queue[0][3]
        return None

    def empty(self) -> bool:
        """队列是否为空"""
        return len(self._queue) == 0

    def size(self) -> int:
        """队列大小"""
        return len(self._queue)

    def clear(self):
        """清空队列"""
        self._queue.clear()
        self._counter = 0

    def has_events(self) -> bool:
        """是否有事件"""
        return not self.empty()


class EventHandler(ABC):
    """事件处理器抽象基类"""

    @abstractmethod
    def handle(self, event: Event) -> Optional[List[Event]]:
        """
        处理事件

        Args:
            event: 事件对象

        Returns:
            可能产生的新事件列表
        """
        pass

    @property
    @abstractmethod
    def handles(self) -> List[EventType]:
        """返回此处理器处理的事件类型列表"""
        pass


class EventBus:
    """
    事件总线

    管理事件处理器的注册和事件分发
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[Callable[[Event], None]] = []

    def register(self, handler: EventHandler):
        """
        注册事件处理器

        Args:
            handler: 事件处理器
        """
        for event_type in handler.handles:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def register_global(self, callback: Callable[[Event], None]):
        """
        注册全局事件监听器

        Args:
            callback: 回调函数
        """
        self._global_handlers.append(callback)

    def dispatch(self, event: Event) -> List[Event]:
        """
        分发事件

        Args:
            event: 事件对象

        Returns:
            产生的新事件列表
        """
        new_events = []

        # 调用全局处理器
        for callback in self._global_handlers:
            try:
                callback(event)
            except Exception as e:
                print(f"Global handler error: {e}")

        # 调用类型特定处理器
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler.handle(event)
                if result:
                    new_events.extend(result)
            except Exception as e:
                print(f"Handler error: {e}")

        return new_events


class DataHandler:
    """
    数据处理器

    将历史数据转换为事件流
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        初始化数据处理器

        Args:
            data: 数据字典 {symbol: DataFrame}
        """
        self.data = data
        self._current_index = 0
        self._merged_index: Optional[pd.DatetimeIndex] = None
        self._prepare_data()

    def _prepare_data(self):
        """准备合并后的时间索引"""
        all_indices = []
        for df in self.data.values():
            all_indices.extend(df.index.tolist())

        # 合并并排序所有时间戳
        self._merged_index = pd.DatetimeIndex(sorted(set(all_indices)))

    def generate_events(self, event_queue: EventQueue):
        """
        生成所有数据事件

        Args:
            event_queue: 事件队列
        """
        for timestamp in self._merged_index:
            for symbol, df in self.data.items():
                if timestamp in df.index:
                    bar = df.loc[timestamp]
                    event = BarEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        open=bar.get('open', bar.get('Open', 0)),
                        high=bar.get('high', bar.get('High', 0)),
                        low=bar.get('low', bar.get('Low', 0)),
                        close=bar.get('close', bar.get('Close', 0)),
                        volume=bar.get('volume', bar.get('Volume', 0)),
                        open_interest=bar.get('open_interest', bar.get('OI', 0)),
                    )
                    event_queue.put(event)

    def get_bar(self, symbol: str, timestamp: datetime) -> Optional[pd.Series]:
        """获取指定时间的K线"""
        if symbol in self.data and timestamp in self.data[symbol].index:
            return self.data[symbol].loc[timestamp]
        return None

    def get_history(
        self,
        symbol: str,
        end_time: datetime,
        periods: int
    ) -> Optional[pd.DataFrame]:
        """
        获取历史数据

        Args:
            symbol: 标的代码
            end_time: 结束时间
            periods: 获取的bar数量

        Returns:
            历史数据DataFrame
        """
        if symbol not in self.data:
            return None

        df = self.data[symbol]
        mask = df.index <= end_time
        return df[mask].tail(periods)


# 便捷函数
def create_bar_event(
    timestamp: datetime,
    symbol: str,
    ohlcv: Dict[str, float]
) -> BarEvent:
    """创建K线事件"""
    return BarEvent(
        timestamp=timestamp,
        symbol=symbol,
        open=ohlcv.get('open', 0),
        high=ohlcv.get('high', 0),
        low=ohlcv.get('low', 0),
        close=ohlcv.get('close', 0),
        volume=ohlcv.get('volume', 0),
    )


def create_signal_event(
    timestamp: datetime,
    symbol: str,
    signal_type: str,
    strength: float = 1.0,
    **kwargs
) -> SignalEvent:
    """创建信号事件"""
    return SignalEvent(
        timestamp=timestamp,
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
        **kwargs
    )


if __name__ == "__main__":
    # 测试事件队列
    queue = EventQueue()

    # 添加不同时间的事件
    from datetime import timedelta

    now = datetime.now()

    queue.put(BarEvent(timestamp=now + timedelta(seconds=2), symbol="AAPL"))
    queue.put(BarEvent(timestamp=now, symbol="GOOGL"))
    queue.put(BarEvent(timestamp=now + timedelta(seconds=1), symbol="MSFT"))

    print("Events in order:")
    while not queue.empty():
        event = queue.get()
        print(f"  {event.timestamp}: {event.data.get('symbol')}")
