"""
撮合模拟器
Execution Simulator

实现真实的交易撮合逻辑：
- 滑点模型
- 手续费模型
- 市场冲击模型
- 订单簿模拟

作者: TraderX Team
创建: 2025-12-23
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .events import Event, EventType, OrderEvent, FillEvent, BarEvent


class SlippageModel(ABC):
    """滑点模型抽象基类"""

    @abstractmethod
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        side: str,
        volume: float,
        volatility: float
    ) -> float:
        """
        计算滑点

        Args:
            price: 基准价格
            quantity: 交易数量
            side: 买卖方向
            volume: 当前bar成交量
            volatility: 当前波动率

        Returns:
            滑点金额
        """
        pass


class FixedSlippageModel(SlippageModel):
    """固定滑点模型"""

    def __init__(self, slippage_rate: float = 0.0005):
        """
        Args:
            slippage_rate: 滑点率 (如0.0005表示0.05%)
        """
        self.slippage_rate = slippage_rate

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        side: str,
        volume: float,
        volatility: float
    ) -> float:
        slippage = price * self.slippage_rate
        return slippage if side == 'BUY' else -slippage


class VolumeSlippageModel(SlippageModel):
    """
    成交量滑点模型

    滑点与成交量占比成正比
    """

    def __init__(
        self,
        base_rate: float = 0.0001,
        volume_impact: float = 0.1
    ):
        """
        Args:
            base_rate: 基础滑点率
            volume_impact: 成交量影响系数
        """
        self.base_rate = base_rate
        self.volume_impact = volume_impact

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        side: str,
        volume: float,
        volatility: float
    ) -> float:
        # 成交量占比
        volume_ratio = quantity / volume if volume > 0 else 0.1

        # 滑点 = 基础滑点 + 成交量影响
        slippage_rate = self.base_rate + self.volume_impact * volume_ratio

        slippage = price * slippage_rate
        return slippage if side == 'BUY' else -slippage


class VolatilitySlippageModel(SlippageModel):
    """
    波动率滑点模型

    滑点与波动率成正比
    """

    def __init__(
        self,
        base_rate: float = 0.0001,
        vol_multiplier: float = 0.5,
        volume_impact: float = 0.05
    ):
        self.base_rate = base_rate
        self.vol_multiplier = vol_multiplier
        self.volume_impact = volume_impact

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        side: str,
        volume: float,
        volatility: float
    ) -> float:
        # 成交量影响
        volume_ratio = quantity / volume if volume > 0 else 0.1

        # 综合滑点
        slippage_rate = (
            self.base_rate +
            self.vol_multiplier * volatility +
            self.volume_impact * volume_ratio
        )

        slippage = price * slippage_rate
        return slippage if side == 'BUY' else -slippage


class CommissionModel(ABC):
    """手续费模型抽象基类"""

    @abstractmethod
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        side: str
    ) -> float:
        """计算手续费"""
        pass


class PercentCommissionModel(CommissionModel):
    """百分比手续费模型"""

    def __init__(self, rate: float = 0.001, min_commission: float = 1.0):
        """
        Args:
            rate: 手续费率
            min_commission: 最低手续费
        """
        self.rate = rate
        self.min_commission = min_commission

    def calculate_commission(
        self,
        price: float,
        quantity: float,
        side: str
    ) -> float:
        commission = price * quantity * self.rate
        return max(commission, self.min_commission)


class PerShareCommissionModel(CommissionModel):
    """每股手续费模型"""

    def __init__(
        self,
        per_share: float = 0.005,
        min_commission: float = 1.0,
        max_percent: float = 0.005
    ):
        """
        Args:
            per_share: 每股手续费
            min_commission: 最低手续费
            max_percent: 最高手续费占比
        """
        self.per_share = per_share
        self.min_commission = min_commission
        self.max_percent = max_percent

    def calculate_commission(
        self,
        price: float,
        quantity: float,
        side: str
    ) -> float:
        commission = quantity * self.per_share
        commission = max(commission, self.min_commission)

        # 限制最高比例
        max_comm = price * quantity * self.max_percent
        return min(commission, max_comm)


class TieredCommissionModel(CommissionModel):
    """阶梯手续费模型"""

    def __init__(self, tiers: List[Tuple[float, float]]):
        """
        Args:
            tiers: 阶梯列表 [(阈值, 费率), ...]
                   例如 [(1000000, 0.001), (5000000, 0.0008), (float('inf'), 0.0005)]
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate_commission(
        self,
        price: float,
        quantity: float,
        side: str
    ) -> float:
        value = price * quantity

        for threshold, rate in self.tiers:
            if value <= threshold:
                return value * rate

        return value * self.tiers[-1][1]


@dataclass
class FillResult:
    """成交结果"""
    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    fill_price: float
    slippage: float
    commission: float
    timestamp: datetime
    is_partial: bool = False
    remaining_quantity: float = 0.0


class ExecutionHandler:
    """
    订单执行处理器

    负责模拟真实的订单执行过程
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        fill_ratio_limit: float = 0.1,
        allow_partial_fills: bool = True
    ):
        """
        初始化执行处理器

        Args:
            slippage_model: 滑点模型
            commission_model: 手续费模型
            fill_ratio_limit: 最大成交量占比（防止过大订单）
            allow_partial_fills: 是否允许部分成交
        """
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.commission_model = commission_model or PercentCommissionModel()
        self.fill_ratio_limit = fill_ratio_limit
        self.allow_partial_fills = allow_partial_fills

        # 订单簿
        self._pending_orders: Dict[str, OrderEvent] = {}

        # 历史波动率缓存
        self._volatility_cache: Dict[str, float] = {}

    def submit_order(self, order: OrderEvent):
        """
        提交订单

        Args:
            order: 订单事件
        """
        self._pending_orders[order.order_id] = order

    def process_bar(
        self,
        bar: BarEvent,
        volatility: Optional[float] = None
    ) -> List[FillEvent]:
        """
        处理K线，尝试成交挂单

        Args:
            bar: K线事件
            volatility: 当前波动率

        Returns:
            成交事件列表
        """
        fills = []

        # 更新波动率缓存
        if volatility is not None:
            self._volatility_cache[bar.symbol] = volatility

        # 处理该标的的所有挂单
        orders_to_remove = []

        for order_id, order in self._pending_orders.items():
            if order.symbol != bar.symbol:
                continue

            fill = self._try_fill_order(order, bar)
            if fill is not None:
                fills.append(fill)

                is_partial = fill.data.get('is_partial', False)
                if not is_partial:
                    orders_to_remove.append(order_id)
                else:
                    # 更新部分成交订单的剩余数量
                    remaining = fill.data.get('remaining_quantity', 0)
                    order.quantity = remaining

        # 移除已完成订单
        for order_id in orders_to_remove:
            del self._pending_orders[order_id]

        return fills

    def _try_fill_order(
        self,
        order: OrderEvent,
        bar: BarEvent
    ) -> Optional[FillEvent]:
        """
        尝试成交订单

        Args:
            order: 订单
            bar: 当前K线

        Returns:
            成交事件或None
        """
        # 检查是否触发
        should_fill, base_price = self._check_trigger(order, bar)

        if not should_fill:
            return None

        # 计算成交数量（考虑成交量限制）
        volume = bar.volume if bar.volume > 0 else float('inf')
        max_fill = volume * self.fill_ratio_limit

        if order.quantity > max_fill:
            if self.allow_partial_fills:
                fill_quantity = max_fill
                remaining = order.quantity - max_fill
                is_partial = True
            else:
                return None  # 无法成交
        else:
            fill_quantity = order.quantity
            remaining = 0
            is_partial = False

        # 计算滑点
        vol = self._volatility_cache.get(order.symbol, 0.02)
        slippage = self.slippage_model.calculate_slippage(
            base_price, fill_quantity, order.side, volume, vol
        )

        # 成交价
        fill_price = base_price + slippage

        # 计算手续费
        commission = self.commission_model.calculate_commission(
            fill_price, fill_quantity, order.side
        )

        return FillEvent(
            timestamp=bar.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(slippage),
            data={
                'is_partial': is_partial,
                'remaining_quantity': remaining,
                'base_price': base_price,
            }
        )

    def _check_trigger(
        self,
        order: OrderEvent,
        bar: BarEvent
    ) -> Tuple[bool, float]:
        """
        检查订单是否触发

        Args:
            order: 订单
            bar: K线

        Returns:
            (是否触发, 成交基准价)
        """
        order_type = order.order_type.upper()

        if order_type == 'MARKET':
            # 市价单使用开盘价
            return True, bar.open

        elif order_type == 'LIMIT':
            if order.side == 'BUY':
                # 买入限价单：低于等于限价时成交
                if bar.low <= order.limit_price:
                    return True, min(order.limit_price, bar.open)
            else:
                # 卖出限价单：高于等于限价时成交
                if bar.high >= order.limit_price:
                    return True, max(order.limit_price, bar.open)

        elif order_type == 'STOP':
            if order.side == 'BUY':
                # 买入止损单：高于止损价时触发
                if bar.high >= order.stop_price:
                    return True, max(order.stop_price, bar.open)
            else:
                # 卖出止损单：低于止损价时触发
                if bar.low <= order.stop_price:
                    return True, min(order.stop_price, bar.open)

        elif order_type == 'STOP_LIMIT':
            if order.side == 'BUY':
                # 先检查止损触发
                if bar.high >= order.stop_price:
                    # 再检查限价
                    if bar.low <= order.limit_price:
                        return True, min(order.limit_price, bar.open)
            else:
                if bar.low <= order.stop_price:
                    if bar.high >= order.limit_price:
                        return True, max(order.limit_price, bar.open)

        return False, 0.0

    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤销
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return True
        return False

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[OrderEvent]:
        """获取挂单列表"""
        if symbol:
            return [o for o in self._pending_orders.values() if o.symbol == symbol]
        return list(self._pending_orders.values())


class MarketImpactModel:
    """
    市场冲击模型

    模拟大订单对市场价格的影响
    """

    def __init__(
        self,
        permanent_impact: float = 0.0001,
        temporary_impact: float = 0.0005,
        decay_rate: float = 0.5
    ):
        """
        Args:
            permanent_impact: 永久性冲击系数
            temporary_impact: 临时性冲击系数
            decay_rate: 衰减率
        """
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        self.decay_rate = decay_rate

        # 跟踪累积冲击
        self._cumulative_impact: Dict[str, float] = {}

    def calculate_impact(
        self,
        symbol: str,
        quantity: float,
        volume: float,
        price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        计算市场冲击

        Args:
            symbol: 标的
            quantity: 交易数量
            volume: 当前成交量
            price: 当前价格
            side: 买卖方向

        Returns:
            (永久冲击, 临时冲击)
        """
        # 成交量占比
        volume_ratio = quantity / volume if volume > 0 else 0.1

        # 计算冲击
        direction = 1 if side == 'BUY' else -1

        permanent = direction * self.permanent_impact * np.sqrt(volume_ratio) * price
        temporary = direction * self.temporary_impact * volume_ratio * price

        # 更新累积冲击
        current = self._cumulative_impact.get(symbol, 0)
        self._cumulative_impact[symbol] = current + permanent

        return permanent, temporary

    def decay_impact(self, symbol: str):
        """衰减市场冲击"""
        if symbol in self._cumulative_impact:
            self._cumulative_impact[symbol] *= (1 - self.decay_rate)

    def get_cumulative_impact(self, symbol: str) -> float:
        """获取累积冲击"""
        return self._cumulative_impact.get(symbol, 0)


# 预设的执行配置
class ExecutionPresets:
    """执行配置预设"""

    @staticmethod
    def retail() -> ExecutionHandler:
        """零售交易配置"""
        return ExecutionHandler(
            slippage_model=FixedSlippageModel(0.0005),
            commission_model=PercentCommissionModel(0.001, 1.0),
            fill_ratio_limit=0.05,
            allow_partial_fills=False
        )

    @staticmethod
    def institutional() -> ExecutionHandler:
        """机构交易配置"""
        return ExecutionHandler(
            slippage_model=VolatilitySlippageModel(0.0001, 0.3, 0.02),
            commission_model=TieredCommissionModel([
                (1000000, 0.0008),
                (10000000, 0.0005),
                (float('inf'), 0.0003)
            ]),
            fill_ratio_limit=0.1,
            allow_partial_fills=True
        )

    @staticmethod
    def hft() -> ExecutionHandler:
        """高频交易配置"""
        return ExecutionHandler(
            slippage_model=VolumeSlippageModel(0.00001, 0.01),
            commission_model=PerShareCommissionModel(0.001, 0.1, 0.001),
            fill_ratio_limit=0.01,
            allow_partial_fills=True
        )

    @staticmethod
    def crypto() -> ExecutionHandler:
        """加密货币配置"""
        return ExecutionHandler(
            slippage_model=VolatilitySlippageModel(0.0002, 0.5, 0.1),
            commission_model=PercentCommissionModel(0.001, 0.0),
            fill_ratio_limit=0.2,
            allow_partial_fills=True
        )

    @staticmethod
    def futures() -> ExecutionHandler:
        """期货配置"""
        return ExecutionHandler(
            slippage_model=VolumeSlippageModel(0.0001, 0.05),
            commission_model=PercentCommissionModel(0.0001, 0.5),
            fill_ratio_limit=0.15,
            allow_partial_fills=True
        )


if __name__ == "__main__":
    # 测试执行处理器
    handler = ExecutionPresets.retail()

    # 模拟订单
    order = OrderEvent(
        timestamp=datetime.now(),
        order_id="ORD-001",
        symbol="AAPL",
        side="BUY",
        quantity=100,
        order_type="MARKET"
    )

    handler.submit_order(order)

    # 模拟K线
    bar = BarEvent(
        timestamp=datetime.now(),
        symbol="AAPL",
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1000000
    )

    fills = handler.process_bar(bar)

    for fill in fills:
        print(f"Fill: {fill.quantity} @ ${fill.fill_price:.2f}")
        print(f"  Slippage: ${fill.slippage:.4f}")
        print(f"  Commission: ${fill.commission:.2f}")
