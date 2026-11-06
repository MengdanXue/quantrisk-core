"""
执行质量指标 (15个)

用于评估交易执行质量和市场冲击
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 执行成本指标 (5个) ====================

@indicator_registry.register('execution')
class ImplementationShortfall(BaseIndicator):
    """
    执行差额

    实际执行价格与决策价格之差
    """

    _metadata = IndicatorMetadata(
        name='ImplementationShortfall',
        display_name='执行差额',
        category='execution',
        description='实际执行价格与决策时价格的差异',
        formula='(Execution Price - Decision Price) / Decision Price',
        required_columns=['decision_price', 'execution_price', 'quantity'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['decision_price', 'execution_price', 'quantity']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算执行差额"""
        shortfall = (data['execution_price'] - data['decision_price']) / data['decision_price']
        return shortfall * 10000  # 以基点表示


@indicator_registry.register('execution')
class VWAP(BaseIndicator):
    """
    成交量加权平均价格 (VWAP)

    用于评估执行质量的基准价格
    """

    _metadata = IndicatorMetadata(
        name='VWAP',
        display_name='VWAP',
        category='execution',
        description='成交量加权平均价格',
        formula='Sum(Price * Volume) / Sum(Volume)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=1
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算VWAP"""
        pv = data['close'] * data['volume']
        vwap = pv.rolling(self.window).sum() / data['volume'].rolling(self.window).sum()
        return vwap


@indicator_registry.register('execution')
class TWAP(BaseIndicator):
    """
    时间加权平均价格 (TWAP)

    简单时间加权的平均价格
    """

    _metadata = IndicatorMetadata(
        name='TWAP',
        display_name='TWAP',
        category='execution',
        description='时间加权平均价格',
        formula='Average(Price) over time window',
        required_columns=['close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=1
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算TWAP"""
        twap = data['close'].rolling(self.window).mean()
        return twap


@indicator_registry.register('execution')
class SlippageCost(BaseIndicator):
    """
    滑点成本

    实际成交价与预期价格的偏差
    """

    _metadata = IndicatorMetadata(
        name='SlippageCost',
        display_name='滑点成本',
        category='execution',
        description='实际成交价与预期价格的偏差',
        formula='(Actual Price - Expected Price) / Expected Price',
        required_columns=['expected_price', 'actual_price'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['expected_price', 'actual_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算滑点成本"""
        slippage = (data['actual_price'] - data['expected_price']) / data['expected_price']
        return slippage * 10000  # 基点


@indicator_registry.register('execution')
class TotalTradingCost(BaseIndicator):
    """
    总交易成本

    包含显性和隐性成本的综合指标
    """

    _metadata = IndicatorMetadata(
        name='TotalTradingCost',
        display_name='总交易成本',
        category='execution',
        description='综合交易成本(佣金+滑点+市场冲击)',
        formula='Commission + Slippage + Market Impact',
        required_columns=['commission', 'slippage', 'market_impact'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['commission', 'slippage', 'market_impact']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算总交易成本"""
        total = data['commission'] + data['slippage'] + data['market_impact']
        return total


# ==================== 市场冲击指标 (5个) ====================

@indicator_registry.register('execution')
class PermanentImpact(BaseIndicator):
    """
    永久性冲击

    交易对价格的长期影响
    """

    _metadata = IndicatorMetadata(
        name='PermanentImpact',
        display_name='永久性冲击',
        category='execution',
        description='交易完成后价格的永久性变化',
        formula='(Post-trade Price - Pre-trade Price) / Pre-trade Price',
        required_columns=['pre_trade_price', 'post_trade_price'],
        output_type='series',
        default_params={'settle_period': 5},
        complexity_level=2
    )

    def __init__(self, settle_period: int = 5):
        self.settle_period = settle_period
        super().__init__(settle_period=settle_period)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.settle_period, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['pre_trade_price', 'post_trade_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算永久性冲击"""
        impact = (data['post_trade_price'] - data['pre_trade_price']) / data['pre_trade_price']
        return impact * 10000  # 基点


@indicator_registry.register('execution')
class TemporaryImpact(BaseIndicator):
    """
    临时性冲击

    交易期间的短期价格偏离
    """

    _metadata = IndicatorMetadata(
        name='TemporaryImpact',
        display_name='临时性冲击',
        category='execution',
        description='交易期间价格的临时性偏离',
        formula='Max Price Deviation during execution',
        required_columns=['high', 'low', 'vwap'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'vwap']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算临时性冲击"""
        high_dev = (data['high'] - data['vwap']) / data['vwap']
        low_dev = (data['vwap'] - data['low']) / data['vwap']
        temp_impact = np.maximum(high_dev, low_dev)
        return temp_impact * 10000  # 基点


@indicator_registry.register('execution')
class MarketImpactModel(BaseIndicator):
    """
    市场冲击模型

    基于成交量和波动率估计市场冲击
    """

    _metadata = IndicatorMetadata(
        name='MarketImpactModel',
        display_name='市场冲击模型',
        category='execution',
        description='基于Almgren-Chriss模型估计市场冲击',
        formula='sigma * sqrt(Q / V) * participation_rate^alpha',
        required_columns=['close', 'volume', 'order_size'],
        output_type='series',
        default_params={'volatility_window': 20, 'alpha': 0.5},
        complexity_level=3
    )

    def __init__(self, volatility_window: int = 20, alpha: float = 0.5):
        self.volatility_window = volatility_window
        self.alpha = alpha
        super().__init__(volatility_window=volatility_window, alpha=alpha)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.volatility_window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume', 'order_size']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算市场冲击"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.volatility_window).std()
        participation = data['order_size'] / data['volume']
        impact = volatility * np.sqrt(participation) * (participation ** self.alpha)
        return impact * 10000  # 基点


@indicator_registry.register('execution')
class ParticipationRate(BaseIndicator):
    """
    参与率

    订单量占市场成交量的比例
    """

    _metadata = IndicatorMetadata(
        name='ParticipationRate',
        display_name='参与率',
        category='execution',
        description='订单量占市场成交量的比例',
        formula='Order Volume / Market Volume',
        required_columns=['order_volume', 'market_volume'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['order_volume', 'market_volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算参与率"""
        rate = data['order_volume'] / data['market_volume']
        return rate * 100  # 百分比


@indicator_registry.register('execution')
class PriceReversion(BaseIndicator):
    """
    价格回归

    交易后价格回归的程度
    """

    _metadata = IndicatorMetadata(
        name='PriceReversion',
        display_name='价格回归',
        category='execution',
        description='交易后价格向均值回归的程度',
        formula='(Trade Price - Post Price) / (Trade Price - Pre Price)',
        required_columns=['pre_price', 'trade_price', 'post_price'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['pre_price', 'trade_price', 'post_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算价格回归"""
        move = data['trade_price'] - data['pre_price']
        reversion = data['trade_price'] - data['post_price']
        ratio = np.where(move != 0, reversion / move, 0)
        return pd.Series(ratio, index=data.index)


# ==================== 执行质量指标 (5个) ====================

@indicator_registry.register('execution')
class FillRate(BaseIndicator):
    """
    成交率

    订单成交量占订单总量的比例
    """

    _metadata = IndicatorMetadata(
        name='FillRate',
        display_name='成交率',
        category='execution',
        description='订单成交量占订单总量的比例',
        formula='Filled Quantity / Order Quantity',
        required_columns=['filled_qty', 'order_qty'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['filled_qty', 'order_qty']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算成交率"""
        fill_rate = data['filled_qty'] / data['order_qty']
        return fill_rate * 100  # 百分比


@indicator_registry.register('execution')
class ExecutionSpeed(BaseIndicator):
    """
    执行速度

    订单从下达到完成的时间
    """

    _metadata = IndicatorMetadata(
        name='ExecutionSpeed',
        display_name='执行速度',
        category='execution',
        description='订单执行时间(秒)',
        formula='Execution End Time - Order Submit Time',
        required_columns=['submit_time', 'complete_time'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['submit_time', 'complete_time']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算执行速度"""
        speed = (data['complete_time'] - data['submit_time']).dt.total_seconds()
        return speed


@indicator_registry.register('execution')
class ArrivalPrice(BaseIndicator):
    """
    到达价格

    订单到达市场时的价格
    """

    _metadata = IndicatorMetadata(
        name='ArrivalPrice',
        display_name='到达价格',
        category='execution',
        description='订单到达市场时的中间价',
        formula='(Bid + Ask) / 2 at arrival time',
        required_columns=['bid', 'ask'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算到达价格"""
        arrival = (data['bid'] + data['ask']) / 2
        return arrival


@indicator_registry.register('execution')
class VWAPDeviation(BaseIndicator):
    """
    VWAP偏离度

    执行价格与VWAP的偏差
    """

    _metadata = IndicatorMetadata(
        name='VWAPDeviation',
        display_name='VWAP偏离度',
        category='execution',
        description='执行价格与VWAP的偏差',
        formula='(Execution Price - VWAP) / VWAP',
        required_columns=['execution_price', 'vwap'],
        output_type='series',
        default_params={},
        complexity_level=1
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['execution_price', 'vwap']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算VWAP偏离度"""
        deviation = (data['execution_price'] - data['vwap']) / data['vwap']
        return deviation * 10000  # 基点


@indicator_registry.register('execution')
class ExecutionQualityScore(BaseIndicator):
    """
    执行质量评分

    综合多个维度评估执行质量
    """

    _metadata = IndicatorMetadata(
        name='ExecutionQualityScore',
        display_name='执行质量评分',
        category='execution',
        description='综合执行质量评分(0-100)',
        formula='Weighted Score(Price, Speed, Fill Rate, Cost)',
        required_columns=['price_score', 'speed_score', 'fill_score', 'cost_score'],
        output_type='series',
        default_params={'weights': [0.3, 0.2, 0.2, 0.3]},
        complexity_level=2
    )

    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.3, 0.2, 0.2, 0.3]
        super().__init__(weights=self.weights)

    def _validate_params(self):
        if len(self.weights) != 4:
            raise ValueError("weights must have 4 elements")
        if abs(sum(self.weights) - 1.0) > 0.001:
            raise ValueError("weights must sum to 1")

    @property
    def required_columns(self) -> List[str]:
        return ['price_score', 'speed_score', 'fill_score', 'cost_score']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算执行质量评分"""
        cols = ['price_score', 'speed_score', 'fill_score', 'cost_score']
        score = sum(data[col] * w for col, w in zip(cols, self.weights))
        return score
