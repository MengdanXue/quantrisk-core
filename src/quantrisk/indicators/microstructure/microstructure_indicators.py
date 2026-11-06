"""
市场微观结构指标 (20个)

包含价差、流动性、价格冲击、信息指标、订单流、波动率分解和高频指标
"""

from typing import List, Dict, Union
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 价差指标 (3个) ====================

@indicator_registry.register('microstructure')
class BidAskSpread(BaseIndicator):
    """
    Bid-Ask Spread (买卖价差)

    衡量市场流动性的基本指标,价差越小流动性越好
    """

    _metadata = IndicatorMetadata(
        name='BidAskSpread',
        display_name='买卖价差',
        category='microstructure',
        description='买价与卖价之差,反映市场流动性成本',
        formula='Spread = Ask - Bid',
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
        """计算买卖价差"""
        self.validate_data(data)
        return data['ask'] - data['bid']


@indicator_registry.register('microstructure')
class EffectiveSpread(BaseIndicator):
    """
    Effective Spread (有效价差)

    实际交易价格与中间价的差异,衡量真实交易成本
    """

    _metadata = IndicatorMetadata(
        name='EffectiveSpread',
        display_name='有效价差',
        category='microstructure',
        description='实际交易价格偏离中间价的程度',
        formula='Effective Spread = 2 * |Price - MidPrice|',
        required_columns=['close', 'bid', 'ask'],
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
        return ['close', 'bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算有效价差"""
        self.validate_data(data)
        mid_price = (data['bid'] + data['ask']) / 2
        return 2 * np.abs(data['close'] - mid_price)


@indicator_registry.register('microstructure')
class RealizedSpread(BaseIndicator):
    """
    Realized Spread (已实现价差)

    衡量流动性提供者的短期收益
    """

    _metadata = IndicatorMetadata(
        name='RealizedSpread',
        display_name='已实现价差',
        category='microstructure',
        description='流动性提供者在短期内实现的价差收益',
        formula='Realized Spread = 2 * (Price - MidPrice_future)',
        required_columns=['close', 'bid', 'ask'],
        output_type='series',
        default_params={'lag': 5},
        complexity_level=2
    )

    def __init__(self, lag: int = 5):
        self.lag = lag
        super().__init__(lag=lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.lag, 'lag')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算已实现价差"""
        self.validate_data(data)
        mid_price = (data['bid'] + data['ask']) / 2
        mid_price_future = mid_price.shift(-self.lag)
        return 2 * (data['close'] - mid_price_future)


# ==================== 流动性指标 (3个) ====================

@indicator_registry.register('microstructure')
class AmihudIlliquidity(BaseIndicator):
    """
    Amihud Illiquidity (Amihud非流动性指标)

    衡量价格冲击程度,值越大表示流动性越差
    """

    _metadata = IndicatorMetadata(
        name='AmihudIlliquidity',
        display_name='Amihud非流动性',
        category='microstructure',
        description='每单位交易量引起的价格变动,衡量市场深度',
        formula='Amihud = |Return| / Volume',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算Amihud非流动性指标"""
        self.validate_data(data)
        returns = data['close'].pct_change().abs()
        # 避免除零
        volume_safe = data['volume'].replace(0, np.nan)
        illiquidity = returns / volume_safe
        return illiquidity.rolling(window=self.window).mean()


@indicator_registry.register('microstructure')
class KyleLambda(BaseIndicator):
    """
    Kyle Lambda (Kyle价格冲击系数)

    衡量订单流对价格的影响程度
    """

    _metadata = IndicatorMetadata(
        name='KyleLambda',
        display_name='Kyle价格冲击系数',
        category='microstructure',
        description='订单流对价格影响的敏感度',
        formula='Lambda = Cov(ΔP, Q) / Var(Q)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算Kyle Lambda"""
        self.validate_data(data)
        price_change = data['close'].diff()

        lambda_values = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_price = price_change.iloc[i-self.window:i]
            window_volume = data['volume'].iloc[i-self.window:i]

            # 计算协方差和方差
            if window_volume.var() > 0:
                covariance = window_price.cov(window_volume)
                variance = window_volume.var()
                lambda_values.iloc[i] = covariance / variance
            else:
                lambda_values.iloc[i] = 0

        return lambda_values


@indicator_registry.register('microstructure')
class MarketDepth(BaseIndicator):
    """
    Market Depth (市场深度)

    衡量市场吸收大额订单的能力
    """

    _metadata = IndicatorMetadata(
        name='MarketDepth',
        display_name='市场深度',
        category='microstructure',
        description='市场可承接的订单量,反映流动性储备',
        formula='Depth = (BidVolume + AskVolume) / 2',
        required_columns=['bid_volume', 'ask_volume'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['bid_volume', 'ask_volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算市场深度"""
        self.validate_data(data)
        depth = (data['bid_volume'] + data['ask_volume']) / 2
        return depth.rolling(window=self.window).mean()


# ==================== 价格冲击指标 (3个) ====================

@indicator_registry.register('microstructure')
class TemporaryPriceImpact(BaseIndicator):
    """
    Temporary Price Impact (临时价格冲击)

    交易对价格的短期影响,通常会快速恢复
    """

    _metadata = IndicatorMetadata(
        name='TemporaryPriceImpact',
        display_name='临时价格冲击',
        category='microstructure',
        description='交易引起的短期价格偏离',
        formula='Temp Impact = (Price_t - MidPrice_t)',
        required_columns=['close', 'bid', 'ask'],
        output_type='series',
        default_params={'window': 5},
        complexity_level=2
    )

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算临时价格冲击"""
        self.validate_data(data)
        mid_price = (data['bid'] + data['ask']) / 2
        impact = data['close'] - mid_price
        return impact.rolling(window=self.window).mean()


@indicator_registry.register('microstructure')
class PermanentPriceImpact(BaseIndicator):
    """
    Permanent Price Impact (永久价格冲击)

    交易对价格的长期影响,反映信息含量
    """

    _metadata = IndicatorMetadata(
        name='PermanentPriceImpact',
        display_name='永久价格冲击',
        category='microstructure',
        description='交易引起的长期价格变化',
        formula='Perm Impact = MidPrice_future - MidPrice_t',
        required_columns=['bid', 'ask'],
        output_type='series',
        default_params={'lag': 20},
        complexity_level=2
    )

    def __init__(self, lag: int = 20):
        self.lag = lag
        super().__init__(lag=lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.lag, 'lag')

    @property
    def required_columns(self) -> List[str]:
        return ['bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算永久价格冲击"""
        self.validate_data(data)
        mid_price = (data['bid'] + data['ask']) / 2
        mid_price_future = mid_price.shift(-self.lag)
        return mid_price_future - mid_price


@indicator_registry.register('microstructure')
class VPIN(BaseIndicator):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN)

    基于成交量同步的知情交易概率
    """

    _metadata = IndicatorMetadata(
        name='VPIN',
        display_name='知情交易概率',
        category='microstructure',
        description='衡量市场中知情交易者的比例',
        formula='VPIN = |BuyVolume - SellVolume| / TotalVolume',
        required_columns=['volume', 'close'],
        output_type='series',
        default_params={'window': 50},
        complexity_level=3
    )

    def __init__(self, window: int = 50):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['volume', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算VPIN"""
        self.validate_data(data)

        # 使用价格变化判断买卖方向
        price_change = data['close'].diff()
        buy_volume = data['volume'].where(price_change > 0, 0)
        sell_volume = data['volume'].where(price_change < 0, 0)

        # 计算订单流不平衡
        ofi = np.abs(buy_volume - sell_volume)
        total_volume = data['volume']

        # 滚动窗口计算VPIN
        vpin = (ofi.rolling(window=self.window).sum() /
                total_volume.rolling(window=self.window).sum())

        return vpin


# ==================== 信息指标 (3个) ====================

@indicator_registry.register('microstructure')
class PIN(BaseIndicator):
    """
    Probability of Informed Trading (知情交易概率)

    估计市场中知情交易者的比例
    """

    _metadata = IndicatorMetadata(
        name='PIN',
        display_name='知情交易概率',
        category='microstructure',
        description='基于买卖订单到达率估计知情交易比例',
        formula='PIN = α * μ / (α * μ + ε_b + ε_s)',
        required_columns=['volume', 'close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=4
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['volume', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算PIN (简化版本)"""
        self.validate_data(data)

        # 分类买卖订单
        price_change = data['close'].diff()
        buy_volume = data['volume'].where(price_change > 0, 0)
        sell_volume = data['volume'].where(price_change < 0, 0)

        pin_values = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_buy = buy_volume.iloc[i-self.window:i]
            window_sell = sell_volume.iloc[i-self.window:i]

            # 简化的PIN估计
            epsilon_b = window_buy.mean()
            epsilon_s = window_sell.mean()

            # 估计知情交易到达率
            mu = np.abs(epsilon_b - epsilon_s)
            alpha = 0.5  # 简化假设

            if (alpha * mu + epsilon_b + epsilon_s) > 0:
                pin_values.iloc[i] = alpha * mu / (alpha * mu + epsilon_b + epsilon_s)
            else:
                pin_values.iloc[i] = 0

        return pin_values


@indicator_registry.register('microstructure')
class InformationShare(BaseIndicator):
    """
    Information Share (信息份额)

    衡量某个市场对价格发现的贡献
    """

    _metadata = IndicatorMetadata(
        name='InformationShare',
        display_name='信息份额',
        category='microstructure',
        description='衡量市场对价格发现过程的贡献度',
        formula='IS = Var(innovation_i) / Var(total_innovation)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'window': 30},
        complexity_level=3
    )

    def __init__(self, window: int = 30):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算信息份额"""
        self.validate_data(data)

        returns = data['close'].pct_change()
        volume_weighted_returns = returns * data['volume']

        is_values = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_returns = returns.iloc[i-self.window:i]
            window_vw_returns = volume_weighted_returns.iloc[i-self.window:i]

            if window_returns.var() > 0:
                is_values.iloc[i] = window_vw_returns.var() / window_returns.var()
            else:
                is_values.iloc[i] = 0

        return is_values


@indicator_registry.register('microstructure')
class PriceContribution(BaseIndicator):
    """
    Price Contribution (价格贡献度)

    衡量交易对价格发现的贡献
    """

    _metadata = IndicatorMetadata(
        name='PriceContribution',
        display_name='价格贡献度',
        category='microstructure',
        description='交易活动对价格变化的解释力度',
        formula='PC = Corr(Volume, |PriceChange|)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算价格贡献度"""
        self.validate_data(data)

        price_change = data['close'].diff().abs()

        pc_values = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_volume = data['volume'].iloc[i-self.window:i]
            window_price = price_change.iloc[i-self.window:i]

            correlation = window_volume.corr(window_price)
            pc_values.iloc[i] = correlation if not pd.isna(correlation) else 0

        return pc_values


# ==================== 订单流指标 (2个) ====================

@indicator_registry.register('microstructure')
class OrderFlowImbalance(BaseIndicator):
    """
    Order Flow Imbalance (订单流不平衡)

    衡量买卖订单流的不平衡程度
    """

    _metadata = IndicatorMetadata(
        name='OrderFlowImbalance',
        display_name='订单流不平衡',
        category='microstructure',
        description='买方订单流与卖方订单流的差异',
        formula='OFI = (BuyVolume - SellVolume) / TotalVolume',
        required_columns=['volume', 'close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['volume', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算订单流不平衡"""
        self.validate_data(data)

        # 使用价格变化判断买卖方向
        price_change = data['close'].diff()
        buy_volume = data['volume'].where(price_change > 0, 0)
        sell_volume = data['volume'].where(price_change < 0, 0)

        # 计算不平衡度
        ofi = (buy_volume - sell_volume) / data['volume']

        return ofi.rolling(window=self.window).mean()


@indicator_registry.register('microstructure')
class VolumeImbalance(BaseIndicator):
    """
    Volume Imbalance (交易量不平衡)

    买卖交易量的相对差异
    """

    _metadata = IndicatorMetadata(
        name='VolumeImbalance',
        display_name='交易量不平衡',
        category='microstructure',
        description='买卖交易量的不对称性',
        formula='VI = (BuyVolume - SellVolume) / (BuyVolume + SellVolume)',
        required_columns=['volume', 'close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['volume', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算交易量不平衡"""
        self.validate_data(data)

        # 分类买卖交易量
        price_change = data['close'].diff()
        buy_volume = data['volume'].where(price_change > 0, 0)
        sell_volume = data['volume'].where(price_change < 0, 0)

        # 计算不平衡
        numerator = buy_volume - sell_volume
        denominator = buy_volume + sell_volume

        vi = self._safe_division(numerator, denominator, fill_value=0)

        return pd.Series(vi).rolling(window=self.window).mean()


# ==================== 波动率分解指标 (3个) ====================

@indicator_registry.register('microstructure')
class RealizedVolatility(BaseIndicator):
    """
    Realized Volatility (已实现波动率)

    基于高频数据计算的实际波动率
    """

    _metadata = IndicatorMetadata(
        name='RealizedVolatility',
        display_name='已实现波动率',
        category='microstructure',
        description='基于高频收益率的平方和计算的波动率',
        formula='RV = sqrt(Sum(r_t^2))',
        required_columns=['close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算已实现波动率"""
        self.validate_data(data)

        returns = data['close'].pct_change()
        squared_returns = returns ** 2

        rv = np.sqrt(squared_returns.rolling(window=self.window).sum())

        return rv


@indicator_registry.register('microstructure')
class BiPowerVariation(BaseIndicator):
    """
    Bi-Power Variation (双幂次变差)

    对跳跃具有稳健性的波动率估计
    """

    _metadata = IndicatorMetadata(
        name='BiPowerVariation',
        display_name='双幂次变差',
        category='microstructure',
        description='对价格跳跃稳健的波动率度量',
        formula='BPV = (π/2) * Sum(|r_t| * |r_{t-1}|)',
        required_columns=['close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算双幂次变差"""
        self.validate_data(data)

        returns = data['close'].pct_change()
        abs_returns = returns.abs()

        # 计算相邻绝对收益的乘积
        product = abs_returns * abs_returns.shift(1)

        # 双幂次变差
        mu_1 = np.sqrt(2 / np.pi)  # E[|Z|] for standard normal Z
        bpv = (np.pi / 2) * product.rolling(window=self.window).sum() / (mu_1 ** 2)

        return np.sqrt(bpv)


@indicator_registry.register('microstructure')
class JumpDetection(BaseIndicator):
    """
    Jump Detection (跳跃检测)

    识别价格中的不连续跳跃
    """

    _metadata = IndicatorMetadata(
        name='JumpDetection',
        display_name='跳跃检测',
        category='microstructure',
        description='检测价格序列中的跳跃行为',
        formula='Jump = (RV - BPV) / RV',
        required_columns=['close'],
        output_type='series',
        default_params={'window': 20, 'threshold': 0.5},
        complexity_level=3
    )

    def __init__(self, window: int = 20, threshold: float = 0.5):
        self.window = window
        self.threshold = threshold
        super().__init__(window=window, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_in_range(self.threshold, 0, 1, 'threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算跳跃检测指标"""
        self.validate_data(data)

        # 计算已实现波动率
        returns = data['close'].pct_change()
        squared_returns = returns ** 2
        rv = squared_returns.rolling(window=self.window).sum()

        # 计算双幂次变差
        abs_returns = returns.abs()
        product = abs_returns * abs_returns.shift(1)
        mu_1 = np.sqrt(2 / np.pi)
        bpv = (np.pi / 2) * product.rolling(window=self.window).sum() / (mu_1 ** 2)

        # 跳跃比率
        jump_ratio = self._safe_division(rv - bpv, rv, fill_value=0)

        # 跳跃信号 (超过阈值时为1)
        jump_signal = (jump_ratio > self.threshold).astype(float)

        return jump_signal


# ==================== 高频指标 (3个) ====================

@indicator_registry.register('microstructure')
class MicroPrice(BaseIndicator):
    """
    Micro Price (微观价格)

    结合订单簿信息的精细价格估计
    """

    _metadata = IndicatorMetadata(
        name='MicroPrice',
        display_name='微观价格',
        category='microstructure',
        description='基于订单簿深度加权的价格',
        formula='MicroPrice = (Bid*AskVol + Ask*BidVol) / (BidVol + AskVol)',
        required_columns=['bid', 'ask', 'bid_volume', 'ask_volume'],
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
        return ['bid', 'ask', 'bid_volume', 'ask_volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算微观价格"""
        self.validate_data(data)

        numerator = data['bid'] * data['ask_volume'] + data['ask'] * data['bid_volume']
        denominator = data['bid_volume'] + data['ask_volume']

        micro_price = self._safe_division(numerator, denominator,
                                         fill_value=(data['bid'] + data['ask']) / 2)

        return pd.Series(micro_price, index=data.index)


@indicator_registry.register('microstructure')
class LeeReadyClassifier(BaseIndicator):
    """
    Lee-Ready Classifier (Lee-Ready交易分类)

    判断交易的主动方向(买方或卖方主动)
    """

    _metadata = IndicatorMetadata(
        name='LeeReadyClassifier',
        display_name='Lee-Ready分类器',
        category='microstructure',
        description='识别交易是买方主动还是卖方主动',
        formula='Sign = 1 if Price > MidPrice, -1 if Price < MidPrice',
        required_columns=['close', 'bid', 'ask'],
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
        return ['close', 'bid', 'ask']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Lee-Ready交易分类"""
        self.validate_data(data)

        mid_price = (data['bid'] + data['ask']) / 2

        # 判断交易方向
        # 1: 买方主动, -1: 卖方主动, 0: 中间价
        trade_sign = pd.Series(0, index=data.index)
        trade_sign[data['close'] > mid_price] = 1
        trade_sign[data['close'] < mid_price] = -1

        # 对于在中间价成交的,使用tick test
        at_mid = data['close'] == mid_price
        if at_mid.any():
            price_change = data['close'].diff()
            trade_sign[at_mid & (price_change > 0)] = 1
            trade_sign[at_mid & (price_change < 0)] = -1

        return trade_sign


@indicator_registry.register('microstructure')
class HighFrequencyVolatility(BaseIndicator):
    """
    High Frequency Volatility (高频波动率)

    基于高频数据的即时波动率估计
    """

    _metadata = IndicatorMetadata(
        name='HighFrequencyVolatility',
        display_name='高频波动率',
        category='microstructure',
        description='捕捉价格短期波动的高频指标',
        formula='HFVol = sqrt(Sum(r_i^2) / n)',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 10},
        complexity_level=2
    )

    def __init__(self, window: int = 10):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算高频波动率"""
        self.validate_data(data)

        # 使用Parkinson波动率估计器(基于高低价)
        hl_ratio = np.log(data['high'] / data['low'])
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)

        # 滚动窗口平均
        hf_vol = np.sqrt(parkinson_var.rolling(window=self.window).mean())

        return hf_vol


# ==================== 导出所有指标 ====================

__all__ = [
    # 价差指标 (3)
    'BidAskSpread',
    'EffectiveSpread',
    'RealizedSpread',

    # 流动性指标 (3)
    'AmihudIlliquidity',
    'KyleLambda',
    'MarketDepth',

    # 价格冲击指标 (3)
    'TemporaryPriceImpact',
    'PermanentPriceImpact',
    'VPIN',

    # 信息指标 (3)
    'PIN',
    'InformationShare',
    'PriceContribution',

    # 订单流指标 (2)
    'OrderFlowImbalance',
    'VolumeImbalance',

    # 波动率分解指标 (3)
    'RealizedVolatility',
    'BiPowerVariation',
    'JumpDetection',

    # 高频指标 (3)
    'MicroPrice',
    'LeeReadyClassifier',
    'HighFrequencyVolatility',
]
