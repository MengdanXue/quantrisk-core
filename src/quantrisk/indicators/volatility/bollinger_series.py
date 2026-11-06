"""
布林带系列指标 (8个)

基于标准差的波动性通道指标
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('volatility')
class BollingerBands(BaseIndicator):
    """
    Bollinger Bands (布林带)

    John Bollinger开发的经典波动性通道
    """

    _metadata = IndicatorMetadata(
        name='BollingerBands',
        display_name='布林带',
        category='volatility',
        description='基于移动平均线和标准差的波动通道',
        formula='Upper = MA + k*σ, Lower = MA - k*σ',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'std_dev': 2.0},
        complexity_level=1
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        super().__init__(period=period, std_dev=std_dev)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        # 带宽
        bandwidth = (upper - lower) / middle * 100

        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Bandwidth': bandwidth
        })


@indicator_registry.register('volatility')
class BollingerPercentB(BaseIndicator):
    """
    Bollinger %B (%B指标)

    价格在布林带中的相对位置
    """

    _metadata = IndicatorMetadata(
        name='BollingerPercentB',
        display_name='布林%B',
        category='volatility',
        description='价格在布林带中的相对位置，0=下轨，1=上轨',
        formula='%B = (Close - Lower) / (Upper - Lower)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'std_dev': 2.0},
        complexity_level=1
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        super().__init__(period=period, std_dev=std_dev)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        percent_b = (close - lower) / (upper - lower)

        return percent_b


@indicator_registry.register('volatility')
class BollingerBandwidth(BaseIndicator):
    """
    Bollinger Bandwidth (布林带宽度)

    衡量布林带的宽度
    """

    _metadata = IndicatorMetadata(
        name='BollingerBandwidth',
        display_name='布林带宽度',
        category='volatility',
        description='布林带宽度占中轨的百分比',
        formula='Bandwidth = (Upper - Lower) / Middle * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'std_dev': 2.0},
        complexity_level=1
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        super().__init__(period=period, std_dev=std_dev)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        bandwidth = (upper - lower) / middle * 100

        return bandwidth


@indicator_registry.register('volatility')
class BollingerSqueeze(BaseIndicator):
    """
    Bollinger Squeeze (布林带挤压)

    检测布林带收窄信号
    """

    _metadata = IndicatorMetadata(
        name='BollingerSqueeze',
        display_name='布林带挤压',
        category='volatility',
        description='当布林带宽度低于历史平均时产生挤压信号',
        formula='Squeeze = Bandwidth < Bandwidth_MA',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'std_dev': 2.0, 'squeeze_period': 125},
        complexity_level=2
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0, squeeze_period: int = 125):
        self.period = period
        self.std_dev = std_dev
        self.squeeze_period = squeeze_period
        super().__init__(period=period, std_dev=std_dev, squeeze_period=squeeze_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')
        IndicatorValidator.validate_positive_int(self.squeeze_period, 'squeeze_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        bandwidth = (upper - lower) / middle * 100
        bandwidth_ma = bandwidth.rolling(window=self.squeeze_period).mean()

        # 挤压信号：当前带宽小于历史平均
        squeeze = (bandwidth < bandwidth_ma).astype(int)

        # 挤压强度
        squeeze_intensity = (bandwidth_ma - bandwidth) / bandwidth_ma * 100

        return pd.DataFrame({
            'BB_Bandwidth': bandwidth,
            'BB_Bandwidth_MA': bandwidth_ma,
            'BB_Squeeze': squeeze,
            'BB_Squeeze_Intensity': squeeze_intensity
        })


@indicator_registry.register('volatility')
class BollingerTrend(BaseIndicator):
    """
    Bollinger Trend (布林趋势)

    基于布林带位置的趋势判断
    """

    _metadata = IndicatorMetadata(
        name='BollingerTrend',
        display_name='布林趋势',
        category='volatility',
        description='根据价格在布林带中的位置判断趋势',
        formula='Trend = 1 if Close > Middle else -1 if Close < Middle else 0',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'std_dev': 2.0},
        complexity_level=1
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        super().__init__(period=period, std_dev=std_dev)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        # 趋势方向
        trend = pd.Series(0, index=close.index)
        trend[close > middle] = 1
        trend[close < middle] = -1

        # 趋势强度（距离中轨的标准差倍数）
        trend_strength = (close - middle) / std

        # 突破信号
        breakout_up = (close > upper).astype(int)
        breakout_down = (close < lower).astype(int)

        return pd.DataFrame({
            'BB_Trend': trend,
            'BB_Trend_Strength': trend_strength,
            'BB_Breakout_Up': breakout_up,
            'BB_Breakout_Down': breakout_down
        })


@indicator_registry.register('volatility')
class DoubleBollingerBands(BaseIndicator):
    """
    Double Bollinger Bands (双布林带)

    使用两组不同标准差的布林带
    """

    _metadata = IndicatorMetadata(
        name='DoubleBollingerBands',
        display_name='双布林带',
        category='volatility',
        description='内外两组布林带，用于识别趋势区间',
        formula='Outer = MA ± 2σ, Inner = MA ± 1σ',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'inner_std': 1.0, 'outer_std': 2.0},
        complexity_level=2
    )

    def __init__(self, period: int = 20, inner_std: float = 1.0, outer_std: float = 2.0):
        self.period = period
        self.inner_std = inner_std
        self.outer_std = outer_std
        super().__init__(period=period, inner_std=inner_std, outer_std=outer_std)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.inner_std, 'inner_std')
        IndicatorValidator.validate_positive_float(self.outer_std, 'outer_std')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        # 外层布林带
        outer_upper = middle + self.outer_std * std
        outer_lower = middle - self.outer_std * std

        # 内层布林带
        inner_upper = middle + self.inner_std * std
        inner_lower = middle - self.inner_std * std

        # 区间判断
        # 3 = 上轨以上（强多），2 = 内外轨之间上方，1 = 中轨以上内轨以下
        # -1 = 中轨以下内轨以上，-2 = 内外轨之间下方，-3 = 下轨以下（强空）
        zone = pd.Series(0, index=close.index)
        zone[close > outer_upper] = 3
        zone[(close <= outer_upper) & (close > inner_upper)] = 2
        zone[(close <= inner_upper) & (close > middle)] = 1
        zone[(close < middle) & (close >= inner_lower)] = -1
        zone[(close < inner_lower) & (close >= outer_lower)] = -2
        zone[close < outer_lower] = -3

        return pd.DataFrame({
            'DBB_Outer_Upper': outer_upper,
            'DBB_Inner_Upper': inner_upper,
            'DBB_Middle': middle,
            'DBB_Inner_Lower': inner_lower,
            'DBB_Outer_Lower': outer_lower,
            'DBB_Zone': zone
        })


@indicator_registry.register('volatility')
class KeltnerBollingerSqueeze(BaseIndicator):
    """
    Keltner-Bollinger Squeeze (凯尔特纳-布林带挤压)

    当布林带在凯尔特纳通道内时产生挤压信号
    """

    _metadata = IndicatorMetadata(
        name='KeltnerBollingerSqueeze',
        display_name='凯尔特纳-布林挤压',
        category='volatility',
        description='布林带与凯尔特纳通道的挤压关系',
        formula='Squeeze = BB_Lower > KC_Lower AND BB_Upper < KC_Upper',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'bb_period': 20, 'bb_std': 2.0, 'kc_period': 20, 'kc_mult': 1.5},
        complexity_level=3
    )

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        super().__init__(bb_period=bb_period, bb_std=bb_std,
                        kc_period=kc_period, kc_mult=kc_mult)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.bb_period, 'bb_period')
        IndicatorValidator.validate_positive_float(self.bb_std, 'bb_std')
        IndicatorValidator.validate_positive_int(self.kc_period, 'kc_period')
        IndicatorValidator.validate_positive_float(self.kc_mult, 'kc_mult')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # 布林带
        bb_middle = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_middle + self.bb_std * bb_std
        bb_lower = bb_middle - self.bb_std * bb_std

        # 凯尔特纳通道
        kc_middle = close.ewm(span=self.kc_period, adjust=False).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.kc_period, adjust=False).mean()

        kc_upper = kc_middle + self.kc_mult * atr
        kc_lower = kc_middle - self.kc_mult * atr

        # 挤压信号：布林带在凯尔特纳通道内
        squeeze_on = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        squeeze_off = ((bb_lower < kc_lower) | (bb_upper > kc_upper)).astype(int)

        # 动量（用于判断方向）
        momentum = close - ((high + low + close) / 3).rolling(window=self.kc_period).mean()

        return pd.DataFrame({
            'KBS_BB_Upper': bb_upper,
            'KBS_BB_Lower': bb_lower,
            'KBS_KC_Upper': kc_upper,
            'KBS_KC_Lower': kc_lower,
            'KBS_Squeeze_On': squeeze_on,
            'KBS_Squeeze_Off': squeeze_off,
            'KBS_Momentum': momentum
        })


@indicator_registry.register('volatility')
class BollingerMomentum(BaseIndicator):
    """
    Bollinger Momentum (布林动量)

    结合布林带位置和动量的综合指标
    """

    _metadata = IndicatorMetadata(
        name='BollingerMomentum',
        display_name='布林动量',
        category='volatility',
        description='基于布林带位置的动量指标',
        formula='BM = %B * Momentum_Normalized',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'std_dev': 2.0, 'momentum_period': 10},
        complexity_level=2
    )

    def __init__(self, period: int = 20, std_dev: float = 2.0, momentum_period: int = 10):
        self.period = period
        self.std_dev = std_dev
        self.momentum_period = momentum_period
        super().__init__(period=period, std_dev=std_dev, momentum_period=momentum_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')
        IndicatorValidator.validate_positive_int(self.momentum_period, 'momentum_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 布林带
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        # %B
        percent_b = (close - lower) / (upper - lower)

        # 动量
        momentum = close - close.shift(self.momentum_period)
        momentum_std = momentum.rolling(window=self.period).std()
        momentum_normalized = momentum / momentum_std

        # 布林动量
        bollinger_momentum = (percent_b - 0.5) * 2 * abs(momentum_normalized)

        # 信号
        signal = pd.Series(0, index=close.index)
        signal[(percent_b > 0.8) & (momentum_normalized > 1)] = 2  # 强多
        signal[(percent_b > 0.5) & (momentum_normalized > 0)] = 1  # 多
        signal[(percent_b < 0.2) & (momentum_normalized < -1)] = -2  # 强空
        signal[(percent_b < 0.5) & (momentum_normalized < 0)] = -1  # 空

        return pd.DataFrame({
            'BM_PercentB': percent_b,
            'BM_Momentum': momentum_normalized,
            'BM_Value': bollinger_momentum,
            'BM_Signal': signal
        })
