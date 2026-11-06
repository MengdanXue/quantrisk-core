"""
ATR系列指标 (7个)

基于真实波动幅度的波动性指标
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('volatility')
class ATR(BaseIndicator):
    """
    Average True Range (平均真实波动幅度)

    J. Welles Wilder开发的经典波动性指标
    """

    _metadata = IndicatorMetadata(
        name='ATR',
        display_name='平均真实波动幅度',
        category='volatility',
        description='衡量价格波动的真实幅度',
        formula='TR = max(H-L, |H-C[1]|, |L-C[1]|), ATR = EMA(TR, n)',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=1
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        return atr


@indicator_registry.register('volatility')
class TrueRange(BaseIndicator):
    """
    True Range (真实波动幅度)

    单周期的真实波动幅度
    """

    _metadata = IndicatorMetadata(
        name='TrueRange',
        display_name='真实波动幅度',
        category='volatility',
        description='当日真实波动幅度',
        formula='TR = max(H-L, |H-C[1]|, |L-C[1]|)',
        required_columns=['high', 'low', 'close'],
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
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr


@indicator_registry.register('volatility')
class ATRPercent(BaseIndicator):
    """
    ATR Percent (ATR百分比)

    ATR占收盘价的百分比
    """

    _metadata = IndicatorMetadata(
        name='ATRPercent',
        display_name='ATR百分比',
        category='volatility',
        description='ATR占价格的百分比，用于跨标的比较',
        formula='ATR% = ATR / Close * 100',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=1
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        return atr / close * 100


@indicator_registry.register('volatility')
class ATRTrailingStop(BaseIndicator):
    """
    ATR Trailing Stop (ATR跟踪止损)

    基于ATR的动态止损位
    """

    _metadata = IndicatorMetadata(
        name='ATRTrailingStop',
        display_name='ATR跟踪止损',
        category='volatility',
        description='基于ATR倍数的动态止损线',
        formula='Stop = Close ± Multiplier * ATR',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14, 'multiplier': 3.0},
        complexity_level=2
    )

    def __init__(self, period: int = 14, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        super().__init__(period=period, multiplier=multiplier)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.multiplier, 'multiplier')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # ATR计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        # 基本止损线
        long_stop = close - self.multiplier * atr
        short_stop = close + self.multiplier * atr

        # 跟踪止损（向有利方向移动，不向不利方向移动）
        trailing_long = long_stop.copy()
        trailing_short = short_stop.copy()

        for i in range(1, len(close)):
            # 多头止损只升不降
            if close.iloc[i] > trailing_long.iloc[i-1]:
                trailing_long.iloc[i] = max(long_stop.iloc[i], trailing_long.iloc[i-1])
            else:
                trailing_long.iloc[i] = long_stop.iloc[i]

            # 空头止损只降不升
            if close.iloc[i] < trailing_short.iloc[i-1]:
                trailing_short.iloc[i] = min(short_stop.iloc[i], trailing_short.iloc[i-1])
            else:
                trailing_short.iloc[i] = short_stop.iloc[i]

        return pd.DataFrame({
            'ATR': atr,
            'ATR_Long_Stop': trailing_long,
            'ATR_Short_Stop': trailing_short
        })


@indicator_registry.register('volatility')
class ATRChannel(BaseIndicator):
    """
    ATR Channel (ATR通道)

    基于ATR的价格通道
    """

    _metadata = IndicatorMetadata(
        name='ATRChannel',
        display_name='ATR通道',
        category='volatility',
        description='以EMA为中轨，ATR为宽度的通道',
        formula='Upper = EMA + k*ATR, Lower = EMA - k*ATR',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14, 'ema_period': 20, 'multiplier': 2.0},
        complexity_level=2
    )

    def __init__(self, period: int = 14, ema_period: int = 20, multiplier: float = 2.0):
        self.period = period
        self.ema_period = ema_period
        self.multiplier = multiplier
        super().__init__(period=period, ema_period=ema_period, multiplier=multiplier)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.ema_period, 'ema_period')
        IndicatorValidator.validate_positive_float(self.multiplier, 'multiplier')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # ATR计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        # 中轨
        middle = close.ewm(span=self.ema_period, adjust=False).mean()

        # 上下轨
        upper = middle + self.multiplier * atr
        lower = middle - self.multiplier * atr

        # 通道位置
        position = (close - lower) / (upper - lower)

        return pd.DataFrame({
            'ATRC_Upper': upper,
            'ATRC_Middle': middle,
            'ATRC_Lower': lower,
            'ATRC_Position': position
        })


@indicator_registry.register('volatility')
class NormalizedATR(BaseIndicator):
    """
    Normalized ATR (标准化ATR)

    ATR的标准化版本，便于历史比较
    """

    _metadata = IndicatorMetadata(
        name='NormalizedATR',
        display_name='标准化ATR',
        category='volatility',
        description='ATR相对于历史的标准化值',
        formula='NATR = (ATR - ATR_MA) / ATR_STD',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14, 'lookback': 100},
        complexity_level=2
    )

    def __init__(self, period: int = 14, lookback: int = 100):
        self.period = period
        self.lookback = lookback
        super().__init__(period=period, lookback=lookback)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.lookback, 'lookback')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # ATR计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        # 标准化
        atr_ma = atr.rolling(window=self.lookback).mean()
        atr_std = atr.rolling(window=self.lookback).std()

        normalized_atr = (atr - atr_ma) / atr_std

        # 百分位排名
        def percentile_rank(x):
            return (x.rank() / len(x) * 100).iloc[-1]

        atr_percentile = atr.rolling(window=self.lookback).apply(percentile_rank, raw=False)

        return pd.DataFrame({
            'ATR': atr,
            'ATR_MA': atr_ma,
            'NATR': normalized_atr,
            'ATR_Percentile': atr_percentile
        })


@indicator_registry.register('volatility')
class ATRRatio(BaseIndicator):
    """
    ATR Ratio (ATR比率)

    短期ATR与长期ATR的比率
    """

    _metadata = IndicatorMetadata(
        name='ATRRatio',
        display_name='ATR比率',
        category='volatility',
        description='短期与长期ATR的比率，用于判断波动变化',
        formula='ATR_Ratio = ATR_Short / ATR_Long',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'short_period': 7, 'long_period': 21},
        complexity_level=2
    )

    def __init__(self, short_period: int = 7, long_period: int = 21):
        self.short_period = short_period
        self.long_period = long_period
        super().__init__(short_period=short_period, long_period=long_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.short_period, 'short_period')
        IndicatorValidator.validate_positive_int(self.long_period, 'long_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # TR计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 短期和长期ATR
        atr_short = tr.ewm(span=self.short_period, adjust=False).mean()
        atr_long = tr.ewm(span=self.long_period, adjust=False).mean()

        # 比率
        atr_ratio = atr_short / atr_long

        # 变化方向
        atr_change = atr_ratio.diff()

        return pd.DataFrame({
            'ATR_Short': atr_short,
            'ATR_Long': atr_long,
            'ATR_Ratio': atr_ratio,
            'ATR_Change': atr_change
        })
