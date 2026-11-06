"""
趋势通道指标 (5个)

用于识别价格运行的通道和回归趋势
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('trend')
class DonchianChannel(BaseIndicator):
    """
    Donchian Channel (唐奇安通道)

    Richard Donchian开发的经典突破交易通道
    """

    _metadata = IndicatorMetadata(
        name='DonchianChannel',
        display_name='唐奇安通道',
        category='trend',
        description='基于最高价和最低价的突破通道',
        formula='Upper = Highest(High, n), Lower = Lowest(Low, n)',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=1
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']

        upper = high.rolling(window=self.period).max()
        lower = low.rolling(window=self.period).min()
        middle = (upper + lower) / 2

        # 通道宽度（百分比）
        width = (upper - lower) / middle * 100

        return pd.DataFrame({
            'DC_Upper': upper,
            'DC_Middle': middle,
            'DC_Lower': lower,
            'DC_Width': width
        })


@indicator_registry.register('trend')
class KeltnerChannel(BaseIndicator):
    """
    Keltner Channel (肯特纳通道)

    基于EMA和ATR的波动性通道
    """

    _metadata = IndicatorMetadata(
        name='KeltnerChannel',
        display_name='肯特纳通道',
        category='trend',
        description='使用EMA作为中轨，ATR作为通道宽度',
        formula='Middle = EMA(Close, n), Upper/Lower = Middle ± Multiplier * ATR',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 20, 'atr_period': 10, 'multiplier': 2.0},
        complexity_level=2
    )

    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        super().__init__(period=period, atr_period=atr_period, multiplier=multiplier)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.atr_period, 'atr_period')
        IndicatorValidator.validate_positive_float(self.multiplier, 'multiplier')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # 中轨 = EMA
        middle = close.ewm(span=self.period, adjust=False).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        # 上下轨
        upper = middle + self.multiplier * atr
        lower = middle - self.multiplier * atr

        # 通道宽度
        width = (upper - lower) / middle * 100

        return pd.DataFrame({
            'KC_Upper': upper,
            'KC_Middle': middle,
            'KC_Lower': lower,
            'KC_Width': width
        })


@indicator_registry.register('trend')
class EnvelopeChannel(BaseIndicator):
    """
    Envelope Channel (包络通道)

    基于移动平均线的固定百分比通道
    """

    _metadata = IndicatorMetadata(
        name='EnvelopeChannel',
        display_name='包络通道',
        category='trend',
        description='移动平均线上下固定百分比的通道',
        formula='Upper = MA * (1 + percent), Lower = MA * (1 - percent)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'percent': 2.5, 'ma_type': 'sma'},
        complexity_level=1
    )

    def __init__(self, period: int = 20, percent: float = 2.5, ma_type: str = 'sma'):
        self.period = period
        self.percent = percent
        self.ma_type = ma_type
        super().__init__(period=period, percent=percent, ma_type=ma_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.percent, 'percent')
        IndicatorValidator.validate_in_list(self.ma_type, 'ma_type', ['sma', 'ema'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 计算移动平均
        if self.ma_type == 'sma':
            middle = close.rolling(window=self.period).mean()
        else:  # ema
            middle = close.ewm(span=self.period, adjust=False).mean()

        # 计算上下轨
        percent_factor = self.percent / 100
        upper = middle * (1 + percent_factor)
        lower = middle * (1 - percent_factor)

        return pd.DataFrame({
            'ENV_Upper': upper,
            'ENV_Middle': middle,
            'ENV_Lower': lower
        })


@indicator_registry.register('trend')
class RegressionChannel(BaseIndicator):
    """
    Regression Channel (回归通道)

    基于线性回归的标准差通道
    """

    _metadata = IndicatorMetadata(
        name='RegressionChannel',
        display_name='回归通道',
        category='trend',
        description='使用线性回归和标准差构建的通道',
        formula='Middle = Linear Regression, Upper/Lower = Middle ± k * StdDev',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'std_dev': 2.0},
        complexity_level=3
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

        middle = pd.Series(index=close.index, dtype=float)
        upper = pd.Series(index=close.index, dtype=float)
        lower = pd.Series(index=close.index, dtype=float)
        slope_series = pd.Series(index=close.index, dtype=float)
        r_squared = pd.Series(index=close.index, dtype=float)

        for i in range(self.period - 1, len(close)):
            y = close.iloc[i - self.period + 1:i + 1].values
            x = np.arange(self.period)

            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # 回归线在当前点的值
            reg_value = intercept + slope * (self.period - 1)

            # 残差的标准差
            y_pred = intercept + slope * x
            residuals = y - y_pred
            std = np.std(residuals, ddof=1)

            middle.iloc[i] = reg_value
            upper.iloc[i] = reg_value + self.std_dev * std
            lower.iloc[i] = reg_value - self.std_dev * std
            slope_series.iloc[i] = slope
            r_squared.iloc[i] = r_value ** 2

        return pd.DataFrame({
            'RC_Upper': upper,
            'RC_Middle': middle,
            'RC_Lower': lower,
            'RC_Slope': slope_series,
            'RC_R2': r_squared
        })


@indicator_registry.register('trend')
class LinearRegressionLine(BaseIndicator):
    """
    Linear Regression Line (线性回归线)

    在滚动窗口内的线性回归值
    """

    _metadata = IndicatorMetadata(
        name='LinearRegressionLine',
        display_name='线性回归线',
        category='trend',
        description='滚动窗口内的线性回归端点值',
        formula='LR = intercept + slope * (n-1)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        lr_value = pd.Series(index=close.index, dtype=float)
        lr_slope = pd.Series(index=close.index, dtype=float)
        lr_intercept = pd.Series(index=close.index, dtype=float)
        lr_angle = pd.Series(index=close.index, dtype=float)

        x = np.arange(self.period)

        for i in range(self.period - 1, len(close)):
            y = close.iloc[i - self.period + 1:i + 1].values

            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            lr_value.iloc[i] = intercept + slope * (self.period - 1)
            lr_slope.iloc[i] = slope
            lr_intercept.iloc[i] = intercept
            lr_angle.iloc[i] = np.degrees(np.arctan(slope))

        return pd.DataFrame({
            'LR_Value': lr_value,
            'LR_Slope': lr_slope,
            'LR_Intercept': lr_intercept,
            'LR_Angle': lr_angle
        })
