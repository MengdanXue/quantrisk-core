"""
价格动量指标 (8个)

基于价格变化率的动量指标
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('momentum')
class Momentum(BaseIndicator):
    """
    Momentum Indicator (动量指标)

    当前价格与N周期前价格的差值
    """

    _metadata = IndicatorMetadata(
        name='Momentum',
        display_name='动量指标',
        category='momentum',
        description='当前价格与N周期前价格的差值',
        formula='Momentum = Close - Close[n]',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 10},
        complexity_level=1
    )

    def __init__(self, period: int = 10):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        return data['close'] - data['close'].shift(self.period)


@indicator_registry.register('momentum')
class ROC(BaseIndicator):
    """
    Rate of Change (变化率)

    当前价格相对于N周期前的百分比变化
    """

    _metadata = IndicatorMetadata(
        name='ROC',
        display_name='变化率',
        category='momentum',
        description='当前价格相对于N周期前的百分比变化',
        formula='ROC = (Close - Close[n]) / Close[n] * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 10},
        complexity_level=1
    )

    def __init__(self, period: int = 10):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        return (close - close.shift(self.period)) / close.shift(self.period) * 100


@indicator_registry.register('momentum')
class APO(BaseIndicator):
    """
    Absolute Price Oscillator (绝对价格振荡器)

    快慢EMA的绝对差值
    """

    _metadata = IndicatorMetadata(
        name='APO',
        display_name='绝对价格振荡器',
        category='momentum',
        description='快速EMA与慢速EMA的绝对差值',
        formula='APO = EMA(fast) - EMA(slow)',
        required_columns=['close'],
        output_type='series',
        default_params={'fast_period': 12, 'slow_period': 26},
        complexity_level=1
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(fast_period=fast_period, slow_period=slow_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        return fast_ema - slow_ema


@indicator_registry.register('momentum')
class CMO(BaseIndicator):
    """
    Chande Momentum Oscillator (钱德动量振荡器)

    Tushar Chande开发的动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='CMO',
        display_name='钱德动量振荡器',
        category='momentum',
        description='涨幅与跌幅的比率，范围-100到100',
        formula='CMO = (Sum(Gain) - Sum(Loss)) / (Sum(Gain) + Sum(Loss)) * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        change = close.diff()
        gain = change.where(change > 0, 0)
        loss = (-change.where(change < 0, 0))

        sum_gain = gain.rolling(window=self.period).sum()
        sum_loss = loss.rolling(window=self.period).sum()

        cmo = (sum_gain - sum_loss) / (sum_gain + sum_loss) * 100

        return cmo


@indicator_registry.register('momentum')
class InertiaIndicator(BaseIndicator):
    """
    Inertia Indicator (惯性指标)

    基于RVI的价格惯性衡量
    """

    _metadata = IndicatorMetadata(
        name='InertiaIndicator',
        display_name='惯性指标',
        category='momentum',
        description='使用线性回归斜率的RVI平滑版本',
        formula='Inertia = LinReg(RVI, n)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'rvi_period': 14},
        complexity_level=3
    )

    def __init__(self, period: int = 20, rvi_period: int = 14):
        self.period = period
        self.rvi_period = rvi_period
        super().__init__(period=period, rvi_period=rvi_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.rvi_period, 'rvi_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 简化的RVI计算
        change = close.diff()
        std = close.rolling(window=self.rvi_period).std()
        rvi = change.rolling(window=self.rvi_period).mean() / std

        # 对RVI进行线性回归
        def linreg(x):
            n = len(x)
            x_idx = np.arange(n)
            slope, intercept = np.polyfit(x_idx, x, 1)
            return intercept + slope * (n - 1)

        inertia = rvi.rolling(window=self.period).apply(linreg, raw=True)

        return inertia


@indicator_registry.register('momentum')
class ForecastOscillator(BaseIndicator):
    """
    Forecast Oscillator (预测振荡器)

    价格与线性回归预测值的偏差百分比
    """

    _metadata = IndicatorMetadata(
        name='ForecastOscillator',
        display_name='预测振荡器',
        category='momentum',
        description='价格与线性回归预测值的偏差',
        formula='FO = (Close - LinReg) / Close * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        def linreg(x):
            n = len(x)
            x_idx = np.arange(n)
            slope, intercept = np.polyfit(x_idx, x, 1)
            return intercept + slope * (n - 1)

        linreg_value = close.rolling(window=self.period).apply(linreg, raw=True)

        forecast_osc = (close - linreg_value) / close * 100

        return forecast_osc


@indicator_registry.register('momentum')
class LinearRegressionSlope(BaseIndicator):
    """
    Linear Regression Slope (线性回归斜率)

    滚动窗口内的线性回归斜率
    """

    _metadata = IndicatorMetadata(
        name='LinearRegressionSlope',
        display_name='线性回归斜率',
        category='momentum',
        description='衡量价格趋势的斜率',
        formula='Slope = LinReg斜率',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        def get_slope(x):
            n = len(x)
            x_idx = np.arange(n)
            slope, _ = np.polyfit(x_idx, x, 1)
            return slope

        return close.rolling(window=self.period).apply(get_slope, raw=True)


@indicator_registry.register('momentum')
class RSquared(BaseIndicator):
    """
    R-Squared (决定系数)

    衡量价格趋势的拟合程度
    """

    _metadata = IndicatorMetadata(
        name='RSquared',
        display_name='决定系数',
        category='momentum',
        description='线性回归的R²，衡量趋势强度和一致性',
        formula='R² = 相关系数的平方',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        def get_r_squared(y):
            n = len(y)
            x = np.arange(n)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return r_value ** 2

        return close.rolling(window=self.period).apply(get_r_squared, raw=True)
