"""
MACD系列指标 (6个)

各种MACD变体和增强版本
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('momentum')
class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (移动平均收敛发散)

    Gerald Appel开发的经典趋势动量指标
    """

    _metadata = IndicatorMetadata(
        name='MACD',
        display_name='MACD',
        category='momentum',
        description='快速EMA与慢速EMA的差值，识别趋势变化',
        formula='MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        complexity_level=2
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal

        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Histogram': histogram
        })


@indicator_registry.register('momentum')
class MACDHistogram(BaseIndicator):
    """
    MACD Histogram (MACD柱状图)

    MACD与信号线的差值
    """

    _metadata = IndicatorMetadata(
        name='MACDHistogram',
        display_name='MACD柱状图',
        category='momentum',
        description='MACD与信号线的差值，显示动量变化',
        formula='Histogram = MACD - Signal',
        required_columns=['close'],
        output_type='series',
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        complexity_level=2
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        macd_data = MACD(self.fast_period, self.slow_period, self.signal_period).calculate(data)
        return macd_data['MACD_Histogram']


@indicator_registry.register('momentum')
class MACDZeroLag(BaseIndicator):
    """
    Zero Lag MACD (零滞后MACD)

    使用ZLEMA计算的MACD，减少滞后
    """

    _metadata = IndicatorMetadata(
        name='MACDZeroLag',
        display_name='零滞后MACD',
        category='momentum',
        description='使用零滞后EMA计算的MACD，更快响应',
        formula='ZLMACD = ZLEMA(12) - ZLEMA(26)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        complexity_level=3
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def _zlema(self, series: pd.Series, period: int) -> pd.Series:
        """计算零滞后EMA"""
        lag = int((period - 1) / 2)
        ema_data = 2 * series - series.shift(lag)
        return ema_data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        fast_zlema = self._zlema(close, self.fast_period)
        slow_zlema = self._zlema(close, self.slow_period)

        macd = fast_zlema - slow_zlema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal

        return pd.DataFrame({
            'ZLMACD': macd,
            'ZLMACD_Signal': signal,
            'ZLMACD_Histogram': histogram
        })


@indicator_registry.register('momentum')
class MACDLeader(BaseIndicator):
    """
    MACD Leader (MACD领先指标)

    预测MACD的领先信号
    """

    _metadata = IndicatorMetadata(
        name='MACDLeader',
        display_name='MACD领先指标',
        category='momentum',
        description='提前预测MACD穿越的领先指标',
        formula='Leader = MACD + MACD偏离度',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        complexity_level=3
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 标准MACD
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        # 计算偏离度
        fast_diff = close - fast_ema
        slow_diff = close - slow_ema

        # Leader = MACD + 偏离度差
        leader = macd + (fast_diff - slow_diff)

        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Leader': leader
        })


@indicator_registry.register('momentum')
class MACDBB(BaseIndicator):
    """
    MACD Bollinger Bands (MACD布林带)

    在MACD上叠加布林带
    """

    _metadata = IndicatorMetadata(
        name='MACDBB',
        display_name='MACD布林带',
        category='momentum',
        description='MACD的布林带通道，识别极端值',
        formula='BB = MACD ± k * StdDev(MACD)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 12, 'slow_period': 26, 'bb_period': 10, 'std_dev': 2.0},
        complexity_level=3
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 bb_period: int = 10, std_dev: float = 2.0):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.bb_period = bb_period
        self.std_dev = std_dev
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        bb_period=bb_period, std_dev=std_dev)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.bb_period, 'bb_period')
        IndicatorValidator.validate_positive_float(self.std_dev, 'std_dev')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # MACD
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema

        # 布林带
        macd_sma = macd.rolling(window=self.bb_period).mean()
        macd_std = macd.rolling(window=self.bb_period).std()

        upper = macd_sma + self.std_dev * macd_std
        lower = macd_sma - self.std_dev * macd_std

        # %B
        percent_b = (macd - lower) / (upper - lower) * 100

        return pd.DataFrame({
            'MACD': macd,
            'MACDBB_Upper': upper,
            'MACDBB_Middle': macd_sma,
            'MACDBB_Lower': lower,
            'MACDBB_PercentB': percent_b
        })


@indicator_registry.register('momentum')
class PPOMACD(BaseIndicator):
    """
    Percentage Price Oscillator MACD (百分比价格振荡器)

    百分比形式的MACD
    """

    _metadata = IndicatorMetadata(
        name='PPOMACD',
        display_name='百分比价格振荡器',
        category='momentum',
        description='百分比形式的MACD，便于跨标的比较',
        formula='PPO = (EMA12 - EMA26) / EMA26 * 100',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        complexity_level=2
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                        signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # PPO = 百分比差异
        ppo = (fast_ema - slow_ema) / slow_ema * 100
        signal = ppo.ewm(span=self.signal_period, adjust=False).mean()
        histogram = ppo - signal

        return pd.DataFrame({
            'PPO': ppo,
            'PPO_Signal': signal,
            'PPO_Histogram': histogram
        })
