"""
振荡器指标 (14个)

衡量价格动量和超买超卖状态的振荡器
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('momentum')
class RSI(BaseIndicator):
    """
    Relative Strength Index (相对强弱指标)

    J. Welles Wilder开发的经典动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='RSI',
        display_name='相对强弱指标',
        category='momentum',
        description='衡量价格涨跌速度和变化幅度，70以上超买，30以下超卖',
        formula='RSI = 100 - 100 / (1 + RS), RS = Avg Gain / Avg Loss',
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

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))

        avg_gain = gain.ewm(alpha=1/self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


@indicator_registry.register('momentum')
class StochasticOscillator(BaseIndicator):
    """
    Stochastic Oscillator (随机振荡器)

    George Lane开发的动量指标
    """

    _metadata = IndicatorMetadata(
        name='StochasticOscillator',
        display_name='随机振荡器',
        category='momentum',
        description='比较收盘价与价格区间的位置，识别反转',
        formula='%K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'k_period': 14, 'd_period': 3, 'smooth_k': 3},
        complexity_level=2
    )

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        super().__init__(k_period=k_period, d_period=d_period, smooth_k=smooth_k)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.k_period, 'k_period')
        IndicatorValidator.validate_positive_int(self.d_period, 'd_period')
        IndicatorValidator.validate_positive_int(self.smooth_k, 'smooth_k')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()

        # Fast %K
        fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # Slow %K (smoothed)
        slow_k = fast_k.rolling(window=self.smooth_k).mean()

        # %D
        d = slow_k.rolling(window=self.d_period).mean()

        return pd.DataFrame({
            'Stoch_K': slow_k,
            'Stoch_D': d,
            'Fast_K': fast_k
        })


@indicator_registry.register('momentum')
class StochasticRSI(BaseIndicator):
    """
    Stochastic RSI (随机RSI)

    将随机振荡器应用于RSI的值
    """

    _metadata = IndicatorMetadata(
        name='StochasticRSI',
        display_name='随机RSI',
        category='momentum',
        description='RSI的随机振荡器版本，更敏感',
        formula='StochRSI = (RSI - LowestRSI) / (HighestRSI - LowestRSI)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'rsi_period': 14, 'stoch_period': 14, 'k_smooth': 3, 'd_smooth': 3},
        complexity_level=3
    )

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14,
                 k_smooth: int = 3, d_smooth: int = 3):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        super().__init__(rsi_period=rsi_period, stoch_period=stoch_period,
                        k_smooth=k_smooth, d_smooth=d_smooth)

    def _validate_params(self):
        for p in [self.rsi_period, self.stoch_period, self.k_smooth, self.d_smooth]:
            IndicatorValidator.validate_positive_int(p, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 先计算RSI
        rsi = RSI(self.rsi_period).calculate(data)

        # 对RSI应用随机振荡器
        lowest_rsi = rsi.rolling(window=self.stoch_period).min()
        highest_rsi = rsi.rolling(window=self.stoch_period).max()

        stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)

        # %K and %D
        k = stoch_rsi.rolling(window=self.k_smooth).mean() * 100
        d = k.rolling(window=self.d_smooth).mean()

        return pd.DataFrame({
            'StochRSI_K': k,
            'StochRSI_D': d,
            'StochRSI': stoch_rsi * 100
        })


@indicator_registry.register('momentum')
class CCI(BaseIndicator):
    """
    Commodity Channel Index (商品通道指数)

    Donald Lambert开发的动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='CCI',
        display_name='商品通道指数',
        category='momentum',
        description='衡量价格与统计平均价的偏离程度',
        formula='CCI = (TP - SMA(TP)) / (0.015 * MAD)',
        required_columns=['high', 'low', 'close'],
        output_type='series',
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
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 典型价格
        tp = (data['high'] + data['low'] + data['close']) / 3

        # SMA of TP
        sma_tp = tp.rolling(window=self.period).mean()

        # Mean Absolute Deviation
        mad = tp.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        # CCI
        cci = (tp - sma_tp) / (0.015 * mad)

        return cci


@indicator_registry.register('momentum')
class WilliamsR(BaseIndicator):
    """
    Williams %R (威廉指标)

    Larry Williams开发的超买超卖振荡器
    """

    _metadata = IndicatorMetadata(
        name='WilliamsR',
        display_name='威廉指标',
        category='momentum',
        description='衡量收盘价在周期高低范围内的位置，-20以上超买，-80以下超卖',
        formula='%R = (Highest - Close) / (Highest - Lowest) * -100',
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

        highest = data['high'].rolling(window=self.period).max()
        lowest = data['low'].rolling(window=self.period).min()

        wr = -100 * (highest - data['close']) / (highest - lowest)

        return wr


@indicator_registry.register('momentum')
class UltimateOscillator(BaseIndicator):
    """
    Ultimate Oscillator (终极振荡器)

    Larry Williams开发的多周期动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='UltimateOscillator',
        display_name='终极振荡器',
        category='momentum',
        description='结合三个周期的买入压力，减少虚假信号',
        formula='UO = 100 * (4*BP7/TR7 + 2*BP14/TR14 + BP28/TR28) / 7',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'period1': 7, 'period2': 14, 'period3': 28},
        complexity_level=3
    )

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        super().__init__(period1=period1, period2=period2, period3=period3)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period1, 'period1')
        IndicatorValidator.validate_positive_int(self.period2, 'period2')
        IndicatorValidator.validate_positive_int(self.period3, 'period3')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Buying Pressure
        prior_close = close.shift(1)
        bp = close - pd.concat([low, prior_close], axis=1).min(axis=1)

        # True Range
        tr1 = high - low
        tr2 = abs(high - prior_close)
        tr3 = abs(low - prior_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Averages
        avg1 = bp.rolling(window=self.period1).sum() / tr.rolling(window=self.period1).sum()
        avg2 = bp.rolling(window=self.period2).sum() / tr.rolling(window=self.period2).sum()
        avg3 = bp.rolling(window=self.period3).sum() / tr.rolling(window=self.period3).sum()

        # Ultimate Oscillator
        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return uo


@indicator_registry.register('momentum')
class AwesomeOscillator(BaseIndicator):
    """
    Awesome Oscillator (动量振荡器)

    Bill Williams开发的简单动量指标
    """

    _metadata = IndicatorMetadata(
        name='AwesomeOscillator',
        display_name='动量振荡器',
        category='momentum',
        description='中间价的5期SMA与34期SMA的差值',
        formula='AO = SMA(HL/2, 5) - SMA(HL/2, 34)',
        required_columns=['high', 'low'],
        output_type='series',
        default_params={'fast_period': 5, 'slow_period': 34},
        complexity_level=1
    )

    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(fast_period=fast_period, slow_period=slow_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        median_price = (data['high'] + data['low']) / 2

        fast_sma = median_price.rolling(window=self.fast_period).mean()
        slow_sma = median_price.rolling(window=self.slow_period).mean()

        return fast_sma - slow_sma


@indicator_registry.register('momentum')
class AcceleratorOscillator(BaseIndicator):
    """
    Accelerator Oscillator (加速振荡器)

    Bill Williams开发的AO加速度指标
    """

    _metadata = IndicatorMetadata(
        name='AcceleratorOscillator',
        display_name='加速振荡器',
        category='momentum',
        description='AO与其5期SMA的差值，衡量动量变化速度',
        formula='AC = AO - SMA(AO, 5)',
        required_columns=['high', 'low'],
        output_type='series',
        default_params={'ao_fast': 5, 'ao_slow': 34, 'ac_period': 5},
        complexity_level=2
    )

    def __init__(self, ao_fast: int = 5, ao_slow: int = 34, ac_period: int = 5):
        self.ao_fast = ao_fast
        self.ao_slow = ao_slow
        self.ac_period = ac_period
        super().__init__(ao_fast=ao_fast, ao_slow=ao_slow, ac_period=ac_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.ao_fast, 'ao_fast')
        IndicatorValidator.validate_positive_int(self.ao_slow, 'ao_slow')
        IndicatorValidator.validate_positive_int(self.ac_period, 'ac_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        ao = AwesomeOscillator(self.ao_fast, self.ao_slow).calculate(data)
        ao_sma = ao.rolling(window=self.ac_period).mean()

        return ao - ao_sma


@indicator_registry.register('momentum')
class PrettyGoodOscillator(BaseIndicator):
    """
    Pretty Good Oscillator (优质振荡器)

    使用ATR标准化的价格变化
    """

    _metadata = IndicatorMetadata(
        name='PrettyGoodOscillator',
        display_name='优质振荡器',
        category='momentum',
        description='使用ATR标准化的价格与SMA的偏离',
        formula='PGO = (Close - SMA) / EMA(ATR)',
        required_columns=['high', 'low', 'close'],
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
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        close = data['close']
        high = data['high']
        low = data['low']

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        # SMA
        sma = close.rolling(window=self.period).mean()

        # PGO
        pgo = (close - sma) / atr

        return pgo


@indicator_registry.register('momentum')
class TSI(BaseIndicator):
    """
    True Strength Index (真实强度指数)

    双重平滑的动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='TSI',
        display_name='真实强度指数',
        category='momentum',
        description='双重平滑的价格变化比率',
        formula='TSI = 100 * EMA(EMA(PC, r), s) / EMA(EMA(|PC|, r), s)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_period': 13, 'slow_period': 25, 'signal_period': 13},
        complexity_level=3
    )

    def __init__(self, fast_period: int = 13, slow_period: int = 25, signal_period: int = 13):
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

        # Price Change
        pc = close.diff()

        # Double smoothed PC
        pc_ema1 = pc.ewm(span=self.slow_period, adjust=False).mean()
        pc_ema2 = pc_ema1.ewm(span=self.fast_period, adjust=False).mean()

        # Double smoothed |PC|
        apc_ema1 = pc.abs().ewm(span=self.slow_period, adjust=False).mean()
        apc_ema2 = apc_ema1.ewm(span=self.fast_period, adjust=False).mean()

        # TSI
        tsi = 100 * pc_ema2 / apc_ema2

        # Signal
        tsi_signal = tsi.ewm(span=self.signal_period, adjust=False).mean()

        return pd.DataFrame({
            'TSI': tsi,
            'TSI_Signal': tsi_signal
        })


@indicator_registry.register('momentum')
class ConnorsRSI(BaseIndicator):
    """
    Connors RSI (康纳斯RSI)

    Larry Connors开发的复合RSI指标
    """

    _metadata = IndicatorMetadata(
        name='ConnorsRSI',
        display_name='康纳斯RSI',
        category='momentum',
        description='结合价格RSI、连续涨跌RSI和ROC百分位的复合指标',
        formula='CRSI = (RSI + Streak RSI + ROC Percentile) / 3',
        required_columns=['close'],
        output_type='series',
        default_params={'rsi_period': 3, 'streak_period': 2, 'roc_period': 100},
        complexity_level=4
    )

    def __init__(self, rsi_period: int = 3, streak_period: int = 2, roc_period: int = 100):
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.roc_period = roc_period
        super().__init__(rsi_period=rsi_period, streak_period=streak_period,
                        roc_period=roc_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.rsi_period, 'rsi_period')
        IndicatorValidator.validate_positive_int(self.streak_period, 'streak_period')
        IndicatorValidator.validate_positive_int(self.roc_period, 'roc_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 1. Standard RSI
        rsi = RSI(self.rsi_period).calculate(data)

        # 2. Streak RSI (连续涨跌天数的RSI)
        change = close.diff()
        streak = pd.Series(0, index=close.index)

        for i in range(1, len(close)):
            if change.iloc[i] > 0:
                streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] > 0 else 1
            elif change.iloc[i] < 0:
                streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] < 0 else -1
            else:
                streak.iloc[i] = 0

        # 计算streak的RSI
        streak_delta = streak.diff()
        streak_gain = streak_delta.where(streak_delta > 0, 0)
        streak_loss = (-streak_delta.where(streak_delta < 0, 0))
        streak_avg_gain = streak_gain.ewm(alpha=1/self.streak_period, adjust=False).mean()
        streak_avg_loss = streak_loss.ewm(alpha=1/self.streak_period, adjust=False).mean()
        streak_rs = streak_avg_gain / streak_avg_loss
        streak_rsi = 100 - (100 / (1 + streak_rs))

        # 3. ROC Percentile
        roc = (close - close.shift(1)) / close.shift(1) * 100

        def percentile_rank(x):
            return (x < x.iloc[-1]).sum() / (len(x) - 1) * 100

        roc_percentile = roc.rolling(window=self.roc_period).apply(percentile_rank, raw=False)

        # Connors RSI
        crsi = (rsi + streak_rsi + roc_percentile) / 3

        return crsi


@indicator_registry.register('momentum')
class LaguerreRSI(BaseIndicator):
    """
    Laguerre RSI (拉盖尔RSI)

    使用Laguerre滤波器的RSI变体
    """

    _metadata = IndicatorMetadata(
        name='LaguerreRSI',
        display_name='拉盖尔RSI',
        category='momentum',
        description='使用Laguerre多项式滤波的RSI，减少噪音',
        formula='基于Laguerre滤波器的RSI计算',
        required_columns=['close'],
        output_type='series',
        default_params={'gamma': 0.5},
        complexity_level=4
    )

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma
        super().__init__(gamma=gamma)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.gamma, 'gamma', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close'].values
        n = len(close)

        L0 = np.zeros(n)
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        L3 = np.zeros(n)
        lrsi = np.zeros(n)

        for i in range(1, n):
            L0[i] = (1 - self.gamma) * close[i] + self.gamma * L0[i-1]
            L1[i] = -self.gamma * L0[i] + L0[i-1] + self.gamma * L1[i-1]
            L2[i] = -self.gamma * L1[i] + L1[i-1] + self.gamma * L2[i-1]
            L3[i] = -self.gamma * L2[i] + L2[i-1] + self.gamma * L3[i-1]

            cu = 0
            cd = 0

            if L0[i] >= L1[i]:
                cu += L0[i] - L1[i]
            else:
                cd += L1[i] - L0[i]

            if L1[i] >= L2[i]:
                cu += L1[i] - L2[i]
            else:
                cd += L2[i] - L1[i]

            if L2[i] >= L3[i]:
                cu += L2[i] - L3[i]
            else:
                cd += L3[i] - L2[i]

            if cu + cd != 0:
                lrsi[i] = 100 * cu / (cu + cd)
            else:
                lrsi[i] = 50

        return pd.Series(lrsi, index=data.index)


@indicator_registry.register('momentum')
class SMI(BaseIndicator):
    """
    Stochastic Momentum Index (随机动量指数)

    更精确的随机振荡器变体
    """

    _metadata = IndicatorMetadata(
        name='SMI',
        display_name='随机动量指数',
        category='momentum',
        description='衡量收盘价相对于高低中点的位置',
        formula='SMI = EMA(EMA(Close - HL/2, r), s) / EMA(EMA(H-L, r), s) * 100',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'k_period': 5, 'd_period': 3, 'ema_period': 3},
        complexity_level=3
    )

    def __init__(self, k_period: int = 5, d_period: int = 3, ema_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.ema_period = ema_period
        super().__init__(k_period=k_period, d_period=d_period, ema_period=ema_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.k_period, 'k_period')
        IndicatorValidator.validate_positive_int(self.d_period, 'd_period')
        IndicatorValidator.validate_positive_int(self.ema_period, 'ema_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # 周期内的最高和最低
        highest = high.rolling(window=self.k_period).max()
        lowest = low.rolling(window=self.k_period).min()

        # 中点
        mid = (highest + lowest) / 2

        # D = Close - Mid
        d = close - mid

        # HL Range
        hl_range = highest - lowest

        # 双重EMA平滑
        d_ema1 = d.ewm(span=self.d_period, adjust=False).mean()
        d_ema2 = d_ema1.ewm(span=self.ema_period, adjust=False).mean()

        hl_ema1 = hl_range.ewm(span=self.d_period, adjust=False).mean()
        hl_ema2 = hl_ema1.ewm(span=self.ema_period, adjust=False).mean()

        # SMI
        smi = 100 * d_ema2 / (hl_ema2 / 2)

        # Signal
        smi_signal = smi.ewm(span=self.ema_period, adjust=False).mean()

        return pd.DataFrame({
            'SMI': smi,
            'SMI_Signal': smi_signal
        })


@indicator_registry.register('momentum')
class RVI(BaseIndicator):
    """
    Relative Vigor Index (相对活力指数)

    衡量价格收盘趋势的强度
    """

    _metadata = IndicatorMetadata(
        name='RVI',
        display_name='相对活力指数',
        category='momentum',
        description='比较收盘价与开盘价的关系，衡量上涨动能',
        formula='RVI = SMA(Close-Open, n) / SMA(High-Low, n)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 10, 'signal_period': 4},
        complexity_level=2
    )

    def __init__(self, period: int = 10, signal_period: int = 4):
        self.period = period
        self.signal_period = signal_period
        super().__init__(period=period, signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        # 收盘-开盘
        co = close - open_price

        # 高-低
        hl = high - low

        # 对称加权
        co_weighted = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6
        hl_weighted = (hl + 2 * hl.shift(1) + 2 * hl.shift(2) + hl.shift(3)) / 6

        # RVI
        rvi_num = co_weighted.rolling(window=self.period).sum()
        rvi_den = hl_weighted.rolling(window=self.period).sum()
        rvi = rvi_num / rvi_den

        # Signal
        rvi_signal = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6

        return pd.DataFrame({
            'RVI': rvi,
            'RVI_Signal': rvi_signal
        })
