"""
趋势强度指标 (14个)

用于衡量趋势的强度、方向和持续性
"""

from typing import List, Union, Dict
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('trend')
class ADX(BaseIndicator):
    """
    Average Directional Index (平均趋向指数)

    衡量趋势强度的经典指标，由J. Welles Wilder开发
    """

    _metadata = IndicatorMetadata(
        name='ADX',
        display_name='平均趋向指数',
        category='trend',
        description='衡量趋势强度，不分方向。ADX>25表示强趋势',
        formula='ADX = EMA(|+DI - -DI| / (+DI + -DI), n)',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14},
        complexity_level=3
    )

    def __init__(self, period: int = 14):
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
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()

        # +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # +DI and -DI
        plus_di = 100 * (plus_dm.ewm(alpha=1/self.period, adjust=False).mean() /
                        atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/self.period, adjust=False).mean() /
                         atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()

        return pd.DataFrame({
            'ADX': adx,
            'DI_Plus': plus_di,
            'DI_Minus': minus_di,
            'DX': dx
        })


@indicator_registry.register('trend')
class Aroon(BaseIndicator):
    """
    Aroon Indicator (阿隆指标)

    衡量趋势开始时间的指标
    """

    _metadata = IndicatorMetadata(
        name='Aroon',
        display_name='阿隆指标',
        category='trend',
        description='衡量距离最高/最低价的周期数，判断趋势开始',
        formula='Aroon Up = ((n - 距最高价周期数) / n) * 100',
        required_columns=['high', 'low'],
        output_type='dataframe',
        default_params={'period': 25},
        complexity_level=2
    )

    def __init__(self, period: int = 25):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']

        def periods_since_high(x):
            return self.period - x.argmax()

        def periods_since_low(x):
            return self.period - x.argmin()

        aroon_up = high.rolling(window=self.period + 1).apply(
            periods_since_high, raw=True) / self.period * 100
        aroon_down = low.rolling(window=self.period + 1).apply(
            periods_since_low, raw=True) / self.period * 100

        return pd.DataFrame({
            'Aroon_Up': aroon_up,
            'Aroon_Down': aroon_down
        })


@indicator_registry.register('trend')
class AroonOscillator(BaseIndicator):
    """
    Aroon Oscillator (阿隆震荡器)

    Aroon Up和Aroon Down的差值
    """

    _metadata = IndicatorMetadata(
        name='AroonOscillator',
        display_name='阿隆震荡器',
        category='trend',
        description='Aroon Up - Aroon Down，正值表示上升趋势',
        formula='Aroon Osc = Aroon Up - Aroon Down',
        required_columns=['high', 'low'],
        output_type='series',
        default_params={'period': 25},
        complexity_level=2
    )

    def __init__(self, period: int = 25):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        aroon = Aroon(self.period).calculate(data)
        return aroon['Aroon_Up'] - aroon['Aroon_Down']


@indicator_registry.register('trend')
class DI_Plus(BaseIndicator):
    """
    Positive Directional Indicator (+DI)

    正向趋势指标
    """

    _metadata = IndicatorMetadata(
        name='DI_Plus',
        display_name='正向趋向指标',
        category='trend',
        description='衡量上升趋势的强度',
        formula='+DI = 100 * EMA(+DM, n) / ATR',
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
        adx_data = ADX(self.period).calculate(data)
        return adx_data['DI_Plus']


@indicator_registry.register('trend')
class DI_Minus(BaseIndicator):
    """
    Negative Directional Indicator (-DI)

    负向趋势指标
    """

    _metadata = IndicatorMetadata(
        name='DI_Minus',
        display_name='负向趋向指标',
        category='trend',
        description='衡量下降趋势的强度',
        formula='-DI = 100 * EMA(-DM, n) / ATR',
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
        adx_data = ADX(self.period).calculate(data)
        return adx_data['DI_Minus']


@indicator_registry.register('trend')
class DMI(BaseIndicator):
    """
    Directional Movement Index (趋向运动指数)

    综合ADX、+DI、-DI的完整趋向系统
    """

    _metadata = IndicatorMetadata(
        name='DMI',
        display_name='趋向运动系统',
        category='trend',
        description='Wilder的完整趋向运动系统',
        formula='DMI = {ADX, +DI, -DI, DX}',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14},
        complexity_level=3
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return ADX(self.period).calculate(data)


@indicator_registry.register('trend')
class VortexIndicator(BaseIndicator):
    """
    Vortex Indicator (涡旋指标)

    识别趋势开始、结束和反转的指标
    """

    _metadata = IndicatorMetadata(
        name='VortexIndicator',
        display_name='涡旋指标',
        category='trend',
        description='使用真实波幅和方向运动识别趋势',
        formula='VI+ = Sum(VM+, n) / Sum(TR, n)',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
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

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Vortex Movement
        vm_plus = abs(high - low.shift(1))
        vm_minus = abs(low - high.shift(1))

        # 累积和
        tr_sum = tr.rolling(window=self.period).sum()
        vm_plus_sum = vm_plus.rolling(window=self.period).sum()
        vm_minus_sum = vm_minus.rolling(window=self.period).sum()

        # Vortex Indicator
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum

        return pd.DataFrame({
            'VI_Plus': vi_plus,
            'VI_Minus': vi_minus
        })


@indicator_registry.register('trend')
class MassIndex(BaseIndicator):
    """
    Mass Index (质量指数)

    识别趋势反转的指标
    """

    _metadata = IndicatorMetadata(
        name='MassIndex',
        display_name='质量指数',
        category='trend',
        description='通过价格区间波动识别趋势反转，>27表示"膨胀"',
        formula='MI = Sum(EMA(H-L, 9) / EMA(EMA(H-L, 9), 9), 25)',
        required_columns=['high', 'low'],
        output_type='series',
        default_params={'period': 25, 'ema_period': 9},
        complexity_level=3
    )

    def __init__(self, period: int = 25, ema_period: int = 9):
        self.period = period
        self.ema_period = ema_period
        super().__init__(period=period, ema_period=ema_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.ema_period, 'ema_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        hl_range = data['high'] - data['low']
        ema1 = hl_range.ewm(span=self.ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.ema_period, adjust=False).mean()

        ratio = ema1 / ema2
        return ratio.rolling(window=self.period).sum()


@indicator_registry.register('trend')
class DPO(BaseIndicator):
    """
    Detrended Price Oscillator (去趋势价格震荡器)

    消除长期趋势，突出短期周期
    """

    _metadata = IndicatorMetadata(
        name='DPO',
        display_name='去趋势价格震荡器',
        category='trend',
        description='移除趋势以识别价格周期',
        formula='DPO = Close - SMA(Close, n)[n/2 + 1 periods ago]',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        close = data['close']
        sma = close.rolling(window=self.period).mean()

        shift = int(self.period / 2) + 1
        return close - sma.shift(shift)


@indicator_registry.register('trend')
class KST(BaseIndicator):
    """
    Know Sure Thing (确然指标)

    Martin Pring开发的多周期动量振荡器
    """

    _metadata = IndicatorMetadata(
        name='KST',
        display_name='确然指标',
        category='trend',
        description='多周期ROC的加权平均，识别主要趋势',
        formula='KST = w1*SMROC1 + w2*SMROC2 + w3*SMROC3 + w4*SMROC4',
        required_columns=['close'],
        output_type='dataframe',
        default_params={
            'roc1': 10, 'roc2': 15, 'roc3': 20, 'roc4': 30,
            'sma1': 10, 'sma2': 10, 'sma3': 10, 'sma4': 15,
            'signal': 9
        },
        complexity_level=3
    )

    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                 sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15,
                 signal: int = 9):
        self.roc1, self.roc2, self.roc3, self.roc4 = roc1, roc2, roc3, roc4
        self.sma1, self.sma2, self.sma3, self.sma4 = sma1, sma2, sma3, sma4
        self.signal = signal
        super().__init__(
            roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4,
            sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4,
            signal=signal
        )

    def _validate_params(self):
        for p in [self.roc1, self.roc2, self.roc3, self.roc4,
                  self.sma1, self.sma2, self.sma3, self.sma4, self.signal]:
            IndicatorValidator.validate_positive_int(p, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 计算ROC
        roc_1 = ((close - close.shift(self.roc1)) / close.shift(self.roc1)) * 100
        roc_2 = ((close - close.shift(self.roc2)) / close.shift(self.roc2)) * 100
        roc_3 = ((close - close.shift(self.roc3)) / close.shift(self.roc3)) * 100
        roc_4 = ((close - close.shift(self.roc4)) / close.shift(self.roc4)) * 100

        # 平滑ROC
        smroc_1 = roc_1.rolling(window=self.sma1).mean()
        smroc_2 = roc_2.rolling(window=self.sma2).mean()
        smroc_3 = roc_3.rolling(window=self.sma3).mean()
        smroc_4 = roc_4.rolling(window=self.sma4).mean()

        # KST
        kst = smroc_1 * 1 + smroc_2 * 2 + smroc_3 * 3 + smroc_4 * 4

        # Signal
        kst_signal = kst.rolling(window=self.signal).mean()

        return pd.DataFrame({
            'KST': kst,
            'KST_Signal': kst_signal
        })


@indicator_registry.register('trend')
class TRIX(BaseIndicator):
    """
    Triple Exponential Average (三重指数平均)

    三重平滑EMA的变化率
    """

    _metadata = IndicatorMetadata(
        name='TRIX',
        display_name='三重指数平均',
        category='trend',
        description='三重平滑EMA的变化率，过滤噪音',
        formula='TRIX = 100 * (EMA³ - EMA³[1]) / EMA³[1]',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 15, 'signal': 9},
        complexity_level=2
    )

    def __init__(self, period: int = 15, signal: int = 9):
        self.period = period
        self.signal = signal
        super().__init__(period=period, signal=signal)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.signal, 'signal')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 三重EMA
        ema1 = close.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()

        # TRIX = ROC of EMA³
        trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)

        # Signal
        trix_signal = trix.ewm(span=self.signal, adjust=False).mean()

        return pd.DataFrame({
            'TRIX': trix,
            'TRIX_Signal': trix_signal
        })


@indicator_registry.register('trend')
class Supertrend(BaseIndicator):
    """
    Supertrend (超级趋势)

    基于ATR的趋势跟踪指标
    """

    _metadata = IndicatorMetadata(
        name='Supertrend',
        display_name='超级趋势',
        category='trend',
        description='使用ATR通道识别趋势方向',
        formula='Supertrend = HL2 ± Multiplier * ATR',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 10, 'multiplier': 3.0},
        complexity_level=3
    )

    def __init__(self, period: int = 10, multiplier: float = 3.0):
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

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()

        # HL2
        hl2 = (high + low) / 2

        # 上下轨
        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        # Supertrend计算
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1  # 1=上升趋势, -1=下降趋势

        for i in range(1, len(close)):
            # 更新上轨
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                final_upper = upper_band.iloc[i]
            else:
                final_upper = upper_band.iloc[i-1]

            # 更新下轨
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                final_lower = lower_band.iloc[i]
            else:
                final_lower = lower_band.iloc[i-1]

            # 确定趋势方向
            if supertrend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] <= final_upper:
                supertrend.iloc[i] = final_upper
                direction.iloc[i] = -1
            elif supertrend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] > final_upper:
                supertrend.iloc[i] = final_lower
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] >= final_lower:
                supertrend.iloc[i] = final_lower
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] < final_lower:
                supertrend.iloc[i] = final_upper
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

        return pd.DataFrame({
            'Supertrend': supertrend,
            'Direction': direction,
            'Upper_Band': upper_band,
            'Lower_Band': lower_band
        })


@indicator_registry.register('trend')
class ParabolicSAR(BaseIndicator):
    """
    Parabolic SAR (抛物线转向指标)

    J. Welles Wilder开发的趋势跟踪和止损指标
    """

    _metadata = IndicatorMetadata(
        name='ParabolicSAR',
        display_name='抛物线转向指标',
        category='trend',
        description='跟踪价格趋势并提供止损点位',
        formula='SAR = SAR + AF * (EP - SAR)',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'af_start': 0.02, 'af_increment': 0.02, 'af_max': 0.2},
        complexity_level=4
    )

    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        super().__init__(af_start=af_start, af_increment=af_increment, af_max=af_max)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.af_start, 'af_start', 0.001, 1)
        IndicatorValidator.validate_range(self.af_increment, 'af_increment', 0.001, 1)
        IndicatorValidator.validate_range(self.af_max, 'af_max', 0.01, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)

        psar = np.zeros(n)
        trend = np.zeros(n)  # 1=上升, -1=下降
        af = np.zeros(n)
        ep = np.zeros(n)

        # 初始化
        trend[0] = 1 if (n > 1 and close[0] > close[1]) else -1
        psar[0] = low[0] if trend[0] == 1 else high[0]
        af[0] = self.af_start
        ep[0] = high[0] if trend[0] == 1 else low[0]

        for i in range(1, n):
            if trend[i-1] == 1:  # 上升趋势
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])

                if low[i] < psar[i]:  # 趋势反转
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    af[i] = self.af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # 下降趋势
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])

                if high[i] > psar[i]:  # 趋势反转
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    af[i] = self.af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]

        return pd.DataFrame({
            'PSAR': psar,
            'Trend': trend,
            'AF': af,
            'EP': ep
        }, index=data.index)


@indicator_registry.register('trend')
class ChandelierExit(BaseIndicator):
    """
    Chandelier Exit (吊灯止损)

    Chuck LeBeau开发的基于ATR的追踪止损指标
    """

    _metadata = IndicatorMetadata(
        name='ChandelierExit',
        display_name='吊灯止损',
        category='trend',
        description='基于ATR的波动性追踪止损',
        formula='Long Exit = Highest(High, n) - Multiplier * ATR',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 22, 'multiplier': 3.0},
        complexity_level=2
    )

    def __init__(self, period: int = 22, multiplier: float = 3.0):
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

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        # Chandelier Exit
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()

        long_exit = highest_high - self.multiplier * atr
        short_exit = lowest_low + self.multiplier * atr

        return pd.DataFrame({
            'Long_Exit': long_exit,
            'Short_Exit': short_exit
        })
