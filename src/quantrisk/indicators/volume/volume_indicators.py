"""
成交量指标 (15个)

基于成交量的技术分析指标
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('volume')
class OBV(BaseIndicator):
    """
    On Balance Volume (能量潮)

    Joe Granville开发的经典成交量指标
    """

    _metadata = IndicatorMetadata(
        name='OBV',
        display_name='能量潮',
        category='volume',
        description='累积成交量，价格上涨加成交量，下跌减成交量',
        formula='OBV = OBV[1] + Vol * sign(Close - Close[1])',
        required_columns=['close', 'volume'],
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
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 价格变化方向
        direction = np.sign(close.diff())

        # OBV = 累积（成交量 * 方向）
        obv = (volume * direction).cumsum()

        return obv


@indicator_registry.register('volume')
class VolumeMA(BaseIndicator):
    """
    Volume Moving Average (成交量移动平均)

    成交量的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='VolumeMA',
        display_name='成交量均线',
        category='volume',
        description='成交量的移动平均线',
        formula='VMA = SMA(Volume, n)',
        required_columns=['volume'],
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
        return ['volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        volume = data['volume']

        vol_ma = volume.rolling(window=self.period).mean()
        vol_ratio = volume / vol_ma

        return pd.DataFrame({
            'Volume': volume,
            'Volume_MA': vol_ma,
            'Volume_Ratio': vol_ratio
        })


@indicator_registry.register('volume')
class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (成交量加权平均价)

    以成交量为权重的平均价格
    """

    _metadata = IndicatorMetadata(
        name='VWAP',
        display_name='成交量加权均价',
        category='volume',
        description='以成交量为权重的平均价格',
        formula='VWAP = Σ(Typical Price * Volume) / Σ(Volume)',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 典型价格
        typical_price = (high + low + close) / 3

        # VWAP
        cum_tp_vol = (typical_price * volume).rolling(window=self.period).sum()
        cum_vol = volume.rolling(window=self.period).sum()
        vwap = cum_tp_vol / cum_vol

        # 价格与VWAP的偏离
        deviation = (close - vwap) / vwap * 100

        return pd.DataFrame({
            'VWAP': vwap,
            'VWAP_Deviation': deviation
        })


@indicator_registry.register('volume')
class AccumulationDistribution(BaseIndicator):
    """
    Accumulation/Distribution Line (聚散指标)

    Marc Chaikin开发的成交量指标
    """

    _metadata = IndicatorMetadata(
        name='AccumulationDistribution',
        display_name='聚散指标',
        category='volume',
        description='衡量资金流入流出的累积指标',
        formula='AD = Σ((Close-Low)-(High-Close))/(High-Low) * Volume',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 资金流量乘数 (Money Flow Multiplier)
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)

        # 资金流量 (Money Flow Volume)
        mfv = mfm * volume

        # 累积
        ad = mfv.cumsum()

        return ad


@indicator_registry.register('volume')
class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow (蔡金资金流)

    衡量特定周期内的资金流动
    """

    _metadata = IndicatorMetadata(
        name='ChaikinMoneyFlow',
        display_name='蔡金资金流',
        category='volume',
        description='周期内资金流入流出的比率',
        formula='CMF = Σ(MFV, n) / Σ(Volume, n)',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='series',
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 资金流量乘数
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)

        # 资金流量
        mfv = mfm * volume

        # CMF = 周期内MFV之和 / 周期内成交量之和
        cmf = mfv.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

        return cmf


@indicator_registry.register('volume')
class MoneyFlowIndex(BaseIndicator):
    """
    Money Flow Index (资金流量指数)

    成交量加权的RSI
    """

    _metadata = IndicatorMetadata(
        name='MoneyFlowIndex',
        display_name='资金流量指数',
        category='volume',
        description='结合价格和成交量的动量指标，类似成交量加权的RSI',
        formula='MFI = 100 - 100/(1 + Positive MF/Negative MF)',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 典型价格
        typical_price = (high + low + close) / 3

        # 原始资金流
        raw_money_flow = typical_price * volume

        # 判断正负流向
        tp_change = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_change > 0, 0)
        negative_flow = raw_money_flow.where(tp_change < 0, 0)

        # 周期内正负流向之和
        positive_mf = positive_flow.rolling(window=self.period).sum()
        negative_mf = negative_flow.rolling(window=self.period).sum()

        # MFI
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi


@indicator_registry.register('volume')
class ForceIndex(BaseIndicator):
    """
    Force Index (强力指数)

    Alexander Elder开发的成交量动量指标
    """

    _metadata = IndicatorMetadata(
        name='ForceIndex',
        display_name='强力指数',
        category='volume',
        description='价格变化与成交量的乘积',
        formula='FI = (Close - Close[1]) * Volume',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 13},
        complexity_level=1
    )

    def __init__(self, period: int = 13):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 原始强力指数
        force = close.diff() * volume

        # 平滑后的强力指数
        force_ema = force.ewm(span=self.period, adjust=False).mean()

        return pd.DataFrame({
            'Force_Raw': force,
            'Force_EMA': force_ema
        })


@indicator_registry.register('volume')
class EaseOfMovement(BaseIndicator):
    """
    Ease of Movement (移动便利指数)

    Richard Arms开发的成交量效率指标
    """

    _metadata = IndicatorMetadata(
        name='EaseOfMovement',
        display_name='移动便利指数',
        category='volume',
        description='衡量价格相对于成交量移动的难易程度',
        formula='EMV = (H+L)/2 - (H[1]+L[1])/2) / (Volume/Box Ratio)',
        required_columns=['high', 'low', 'volume'],
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
        return ['high', 'low', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        volume = data['volume']

        # 距离移动
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)

        # 箱体比率
        box_ratio = (volume / 100000000) / (high - low)

        # EMV
        emv = distance / box_ratio

        # 平滑
        emv_ma = emv.rolling(window=self.period).mean()

        return pd.DataFrame({
            'EMV': emv,
            'EMV_MA': emv_ma
        })


@indicator_registry.register('volume')
class VolumeOscillator(BaseIndicator):
    """
    Volume Oscillator (成交量振荡器)

    快慢成交量均线的差值百分比
    """

    _metadata = IndicatorMetadata(
        name='VolumeOscillator',
        display_name='成交量振荡器',
        category='volume',
        description='短期与长期成交量均线的差值百分比',
        formula='VO = (Short Vol MA - Long Vol MA) / Long Vol MA * 100',
        required_columns=['volume'],
        output_type='series',
        default_params={'short_period': 5, 'long_period': 20},
        complexity_level=1
    )

    def __init__(self, short_period: int = 5, long_period: int = 20):
        self.short_period = short_period
        self.long_period = long_period
        super().__init__(short_period=short_period, long_period=long_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.short_period, 'short_period')
        IndicatorValidator.validate_positive_int(self.long_period, 'long_period')

    @property
    def required_columns(self) -> List[str]:
        return ['volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        volume = data['volume']

        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()

        vol_osc = (short_ma - long_ma) / long_ma * 100

        return vol_osc


@indicator_registry.register('volume')
class NegativeVolumeIndex(BaseIndicator):
    """
    Negative Volume Index (负成交量指数)

    Paul Dysart开发的成交量指标
    """

    _metadata = IndicatorMetadata(
        name='NegativeVolumeIndex',
        display_name='负成交量指数',
        category='volume',
        description='只在成交量下降时累积价格变化',
        formula='NVI = NVI[1] + (Close-Close[1])/Close[1]*100 when Vol < Vol[1]',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'signal_period': 255},
        complexity_level=2
    )

    def __init__(self, signal_period: int = 255):
        self.signal_period = signal_period
        super().__init__(signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 价格变化率
        roc = close.pct_change() * 100

        # 成交量下降时累积
        vol_decrease = volume < volume.shift(1)
        nvi_change = roc.where(vol_decrease, 0)

        # NVI从1000开始
        nvi = 1000 + nvi_change.cumsum()

        # 信号线
        nvi_signal = nvi.ewm(span=self.signal_period, adjust=False).mean()

        return pd.DataFrame({
            'NVI': nvi,
            'NVI_Signal': nvi_signal
        })


@indicator_registry.register('volume')
class PositiveVolumeIndex(BaseIndicator):
    """
    Positive Volume Index (正成交量指数)

    只在成交量上升时累积价格变化
    """

    _metadata = IndicatorMetadata(
        name='PositiveVolumeIndex',
        display_name='正成交量指数',
        category='volume',
        description='只在成交量上升时累积价格变化',
        formula='PVI = PVI[1] + (Close-Close[1])/Close[1]*100 when Vol > Vol[1]',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'signal_period': 255},
        complexity_level=2
    )

    def __init__(self, signal_period: int = 255):
        self.signal_period = signal_period
        super().__init__(signal_period=signal_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.signal_period, 'signal_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 价格变化率
        roc = close.pct_change() * 100

        # 成交量上升时累积
        vol_increase = volume > volume.shift(1)
        pvi_change = roc.where(vol_increase, 0)

        # PVI从1000开始
        pvi = 1000 + pvi_change.cumsum()

        # 信号线
        pvi_signal = pvi.ewm(span=self.signal_period, adjust=False).mean()

        return pd.DataFrame({
            'PVI': pvi,
            'PVI_Signal': pvi_signal
        })


@indicator_registry.register('volume')
class VolumeProfile(BaseIndicator):
    """
    Volume Profile (成交量分布)

    价格区间内的成交量分布
    """

    _metadata = IndicatorMetadata(
        name='VolumeProfile',
        display_name='成交量分布',
        category='volume',
        description='计算价格区间内的成交量集中度',
        formula='VP = Volume at each price level',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20, 'bins': 10},
        complexity_level=3
    )

    def __init__(self, period: int = 20, bins: int = 10):
        self.period = period
        self.bins = bins
        super().__init__(period=period, bins=bins)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.bins, 'bins')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        poc = pd.Series(index=close.index, dtype=float)  # Point of Control
        vah = pd.Series(index=close.index, dtype=float)  # Value Area High
        val = pd.Series(index=close.index, dtype=float)  # Value Area Low

        for i in range(self.period - 1, len(close)):
            window_high = high.iloc[i - self.period + 1:i + 1]
            window_low = low.iloc[i - self.period + 1:i + 1]
            window_close = close.iloc[i - self.period + 1:i + 1]
            window_volume = volume.iloc[i - self.period + 1:i + 1]

            # 价格区间
            price_min = window_low.min()
            price_max = window_high.max()
            price_range = price_max - price_min

            if price_range > 0:
                # 创建价格档位
                bin_size = price_range / self.bins
                bins_edges = [price_min + i * bin_size for i in range(self.bins + 1)]

                # 计算每个档位的成交量
                vol_profile = np.zeros(self.bins)
                for j in range(len(window_close)):
                    bin_idx = int((window_close.iloc[j] - price_min) / bin_size)
                    bin_idx = min(bin_idx, self.bins - 1)
                    vol_profile[bin_idx] += window_volume.iloc[j]

                # POC: 成交量最大的价格档位
                poc_idx = np.argmax(vol_profile)
                poc.iloc[i] = bins_edges[poc_idx] + bin_size / 2

                # Value Area (70%成交量的价格区间)
                total_vol = vol_profile.sum()
                va_vol = 0
                va_bins = [poc_idx]

                while va_vol < total_vol * 0.7:
                    # 向上或向下扩展
                    upper = max(va_bins) + 1
                    lower = min(va_bins) - 1

                    upper_vol = vol_profile[upper] if upper < self.bins else 0
                    lower_vol = vol_profile[lower] if lower >= 0 else 0

                    if upper_vol >= lower_vol and upper < self.bins:
                        va_bins.append(upper)
                        va_vol += upper_vol
                    elif lower >= 0:
                        va_bins.append(lower)
                        va_vol += lower_vol
                    else:
                        break

                vah.iloc[i] = bins_edges[max(va_bins) + 1]
                val.iloc[i] = bins_edges[min(va_bins)]

        return pd.DataFrame({
            'VP_POC': poc,
            'VP_VAH': vah,
            'VP_VAL': val
        })


@indicator_registry.register('volume')
class KlingerOscillator(BaseIndicator):
    """
    Klinger Volume Oscillator (克林格成交量振荡器)

    Stephen Klinger开发的成交量趋势指标
    """

    _metadata = IndicatorMetadata(
        name='KlingerOscillator',
        display_name='克林格振荡器',
        category='volume',
        description='结合价格趋势和成交量的振荡器',
        formula='KVO = EMA(VF, 34) - EMA(VF, 55)',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'fast_period': 34, 'slow_period': 55, 'signal_period': 13},
        complexity_level=3
    )

    def __init__(self, fast_period: int = 34, slow_period: int = 55, signal_period: int = 13):
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # HLC3
        hlc3 = (high + low + close) / 3

        # 趋势方向
        trend = np.sign(hlc3.diff())

        # 价格变化幅度
        dm = high - low

        # 累积变化
        cm = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if trend.iloc[i] == trend.iloc[i-1]:
                cm.iloc[i] = cm.iloc[i-1] + dm.iloc[i]
            else:
                cm.iloc[i] = dm.iloc[i-1] + dm.iloc[i]

        # Volume Force
        vf = volume * abs(2 * dm / cm - 1) * trend * 100

        # KVO
        kvo = vf.ewm(span=self.fast_period, adjust=False).mean() - \
              vf.ewm(span=self.slow_period, adjust=False).mean()

        # 信号线
        signal = kvo.ewm(span=self.signal_period, adjust=False).mean()

        return pd.DataFrame({
            'KVO': kvo,
            'KVO_Signal': signal,
            'KVO_Histogram': kvo - signal
        })


@indicator_registry.register('volume')
class ChaikinOscillator(BaseIndicator):
    """
    Chaikin Oscillator (蔡金振荡器)

    A/D线的MACD
    """

    _metadata = IndicatorMetadata(
        name='ChaikinOscillator',
        display_name='蔡金振荡器',
        category='volume',
        description='聚散指标的快慢均线差',
        formula='CO = EMA(AD, 3) - EMA(AD, 10)',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='series',
        default_params={'fast_period': 3, 'slow_period': 10},
        complexity_level=2
    )

    def __init__(self, fast_period: int = 3, slow_period: int = 10):
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(fast_period=fast_period, slow_period=slow_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 计算A/D线
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        ad = mfv.cumsum()

        # 蔡金振荡器
        fast_ema = ad.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = ad.ewm(span=self.slow_period, adjust=False).mean()

        return fast_ema - slow_ema
