"""
周期分析指标 (15个)

基于周期理论和傅里叶分析的技术指标
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('cycle')
class SineWaveIndicator(BaseIndicator):
    """正弦波指标"""

    _metadata = IndicatorMetadata(
        name='SineWaveIndicator',
        display_name='正弦波指标',
        category='cycle',
        description='John Ehlers的正弦波周期指标',
        formula='Sine Wave = sin(2π/Period)',
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
        n = len(data)

        # 生成正弦波
        phase = np.arange(n) * 2 * np.pi / self.period
        sine = np.sin(phase)
        lead_sine = np.sin(phase + np.pi / 4)  # 领先45度

        return pd.DataFrame({
            'Sine': pd.Series(sine, index=data.index),
            'LeadSine': pd.Series(lead_sine, index=data.index)
        })


@indicator_registry.register('cycle')
class HilbertTransform(BaseIndicator):
    """希尔伯特变换"""

    _metadata = IndicatorMetadata(
        name='HilbertTransform',
        display_name='希尔伯特变换',
        category='cycle',
        description='使用希尔伯特变换检测周期',
        formula='Hilbert Transform Phase Detection',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 7},
        complexity_level=3
    )

    def __init__(self, period: int = 7):
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

        # 简化的希尔伯特变换（使用移动平均差分近似）
        smooth = close.rolling(window=4).mean()
        detrender = smooth - smooth.shift(2)

        # 相位计算
        in_phase = detrender.shift(3)
        quadrature = detrender

        # 瞬时相位
        phase = np.arctan2(quadrature, in_phase) * 180 / np.pi

        # 瞬时周期
        delta_phase = phase.diff()
        delta_phase[delta_phase < 0] += 360
        inst_period = 360 / delta_phase.replace(0, np.nan)
        inst_period = inst_period.clip(lower=6, upper=50)

        # 趋势模式
        trend_mode = (inst_period.rolling(window=self.period).std() /
                     inst_period.rolling(window=self.period).mean())

        return pd.DataFrame({
            'HT_InPhase': in_phase,
            'HT_Quadrature': quadrature,
            'HT_Phase': phase,
            'HT_Period': inst_period,
            'HT_TrendMode': trend_mode
        })


@indicator_registry.register('cycle')
class FourierAnalysis(BaseIndicator):
    """傅里叶分析"""

    _metadata = IndicatorMetadata(
        name='FourierAnalysis',
        display_name='傅里叶分析',
        category='cycle',
        description='使用FFT分析价格周期',
        formula='FFT Spectral Analysis',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 64},
        complexity_level=3
    )

    def __init__(self, window: int = 64):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        dominant_period = pd.Series(index=data.index, dtype=float)
        spectral_power = pd.Series(index=data.index, dtype=float)

        for i in range(self.window - 1, len(close)):
            window_data = close.iloc[i - self.window + 1:i + 1].values

            # 去趋势
            detrended = signal.detrend(window_data)

            # FFT
            fft_result = fft(detrended)
            power = np.abs(fft_result[:self.window // 2]) ** 2
            freqs = fftfreq(self.window)[:self.window // 2]

            # 找到主要周期（排除直流分量）
            power[0] = 0
            if np.max(power) > 0:
                peak_idx = np.argmax(power)
                if freqs[peak_idx] > 0:
                    dominant_period.iloc[i] = 1 / freqs[peak_idx]
                spectral_power.iloc[i] = power[peak_idx]

        # 归一化功率
        spectral_power = spectral_power / spectral_power.rolling(window=20).mean()

        return pd.DataFrame({
            'Dominant_Period': dominant_period,
            'Spectral_Power': spectral_power
        })


@indicator_registry.register('cycle')
class CyclePeriod(BaseIndicator):
    """周期长度检测"""

    _metadata = IndicatorMetadata(
        name='CyclePeriod',
        display_name='周期长度',
        category='cycle',
        description='动态检测价格周期长度',
        formula='Adaptive Cycle Period Detection',
        required_columns=['close'],
        output_type='series',
        default_params={'min_period': 6, 'max_period': 50},
        complexity_level=3
    )

    def __init__(self, min_period: int = 6, max_period: int = 50):
        self.min_period = min_period
        self.max_period = max_period
        super().__init__(min_period=min_period, max_period=max_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.min_period, 'min_period')
        IndicatorValidator.validate_positive_int(self.max_period, 'max_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        cycle_period = pd.Series(index=data.index, dtype=float)

        for i in range(self.max_period, len(close)):
            window = close.iloc[i - self.max_period:i + 1].values

            best_period = self.min_period
            best_correlation = -1

            for period in range(self.min_period, self.max_period + 1):
                if i >= 2 * period:
                    current = window[-period:]
                    lagged = window[-2*period:-period]
                    if len(current) == len(lagged):
                        corr = np.corrcoef(current, lagged)[0, 1]
                        if not np.isnan(corr) and corr > best_correlation:
                            best_correlation = corr
                            best_period = period

            cycle_period.iloc[i] = best_period

        return cycle_period


@indicator_registry.register('cycle')
class MESAAdaptiveMA(BaseIndicator):
    """MESA自适应移动平均"""

    _metadata = IndicatorMetadata(
        name='MESAAdaptiveMA',
        display_name='MESA自适应均线',
        category='cycle',
        description='John Ehlers的MAMA指标',
        formula='MESA Adaptive Moving Average',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'fast_limit': 0.5, 'slow_limit': 0.05},
        complexity_level=3
    )

    def __init__(self, fast_limit: float = 0.5, slow_limit: float = 0.05):
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        super().__init__(fast_limit=fast_limit, slow_limit=slow_limit)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.fast_limit, 'fast_limit')
        IndicatorValidator.validate_positive_float(self.slow_limit, 'slow_limit')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        mama = np.zeros(n)
        fama = np.zeros(n)  # Following AMA

        mama[0] = close.iloc[0]
        fama[0] = close.iloc[0]

        for i in range(1, n):
            # 简化的相位变化估计
            smooth = close.iloc[max(0, i-4):i+1].mean()
            period = 20  # 简化为固定周期

            # 自适应alpha
            delta_phase = abs(close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] if close.iloc[i-1] != 0 else 0
            alpha = max(self.slow_limit, min(self.fast_limit, delta_phase))

            mama[i] = alpha * close.iloc[i] + (1 - alpha) * mama[i-1]
            fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]

        return pd.DataFrame({
            'MAMA': pd.Series(mama, index=data.index),
            'FAMA': pd.Series(fama, index=data.index)
        })


@indicator_registry.register('cycle')
class EvenBetterSinewave(BaseIndicator):
    """更优正弦波"""

    _metadata = IndicatorMetadata(
        name='EvenBetterSinewave',
        display_name='EBSW指标',
        category='cycle',
        description='John Ehlers的改进正弦波指标',
        formula='Even Better Sinewave',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'hp_period': 40, 'smooth_period': 10},
        complexity_level=3
    )

    def __init__(self, hp_period: int = 40, smooth_period: int = 10):
        self.hp_period = hp_period
        self.smooth_period = smooth_period
        super().__init__(hp_period=hp_period, smooth_period=smooth_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.hp_period, 'hp_period')
        IndicatorValidator.validate_positive_int(self.smooth_period, 'smooth_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        # 高通滤波器
        alpha_hp = (1 - np.sin(2 * np.pi / self.hp_period)) / np.cos(2 * np.pi / self.hp_period)
        hp = np.zeros(n)

        for i in range(1, n):
            hp[i] = 0.5 * (1 + alpha_hp) * (close.iloc[i] - close.iloc[i-1]) + alpha_hp * hp[i-1]

        # 超级平滑滤波器
        a1 = np.exp(-1.414 * np.pi / self.smooth_period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / self.smooth_period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3

        filt = np.zeros(n)
        for i in range(2, n):
            filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]

        # 波形
        wave = pd.Series(filt, index=data.index)
        power = wave.rolling(window=self.smooth_period).apply(lambda x: np.mean(x**2))
        wave_normalized = wave / np.sqrt(power.replace(0, np.nan))

        # 触发信号
        trigger = wave_normalized.shift(1)

        return pd.DataFrame({
            'EBSW_Wave': wave_normalized,
            'EBSW_Trigger': trigger
        })


@indicator_registry.register('cycle')
class SchaeffTrendCycle(BaseIndicator):
    """Schaff趋势周期"""

    _metadata = IndicatorMetadata(
        name='SchaeffTrendCycle',
        display_name='Schaff趋势周期',
        category='cycle',
        description='结合MACD和随机指标的周期指标',
        formula='STC = Stochastic(MACD)',
        required_columns=['close'],
        output_type='series',
        default_params={'fast': 23, 'slow': 50, 'cycle': 10},
        complexity_level=2
    )

    def __init__(self, fast: int = 23, slow: int = 50, cycle: int = 10):
        self.fast = fast
        self.slow = slow
        self.cycle = cycle
        super().__init__(fast=fast, slow=slow, cycle=cycle)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.fast, 'fast')
        IndicatorValidator.validate_positive_int(self.slow, 'slow')
        IndicatorValidator.validate_positive_int(self.cycle, 'cycle')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # MACD线
        fast_ema = close.ewm(span=self.fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow, adjust=False).mean()
        macd = fast_ema - slow_ema

        # 第一次随机化
        lowest = macd.rolling(window=self.cycle).min()
        highest = macd.rolling(window=self.cycle).max()
        stoch1 = (macd - lowest) / (highest - lowest) * 100

        # 平滑
        pf = stoch1.ewm(span=self.cycle, adjust=False).mean()

        # 第二次随机化
        lowest_pf = pf.rolling(window=self.cycle).min()
        highest_pf = pf.rolling(window=self.cycle).max()
        stoch2 = (pf - lowest_pf) / (highest_pf - lowest_pf) * 100

        # 最终STC
        stc = stoch2.ewm(span=self.cycle, adjust=False).mean()

        return stc


@indicator_registry.register('cycle')
class InstantaneousTrendline(BaseIndicator):
    """瞬时趋势线"""

    _metadata = IndicatorMetadata(
        name='InstantaneousTrendline',
        display_name='瞬时趋势线',
        category='cycle',
        description='John Ehlers的瞬时趋势线',
        formula='ITrend Instantaneous Trendline',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'alpha': 0.07},
        complexity_level=2
    )

    def __init__(self, alpha: float = 0.07):
        self.alpha = alpha
        super().__init__(alpha=alpha)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.alpha, 'alpha', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        itrend = np.zeros(n)
        trigger = np.zeros(n)

        for i in range(7, n):
            itrend[i] = (self.alpha - self.alpha**2/4) * close.iloc[i] + \
                        0.5 * self.alpha**2 * close.iloc[i-1] - \
                        (self.alpha - 0.75 * self.alpha**2) * close.iloc[i-2] + \
                        2 * (1 - self.alpha) * itrend[i-1] - \
                        (1 - self.alpha)**2 * itrend[i-2]

            trigger[i] = 2 * itrend[i] - itrend[i-2]

        return pd.DataFrame({
            'ITrend': pd.Series(itrend, index=data.index),
            'ITrend_Trigger': pd.Series(trigger, index=data.index)
        })


@indicator_registry.register('cycle')
class CyberCycle(BaseIndicator):
    """网络周期"""

    _metadata = IndicatorMetadata(
        name='CyberCycle',
        display_name='网络周期',
        category='cycle',
        description='John Ehlers的网络周期指标',
        formula='Cyber Cycle Oscillator',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'alpha': 0.07},
        complexity_level=2
    )

    def __init__(self, alpha: float = 0.07):
        self.alpha = alpha
        super().__init__(alpha=alpha)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.alpha, 'alpha', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        # 平滑价格
        smooth = np.zeros(n)
        for i in range(3, n):
            smooth[i] = (close.iloc[i] + 2*close.iloc[i-1] + 2*close.iloc[i-2] + close.iloc[i-3]) / 6

        # 网络周期
        cycle = np.zeros(n)
        for i in range(6, n):
            cycle[i] = (1 - 0.5*self.alpha)**2 * (smooth[i] - 2*smooth[i-1] + smooth[i-2]) + \
                       2 * (1 - self.alpha) * cycle[i-1] - \
                       (1 - self.alpha)**2 * cycle[i-2]

        # 触发线
        trigger = cycle.copy()
        for i in range(1, n):
            trigger[i] = cycle[i-1]

        return pd.DataFrame({
            'CyberCycle': pd.Series(cycle, index=data.index),
            'CyberCycle_Trigger': pd.Series(trigger, index=data.index)
        })


@indicator_registry.register('cycle')
class DetrenderIndicator(BaseIndicator):
    """去趋势指标"""

    _metadata = IndicatorMetadata(
        name='DetrenderIndicator',
        display_name='去趋势指标',
        category='cycle',
        description='去除趋势后的周期成分',
        formula='Detrended Price = Price - Trend',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 趋势（移动平均）
        trend = close.rolling(window=self.period).mean()

        # 去趋势成分
        detrended = close - trend

        # 标准化
        std = detrended.rolling(window=self.period).std()
        detrended_norm = detrended / std

        # 振幅
        amplitude = abs(detrended)
        avg_amplitude = amplitude.rolling(window=self.period).mean()

        return pd.DataFrame({
            'Trend': trend,
            'Detrended': detrended,
            'Detrended_Norm': detrended_norm,
            'Amplitude': avg_amplitude
        })


@indicator_registry.register('cycle')
class BandpassFilter(BaseIndicator):
    """带通滤波器"""

    _metadata = IndicatorMetadata(
        name='BandpassFilter',
        display_name='带通滤波器',
        category='cycle',
        description='提取特定周期范围的价格成分',
        formula='Bandpass = Highpass - Lowpass',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'center_period': 20, 'bandwidth': 0.3},
        complexity_level=2
    )

    def __init__(self, center_period: int = 20, bandwidth: float = 0.3):
        self.center_period = center_period
        self.bandwidth = bandwidth
        super().__init__(center_period=center_period, bandwidth=bandwidth)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.center_period, 'center_period')
        IndicatorValidator.validate_range(self.bandwidth, 'bandwidth', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        # 带通滤波参数
        delta = self.bandwidth
        beta = np.cos(2 * np.pi / self.center_period)
        gamma = 1 / np.cos(2 * np.pi * delta / self.center_period)
        alpha = gamma - np.sqrt(gamma**2 - 1)

        bp = np.zeros(n)
        for i in range(2, n):
            bp[i] = 0.5 * (1 - alpha) * (close.iloc[i] - close.iloc[i-2]) + \
                    beta * (1 + alpha) * bp[i-1] - alpha * bp[i-2]

        # 功率
        power = pd.Series(bp, index=data.index).rolling(window=self.center_period).apply(
            lambda x: np.mean(x**2)
        )
        power_norm = np.sqrt(power)

        # 归一化带通
        bp_norm = pd.Series(bp, index=data.index) / power_norm.replace(0, np.nan)

        # 触发
        trigger = bp_norm.shift(1)

        return pd.DataFrame({
            'Bandpass': pd.Series(bp, index=data.index),
            'Bandpass_Norm': bp_norm,
            'Bandpass_Trigger': trigger,
            'Bandpass_Power': power_norm
        })


@indicator_registry.register('cycle')
class HighpassFilter(BaseIndicator):
    """高通滤波器"""

    _metadata = IndicatorMetadata(
        name='HighpassFilter',
        display_name='高通滤波器',
        category='cycle',
        description='去除长周期趋势，保留短周期波动',
        formula='Highpass Filter',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 48},
        complexity_level=2
    )

    def __init__(self, period: int = 48):
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
        n = len(close)

        alpha = (1 - np.sin(2 * np.pi / self.period)) / np.cos(2 * np.pi / self.period)
        hp = np.zeros(n)

        for i in range(1, n):
            hp[i] = 0.5 * (1 + alpha) * (close.iloc[i] - close.iloc[i-1]) + alpha * hp[i-1]

        return pd.Series(hp, index=data.index)


@indicator_registry.register('cycle')
class SmoothingFilter(BaseIndicator):
    """超级平滑滤波器"""

    _metadata = IndicatorMetadata(
        name='SmoothingFilter',
        display_name='超级平滑滤波器',
        category='cycle',
        description='John Ehlers的超级平滑滤波器',
        formula='SuperSmoother Filter',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 10},
        complexity_level=2
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
        n = len(close)

        # 滤波器系数
        a = np.exp(-1.414 * np.pi / self.period)
        b = 2 * a * np.cos(1.414 * np.pi / self.period)

        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        filt = np.zeros(n)
        for i in range(2, n):
            filt[i] = c1 * (close.iloc[i] + close.iloc[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]

        return pd.Series(filt, index=data.index)


@indicator_registry.register('cycle')
class RoofingFilter(BaseIndicator):
    """屋顶滤波器"""

    _metadata = IndicatorMetadata(
        name='RoofingFilter',
        display_name='屋顶滤波器',
        category='cycle',
        description='高通+低通组合滤波器',
        formula='Roofing = Highpass(Supersmoother(Price))',
        required_columns=['close'],
        output_type='series',
        default_params={'hp_period': 48, 'lp_period': 10},
        complexity_level=2
    )

    def __init__(self, hp_period: int = 48, lp_period: int = 10):
        self.hp_period = hp_period
        self.lp_period = lp_period
        super().__init__(hp_period=hp_period, lp_period=lp_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.hp_period, 'hp_period')
        IndicatorValidator.validate_positive_int(self.lp_period, 'lp_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        # 高通滤波器
        alpha_hp = (1 - np.sin(2 * np.pi / self.hp_period)) / np.cos(2 * np.pi / self.hp_period)
        hp = np.zeros(n)
        for i in range(1, n):
            hp[i] = 0.5 * (1 + alpha_hp) * (close.iloc[i] - close.iloc[i-1]) + alpha_hp * hp[i-1]

        # 超级平滑滤波器
        a = np.exp(-1.414 * np.pi / self.lp_period)
        b = 2 * a * np.cos(1.414 * np.pi / self.lp_period)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        filt = np.zeros(n)
        for i in range(2, n):
            filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]

        return pd.Series(filt, index=data.index)
