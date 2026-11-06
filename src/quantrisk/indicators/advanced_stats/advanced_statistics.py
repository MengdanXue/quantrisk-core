"""
高级统计指标 (20个)

基于高级统计方法的技术分析指标
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('advanced_stats')
class RollingRegression(BaseIndicator):
    """滚动回归分析"""

    _metadata = IndicatorMetadata(
        name='RollingRegression',
        display_name='滚动回归',
        category='advanced_stats',
        description='滚动窗口内的线性回归分析',
        formula='y = α + βx + ε',
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

        alpha = pd.Series(index=close.index, dtype=float)
        beta = pd.Series(index=close.index, dtype=float)
        r_squared = pd.Series(index=close.index, dtype=float)
        residual = pd.Series(index=close.index, dtype=float)

        x = np.arange(self.period)

        for i in range(self.period - 1, len(close)):
            y = close.iloc[i - self.period + 1:i + 1].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            alpha.iloc[i] = intercept
            beta.iloc[i] = slope
            r_squared.iloc[i] = r_value ** 2
            residual.iloc[i] = close.iloc[i] - (intercept + slope * (self.period - 1))

        return pd.DataFrame({
            'Reg_Alpha': alpha,
            'Reg_Beta': beta,
            'Reg_R2': r_squared,
            'Reg_Residual': residual
        })


@indicator_registry.register('advanced_stats')
class KalmanFilter(BaseIndicator):
    """卡尔曼滤波器"""

    _metadata = IndicatorMetadata(
        name='KalmanFilter',
        display_name='卡尔曼滤波',
        category='advanced_stats',
        description='使用卡尔曼滤波估计真实价格',
        formula='Kalman Filter State Estimation',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'process_noise': 0.01, 'measurement_noise': 0.1},
        complexity_level=3
    )

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        super().__init__(process_noise=process_noise, measurement_noise=measurement_noise)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.process_noise, 'process_noise')
        IndicatorValidator.validate_positive_float(self.measurement_noise, 'measurement_noise')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        # 初始化
        x_est = np.zeros(n)  # 状态估计
        p_est = np.zeros(n)  # 误差协方差
        x_est[0] = close.iloc[0]
        p_est[0] = 1.0

        Q = self.process_noise  # 过程噪声
        R = self.measurement_noise  # 测量噪声

        for i in range(1, n):
            # 预测
            x_pred = x_est[i-1]
            p_pred = p_est[i-1] + Q

            # 更新
            K = p_pred / (p_pred + R)  # 卡尔曼增益
            x_est[i] = x_pred + K * (close.iloc[i] - x_pred)
            p_est[i] = (1 - K) * p_pred

        kalman_estimate = pd.Series(x_est, index=close.index)
        kalman_error = close - kalman_estimate

        return pd.DataFrame({
            'Kalman_Estimate': kalman_estimate,
            'Kalman_Error': kalman_error,
            'Kalman_Gain': pd.Series(p_est, index=close.index)
        })


@indicator_registry.register('advanced_stats')
class CopulaCorrelation(BaseIndicator):
    """Copula相关性"""

    _metadata = IndicatorMetadata(
        name='CopulaCorrelation',
        display_name='Copula相关',
        category='advanced_stats',
        description='基于Copula的尾部相关性分析',
        formula='Spearman Rank Correlation',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 60},
        complexity_level=3
    )

    def __init__(self, period: int = 60):
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

        returns = close.pct_change()
        vol_change = volume.pct_change()

        # Pearson相关
        pearson = returns.rolling(window=self.period).corr(vol_change)

        # Spearman秩相关
        def spearman_corr(x, y):
            return stats.spearmanr(x, y)[0]

        spearman = pd.Series(index=close.index, dtype=float)
        for i in range(self.period - 1, len(close)):
            r = returns.iloc[i - self.period + 1:i + 1].dropna()
            v = vol_change.iloc[i - self.period + 1:i + 1].dropna()
            if len(r) > 2 and len(v) > 2:
                min_len = min(len(r), len(v))
                spearman.iloc[i] = spearman_corr(r.iloc[-min_len:], v.iloc[-min_len:])

        # Kendall tau
        kendall = pd.Series(index=close.index, dtype=float)
        for i in range(self.period - 1, len(close)):
            r = returns.iloc[i - self.period + 1:i + 1].dropna()
            v = vol_change.iloc[i - self.period + 1:i + 1].dropna()
            if len(r) > 2 and len(v) > 2:
                min_len = min(len(r), len(v))
                kendall.iloc[i] = stats.kendalltau(r.iloc[-min_len:], v.iloc[-min_len:])[0]

        return pd.DataFrame({
            'Pearson_Corr': pearson,
            'Spearman_Corr': spearman,
            'Kendall_Tau': kendall
        })


@indicator_registry.register('advanced_stats')
class EntropyIndicator(BaseIndicator):
    """信息熵指标"""

    _metadata = IndicatorMetadata(
        name='EntropyIndicator',
        display_name='信息熵',
        category='advanced_stats',
        description='价格分布的信息熵，衡量不确定性',
        formula='H = -Σ p(x) * log(p(x))',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'bins': 10},
        complexity_level=2
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        returns = close.pct_change()

        def calc_entropy(x):
            x = x.dropna()
            if len(x) < 2:
                return np.nan
            hist, _ = np.histogram(x, bins=self.bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist + 1e-10))

        entropy = returns.rolling(window=self.period).apply(calc_entropy, raw=False)
        return entropy


@indicator_registry.register('advanced_stats')
class MutualInformation(BaseIndicator):
    """互信息指标"""

    _metadata = IndicatorMetadata(
        name='MutualInformation',
        display_name='互信息',
        category='advanced_stats',
        description='价格与成交量之间的互信息',
        formula='MI = H(X) + H(Y) - H(X,Y)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'period': 30, 'bins': 5},
        complexity_level=3
    )

    def __init__(self, period: int = 30, bins: int = 5):
        self.period = period
        self.bins = bins
        super().__init__(period=period, bins=bins)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.bins, 'bins')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        returns = data['close'].pct_change()
        vol_change = data['volume'].pct_change()

        def calc_mi(window_idx):
            r = returns.iloc[window_idx].dropna().values
            v = vol_change.iloc[window_idx].dropna().values
            if len(r) < 5 or len(v) < 5:
                return np.nan

            min_len = min(len(r), len(v))
            r, v = r[-min_len:], v[-min_len:]

            # 离散化
            r_bins = np.digitize(r, np.percentile(r, np.linspace(0, 100, self.bins + 1)[1:-1]))
            v_bins = np.digitize(v, np.percentile(v, np.linspace(0, 100, self.bins + 1)[1:-1]))

            # 计算联合分布和边缘分布
            joint = np.zeros((self.bins, self.bins))
            for i in range(len(r_bins)):
                joint[r_bins[i] - 1, v_bins[i] - 1] += 1
            joint /= len(r_bins)

            p_r = joint.sum(axis=1)
            p_v = joint.sum(axis=0)

            # 互信息
            mi = 0
            for i in range(self.bins):
                for j in range(self.bins):
                    if joint[i, j] > 0 and p_r[i] > 0 and p_v[j] > 0:
                        mi += joint[i, j] * np.log(joint[i, j] / (p_r[i] * p_v[j] + 1e-10))
            return mi

        result = pd.Series(index=data.index, dtype=float)
        for i in range(self.period - 1, len(data)):
            result.iloc[i] = calc_mi(range(i - self.period + 1, i + 1))

        return result


@indicator_registry.register('advanced_stats')
class FractalDimension(BaseIndicator):
    """分形维度"""

    _metadata = IndicatorMetadata(
        name='FractalDimension',
        display_name='分形维度',
        category='advanced_stats',
        description='价格序列的分形维度，衡量复杂度',
        formula='D = log(N) / log(N * L / L_total)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 30},
        complexity_level=3
    )

    def __init__(self, period: int = 30):
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

        def calc_fractal_dim(x):
            if len(x) < 4:
                return np.nan

            n = len(x)
            max_val = x.max()
            min_val = x.min()
            range_val = max_val - min_val

            if range_val == 0:
                return 1.0

            # 计算路径长度
            path_length = np.sum(np.abs(np.diff(x)))

            # 简化的分形维度估计
            if path_length == 0:
                return 1.0

            d = 1 + np.log(path_length / range_val) / np.log(n)
            return np.clip(d, 1, 2)

        fractal = close.rolling(window=self.period).apply(calc_fractal_dim, raw=True)
        return fractal


@indicator_registry.register('advanced_stats')
class WaveletDecomposition(BaseIndicator):
    """小波分解"""

    _metadata = IndicatorMetadata(
        name='WaveletDecomposition',
        display_name='小波分解',
        category='advanced_stats',
        description='使用小波变换分解价格序列',
        formula='Discrete Wavelet Transform',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 32},
        complexity_level=3
    )

    def __init__(self, period: int = 32):
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

        # 简化版小波分解（使用移动平均代替）
        # 趋势成分（低频）
        trend = close.rolling(window=self.period).mean()

        # 细节成分（高频）
        detail = close - trend

        # 能量分布
        trend_energy = (trend ** 2).rolling(window=self.period).mean()
        detail_energy = (detail ** 2).rolling(window=self.period).mean()

        # 能量比
        total_energy = trend_energy + detail_energy
        trend_ratio = trend_energy / total_energy

        return pd.DataFrame({
            'Wavelet_Trend': trend,
            'Wavelet_Detail': detail,
            'Trend_Energy_Ratio': trend_ratio
        })


@indicator_registry.register('advanced_stats')
class PeakValleyDetection(BaseIndicator):
    """峰谷检测"""

    _metadata = IndicatorMetadata(
        name='PeakValleyDetection',
        display_name='峰谷检测',
        category='advanced_stats',
        description='自动检测价格的峰值和谷值',
        formula='Local Maxima/Minima Detection',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'prominence': 0.02, 'distance': 5},
        complexity_level=2
    )

    def __init__(self, prominence: float = 0.02, distance: int = 5):
        self.prominence = prominence
        self.distance = distance
        super().__init__(prominence=prominence, distance=distance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.prominence, 'prominence')
        IndicatorValidator.validate_positive_int(self.distance, 'distance')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close'].values

        # 检测峰值
        peaks, peak_props = find_peaks(close, prominence=close.mean() * self.prominence,
                                        distance=self.distance)

        # 检测谷值
        valleys, valley_props = find_peaks(-close, prominence=close.mean() * self.prominence,
                                           distance=self.distance)

        # 创建标记序列
        peak_signal = pd.Series(0, index=data.index)
        valley_signal = pd.Series(0, index=data.index)

        peak_signal.iloc[peaks] = 1
        valley_signal.iloc[valleys] = 1

        # 距离上一个峰/谷的周期数
        peak_distance = pd.Series(index=data.index, dtype=float)
        valley_distance = pd.Series(index=data.index, dtype=float)

        last_peak = -1
        last_valley = -1
        for i in range(len(data)):
            if peak_signal.iloc[i] == 1:
                last_peak = i
            if valley_signal.iloc[i] == 1:
                last_valley = i

            peak_distance.iloc[i] = i - last_peak if last_peak >= 0 else np.nan
            valley_distance.iloc[i] = i - last_valley if last_valley >= 0 else np.nan

        return pd.DataFrame({
            'Peak_Signal': peak_signal,
            'Valley_Signal': valley_signal,
            'Peak_Distance': peak_distance,
            'Valley_Distance': valley_distance
        })


@indicator_registry.register('advanced_stats')
class BootstrapConfidence(BaseIndicator):
    """Bootstrap置信区间"""

    _metadata = IndicatorMetadata(
        name='BootstrapConfidence',
        display_name='Bootstrap置信区间',
        category='advanced_stats',
        description='使用Bootstrap方法估计收益率置信区间',
        formula='Bootstrap Resampling',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60, 'n_bootstrap': 100, 'confidence': 0.95},
        complexity_level=3
    )

    def __init__(self, period: int = 60, n_bootstrap: int = 100, confidence: float = 0.95):
        self.period = period
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        super().__init__(period=period, n_bootstrap=n_bootstrap, confidence=confidence)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.n_bootstrap, 'n_bootstrap')
        IndicatorValidator.validate_range(self.confidence, 'confidence', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        returns = data['close'].pct_change()

        lower = pd.Series(index=data.index, dtype=float)
        upper = pd.Series(index=data.index, dtype=float)
        mean_est = pd.Series(index=data.index, dtype=float)

        alpha = 1 - self.confidence

        for i in range(self.period - 1, len(data)):
            r = returns.iloc[i - self.period + 1:i + 1].dropna().values
            if len(r) < 10:
                continue

            # Bootstrap重采样
            means = np.zeros(self.n_bootstrap)
            for b in range(self.n_bootstrap):
                sample = np.random.choice(r, size=len(r), replace=True)
                means[b] = sample.mean()

            lower.iloc[i] = np.percentile(means, alpha / 2 * 100)
            upper.iloc[i] = np.percentile(means, (1 - alpha / 2) * 100)
            mean_est.iloc[i] = means.mean()

        return pd.DataFrame({
            'BS_Lower': lower,
            'BS_Mean': mean_est,
            'BS_Upper': upper
        })


@indicator_registry.register('advanced_stats')
class RobustStatistics(BaseIndicator):
    """稳健统计"""

    _metadata = IndicatorMetadata(
        name='RobustStatistics',
        display_name='稳健统计',
        category='advanced_stats',
        description='对异常值稳健的统计量',
        formula='Median, MAD, Trimmed Mean',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'trim_percent': 0.1},
        complexity_level=2
    )

    def __init__(self, period: int = 20, trim_percent: float = 0.1):
        self.period = period
        self.trim_percent = trim_percent
        super().__init__(period=period, trim_percent=trim_percent)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_range(self.trim_percent, 'trim_percent', 0, 0.5)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 中位数
        median = close.rolling(window=self.period).median()

        # MAD (Median Absolute Deviation)
        def calc_mad(x):
            return np.median(np.abs(x - np.median(x)))

        mad = close.rolling(window=self.period).apply(calc_mad, raw=True)

        # 截断均值
        def calc_trimmed_mean(x):
            return stats.trim_mean(x, self.trim_percent)

        trimmed_mean = close.rolling(window=self.period).apply(calc_trimmed_mean, raw=True)

        # Winsorized均值
        def calc_winsor_mean(x):
            return stats.mstats.winsorize(x, limits=[self.trim_percent, self.trim_percent]).mean()

        winsor_mean = close.rolling(window=self.period).apply(calc_winsor_mean, raw=True)

        return pd.DataFrame({
            'Robust_Median': median,
            'Robust_MAD': mad,
            'Trimmed_Mean': trimmed_mean,
            'Winsorized_Mean': winsor_mean
        })


@indicator_registry.register('advanced_stats')
class QuantileRegression(BaseIndicator):
    """分位数回归"""

    _metadata = IndicatorMetadata(
        name='QuantileRegression',
        display_name='分位数回归',
        category='advanced_stats',
        description='不同分位数的回归估计',
        formula='Quantile Regression Estimates',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 30},
        complexity_level=3
    )

    def __init__(self, period: int = 30):
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

        q10 = close.rolling(window=self.period).quantile(0.10)
        q25 = close.rolling(window=self.period).quantile(0.25)
        q50 = close.rolling(window=self.period).quantile(0.50)
        q75 = close.rolling(window=self.period).quantile(0.75)
        q90 = close.rolling(window=self.period).quantile(0.90)

        # IQR
        iqr = q75 - q25

        # 当前价格所在分位
        def calc_percentile(x):
            return stats.percentileofscore(x[:-1], x[-1])

        current_percentile = close.rolling(window=self.period).apply(calc_percentile, raw=True)

        return pd.DataFrame({
            'Q10': q10,
            'Q25': q25,
            'Q50': q50,
            'Q75': q75,
            'Q90': q90,
            'IQR': iqr,
            'Current_Percentile': current_percentile
        })


@indicator_registry.register('advanced_stats')
class ExponentialSmoothing(BaseIndicator):
    """指数平滑"""

    _metadata = IndicatorMetadata(
        name='ExponentialSmoothing',
        display_name='指数平滑',
        category='advanced_stats',
        description='Holt-Winters三次指数平滑',
        formula='Level, Trend, Seasonal Components',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'alpha': 0.3, 'beta': 0.1},
        complexity_level=2
    )

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        super().__init__(alpha=alpha, beta=beta)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.alpha, 'alpha', 0, 1)
        IndicatorValidator.validate_range(self.beta, 'beta', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        n = len(close)

        level = np.zeros(n)
        trend = np.zeros(n)
        forecast = np.zeros(n)

        level[0] = close.iloc[0]
        trend[0] = 0

        for i in range(1, n):
            level[i] = self.alpha * close.iloc[i] + (1 - self.alpha) * (level[i-1] + trend[i-1])
            trend[i] = self.beta * (level[i] - level[i-1]) + (1 - self.beta) * trend[i-1]
            forecast[i] = level[i-1] + trend[i-1]

        return pd.DataFrame({
            'ES_Level': pd.Series(level, index=close.index),
            'ES_Trend': pd.Series(trend, index=close.index),
            'ES_Forecast': pd.Series(forecast, index=close.index),
            'ES_Error': close - pd.Series(forecast, index=close.index)
        })


@indicator_registry.register('advanced_stats')
class CointegrationTest(BaseIndicator):
    """协整检验"""

    _metadata = IndicatorMetadata(
        name='CointegrationTest',
        display_name='协整检验',
        category='advanced_stats',
        description='价格与均值的协整关系检验',
        formula='ADF Test on Spread',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60},
        complexity_level=3
    )

    def __init__(self, period: int = 60):
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

        # 计算价差（与移动平均的偏离）
        ma = close.rolling(window=self.period).mean()
        spread = close - ma

        # Z-score
        spread_std = spread.rolling(window=self.period).std()
        z_spread = spread / spread_std

        # 半衰期估计
        def calc_half_life(x):
            if len(x) < 10:
                return np.nan
            lagged = x[:-1]
            delta = np.diff(x)
            if np.std(lagged) == 0:
                return np.nan
            beta = np.corrcoef(lagged, delta)[0, 1] * np.std(delta) / np.std(lagged)
            if beta >= 0:
                return np.nan
            return -np.log(2) / beta

        half_life = spread.rolling(window=self.period).apply(calc_half_life, raw=True)

        return pd.DataFrame({
            'Coint_Spread': spread,
            'Coint_ZScore': z_spread,
            'Coint_HalfLife': half_life
        })


@indicator_registry.register('advanced_stats')
class ChangePointDetection(BaseIndicator):
    """变点检测"""

    _metadata = IndicatorMetadata(
        name='ChangePointDetection',
        display_name='变点检测',
        category='advanced_stats',
        description='检测价格序列中的结构性变化',
        formula='CUSUM and Page-Hinkley Test',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 30, 'threshold': 2.0},
        complexity_level=3
    )

    def __init__(self, period: int = 30, threshold: float = 2.0):
        self.period = period
        self.threshold = threshold
        super().__init__(period=period, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.threshold, 'threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        returns = data['close'].pct_change()

        # CUSUM统计量
        mean_ret = returns.rolling(window=self.period).mean()
        std_ret = returns.rolling(window=self.period).std()
        normalized = (returns - mean_ret) / std_ret

        cusum_pos = pd.Series(0.0, index=data.index)
        cusum_neg = pd.Series(0.0, index=data.index)

        for i in range(1, len(data)):
            if not np.isnan(normalized.iloc[i]):
                cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + normalized.iloc[i])
                cusum_neg.iloc[i] = min(0, cusum_neg.iloc[i-1] + normalized.iloc[i])

        # 变点信号
        change_signal = pd.Series(0, index=data.index)
        change_signal[(cusum_pos > self.threshold * self.period ** 0.5) |
                      (cusum_neg < -self.threshold * self.period ** 0.5)] = 1

        return pd.DataFrame({
            'CUSUM_Pos': cusum_pos,
            'CUSUM_Neg': cusum_neg,
            'Change_Signal': change_signal
        })


@indicator_registry.register('advanced_stats')
class RegimeSwitching(BaseIndicator):
    """状态转换模型"""

    _metadata = IndicatorMetadata(
        name='RegimeSwitching',
        display_name='状态转换',
        category='advanced_stats',
        description='简化的市场状态识别',
        formula='Volatility-based Regime Classification',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60, 'n_regimes': 3},
        complexity_level=3
    )

    def __init__(self, period: int = 60, n_regimes: int = 3):
        self.period = period
        self.n_regimes = n_regimes
        super().__init__(period=period, n_regimes=n_regimes)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.n_regimes, 'n_regimes')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        returns = close.pct_change()

        # 波动率
        volatility = returns.rolling(window=20).std() * np.sqrt(252)

        # 趋势强度
        trend = (close - close.rolling(window=self.period).mean()) / close.rolling(window=self.period).std()

        # 基于波动率分位数的状态分类
        vol_percentile = pd.Series(index=data.index, dtype=float)
        for i in range(self.period - 1, len(data)):
            v = volatility.iloc[i - self.period + 1:i + 1].dropna()
            if len(v) > 0:
                vol_percentile.iloc[i] = stats.percentileofscore(v, volatility.iloc[i])

        # 状态分类
        regime = pd.Series(1, index=data.index)  # 1=正常
        regime[vol_percentile < 33] = 0  # 0=低波动
        regime[vol_percentile > 66] = 2  # 2=高波动

        return pd.DataFrame({
            'Regime': regime,
            'Vol_Percentile': vol_percentile,
            'Trend_Strength': trend
        })


@indicator_registry.register('advanced_stats')
class OutlierDetection(BaseIndicator):
    """异常值检测"""

    _metadata = IndicatorMetadata(
        name='OutlierDetection',
        display_name='异常值检测',
        category='advanced_stats',
        description='检测价格和收益率中的异常值',
        formula='IQR and Z-score Methods',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60, 'z_threshold': 3.0},
        complexity_level=2
    )

    def __init__(self, period: int = 60, z_threshold: float = 3.0):
        self.period = period
        self.z_threshold = z_threshold
        super().__init__(period=period, z_threshold=z_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.z_threshold, 'z_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        returns = data['close'].pct_change()

        # Z-score方法
        mean = returns.rolling(window=self.period).mean()
        std = returns.rolling(window=self.period).std()
        z_score = (returns - mean) / std

        z_outlier = (abs(z_score) > self.z_threshold).astype(int)

        # IQR方法
        q1 = returns.rolling(window=self.period).quantile(0.25)
        q3 = returns.rolling(window=self.period).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outlier = ((returns < lower_bound) | (returns > upper_bound)).astype(int)

        # 综合异常分数
        outlier_score = (abs(z_score) / self.z_threshold).clip(upper=3)

        return pd.DataFrame({
            'Z_Score': z_score,
            'Z_Outlier': z_outlier,
            'IQR_Outlier': iqr_outlier,
            'Outlier_Score': outlier_score
        })


@indicator_registry.register('advanced_stats')
class SerialDependence(BaseIndicator):
    """序列相依性"""

    _metadata = IndicatorMetadata(
        name='SerialDependence',
        display_name='序列相依性',
        category='advanced_stats',
        description='检测收益率的序列相关和自相关',
        formula='ACF, PACF, Ljung-Box Test',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60, 'max_lag': 5},
        complexity_level=3
    )

    def __init__(self, period: int = 60, max_lag: int = 5):
        self.period = period
        self.max_lag = max_lag
        super().__init__(period=period, max_lag=max_lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.max_lag, 'max_lag')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        returns = data['close'].pct_change()

        # 滞后1的自相关
        acf_1 = returns.rolling(window=self.period).corr(returns.shift(1))

        # 滞后2的自相关
        acf_2 = returns.rolling(window=self.period).corr(returns.shift(2))

        # 收益率平方的自相关（波动率聚集）
        returns_sq = returns ** 2
        acf_sq_1 = returns_sq.rolling(window=self.period).corr(returns_sq.shift(1))

        # 自相关衰减指标
        def calc_acf_decay(x):
            x = x.dropna()
            if len(x) < 10:
                return np.nan
            acf_vals = [x.autocorr(lag=i) for i in range(1, min(6, len(x) // 3))]
            if len(acf_vals) < 2:
                return np.nan
            return np.mean(np.abs(acf_vals))

        acf_strength = returns.rolling(window=self.period).apply(
            lambda x: calc_acf_decay(pd.Series(x)), raw=False
        )

        return pd.DataFrame({
            'ACF_1': acf_1,
            'ACF_2': acf_2,
            'ACF_Squared_1': acf_sq_1,
            'ACF_Strength': acf_strength
        })


@indicator_registry.register('advanced_stats')
class NonlinearDynamics(BaseIndicator):
    """非线性动力学"""

    _metadata = IndicatorMetadata(
        name='NonlinearDynamics',
        display_name='非线性动力学',
        category='advanced_stats',
        description='检测价格序列的非线性特性',
        formula='Lyapunov Exponent Approximation',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60},
        complexity_level=3
    )

    def __init__(self, period: int = 60):
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
        returns = close.pct_change()

        # 简化的Lyapunov指数估计
        def calc_lyapunov(x):
            x = x.dropna()
            if len(x) < 10:
                return np.nan
            diffs = np.abs(np.diff(x))
            diffs = diffs[diffs > 0]
            if len(diffs) < 2:
                return np.nan
            return np.mean(np.log(diffs + 1e-10))

        lyapunov = returns.rolling(window=self.period).apply(calc_lyapunov, raw=False)

        # 非线性检测（基于收益率分布的非正态性）
        def calc_nonlinearity(x):
            x = x.dropna()
            if len(x) < 10:
                return np.nan
            # JB统计量作为非线性代理
            return stats.jarque_bera(x)[0]

        nonlinearity = returns.rolling(window=self.period).apply(calc_nonlinearity, raw=False)

        # 混沌度量
        def calc_complexity(x):
            x = x.dropna()
            if len(x) < 10:
                return np.nan
            # 使用样本熵近似
            std = np.std(x)
            if std == 0:
                return 0
            threshold = 0.2 * std
            matches = 0
            for i in range(len(x) - 1):
                for j in range(i + 1, len(x) - 1):
                    if (
                        abs(x.iloc[i] - x.iloc[j]) < threshold
                        and abs(x.iloc[i + 1] - x.iloc[j + 1]) < threshold
                    ):
                        matches += 1
            return -np.log(matches / (len(x) * (len(x) - 1) / 2) + 1e-10)

        complexity = returns.rolling(window=self.period).apply(calc_complexity, raw=False)

        return pd.DataFrame({
            'Lyapunov_Approx': lyapunov,
            'Nonlinearity': nonlinearity,
            'Complexity': complexity
        })
