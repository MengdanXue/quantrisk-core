"""
机器学习特征指标 (20个)

用于机器学习模型的特征工程指标，包括：
- 时间序列特征生成
- 统计特征提取
- 技术指标组合与变换
- 标签生成器
"""

from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures as SKPolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ============================================================================
# 时间序列特征 (5个)
# ============================================================================

@indicator_registry.register('ml_features')
class LagFeatures(BaseIndicator):
    """
    滞后特征生成器

    生成指定列的多个滞后期特征，用于捕捉时间依赖性
    """

    _metadata = IndicatorMetadata(
        name='LagFeatures',
        display_name='滞后特征',
        category='ml_features',
        description='生成多个滞后期特征，用于时间序列建模',
        formula='lag_t = value_{t-n}',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'lags': [1, 2, 3, 5, 10], 'column': 'close'},
        complexity_level=1
    )

    def __init__(self, lags: List[int] = None, column: str = 'close'):
        self.lags = lags or [1, 2, 3, 5, 10]
        self.column = column
        super().__init__(lags=self.lags, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_type(self.lags, 'lags', list)
        for lag in self.lags:
            IndicatorValidator.validate_positive_int(lag, f'lag_{lag}')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        for lag in self.lags:
            result[f'{self.column}_lag_{lag}'] = data[self.column].shift(lag)

        return result


@indicator_registry.register('ml_features')
class RollingFeatures(BaseIndicator):
    """
    滚动窗口特征

    计算滚动窗口内的统计特征（均值、标准差、最大值、最小值等）
    """

    _metadata = IndicatorMetadata(
        name='RollingFeatures',
        display_name='滚动窗口特征',
        category='ml_features',
        description='计算滚动窗口的统计特征',
        formula='rolling_stat = func(window)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'column': 'close', 'features': ['mean', 'std', 'min', 'max']},
        complexity_level=2
    )

    def __init__(self, window: int = 20, column: str = 'close',
                 features: List[str] = None):
        self.window = window
        self.column = column
        self.features = features or ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
        super().__init__(window=window, column=column, features=self.features)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        valid_features = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt', 'sum']
        for feat in self.features:
            if feat not in valid_features:
                raise ValueError(f"Invalid feature: {feat}. Must be one of {valid_features}")

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        rolling = data[self.column].rolling(window=self.window)

        for feat in self.features:
            if feat == 'mean':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.mean()
            elif feat == 'std':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.std()
            elif feat == 'min':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.min()
            elif feat == 'max':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.max()
            elif feat == 'median':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.median()
            elif feat == 'skew':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.skew()
            elif feat == 'kurt':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.kurt()
            elif feat == 'sum':
                result[f'{self.column}_rolling_{feat}_{self.window}'] = rolling.sum()

        return result


@indicator_registry.register('ml_features')
class DiffFeatures(BaseIndicator):
    """
    差分特征

    计算一阶、二阶差分，以及百分比变化
    """

    _metadata = IndicatorMetadata(
        name='DiffFeatures',
        display_name='差分特征',
        category='ml_features',
        description='计算差分和百分比变化特征',
        formula='diff1 = value_t - value_{t-1}',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'column': 'close', 'orders': [1, 2]},
        complexity_level=1
    )

    def __init__(self, column: str = 'close', orders: List[int] = None):
        self.column = column
        self.orders = orders or [1, 2]
        super().__init__(column=column, orders=self.orders)

    def _validate_params(self):
        for order in self.orders:
            IndicatorValidator.validate_positive_int(order, f'order_{order}')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 差分
        for order in self.orders:
            result[f'{self.column}_diff_{order}'] = data[self.column].diff(order)

        # 百分比变化
        result[f'{self.column}_pct_change'] = data[self.column].pct_change()

        # 对数收益率
        result[f'{self.column}_log_return'] = np.log(data[self.column] / data[self.column].shift(1))

        return result


@indicator_registry.register('ml_features')
class AutocorrelationFeature(BaseIndicator):
    """
    自相关特征

    计算不同滞后期的自相关系数
    """

    _metadata = IndicatorMetadata(
        name='AutocorrelationFeature',
        display_name='自相关特征',
        category='ml_features',
        description='计算时间序列的自相关系数',
        formula='ACF(lag) = corr(X_t, X_{t-lag})',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'lags': [1, 5, 10, 20], 'window': 50, 'column': 'close'},
        complexity_level=3
    )

    def __init__(self, lags: List[int] = None, window: int = 50, column: str = 'close'):
        self.lags = lags or [1, 5, 10, 20]
        self.window = window
        self.column = column
        super().__init__(lags=self.lags, window=window, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        for lag in self.lags:
            IndicatorValidator.validate_positive_int(lag, f'lag_{lag}')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        for lag in self.lags:
            acf_values = []
            for i in range(len(series)):
                if i < self.window + lag:
                    acf_values.append(np.nan)
                else:
                    window_data = series.iloc[i-self.window:i]
                    lagged_data = series.iloc[i-self.window-lag:i-lag]
                    if len(window_data) == len(lagged_data) and len(window_data) > 0:
                        corr = np.corrcoef(window_data, lagged_data)[0, 1]
                        acf_values.append(corr if np.isfinite(corr) else 0)
                    else:
                        acf_values.append(np.nan)

            result[f'{self.column}_acf_{lag}'] = acf_values

        return result


@indicator_registry.register('ml_features')
class PartialAutocorrelation(BaseIndicator):
    """
    偏自相关特征

    计算偏自相关系数，用于识别AR模型的阶数
    """

    _metadata = IndicatorMetadata(
        name='PartialAutocorrelation',
        display_name='偏自相关特征',
        category='ml_features',
        description='计算偏自相关系数',
        formula='PACF(lag) = partial correlation with lag',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'lags': [1, 5, 10], 'window': 50, 'column': 'close'},
        complexity_level=4
    )

    def __init__(self, lags: List[int] = None, window: int = 50, column: str = 'close'):
        self.lags = lags or [1, 5, 10]
        self.window = window
        self.column = column
        super().__init__(lags=self.lags, window=window, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        for lag in self.lags:
            IndicatorValidator.validate_positive_int(lag, f'lag_{lag}')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # 简化的PACF计算（通过Yule-Walker方程）
        for target_lag in self.lags:
            pacf_values = []
            for i in range(len(series)):
                if i < self.window + target_lag:
                    pacf_values.append(np.nan)
                else:
                    window_data = series.iloc[i-self.window:i].values

                    # 构建自回归矩阵
                    try:
                        # 使用OLS估计PACF
                        X = np.column_stack([window_data[target_lag-j:-j if j > 0 else None]
                                           for j in range(1, target_lag+1)])
                        y = window_data[target_lag:]

                        if len(X) > target_lag and np.linalg.matrix_rank(X) == target_lag:
                            coef = np.linalg.lstsq(X, y, rcond=None)[0]
                            pacf_values.append(coef[-1])
                        else:
                            pacf_values.append(0)
                    except:
                        pacf_values.append(0)

            result[f'{self.column}_pacf_{target_lag}'] = pacf_values

        return result


# ============================================================================
# 统计特征 (5个)
# ============================================================================

@indicator_registry.register('ml_features')
class DistributionFeatures(BaseIndicator):
    """
    分布特征

    计算偏度、峰度等分布统计量
    """

    _metadata = IndicatorMetadata(
        name='DistributionFeatures',
        display_name='分布特征',
        category='ml_features',
        description='计算价格分布的统计特征（偏度、峰度等）',
        formula='skew = E[(X-μ)³]/σ³, kurt = E[(X-μ)⁴]/σ⁴',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'column': 'close'},
        complexity_level=2
    )

    def __init__(self, window: int = 20, column: str = 'close'):
        self.window = window
        self.column = column
        super().__init__(window=window, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        returns = data[self.column].pct_change()

        # 偏度
        result[f'{self.column}_skew_{self.window}'] = returns.rolling(self.window).skew()

        # 峰度
        result[f'{self.column}_kurt_{self.window}'] = returns.rolling(self.window).kurt()

        # 四分位数
        result[f'{self.column}_q25_{self.window}'] = returns.rolling(self.window).quantile(0.25)
        result[f'{self.column}_q75_{self.window}'] = returns.rolling(self.window).quantile(0.75)
        result[f'{self.column}_iqr_{self.window}'] = (
            result[f'{self.column}_q75_{self.window}'] - result[f'{self.column}_q25_{self.window}']
        )

        # Jarque-Bera统计量 (正态性检验)
        def jb_stat(x):
            if len(x) < 4:
                return np.nan
            n = len(x)
            s = stats.skew(x)
            k = stats.kurtosis(x)
            return n/6 * (s**2 + (k**2)/4)

        result[f'{self.column}_jb_stat_{self.window}'] = (
            returns.rolling(self.window).apply(jb_stat, raw=True)
        )

        return result


@indicator_registry.register('ml_features')
class OutlierScore(BaseIndicator):
    """
    异常值评分

    使用Z-score和IQR方法检测异常值
    """

    _metadata = IndicatorMetadata(
        name='OutlierScore',
        display_name='异常值评分',
        category='ml_features',
        description='计算异常值评分，用于识别极端市场行为',
        formula='z_score = (x - μ) / σ',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'column': 'close'},
        complexity_level=2
    )

    def __init__(self, window: int = 20, column: str = 'close'):
        self.window = window
        self.column = column
        super().__init__(window=window, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # Z-score
        rolling_mean = series.rolling(self.window).mean()
        rolling_std = series.rolling(self.window).std()
        result[f'{self.column}_zscore_{self.window}'] = (series - rolling_mean) / rolling_std

        # Modified Z-score (使用中位数)
        rolling_median = series.rolling(self.window).median()
        mad = (series - rolling_median).abs().rolling(self.window).median()
        result[f'{self.column}_mod_zscore_{self.window}'] = 0.6745 * (series - rolling_median) / mad

        # IQR方法
        q25 = series.rolling(self.window).quantile(0.25)
        q75 = series.rolling(self.window).quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        result[f'{self.column}_outlier_iqr_{self.window}'] = (
            ((series < lower_bound) | (series > upper_bound)).astype(int)
        )

        # 距离边界的距离
        result[f'{self.column}_distance_to_bound_{self.window}'] = np.minimum(
            (series - lower_bound).abs(),
            (series - upper_bound).abs()
        )

        return result


@indicator_registry.register('ml_features')
class ChangePointDetection(BaseIndicator):
    """
    变点检测

    检测时间序列中的结构性变化点
    """

    _metadata = IndicatorMetadata(
        name='ChangePointDetection',
        display_name='变点检测',
        category='ml_features',
        description='检测时间序列的结构性变化点',
        formula='CUSUM = sum(x_i - μ)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'threshold': 2.0, 'column': 'close'},
        complexity_level=3
    )

    def __init__(self, window: int = 20, threshold: float = 2.0, column: str = 'close'):
        self.window = window
        self.threshold = threshold
        self.column = column
        super().__init__(window=window, threshold=threshold, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.threshold, 'threshold')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # CUSUM (累积和控制图)
        returns = series.pct_change()
        mean_return = returns.rolling(self.window).mean()
        cusum_pos = pd.Series(0.0, index=data.index)
        cusum_neg = pd.Series(0.0, index=data.index)

        for i in range(1, len(returns)):
            if i >= self.window:
                deviation = returns.iloc[i] - mean_return.iloc[i]
                cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + deviation)
                cusum_neg.iloc[i] = min(0, cusum_neg.iloc[i-1] + deviation)

        result[f'{self.column}_cusum_pos_{self.window}'] = cusum_pos
        result[f'{self.column}_cusum_neg_{self.window}'] = cusum_neg

        # 变点信号
        result[f'{self.column}_changepoint_{self.window}'] = (
            (cusum_pos.abs() > self.threshold) | (cusum_neg.abs() > self.threshold)
        ).astype(int)

        # 均值变化检测
        window_mean = series.rolling(self.window).mean()
        mean_change = window_mean.diff(self.window)
        result[f'{self.column}_mean_change_{self.window}'] = mean_change

        # 方差变化检测
        window_std = series.rolling(self.window).std()
        std_change = window_std.diff(self.window)
        result[f'{self.column}_std_change_{self.window}'] = std_change

        return result


@indicator_registry.register('ml_features')
class TrendDecomposition(BaseIndicator):
    """
    趋势分解特征

    将时间序列分解为趋势、季节性和残差成分
    """

    _metadata = IndicatorMetadata(
        name='TrendDecomposition',
        display_name='趋势分解',
        category='ml_features',
        description='分解时间序列的趋势成分',
        formula='Y = Trend + Seasonal + Residual',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'column': 'close'},
        complexity_level=3
    )

    def __init__(self, window: int = 20, column: str = 'close'):
        self.window = window
        self.column = column
        super().__init__(window=window, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # 趋势成分（移动平均）
        trend = series.rolling(window=self.window, center=True).mean()
        result[f'{self.column}_trend_{self.window}'] = trend

        # 去趋势
        detrended = series - trend
        result[f'{self.column}_detrended_{self.window}'] = detrended

        # 趋势强度
        trend_strength = 1 - (detrended.var() / series.var()) if series.var() > 0 else 0
        result[f'{self.column}_trend_strength_{self.window}'] = trend_strength

        # 趋势方向
        trend_slope = trend.diff(self.window) / self.window
        result[f'{self.column}_trend_slope_{self.window}'] = trend_slope

        # 趋势加速度
        trend_accel = trend_slope.diff(self.window)
        result[f'{self.column}_trend_accel_{self.window}'] = trend_accel

        return result


@indicator_registry.register('ml_features')
class SeasonalDecomposition(BaseIndicator):
    """
    季节性分解

    提取时间序列的周期性模式
    """

    _metadata = IndicatorMetadata(
        name='SeasonalDecomposition',
        display_name='季节性分解',
        category='ml_features',
        description='提取周期性和季节性特征',
        formula='Seasonal = Y - Trend',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 5, 'column': 'close'},
        complexity_level=3
    )

    def __init__(self, period: int = 5, column: str = 'close'):
        self.period = period
        self.column = column
        super().__init__(period=period, column=column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # 计算周期性模式
        seasonal_avg = pd.Series(0.0, index=data.index)
        for i in range(self.period):
            mask = np.arange(len(series)) % self.period == i
            period_mean = series[mask].mean()
            seasonal_avg[mask] = period_mean

        result[f'{self.column}_seasonal_{self.period}'] = seasonal_avg

        # 去季节性
        deseasonalized = series - seasonal_avg
        result[f'{self.column}_deseasonalized_{self.period}'] = deseasonalized

        # 季节性强度
        seasonal_strength = (seasonal_avg.var() / series.var()) if series.var() > 0 else 0
        result[f'{self.column}_seasonal_strength_{self.period}'] = seasonal_strength

        # 周期内位置
        result[f'{self.column}_period_position'] = np.arange(len(series)) % self.period

        # 周期幅度
        period_range = series.rolling(self.period).max() - series.rolling(self.period).min()
        result[f'{self.column}_period_range_{self.period}'] = period_range

        return result


# ============================================================================
# 技术特征组合 (5个)
# ============================================================================

@indicator_registry.register('ml_features')
class IndicatorClusters(BaseIndicator):
    """
    指标聚类特征

    将多个技术指标进行聚类，识别市场状态
    """

    _metadata = IndicatorMetadata(
        name='IndicatorClusters',
        display_name='指标聚类',
        category='ml_features',
        description='对技术指标进行聚类分析，识别市场状态',
        formula='cluster = KMeans(indicators)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'n_clusters': 3, 'window': 20},
        complexity_level=4
    )

    def __init__(self, n_clusters: int = 3, window: int = 20):
        self.n_clusters = n_clusters
        self.window = window
        super().__init__(n_clusters=n_clusters, window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.n_clusters, 'n_clusters')
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 创建特征矩阵
        features = pd.DataFrame()

        # 价格特征
        features['return'] = data['close'].pct_change()
        features['volatility'] = features['return'].rolling(self.window).std()
        features['rsi'] = self._calculate_rsi(data['close'], self.window)

        # 成交量特征
        features['volume_ma'] = data['volume'].rolling(self.window).mean()
        features['volume_std'] = data['volume'].rolling(self.window).std()

        # 填充NaN
        features = features.bfill().fillna(0)

        # 标准化
        features_norm = (features - features.mean()) / (features.std() + 1e-8)

        # 聚类
        cluster_labels = np.full(len(data), -1)

        for i in range(self.window, len(data)):
            window_data = features_norm.iloc[i-self.window:i+1].values
            if len(window_data) >= self.n_clusters:
                try:
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(window_data)
                    cluster_labels[i] = labels[-1]
                except:
                    cluster_labels[i] = -1

        result['market_cluster'] = cluster_labels

        # 聚类稳定性（最近N个点的聚类一致性）
        stability_window = min(5, self.window)
        result['cluster_stability'] = pd.Series(cluster_labels).rolling(
            stability_window
        ).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)

        return result

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))


@indicator_registry.register('ml_features')
class TechnicalPCA(BaseIndicator):
    """
    技术指标PCA降维

    对多个技术指标进行主成分分析，提取主要特征
    """

    _metadata = IndicatorMetadata(
        name='TechnicalPCA',
        display_name='技术指标PCA',
        category='ml_features',
        description='对技术指标进行PCA降维',
        formula='PC = PCA(indicators)',
        required_columns=['close', 'high', 'low'],
        output_type='dataframe',
        default_params={'n_components': 3, 'window': 50},
        complexity_level=4
    )

    def __init__(self, n_components: int = 3, window: int = 50):
        self.n_components = n_components
        self.window = window
        super().__init__(n_components=n_components, window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.n_components, 'n_components')
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 创建技术指标特征
        features = pd.DataFrame()
        features['return'] = data['close'].pct_change()
        features['high_low_range'] = (data['high'] - data['low']) / data['close']
        features['ma_20'] = data['close'].rolling(20).mean() / data['close']
        features['ma_50'] = data['close'].rolling(50).mean() / data['close']
        features['volatility'] = features['return'].rolling(20).std()

        # 填充NaN
        features = features.bfill().fillna(0)

        # 滚动PCA
        for i in range(n_components := self.n_components):
            result[f'pca_component_{i+1}'] = np.nan

        result['pca_explained_variance'] = np.nan

        for i in range(self.window, len(data)):
            window_features = features.iloc[i-self.window:i].values

            try:
                pca = PCA(n_components=self.n_components)
                transformed = pca.fit_transform(window_features)

                # 最新点的主成分
                for j in range(self.n_components):
                    result.iloc[i, j] = transformed[-1, j]

                # 累积解释方差
                result.iloc[i, self.n_components] = pca.explained_variance_ratio_.sum()
            except:
                pass

        return result


@indicator_registry.register('ml_features')
class FeatureInteraction(BaseIndicator):
    """
    特征交互

    创建特征之间的交互项（乘积、比率等）
    """

    _metadata = IndicatorMetadata(
        name='FeatureInteraction',
        display_name='特征交互',
        category='ml_features',
        description='创建特征间的交互项',
        formula='interaction = f1 * f2, f1 / f2',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 基础特征
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.window).std()
        volume_ma = data['volume'].rolling(self.window).mean()

        # 交互特征
        # 1. 收益率 × 波动率
        result['return_vol_interaction'] = returns * volatility

        # 2. 收益率 × 成交量
        result['return_volume_interaction'] = returns * (data['volume'] / volume_ma)

        # 3. 波动率 × 成交量
        result['vol_volume_interaction'] = volatility * (data['volume'] / volume_ma)

        # 4. 价格动量 × 波动率
        momentum = data['close'] / data['close'].rolling(self.window).mean() - 1
        result['momentum_vol_interaction'] = momentum * volatility

        # 5. 相对成交量 × 相对价格
        rel_volume = data['volume'] / volume_ma
        rel_price = data['close'] / data['close'].rolling(self.window).mean()
        result['rel_volume_price_interaction'] = rel_volume * rel_price

        # 6. 收益率符号 × 成交量
        result['return_sign_volume'] = np.sign(returns) * rel_volume

        return result


@indicator_registry.register('ml_features')
class NonlinearTransform(BaseIndicator):
    """
    非线性变换

    对特征进行非线性变换（log, sqrt, power等）
    """

    _metadata = IndicatorMetadata(
        name='NonlinearTransform',
        display_name='非线性变换',
        category='ml_features',
        description='对特征进行非线性变换',
        formula='f(x) = log(x), sqrt(x), x^2, exp(x)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'column': 'close'},
        complexity_level=2
    )

    def __init__(self, column: str = 'close'):
        self.column = column
        super().__init__(column=column)

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)
        series = data[self.column]

        # 对数变换
        result[f'{self.column}_log'] = np.log(series + 1e-8)

        # 平方根变换
        result[f'{self.column}_sqrt'] = np.sqrt(np.abs(series))

        # 平方变换
        result[f'{self.column}_square'] = series ** 2

        # 立方变换
        result[f'{self.column}_cube'] = series ** 3

        # Box-Cox类似变换
        result[f'{self.column}_boxcox'] = (series ** 0.5 - 1) / 0.5

        # 反转
        result[f'{self.column}_reciprocal'] = 1 / (series + 1e-8)

        # Sigmoid变换（标准化到0-1）
        mean_val = series.mean()
        std_val = series.std()
        z_score = (series - mean_val) / (std_val + 1e-8)
        result[f'{self.column}_sigmoid'] = 1 / (1 + np.exp(-z_score))

        # Tanh变换
        result[f'{self.column}_tanh'] = np.tanh(z_score)

        return result


@indicator_registry.register('ml_features')
class PolynomialFeatures(BaseIndicator):
    """
    多项式特征

    生成多项式和交互特征
    """

    _metadata = IndicatorMetadata(
        name='PolynomialFeatures',
        display_name='多项式特征',
        category='ml_features',
        description='生成多项式特征',
        formula='poly = [1, x, x^2, x*y, y^2, ...]',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'degree': 2, 'window': 20},
        complexity_level=3
    )

    def __init__(self, degree: int = 2, window: int = 20):
        self.degree = degree
        self.window = window
        super().__init__(degree=degree, window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.degree, 'degree')
        IndicatorValidator.validate_range(self.degree, 'degree', 1, 3)
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 创建基础特征
        base_features = pd.DataFrame()
        base_features['return'] = data['close'].pct_change()
        base_features['ma_ratio'] = data['close'] / data['close'].rolling(self.window).mean()
        base_features['volatility'] = base_features['return'].rolling(self.window).std()

        # 填充NaN
        base_features = base_features.bfill().fillna(0)

        # 生成多项式特征
        if self.degree >= 2:
            # 二次项
            for col in base_features.columns:
                result[f'{col}_squared'] = base_features[col] ** 2

            # 交叉项
            cols = list(base_features.columns)
            for i, col1 in enumerate(cols):
                for col2 in cols[i+1:]:
                    result[f'{col1}_{col2}_interaction'] = (
                        base_features[col1] * base_features[col2]
                    )

        if self.degree >= 3:
            # 三次项
            for col in base_features.columns:
                result[f'{col}_cubed'] = base_features[col] ** 3

        return result


# ============================================================================
# 标签生成 (5个)
# ============================================================================

@indicator_registry.register('ml_features')
class TrendLabeler(BaseIndicator):
    """
    趋势标签生成

    基于未来收益率生成趋势标签（上涨/下跌/横盘）
    """

    _metadata = IndicatorMetadata(
        name='TrendLabeler',
        display_name='趋势标签',
        category='ml_features',
        description='生成趋势分类标签',
        formula='label = sign(future_return)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'forward_period': 5, 'threshold': 0.01},
        complexity_level=2
    )

    def __init__(self, forward_period: int = 5, threshold: float = 0.01):
        self.forward_period = forward_period
        self.threshold = threshold
        super().__init__(forward_period=forward_period, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.forward_period, 'forward_period')
        IndicatorValidator.validate_positive_float(self.threshold, 'threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 未来收益率
        future_return = data['close'].pct_change(self.forward_period).shift(-self.forward_period)

        # 分类标签 (0: 下跌, 1: 横盘, 2: 上涨)
        labels = pd.Series(1, index=data.index)  # 默认横盘
        labels[future_return > self.threshold] = 2  # 上涨
        labels[future_return < -self.threshold] = 0  # 下跌

        result[f'trend_label_{self.forward_period}'] = labels

        # 二元标签 (0: 下跌, 1: 上涨)
        binary_labels = (future_return > 0).astype(int)
        result[f'binary_label_{self.forward_period}'] = binary_labels

        # 连续值（实际未来收益率）
        result[f'future_return_{self.forward_period}'] = future_return

        # 趋势强度
        result[f'trend_strength_{self.forward_period}'] = future_return.abs()

        return result


@indicator_registry.register('ml_features')
class VolatilityRegime(BaseIndicator):
    """
    波动率区制

    基于波动率水平划分市场区制
    """

    _metadata = IndicatorMetadata(
        name='VolatilityRegime',
        display_name='波动率区制',
        category='ml_features',
        description='识别高/中/低波动率区制',
        formula='regime = quantile(volatility)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'window': 20, 'n_regimes': 3},
        complexity_level=2
    )

    def __init__(self, window: int = 20, n_regimes: int = 3):
        self.window = window
        self.n_regimes = n_regimes
        super().__init__(window=window, n_regimes=n_regimes)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.n_regimes, 'n_regimes')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 计算波动率
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.window).std()

        # 基于历史分位数划分区制
        regime_labels = pd.Series(0, index=data.index)

        for i in range(self.window * 2, len(data)):
            hist_vol = volatility.iloc[:i]
            current_vol = volatility.iloc[i]

            if pd.notna(current_vol):
                # 计算分位数边界
                quantiles = [hist_vol.quantile(j / self.n_regimes)
                           for j in range(1, self.n_regimes)]

                # 确定区制
                regime = 0
                for q in quantiles:
                    if current_vol > q:
                        regime += 1

                regime_labels.iloc[i] = regime

        result[f'volatility_regime_{self.window}'] = regime_labels
        result[f'volatility_{self.window}'] = volatility

        # 波动率变化率
        result[f'volatility_change_{self.window}'] = volatility.pct_change()

        # 相对波动率
        vol_ma = volatility.rolling(self.window * 2).mean()
        result[f'relative_volatility_{self.window}'] = volatility / (vol_ma + 1e-8)

        return result


@indicator_registry.register('ml_features')
class ReturnBuckets(BaseIndicator):
    """
    收益率分桶

    将收益率离散化为桶/类别
    """

    _metadata = IndicatorMetadata(
        name='ReturnBuckets',
        display_name='收益率分桶',
        category='ml_features',
        description='将收益率分桶为离散类别',
        formula='bucket = quantile_bin(return)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'n_buckets': 5, 'forward_period': 1},
        complexity_level=2
    )

    def __init__(self, n_buckets: int = 5, forward_period: int = 1):
        self.n_buckets = n_buckets
        self.forward_period = forward_period
        super().__init__(n_buckets=n_buckets, forward_period=forward_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.n_buckets, 'n_buckets')
        IndicatorValidator.validate_positive_int(self.forward_period, 'forward_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 未来收益率
        future_return = data['close'].pct_change(self.forward_period).shift(-self.forward_period)

        # 基于分位数分桶
        buckets = pd.qcut(
            future_return.dropna(),
            q=self.n_buckets,
            labels=False,
            duplicates='drop'
        )

        # 对齐到原索引
        result[f'return_bucket_{self.forward_period}'] = buckets.reindex(data.index)

        # 等宽分桶
        bins = pd.cut(
            future_return.dropna(),
            bins=self.n_buckets,
            labels=False
        )
        result[f'return_bin_equal_{self.forward_period}'] = bins.reindex(data.index)

        # 标准化分桶（基于标准差）
        mean_return = future_return.mean()
        std_return = future_return.std()

        std_buckets = pd.Series(self.n_buckets // 2, index=data.index)
        for i in range(self.n_buckets):
            lower = mean_return + (i - self.n_buckets // 2) * std_return
            upper = mean_return + (i - self.n_buckets // 2 + 1) * std_return
            mask = (future_return >= lower) & (future_return < upper)
            std_buckets[mask] = i

        result[f'return_bucket_std_{self.forward_period}'] = std_buckets

        return result


@indicator_registry.register('ml_features')
class TripleBarrier(BaseIndicator):
    """
    三重障碍标签

    基于止盈、止损和时间障碍生成标签
    """

    _metadata = IndicatorMetadata(
        name='TripleBarrier',
        display_name='三重障碍标签',
        category='ml_features',
        description='基于止盈/止损/时间障碍生成标签',
        formula='label = first(take_profit, stop_loss, max_holding)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={
            'take_profit': 0.02,
            'stop_loss': 0.02,
            'max_holding': 10
        },
        complexity_level=4
    )

    def __init__(self, take_profit: float = 0.02, stop_loss: float = 0.02,
                 max_holding: int = 10):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_holding = max_holding
        super().__init__(
            take_profit=take_profit,
            stop_loss=stop_loss,
            max_holding=max_holding
        )

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.take_profit, 'take_profit')
        IndicatorValidator.validate_positive_float(self.stop_loss, 'stop_loss')
        IndicatorValidator.validate_positive_int(self.max_holding, 'max_holding')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        labels = pd.Series(0, index=data.index)  # 0: 未知, 1: 止盈, -1: 止损, 2: 时间到期
        exit_time = pd.Series(np.nan, index=data.index)
        exit_return = pd.Series(np.nan, index=data.index)

        close_prices = data['close'].values

        for i in range(len(data) - self.max_holding):
            entry_price = close_prices[i]

            # 检查未来价格
            for j in range(1, min(self.max_holding + 1, len(data) - i)):
                future_price = close_prices[i + j]
                ret = (future_price - entry_price) / entry_price

                # 检查止盈
                if ret >= self.take_profit:
                    labels.iloc[i] = 1
                    exit_time.iloc[i] = j
                    exit_return.iloc[i] = ret
                    break

                # 检查止损
                if ret <= -self.stop_loss:
                    labels.iloc[i] = -1
                    exit_time.iloc[i] = j
                    exit_return.iloc[i] = ret
                    break

                # 时间到期
                if j == self.max_holding:
                    labels.iloc[i] = 2
                    exit_time.iloc[i] = j
                    exit_return.iloc[i] = ret

        result['triple_barrier_label'] = labels
        result['triple_barrier_exit_time'] = exit_time
        result['triple_barrier_return'] = exit_return

        # 二元标签（盈利/亏损）
        result['triple_barrier_binary'] = (exit_return > 0).astype(int)

        return result


@indicator_registry.register('ml_features')
class MetaLabeling(BaseIndicator):
    """
    元标签

    生成二级标签，用于过滤主模型的预测
    """

    _metadata = IndicatorMetadata(
        name='MetaLabeling',
        display_name='元标签',
        category='ml_features',
        description='生成元标签用于模型过滤',
        formula='meta_label = should_trade(primary_signal)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={
            'forward_period': 5,
            'confidence_threshold': 0.01
        },
        complexity_level=3
    )

    def __init__(self, forward_period: int = 5, confidence_threshold: float = 0.01):
        self.forward_period = forward_period
        self.confidence_threshold = confidence_threshold
        super().__init__(
            forward_period=forward_period,
            confidence_threshold=confidence_threshold
        )

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.forward_period, 'forward_period')
        IndicatorValidator.validate_positive_float(self.confidence_threshold, 'confidence_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        result = pd.DataFrame(index=data.index)

        # 计算未来收益
        future_return = data['close'].pct_change(self.forward_period).shift(-self.forward_period)

        # 元标签: 收益率绝对值是否超过阈值（是否应该交易）
        should_trade = (future_return.abs() > self.confidence_threshold).astype(int)
        result[f'meta_should_trade_{self.forward_period}'] = should_trade

        # 收益质量评分
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()

        # Sharpe比率（简化版）
        rolling_sharpe = (returns.rolling(self.forward_period).mean() /
                         (volatility + 1e-8))
        result[f'meta_quality_score_{self.forward_period}'] = rolling_sharpe

        # 趋势一致性
        ma_short = data['close'].rolling(5).mean()
        ma_long = data['close'].rolling(20).mean()
        trend_alignment = ((ma_short > ma_long) & (future_return > 0)) | \
                         ((ma_short < ma_long) & (future_return < 0))
        result[f'meta_trend_aligned_{self.forward_period}'] = trend_alignment.astype(int)

        # 信号强度
        momentum = data['close'] / data['close'].rolling(self.forward_period).mean() - 1
        signal_strength = momentum.abs()
        result[f'meta_signal_strength_{self.forward_period}'] = signal_strength

        # 组合元标签
        combined_meta = (should_trade & trend_alignment).astype(int)
        result[f'meta_combined_{self.forward_period}'] = combined_meta

        return result
