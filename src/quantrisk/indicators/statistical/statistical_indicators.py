"""
统计类指标 (20个)

基于统计学方法的技术分析指标
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('statistical')
class ZScore(BaseIndicator):
    """
    Z-Score (Z分数)

    价格偏离均值的标准差倍数
    """

    _metadata = IndicatorMetadata(
        name='ZScore',
        display_name='Z分数',
        category='statistical',
        description='价格偏离均值的标准差倍数',
        formula='Z = (X - μ) / σ',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        mean = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        z_score = (close - mean) / std

        return z_score


@indicator_registry.register('statistical')
class PercentRank(BaseIndicator):
    """
    Percent Rank (百分位排名)

    当前值在历史分布中的位置
    """

    _metadata = IndicatorMetadata(
        name='PercentRank',
        display_name='百分位排名',
        category='statistical',
        description='当前值在过去N周期中的百分位位置',
        formula='%Rank = (小于当前值的数量 / N) * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 100},
        complexity_level=1
    )

    def __init__(self, period: int = 100):
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

        def calc_percentile(x):
            return stats.percentileofscore(x[:-1], x[-1])

        percent_rank = close.rolling(window=self.period).apply(calc_percentile, raw=True)

        return percent_rank


@indicator_registry.register('statistical')
class StandardDeviation(BaseIndicator):
    """
    Standard Deviation (标准差)

    价格的滚动标准差
    """

    _metadata = IndicatorMetadata(
        name='StandardDeviation',
        display_name='标准差',
        category='statistical',
        description='衡量价格波动的离散程度',
        formula='σ = √(Σ(X - μ)² / N)',
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

        std = close.rolling(window=self.period).std()
        mean = close.rolling(window=self.period).mean()

        # 变异系数
        cv = std / mean * 100

        return pd.DataFrame({
            'StdDev': std,
            'Mean': mean,
            'CV': cv
        })


@indicator_registry.register('statistical')
class Variance(BaseIndicator):
    """
    Variance (方差)

    价格的滚动方差
    """

    _metadata = IndicatorMetadata(
        name='Variance',
        display_name='方差',
        category='statistical',
        description='衡量价格波动的离散程度的平方',
        formula='σ² = Σ(X - μ)² / N',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        variance = close.rolling(window=self.period).var()

        return variance


@indicator_registry.register('statistical')
class Skewness(BaseIndicator):
    """
    Skewness (偏度)

    价格分布的不对称性
    """

    _metadata = IndicatorMetadata(
        name='Skewness',
        display_name='偏度',
        category='statistical',
        description='衡量价格分布的不对称程度',
        formula='Skew = E[(X-μ)³] / σ³',
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

        skewness = close.rolling(window=self.period).skew()

        return skewness


@indicator_registry.register('statistical')
class Kurtosis(BaseIndicator):
    """
    Kurtosis (峰度)

    价格分布的尖峭程度
    """

    _metadata = IndicatorMetadata(
        name='Kurtosis',
        display_name='峰度',
        category='statistical',
        description='衡量价格分布的尾部厚度',
        formula='Kurt = E[(X-μ)⁴] / σ⁴ - 3',
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

        kurtosis = close.rolling(window=self.period).kurt()

        return kurtosis


@indicator_registry.register('statistical')
class Correlation(BaseIndicator):
    """
    Correlation (相关系数)

    两个序列之间的皮尔逊相关系数
    """

    _metadata = IndicatorMetadata(
        name='Correlation',
        display_name='相关系数',
        category='statistical',
        description='两个价格序列的滚动相关性',
        formula='r = Cov(X,Y) / (σX * σY)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'compare_column': 'volume'},
        complexity_level=2
    )

    def __init__(self, period: int = 20, compare_column: str = 'volume'):
        self.period = period
        self.compare_column = compare_column
        super().__init__(period=period, compare_column=compare_column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', self.compare_column]

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        compare = data[self.compare_column]

        correlation = close.rolling(window=self.period).corr(compare)

        return correlation


@indicator_registry.register('statistical')
class Beta(BaseIndicator):
    """
    Beta (贝塔系数)

    相对于基准的系统性风险
    """

    _metadata = IndicatorMetadata(
        name='Beta',
        display_name='贝塔系数',
        category='statistical',
        description='相对于基准的波动性比率',
        formula='β = Cov(Ra, Rb) / Var(Rb)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 60, 'benchmark_column': 'benchmark'},
        complexity_level=2
    )

    def __init__(self, period: int = 60, benchmark_column: str = 'benchmark'):
        self.period = period
        self.benchmark_column = benchmark_column
        super().__init__(period=period, benchmark_column=benchmark_column)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 如果没有基准列，使用价格均值作为模拟基准
        if self.benchmark_column in data.columns:
            benchmark = data[self.benchmark_column]
        else:
            benchmark = close.rolling(window=self.period).mean()

        # 收益率
        returns = close.pct_change()
        bench_returns = benchmark.pct_change()

        # 协方差和方差
        covariance = returns.rolling(window=self.period).cov(bench_returns)
        variance = bench_returns.rolling(window=self.period).var()

        beta = covariance / variance

        # Alpha = 实际收益 - 预期收益(基于Beta)
        expected_return = beta * bench_returns
        alpha = returns - expected_return

        return pd.DataFrame({
            'Beta': beta,
            'Alpha': alpha.rolling(window=self.period).mean() * 252
        })


@indicator_registry.register('statistical')
class Sharpe(BaseIndicator):
    """
    Sharpe Ratio (夏普比率)

    风险调整后收益
    """

    _metadata = IndicatorMetadata(
        name='Sharpe',
        display_name='夏普比率',
        category='statistical',
        description='每单位风险的超额收益',
        formula='Sharpe = (Return - Rf) / σ',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252, 'risk_free': 0.02},
        complexity_level=2
    )

    def __init__(self, period: int = 252, risk_free: float = 0.02):
        self.period = period
        self.risk_free = risk_free
        super().__init__(period=period, risk_free=risk_free)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 日收益率
        returns = close.pct_change()

        # 年化收益率和波动率
        ann_return = returns.rolling(window=self.period).mean() * 252
        ann_vol = returns.rolling(window=self.period).std() * np.sqrt(252)

        # 夏普比率
        sharpe = (ann_return - self.risk_free) / ann_vol

        return sharpe


@indicator_registry.register('statistical')
class Sortino(BaseIndicator):
    """
    Sortino Ratio (索提诺比率)

    只考虑下行风险的风险调整收益
    """

    _metadata = IndicatorMetadata(
        name='Sortino',
        display_name='索提诺比率',
        category='statistical',
        description='每单位下行风险的超额收益',
        formula='Sortino = (Return - Rf) / Downside σ',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252, 'risk_free': 0.02},
        complexity_level=2
    )

    def __init__(self, period: int = 252, risk_free: float = 0.02):
        self.period = period
        self.risk_free = risk_free
        super().__init__(period=period, risk_free=risk_free)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 日收益率
        returns = close.pct_change()

        # 年化收益率
        ann_return = returns.rolling(window=self.period).mean() * 252

        # 下行波动率（只考虑负收益）
        negative_returns = returns.where(returns < 0, 0)
        downside_vol = negative_returns.rolling(window=self.period).std() * np.sqrt(252)

        # 索提诺比率
        sortino = (ann_return - self.risk_free) / downside_vol

        return sortino


@indicator_registry.register('statistical')
class Calmar(BaseIndicator):
    """
    Calmar Ratio (卡尔玛比率)

    收益与最大回撤的比率
    """

    _metadata = IndicatorMetadata(
        name='Calmar',
        display_name='卡尔玛比率',
        category='statistical',
        description='年化收益与最大回撤的比率',
        formula='Calmar = Annual Return / Max Drawdown',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 252):
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

        # 年化收益率
        returns = close.pct_change()
        ann_return = returns.rolling(window=self.period).mean() * 252

        # 滚动最大回撤
        rolling_max = close.rolling(window=self.period).max()
        drawdown = (close - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=self.period).min().abs()

        # 卡尔玛比率
        calmar = ann_return / max_drawdown

        return pd.DataFrame({
            'Calmar': calmar,
            'MaxDrawdown': max_drawdown,
            'AnnualReturn': ann_return
        })


@indicator_registry.register('statistical')
class MaxDrawdown(BaseIndicator):
    """
    Maximum Drawdown (最大回撤)

    从峰值到谷值的最大跌幅
    """

    _metadata = IndicatorMetadata(
        name='MaxDrawdown',
        display_name='最大回撤',
        category='statistical',
        description='滚动窗口内的最大回撤',
        formula='MDD = max((Peak - Trough) / Peak)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 252):
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

        # 滚动最高点
        rolling_max = close.rolling(window=self.period).max()

        # 当前回撤
        drawdown = (close - rolling_max) / rolling_max

        # 最大回撤（绝对值）
        max_drawdown = drawdown.rolling(window=self.period).min().abs()

        # 回撤持续时间（简化版）
        underwater = (drawdown < 0).astype(int)

        return pd.DataFrame({
            'Drawdown': drawdown,
            'MaxDrawdown': max_drawdown,
            'Peak': rolling_max,
            'Underwater': underwater
        })


@indicator_registry.register('statistical')
class VaR(BaseIndicator):
    """
    Value at Risk (风险价值)

    在给定置信水平下的最大可能损失
    """

    _metadata = IndicatorMetadata(
        name='VaR',
        display_name='风险价值',
        category='statistical',
        description='在给定置信水平下的最大可能损失',
        formula='VaR = Percentile(Returns, 1-confidence)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 252, 'confidence': 0.95},
        complexity_level=2
    )

    def __init__(self, period: int = 252, confidence: float = 0.95):
        self.period = period
        self.confidence = confidence
        super().__init__(period=period, confidence=confidence)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        returns = close.pct_change()

        # 历史VaR
        var = returns.rolling(window=self.period).quantile(1 - self.confidence)

        # 参数VaR（假设正态分布）
        mean = returns.rolling(window=self.period).mean()
        std = returns.rolling(window=self.period).std()
        z_score = stats.norm.ppf(1 - self.confidence)
        parametric_var = mean + z_score * std

        return pd.DataFrame({
            'VaR_Historical': var.abs(),
            'VaR_Parametric': parametric_var.abs()
        })


@indicator_registry.register('statistical')
class CVaR(BaseIndicator):
    """
    Conditional VaR / Expected Shortfall (条件风险价值)

    超过VaR水平的平均损失
    """

    _metadata = IndicatorMetadata(
        name='CVaR',
        display_name='条件风险价值',
        category='statistical',
        description='尾部风险的平均损失',
        formula='CVaR = E[Loss | Loss > VaR]',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252, 'confidence': 0.95},
        complexity_level=2
    )

    def __init__(self, period: int = 252, confidence: float = 0.95):
        self.period = period
        self.confidence = confidence
        super().__init__(period=period, confidence=confidence)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        returns = close.pct_change()

        # 计算CVaR
        def calc_cvar(x):
            var_threshold = np.percentile(x, (1 - self.confidence) * 100)
            return x[x <= var_threshold].mean()

        cvar = returns.rolling(window=self.period).apply(calc_cvar, raw=True)

        return cvar.abs()


@indicator_registry.register('statistical')
class InformationRatio(BaseIndicator):
    """
    Information Ratio (信息比率)

    相对于基准的超额收益与跟踪误差的比率
    """

    _metadata = IndicatorMetadata(
        name='InformationRatio',
        display_name='信息比率',
        category='statistical',
        description='主动管理的效率指标',
        formula='IR = (Rp - Rb) / Tracking Error',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 252):
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

        returns = close.pct_change()

        # 模拟基准（移动平均收益）
        benchmark_returns = returns.rolling(window=self.period).mean()

        # 超额收益
        excess_returns = returns - benchmark_returns

        # 跟踪误差
        tracking_error = excess_returns.rolling(window=self.period).std() * np.sqrt(252)

        # 年化超额收益
        ann_excess = excess_returns.rolling(window=self.period).mean() * 252

        # 信息比率
        ir = ann_excess / tracking_error

        return ir


@indicator_registry.register('statistical')
class OmegaRatio(BaseIndicator):
    """
    Omega Ratio (欧米伽比率)

    收益高于阈值的概率加权收益与低于阈值的概率加权损失之比
    """

    _metadata = IndicatorMetadata(
        name='OmegaRatio',
        display_name='欧米伽比率',
        category='statistical',
        description='考虑整个收益分布的风险调整指标',
        formula='Ω = ∫(1-F(r))dr / ∫F(r)dr for r above/below threshold',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252, 'threshold': 0.0},
        complexity_level=3
    )

    def __init__(self, period: int = 252, threshold: float = 0.0):
        self.period = period
        self.threshold = threshold
        super().__init__(period=period, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        returns = close.pct_change()

        def calc_omega(x):
            gains = x[x > self.threshold] - self.threshold
            losses = self.threshold - x[x <= self.threshold]

            sum_gains = gains.sum() if len(gains) > 0 else 0
            sum_losses = losses.sum() if len(losses) > 0 else 0.0001

            return sum_gains / sum_losses

        omega = returns.rolling(window=self.period).apply(calc_omega, raw=True)

        return omega


@indicator_registry.register('statistical')
class TailRatio(BaseIndicator):
    """
    Tail Ratio (尾部比率)

    上尾与下尾极端收益的比率
    """

    _metadata = IndicatorMetadata(
        name='TailRatio',
        display_name='尾部比率',
        category='statistical',
        description='正极端收益与负极端收益的比率',
        formula='TailRatio = |Percentile(95)| / |Percentile(5)|',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 252):
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

        returns = close.pct_change()

        upper_tail = returns.rolling(window=self.period).quantile(0.95)
        lower_tail = returns.rolling(window=self.period).quantile(0.05)

        tail_ratio = upper_tail.abs() / lower_tail.abs()

        return tail_ratio


@indicator_registry.register('statistical')
class HurstExponent(BaseIndicator):
    """
    Hurst Exponent (赫斯特指数)

    衡量时间序列的长期记忆性
    """

    _metadata = IndicatorMetadata(
        name='HurstExponent',
        display_name='赫斯特指数',
        category='statistical',
        description='H>0.5趋势性，H<0.5均值回归，H=0.5随机游走',
        formula='H = log(R/S) / log(n)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 100},
        complexity_level=3
    )

    def __init__(self, period: int = 100):
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

        returns = close.pct_change()

        def calc_hurst(ts):
            if len(ts) < 20:
                return np.nan

            lags = range(2, min(20, len(ts) // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2
            except (np.linalg.LinAlgError, ValueError):
                return np.nan

        hurst = returns.rolling(window=self.period).apply(calc_hurst, raw=True)

        return hurst


@indicator_registry.register('statistical')
class AutoCorrelation(BaseIndicator):
    """
    Auto Correlation (自相关系数)

    序列与自身滞后值的相关性
    """

    _metadata = IndicatorMetadata(
        name='AutoCorrelation',
        display_name='自相关系数',
        category='statistical',
        description='收益率与其滞后值的相关性',
        formula='AC = Corr(r_t, r_{t-lag})',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'lag': 1},
        complexity_level=2
    )

    def __init__(self, period: int = 20, lag: int = 1):
        self.period = period
        self.lag = lag
        super().__init__(period=period, lag=lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.lag, 'lag')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        returns = close.pct_change()
        lagged_returns = returns.shift(self.lag)

        autocorr = returns.rolling(window=self.period).corr(lagged_returns)

        return autocorr
