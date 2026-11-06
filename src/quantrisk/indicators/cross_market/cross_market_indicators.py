"""
跨市场分析指标模块 (20个)

包含相关性、套利、市场间传导、相对强弱等跨市场分析指标
"""

from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 相关性指标 (5个) ====================

@indicator_registry.register('cross_market')
class RollingCorrelation(BaseIndicator):
    """
    Rolling Correlation (滚动相关性)

    计算两个市场间的滚动相关系数
    """

    _metadata = IndicatorMetadata(
        name='RollingCorrelation',
        display_name='滚动相关性',
        category='cross_market',
        description='计算两个资产价格序列的滚动相关系数',
        formula='Corr(X, Y, window) = Cov(X, Y) / (σ_X * σ_Y)',
        required_columns=['close', 'reference_close'],
        output_type='series',
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
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        correlation = data['close'].rolling(window=self.window).corr(
            data['reference_close']
        )

        return correlation


@indicator_registry.register('cross_market')
class DynamicCorrelation(BaseIndicator):
    """
    Dynamic Conditional Correlation (动态条件相关)

    使用GARCH模型计算动态条件相关系数
    """

    _metadata = IndicatorMetadata(
        name='DynamicCorrelation',
        display_name='动态条件相关',
        category='cross_market',
        description='基于GARCH的动态条件相关系数(DCC-GARCH简化版)',
        formula='DCC = Q_t / sqrt(diag(Q_t) * diag(Q_t)\')',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 20, 'decay': 0.94},
        complexity_level=4
    )

    def __init__(self, window: int = 20, decay: float = 0.94):
        self.window = window
        self.decay = decay
        super().__init__(window=window, decay=decay)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_range(self.decay, 'decay', 0.0, 1.0)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        # 简化版DCC: 使用EWMA估计协方差
        cov = ret1.rolling(window=self.window).cov(ret2)
        std1 = ret1.rolling(window=self.window).std()
        std2 = ret2.rolling(window=self.window).std()

        # 动态相关系数
        dcc = cov / (std1 * std2)

        # EWMA平滑
        dcc_smooth = dcc.ewm(alpha=1-self.decay, adjust=False).mean()

        return dcc_smooth


@indicator_registry.register('cross_market')
class CointegrationTest(BaseIndicator):
    """
    Cointegration Test (协整检验)

    检验两个市场之间是否存在长期均衡关系
    """

    _metadata = IndicatorMetadata(
        name='CointegrationTest',
        display_name='协整检验',
        category='cross_market',
        description='使用Engle-Granger检验判断两个序列的协整关系',
        formula='ADF test on residuals: Y_t = β*X_t + ε_t',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=4
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        if self.window < 30:
            raise ValueError("window must be at least 30 for reliable cointegration test")

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        def rolling_coint(x, y):
            """滚动协整检验"""
            try:
                if len(x) < 30 or x.isna().any() or y.isna().any():
                    return np.nan
                _, p_value, _ = coint(x, y)
                # 返回1-p_value，值越大说明协整关系越强
                return 1 - p_value
            except:
                return np.nan

        # 滚动窗口协整检验
        coint_score = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i]
            score = rolling_coint(
                window_data['close'].values,
                window_data['reference_close'].values
            )
            coint_score.iloc[i] = score

        return coint_score


@indicator_registry.register('cross_market')
class GrangerCausality(BaseIndicator):
    """
    Granger Causality (格兰杰因果)

    检验一个市场是否对另一个市场有预测作用
    """

    _metadata = IndicatorMetadata(
        name='GrangerCausality',
        display_name='格兰杰因果',
        category='cross_market',
        description='检验X是否格兰杰引起Y',
        formula='Y_t = Σα_i*Y_{t-i} + Σβ_i*X_{t-i} + ε_t',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60, 'max_lag': 5},
        complexity_level=4
    )

    def __init__(self, window: int = 60, max_lag: int = 5):
        self.window = window
        self.max_lag = max_lag
        super().__init__(window=window, max_lag=max_lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.max_lag, 'max_lag')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        def rolling_granger(y, x, maxlag):
            """滚动格兰杰因果检验"""
            try:
                if len(y) < 30 or y.isna().any() or x.isna().any():
                    return np.nan

                # 组合数据
                test_data = pd.DataFrame({'y': y, 'x': x})

                # 执行格兰杰因果检验
                result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

                # 取最优滞后期的p值
                min_p_value = min([result[i+1][0]['ssr_ftest'][1] for i in range(maxlag)])

                # 返回1-p_value，值越大说明因果关系越强
                return 1 - min_p_value
            except:
                return np.nan

        # 滚动窗口格兰杰因果检验
        granger_score = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i]
            score = rolling_granger(
                window_data['close'].values,
                window_data['reference_close'].values,
                self.max_lag
            )
            granger_score.iloc[i] = score

        return granger_score


@indicator_registry.register('cross_market')
class LeadLagRelation(BaseIndicator):
    """
    Lead-Lag Relation (领先滞后关系)

    分析两个市场之间的领先滞后关系
    """

    _metadata = IndicatorMetadata(
        name='LeadLagRelation',
        display_name='领先滞后关系',
        category='cross_market',
        description='计算两个市场间的最优领先滞后期数',
        formula='max_corr(X_t, Y_{t+k}) for k in [-max_lag, max_lag]',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 20, 'max_lag': 5},
        complexity_level=3
    )

    def __init__(self, window: int = 20, max_lag: int = 5):
        self.window = window
        self.max_lag = max_lag
        super().__init__(window=window, max_lag=max_lag)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.max_lag, 'max_lag')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        def find_lead_lag(x, y, max_lag):
            """寻找最优领先滞后期"""
            correlations = {}

            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Y领先X
                    corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
                elif lag > 0:
                    # X领先Y
                    corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
                else:
                    corr = np.corrcoef(x, y)[0, 1]

                correlations[lag] = corr if not np.isnan(corr) else 0

            # 找到相关性最大的滞后期
            optimal_lag = max(correlations, key=correlations.get)
            max_corr = correlations[optimal_lag]

            return optimal_lag, max_corr

        # 滚动计算领先滞后关系
        optimal_lags = []
        max_corrs = []

        for i in range(self.window + self.max_lag, len(data)):
            window_data = data.iloc[i-self.window-self.max_lag:i+1]
            lag, corr = find_lead_lag(
                window_data['close'].values,
                window_data['reference_close'].values,
                self.max_lag
            )
            optimal_lags.append(lag)
            max_corrs.append(corr)

        # 构建结果DataFrame
        result = pd.DataFrame(index=data.index)
        result['optimal_lag'] = np.nan
        result['max_correlation'] = np.nan

        output_index = result.index[self.window + self.max_lag:]
        result.loc[output_index, 'optimal_lag'] = optimal_lags
        result.loc[output_index, 'max_correlation'] = max_corrs

        return result


# ==================== 套利指标 (5个) ====================

@indicator_registry.register('cross_market')
class PairSpread(BaseIndicator):
    """
    Pair Spread (配对价差)

    计算配对交易的价差序列
    """

    _metadata = IndicatorMetadata(
        name='PairSpread',
        display_name='配对价差',
        category='cross_market',
        description='计算两个资产的价差(使用对数价格)',
        formula='Spread = log(P1) - β*log(P2)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=2
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 对数价格
        log_p1 = np.log(data['close'])
        log_p2 = np.log(data['reference_close'])

        # 滚动计算对冲比率(beta)
        def calc_hedge_ratio(x, y):
            """计算对冲比率"""
            if len(x) < 2:
                return np.nan
            coef = np.polyfit(y, x, 1)[0]
            return coef

        beta = pd.Series(index=data.index, dtype=float)
        for i in range(self.window, len(data)):
            window_p1 = log_p1.iloc[i-self.window:i]
            window_p2 = log_p2.iloc[i-self.window:i]
            beta.iloc[i] = calc_hedge_ratio(window_p1.values, window_p2.values)

        # 计算价差
        spread = log_p1 - beta * log_p2

        return spread


@indicator_registry.register('cross_market')
class ZScore(BaseIndicator):
    """
    Z-Score (标准化价差)

    价差的标准化得分，用于识别套利机会
    """

    _metadata = IndicatorMetadata(
        name='ZScore',
        display_name='标准化价差',
        category='cross_market',
        description='价差相对历史均值的标准差倍数',
        formula='Z = (Spread - μ) / σ',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=2
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 先计算价差
        spread_indicator = PairSpread(window=self.window)
        spread = spread_indicator.calculate(data)

        # 标准化
        mean = spread.rolling(window=self.window).mean()
        std = spread.rolling(window=self.window).std()

        z_score = (spread - mean) / std

        return z_score


@indicator_registry.register('cross_market')
class HedgeRatio(BaseIndicator):
    """
    Hedge Ratio (对冲比率)

    配对交易中的最优对冲比率
    """

    _metadata = IndicatorMetadata(
        name='HedgeRatio',
        display_name='对冲比率',
        category='cross_market',
        description='通过回归计算最优对冲比率',
        formula='β = Cov(P1, P2) / Var(P2)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60, 'method': 'ols'},
        complexity_level=2
    )

    def __init__(self, window: int = 60, method: str = 'ols'):
        self.window = window
        self.method = method
        super().__init__(window=window, method=method)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_in_list(self.method, 'method', ['ols', 'tls'])

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        def calc_ratio_ols(x, y):
            """OLS回归"""
            if len(x) < 2:
                return np.nan
            return np.cov(x, y)[0, 1] / np.var(y)

        def calc_ratio_tls(x, y):
            """Total Least Squares"""
            if len(x) < 2:
                return np.nan
            # 使用主成分分析
            data_matrix = np.vstack([y, x]).T
            _, _, vt = np.linalg.svd(data_matrix - data_matrix.mean(axis=0))
            return -vt[0, 0] / vt[0, 1]

        calc_func = calc_ratio_ols if self.method == 'ols' else calc_ratio_tls

        hedge_ratio = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i]
            ratio = calc_func(
                window_data['close'].values,
                window_data['reference_close'].values
            )
            hedge_ratio.iloc[i] = ratio

        return hedge_ratio


@indicator_registry.register('cross_market')
class MeanReversionSpeed(BaseIndicator):
    """
    Mean Reversion Speed (均值回归速度)

    价差回归到均值的速度
    """

    _metadata = IndicatorMetadata(
        name='MeanReversionSpeed',
        display_name='均值回归速度',
        category='cross_market',
        description='使用Ornstein-Uhlenbeck过程估计均值回归速度',
        formula='dS = θ(μ - S)dt + σdW, λ = -log(ρ)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=4
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 先计算价差
        spread_indicator = PairSpread(window=self.window)
        spread = spread_indicator.calculate(data)

        def estimate_reversion_speed(x):
            """估计均值回归速度"""
            if len(x) < 3 or np.isnan(x).any():
                return np.nan

            # 使用AR(1)模型估计
            x_lag = x[:-1]
            x_current = x[1:]

            # 去中心化
            x_lag_mean = x_lag - np.mean(x_lag)
            x_current_mean = x_current - np.mean(x_current)

            # 计算自相关系数
            rho = np.corrcoef(x_lag_mean, x_current_mean)[0, 1]

            # 均值回归速度
            if rho >= 1:
                return 0.0
            speed = -np.log(abs(rho))

            return speed

        # 滚动计算均值回归速度
        reversion_speed = spread.rolling(window=self.window).apply(
            estimate_reversion_speed, raw=True
        )

        return reversion_speed


@indicator_registry.register('cross_market')
class ArbitrageSignal(BaseIndicator):
    """
    Arbitrage Signal (套利信号)

    综合套利信号生成器
    """

    _metadata = IndicatorMetadata(
        name='ArbitrageSignal',
        display_name='套利信号',
        category='cross_market',
        description='基于Z-Score生成套利交易信号',
        formula='Signal: Z > threshold → Short, Z < -threshold → Long',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 60, 'entry_threshold': 2.0, 'exit_threshold': 0.5},
        complexity_level=3
    )

    def __init__(self, window: int = 60, entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        super().__init__(window=window, entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.entry_threshold, 'entry_threshold')
        IndicatorValidator.validate_positive_float(self.exit_threshold, 'exit_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 计算Z-Score
        zscore_indicator = ZScore(window=self.window)
        z_score = zscore_indicator.calculate(data)

        # 生成信号
        result = pd.DataFrame(index=data.index)
        result['z_score'] = z_score
        result['signal'] = 0

        # 开仓信号
        result.loc[z_score > self.entry_threshold, 'signal'] = -1  # 做空价差
        result.loc[z_score < -self.entry_threshold, 'signal'] = 1   # 做多价差

        # 平仓信号
        result.loc[abs(z_score) < self.exit_threshold, 'signal'] = 0

        # 前向填充保持持仓
        result['position'] = result['signal'].replace(0, np.nan).ffill().fillna(0)

        return result


# ==================== 市场间传导 (5个) ====================

@indicator_registry.register('cross_market')
class SpilloverIndex(BaseIndicator):
    """
    Spillover Index (溢出指数)

    测量市场间的波动率溢出效应
    """

    _metadata = IndicatorMetadata(
        name='SpilloverIndex',
        display_name='溢出指数',
        category='cross_market',
        description='基于VAR模型的波动率溢出指数',
        formula='Spillover = (Σ off-diagonal variance decomposition) / Total',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60, 'horizon': 10},
        complexity_level=5
    )

    def __init__(self, window: int = 60, horizon: int = 10):
        self.window = window
        self.horizon = horizon
        super().__init__(window=window, horizon=horizon)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.horizon, 'horizon')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        def calc_spillover(r1, r2):
            """计算溢出指数(简化版)"""
            if len(r1) < 10 or np.isnan(r1).any() or np.isnan(r2).any():
                return np.nan

            # 简化方法：使用滚动相关性和波动率比值
            corr = np.corrcoef(r1, r2)[0, 1]
            vol1 = np.std(r1)
            vol2 = np.std(r2)

            # 溢出指数近似
            spillover = abs(corr) * min(vol1, vol2) / max(vol1, vol2)

            return spillover

        # 滚动计算溢出指数
        spillover = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_ret1 = ret1.iloc[i-self.window:i].values
            window_ret2 = ret2.iloc[i-self.window:i].values
            spillover.iloc[i] = calc_spillover(window_ret1, window_ret2)

        return spillover


@indicator_registry.register('cross_market')
class ContagionRisk(BaseIndicator):
    """
    Contagion Risk (传染风险)

    测量极端事件的跨市场传染风险
    """

    _metadata = IndicatorMetadata(
        name='ContagionRisk',
        display_name='传染风险',
        category='cross_market',
        description='基于尾部相关性的传染风险指标',
        formula='Contagion = P(X < q_α | Y < q_α) / P(X < q_α)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60, 'quantile': 0.05},
        complexity_level=4
    )

    def __init__(self, window: int = 60, quantile: float = 0.05):
        self.window = window
        self.quantile = quantile
        super().__init__(window=window, quantile=quantile)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_range(self.quantile, 'quantile', 0.01, 0.2)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        def calc_contagion(r1, r2, q):
            """计算传染风险"""
            if len(r1) < 20 or np.isnan(r1).any() or np.isnan(r2).any():
                return np.nan

            # 计算分位数阈值
            threshold1 = np.percentile(r1, q * 100)
            threshold2 = np.percentile(r2, q * 100)

            # 极端事件标识
            extreme1 = r1 < threshold1
            extreme2 = r2 < threshold2

            # 条件概率
            p_both = np.mean(extreme1 & extreme2)
            p_x = np.mean(extreme1)

            if p_x == 0:
                return 0.0

            contagion = p_both / p_x

            return contagion

        # 滚动计算传染风险
        contagion = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_ret1 = ret1.iloc[i-self.window:i].values
            window_ret2 = ret2.iloc[i-self.window:i].values
            contagion.iloc[i] = calc_contagion(window_ret1, window_ret2, self.quantile)

        return contagion


@indicator_registry.register('cross_market')
class SystemicRisk(BaseIndicator):
    """
    Systemic Risk (系统性风险)

    测量系统性风险贡献度
    """

    _metadata = IndicatorMetadata(
        name='SystemicRisk',
        display_name='系统性风险',
        category='cross_market',
        description='使用CoVaR方法估计系统性风险',
        formula='ΔCoVaR = CoVaR(Y|X=VaR_X) - CoVaR(Y|X=Median_X)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60, 'confidence': 0.95},
        complexity_level=5
    )

    def __init__(self, window: int = 60, confidence: float = 0.95):
        self.window = window
        self.confidence = confidence
        super().__init__(window=window, confidence=confidence)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_range(self.confidence, 'confidence', 0.9, 0.99)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        alpha = 1 - self.confidence

        def calc_systemic_risk(r1, r2, alpha):
            """计算系统性风险(简化版CoVaR)"""
            if len(r1) < 20 or np.isnan(r1).any() or np.isnan(r2).any():
                return np.nan

            # VaR of market 1
            var_1 = np.percentile(r1, alpha * 100)

            # Conditional VaR of market 2
            # 当市场1处于VaR水平时，市场2的VaR
            stress_condition = r1 <= var_1
            if stress_condition.sum() < 5:
                return np.nan

            r2_stressed = r2[stress_condition]
            covar_stress = np.percentile(r2_stressed, alpha * 100)

            # 正常情况下的VaR
            covar_normal = np.percentile(r2, alpha * 100)

            # ΔCoVaR
            delta_covar = covar_stress - covar_normal

            return delta_covar

        # 滚动计算系统性风险
        systemic_risk = pd.Series(index=data.index, dtype=float)

        for i in range(self.window, len(data)):
            window_ret1 = ret1.iloc[i-self.window:i].values
            window_ret2 = ret2.iloc[i-self.window:i].values
            systemic_risk.iloc[i] = calc_systemic_risk(window_ret1, window_ret2, alpha)

        return systemic_risk


@indicator_registry.register('cross_market')
class MarketIntegration(BaseIndicator):
    """
    Market Integration (市场整合度)

    测量两个市场的整合程度
    """

    _metadata = IndicatorMetadata(
        name='MarketIntegration',
        display_name='市场整合度',
        category='cross_market',
        description='基于信息共享和价格发现的市场整合度',
        formula='Integration = f(Correlation, Cointegration, InfoShare)',
        required_columns=['close', 'reference_close'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=4
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算多个整合度指标
        # 1. 相关性
        corr = RollingCorrelation(window=self.window).calculate(data)

        # 2. 价格收敛度(价差标准差)
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()
        spread = ret1 - ret2
        spread_std = spread.rolling(window=self.window).std()
        convergence = 1 / (1 + spread_std)  # 标准化到[0,1]

        # 3. 联动性(同向运动频率)
        same_direction = ((ret1 > 0) & (ret2 > 0)) | ((ret1 < 0) & (ret2 < 0))
        linkage = same_direction.rolling(window=self.window).mean()

        # 综合整合度指标
        integration = (abs(corr) * 0.4 + convergence * 0.3 + linkage * 0.3)

        return integration


@indicator_registry.register('cross_market')
class InformationFlow(BaseIndicator):
    """
    Information Flow (信息流向)

    分析信息在市场间的传播方向和强度
    """

    _metadata = IndicatorMetadata(
        name='InformationFlow',
        display_name='信息流向',
        category='cross_market',
        description='使用信息论方法测量市场间的信息流',
        formula='InfoFlow = I(X_t; Y_{t+1}) - I(Y_t; X_{t+1})',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 60, 'bins': 10},
        complexity_level=5
    )

    def __init__(self, window: int = 60, bins: int = 10):
        self.window = window
        self.bins = bins
        super().__init__(window=window, bins=bins)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.bins, 'bins')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 计算收益率
        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        def calc_transfer_entropy(x, y, bins):
            """计算传递熵(简化版)"""
            if len(x) < 20 or np.isnan(x).any() or np.isnan(y).any():
                return np.nan, np.nan

            # 离散化
            x_binned = pd.cut(x, bins=bins, labels=False, duplicates='drop')
            y_binned = pd.cut(y, bins=bins, labels=False, duplicates='drop')

            # 使用滞后相关性近似信息流
            # X到Y的信息流
            x_to_y = np.corrcoef(x[:-1], y[1:])[0, 1] ** 2 if len(x) > 1 else 0
            # Y到X的信息流
            y_to_x = np.corrcoef(y[:-1], x[1:])[0, 1] ** 2 if len(y) > 1 else 0

            return x_to_y, y_to_x

        # 滚动计算信息流
        flow_x_to_y = []
        flow_y_to_x = []

        for i in range(self.window, len(data)):
            window_ret1 = ret1.iloc[i-self.window:i].values
            window_ret2 = ret2.iloc[i-self.window:i].values

            x_to_y, y_to_x = calc_transfer_entropy(window_ret1, window_ret2, self.bins)
            flow_x_to_y.append(x_to_y)
            flow_y_to_x.append(y_to_x)

        # 构建结果
        result = pd.DataFrame(index=data.index)
        result['flow_x_to_y'] = np.nan
        result['flow_y_to_x'] = np.nan
        result['net_flow'] = np.nan

        output_index = result.index[self.window:]
        result.loc[output_index, 'flow_x_to_y'] = flow_x_to_y
        result.loc[output_index, 'flow_y_to_x'] = flow_y_to_x
        result['net_flow'] = result['flow_x_to_y'] - result['flow_y_to_x']

        return result


# ==================== 相对强弱 (5个) ====================

@indicator_registry.register('cross_market')
class RelativeStrength(BaseIndicator):
    """
    Relative Strength (相对强弱)

    比较两个市场的相对表现
    """

    _metadata = IndicatorMetadata(
        name='RelativeStrength',
        display_name='相对强弱',
        category='cross_market',
        description='计算资产相对于参考资产的强度',
        formula='RS = (P1 / P1_base) / (P2 / P2_base)',
        required_columns=['close', 'reference_close'],
        output_type='series',
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
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算相对强度比率
        p1_return = data['close'] / data['close'].shift(self.window)
        p2_return = data['reference_close'] / data['reference_close'].shift(self.window)

        relative_strength = p1_return / p2_return

        return relative_strength


@indicator_registry.register('cross_market')
class SectorRotation(BaseIndicator):
    """
    Sector Rotation (板块轮动)

    检测板块轮动信号
    """

    _metadata = IndicatorMetadata(
        name='SectorRotation',
        display_name='板块轮动',
        category='cross_market',
        description='基于相对动量的板块轮动指标',
        formula='Rotation = RoC(RS, period)',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 20, 'roc_period': 5},
        complexity_level=3
    )

    def __init__(self, window: int = 20, roc_period: int = 5):
        self.window = window
        self.roc_period = roc_period
        super().__init__(window=window, roc_period=roc_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.roc_period, 'roc_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 计算相对强度
        rs = RelativeStrength(window=self.window).calculate(data)

        # 相对强度的变化率
        rs_roc = rs.pct_change(self.roc_period)

        # 动量得分
        momentum_score = rs_roc.rolling(window=self.window).rank(pct=True)

        # 轮动信号
        result = pd.DataFrame(index=data.index)
        result['relative_strength'] = rs
        result['momentum'] = rs_roc
        result['momentum_score'] = momentum_score
        result['rotation_signal'] = 0

        # 强势轮入
        result.loc[momentum_score > 0.7, 'rotation_signal'] = 1
        # 弱势轮出
        result.loc[momentum_score < 0.3, 'rotation_signal'] = -1

        return result


@indicator_registry.register('cross_market')
class StyleMomentum(BaseIndicator):
    """
    Style Momentum (风格动量)

    分析不同投资风格的动量
    """

    _metadata = IndicatorMetadata(
        name='StyleMomentum',
        display_name='风格动量',
        category='cross_market',
        description='基于相对表现的风格动量指标',
        formula='StyleMom = Σw_i * Mom_i',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'short_window': 5, 'long_window': 20},
        complexity_level=3
    )

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(short_window=short_window, long_window=long_window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.short_window, 'short_window')
        IndicatorValidator.validate_positive_int(self.long_window, 'long_window')
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 短期动量
        ret1_short = data['close'].pct_change(self.short_window)
        ret2_short = data['reference_close'].pct_change(self.short_window)

        # 长期动量
        ret1_long = data['close'].pct_change(self.long_window)
        ret2_long = data['reference_close'].pct_change(self.long_window)

        # 相对动量
        relative_momentum_short = ret1_short - ret2_short
        relative_momentum_long = ret1_long - ret2_long

        # 风格偏好(短期vs长期)
        style_preference = relative_momentum_short - relative_momentum_long

        result = pd.DataFrame(index=data.index)
        result['short_momentum'] = relative_momentum_short
        result['long_momentum'] = relative_momentum_long
        result['style_preference'] = style_preference
        result['style_signal'] = np.sign(style_preference)

        return result


@indicator_registry.register('cross_market')
class AssetAllocationSignal(BaseIndicator):
    """
    Asset Allocation Signal (资产配置信号)

    生成资产配置调整信号
    """

    _metadata = IndicatorMetadata(
        name='AssetAllocationSignal',
        display_name='资产配置信号',
        category='cross_market',
        description='基于相对估值和动量的配置信号',
        formula='Allocation = f(RelVal, Momentum, Volatility)',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 60, 'rebalance_threshold': 0.15},
        complexity_level=4
    )

    def __init__(self, window: int = 60, rebalance_threshold: float = 0.15):
        self.window = window
        self.rebalance_threshold = rebalance_threshold
        super().__init__(window=window, rebalance_threshold=rebalance_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_range(self.rebalance_threshold, 'rebalance_threshold', 0.05, 0.5)

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        # 1. 动量得分
        momentum1 = ret1.rolling(window=self.window).mean()
        momentum2 = ret2.rolling(window=self.window).mean()

        # 2. 风险调整收益
        sharpe1 = momentum1 / ret1.rolling(window=self.window).std()
        sharpe2 = momentum2 / ret2.rolling(window=self.window).std()

        # 3. 相对波动率
        vol1 = ret1.rolling(window=self.window).std()
        vol2 = ret2.rolling(window=self.window).std()

        # 综合得分(标准化到[0,1])
        score1 = sharpe1 / (sharpe1 + sharpe2)
        score2 = 1 - score1

        # 最优权重(考虑风险)
        weight1 = score1 * (1 / vol1) / ((1 / vol1) + (1 / vol2))
        weight2 = 1 - weight1

        result = pd.DataFrame(index=data.index)
        result['momentum_asset1'] = momentum1
        result['momentum_asset2'] = momentum2
        result['sharpe_asset1'] = sharpe1
        result['sharpe_asset2'] = sharpe2
        result['weight_asset1'] = weight1
        result['weight_asset2'] = weight2

        # 再平衡信号
        weight_change = abs(weight1 - 0.5)
        result['rebalance_signal'] = (weight_change > self.rebalance_threshold).astype(int)

        return result


@indicator_registry.register('cross_market')
class RiskParityWeight(BaseIndicator):
    """
    Risk Parity Weight (风险平价权重)

    计算风险平价配置权重
    """

    _metadata = IndicatorMetadata(
        name='RiskParityWeight',
        display_name='风险平价权重',
        category='cross_market',
        description='基于等风险贡献的资产配置权重',
        formula='w_i = (1/σ_i) / Σ(1/σ_j)',
        required_columns=['close', 'reference_close'],
        output_type='dataframe',
        default_params={'window': 60},
        complexity_level=3
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'reference_close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        ret1 = data['close'].pct_change()
        ret2 = data['reference_close'].pct_change()

        # 计算波动率
        vol1 = ret1.rolling(window=self.window).std()
        vol2 = ret2.rolling(window=self.window).std()

        # 计算协方差
        cov = ret1.rolling(window=self.window).cov(ret2)

        # 风险平价权重(逆波动率加权)
        inv_vol1 = 1 / vol1
        inv_vol2 = 1 / vol2

        weight1 = inv_vol1 / (inv_vol1 + inv_vol2)
        weight2 = inv_vol2 / (inv_vol1 + inv_vol2)

        # 计算组合风险
        portfolio_var = (weight1 ** 2 * vol1 ** 2 +
                        weight2 ** 2 * vol2 ** 2 +
                        2 * weight1 * weight2 * cov)
        portfolio_vol = np.sqrt(portfolio_var)

        # 风险贡献
        risk_contrib1 = weight1 * vol1 ** 2 / portfolio_var
        risk_contrib2 = weight2 * vol2 ** 2 / portfolio_var

        result = pd.DataFrame(index=data.index)
        result['vol_asset1'] = vol1
        result['vol_asset2'] = vol2
        result['weight_asset1'] = weight1
        result['weight_asset2'] = weight2
        result['portfolio_vol'] = portfolio_vol
        result['risk_contrib_asset1'] = risk_contrib1
        result['risk_contrib_asset2'] = risk_contrib2

        return result


# 导出所有指标
__all__ = [
    # 相关性指标 (5个)
    'RollingCorrelation',
    'DynamicCorrelation',
    'CointegrationTest',
    'GrangerCausality',
    'LeadLagRelation',

    # 套利指标 (5个)
    'PairSpread',
    'ZScore',
    'HedgeRatio',
    'MeanReversionSpeed',
    'ArbitrageSignal',

    # 市场间传导 (5个)
    'SpilloverIndex',
    'ContagionRisk',
    'SystemicRisk',
    'MarketIntegration',
    'InformationFlow',

    # 相对强弱 (5个)
    'RelativeStrength',
    'SectorRotation',
    'StyleMomentum',
    'AssetAllocationSignal',
    'RiskParityWeight',
]
