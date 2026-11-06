"""
历史波动率指标 (10个)

基于不同方法计算的历史波动率
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('volatility')
class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility (历史波动率)

    基于收益率标准差的经典波动率
    """

    _metadata = IndicatorMetadata(
        name='HistoricalVolatility',
        display_name='历史波动率',
        category='volatility',
        description='基于对数收益率标准差的年化波动率',
        formula='HV = StdDev(ln(C/C[1])) * √252 * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=1
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 对数收益率
        log_returns = np.log(close / close.shift(1))

        # 标准差
        hv = log_returns.rolling(window=self.period).std()

        if self.annualize:
            hv = hv * np.sqrt(252) * 100

        return hv


@indicator_registry.register('volatility')
class ParkinsonVolatility(BaseIndicator):
    """
    Parkinson Volatility (帕金森波动率)

    使用最高价和最低价计算的波动率
    """

    _metadata = IndicatorMetadata(
        name='ParkinsonVolatility',
        display_name='帕金森波动率',
        category='volatility',
        description='使用日内高低价计算的波动率，效率更高',
        formula='PV = √(1/(4*n*ln2) * Σ(ln(H/L))²) * √252',
        required_columns=['high', 'low'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=2
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        high = data['high']
        low = data['low']

        # 对数高低价比
        log_hl = np.log(high / low)

        # 帕金森公式
        factor = 1 / (4 * np.log(2))
        pv = np.sqrt(factor * (log_hl ** 2).rolling(window=self.period).mean())

        if self.annualize:
            pv = pv * np.sqrt(252) * 100

        return pv


@indicator_registry.register('volatility')
class GarmanKlassVolatility(BaseIndicator):
    """
    Garman-Klass Volatility (加曼-克拉斯波动率)

    使用OHLC数据计算的高效波动率
    """

    _metadata = IndicatorMetadata(
        name='GarmanKlassVolatility',
        display_name='加曼-克拉斯波动率',
        category='volatility',
        description='使用开高低收计算的波动率，效率比帕金森更高',
        formula='GK = √(0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=2
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)

        # 加曼-克拉斯公式
        gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_vol = np.sqrt(gk.rolling(window=self.period).mean())

        if self.annualize:
            gk_vol = gk_vol * np.sqrt(252) * 100

        return gk_vol


@indicator_registry.register('volatility')
class RogersSatchellVolatility(BaseIndicator):
    """
    Rogers-Satchell Volatility (罗杰斯-萨切尔波动率)

    考虑趋势漂移的波动率估计
    """

    _metadata = IndicatorMetadata(
        name='RogersSatchellVolatility',
        display_name='罗杰斯-萨切尔波动率',
        category='volatility',
        description='考虑趋势影响的波动率，适用于有趋势的市场',
        formula='RS = √(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=2
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        log_hc = np.log(high / close)
        log_ho = np.log(high / open_price)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_price)

        # 罗杰斯-萨切尔公式
        rs = log_hc * log_ho + log_lc * log_lo
        rs_vol = np.sqrt(rs.rolling(window=self.period).mean())

        if self.annualize:
            rs_vol = rs_vol * np.sqrt(252) * 100

        return rs_vol


@indicator_registry.register('volatility')
class YangZhangVolatility(BaseIndicator):
    """
    Yang-Zhang Volatility (杨-张波动率)

    结合隔夜和日内波动的高效估计
    """

    _metadata = IndicatorMetadata(
        name='YangZhangVolatility',
        display_name='杨-张波动率',
        category='volatility',
        description='最高效的波动率估计，考虑开盘跳空',
        formula='YZ = √(σ²_overnight + k*σ²_open + (1-k)*σ²_RS)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=3
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        # 隔夜波动率
        log_oc = np.log(open_price / close.shift(1))
        overnight_var = log_oc.rolling(window=self.period).var()

        # 开盘跳空波动率
        log_co = np.log(close / open_price)
        open_var = log_co.rolling(window=self.period).var()

        # Rogers-Satchell波动率
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_price)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_price)
        rs = log_hc * log_ho + log_lc * log_lo
        rs_var = rs.rolling(window=self.period).mean()

        # 杨-张组合
        k = 0.34 / (1.34 + (self.period + 1) / (self.period - 1))
        yz_var = overnight_var + k * open_var + (1 - k) * rs_var
        yz_vol = np.sqrt(yz_var)

        if self.annualize:
            yz_vol = yz_vol * np.sqrt(252) * 100

        return yz_vol


@indicator_registry.register('volatility')
class RealizedVolatility(BaseIndicator):
    """
    Realized Volatility (已实现波动率)

    基于高频收益的波动率
    """

    _metadata = IndicatorMetadata(
        name='RealizedVolatility',
        display_name='已实现波动率',
        category='volatility',
        description='基于日内或日间收益平方和的波动率',
        formula='RV = √(Σ(r²))',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'annualize': True},
        complexity_level=1
    )

    def __init__(self, period: int = 20, annualize: bool = True):
        self.period = period
        self.annualize = annualize
        super().__init__(period=period, annualize=annualize)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 收益率
        returns = np.log(close / close.shift(1))

        # 已实现波动率（收益平方和的平方根）
        rv = np.sqrt((returns ** 2).rolling(window=self.period).sum())

        if self.annualize:
            rv = rv * np.sqrt(252 / self.period) * 100

        return rv


@indicator_registry.register('volatility')
class VolatilityCone(BaseIndicator):
    """
    Volatility Cone (波动率锥)

    多时间周期的波动率分位数
    """

    _metadata = IndicatorMetadata(
        name='VolatilityCone',
        display_name='波动率锥',
        category='volatility',
        description='当前波动率在历史分布中的位置',
        formula='Percentile = 当前HV在历史HV中的百分位',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'lookback': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 20, lookback: int = 252):
        self.period = period
        self.lookback = lookback
        super().__init__(period=period, lookback=lookback)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.lookback, 'lookback')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 历史波动率
        log_returns = np.log(close / close.shift(1))
        hv = log_returns.rolling(window=self.period).std() * np.sqrt(252) * 100

        # 历史分位数
        def calc_percentile(x):
            return (x.rank() / len(x) * 100).iloc[-1]

        hv_percentile = hv.rolling(window=self.lookback).apply(calc_percentile, raw=False)

        # 历史统计
        hv_min = hv.rolling(window=self.lookback).min()
        hv_max = hv.rolling(window=self.lookback).max()
        hv_mean = hv.rolling(window=self.lookback).mean()
        hv_median = hv.rolling(window=self.lookback).median()

        return pd.DataFrame({
            'HV': hv,
            'HV_Percentile': hv_percentile,
            'HV_Min': hv_min,
            'HV_Max': hv_max,
            'HV_Mean': hv_mean,
            'HV_Median': hv_median
        })


@indicator_registry.register('volatility')
class VolatilityRatio(BaseIndicator):
    """
    Volatility Ratio (波动率比率)

    短期与长期波动率的比率
    """

    _metadata = IndicatorMetadata(
        name='VolatilityRatio',
        display_name='波动率比率',
        category='volatility',
        description='短期波动率与长期波动率的比率',
        formula='VR = HV_Short / HV_Long',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'short_period': 10, 'long_period': 30},
        complexity_level=2
    )

    def __init__(self, short_period: int = 10, long_period: int = 30):
        self.short_period = short_period
        self.long_period = long_period
        super().__init__(short_period=short_period, long_period=long_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.short_period, 'short_period')
        IndicatorValidator.validate_positive_int(self.long_period, 'long_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        log_returns = np.log(close / close.shift(1))

        # 短期和长期波动率
        hv_short = log_returns.rolling(window=self.short_period).std()
        hv_long = log_returns.rolling(window=self.long_period).std()

        # 比率
        vol_ratio = hv_short / hv_long

        # 波动率变化趋势
        vol_trend = vol_ratio.diff()

        return pd.DataFrame({
            'HV_Short': hv_short * np.sqrt(252) * 100,
            'HV_Long': hv_long * np.sqrt(252) * 100,
            'Vol_Ratio': vol_ratio,
            'Vol_Trend': vol_trend
        })


@indicator_registry.register('volatility')
class EWMA_Volatility(BaseIndicator):
    """
    EWMA Volatility (指数加权波动率)

    RiskMetrics使用的EWMA波动率
    """

    _metadata = IndicatorMetadata(
        name='EWMA_Volatility',
        display_name='EWMA波动率',
        category='volatility',
        description='指数加权移动平均波动率，近期数据权重更大',
        formula='σ²_t = λ*σ²_{t-1} + (1-λ)*r²_t',
        required_columns=['close'],
        output_type='series',
        default_params={'lambda_': 0.94, 'annualize': True},
        complexity_level=2
    )

    def __init__(self, lambda_: float = 0.94, annualize: bool = True):
        self.lambda_ = lambda_
        self.annualize = annualize
        super().__init__(lambda_=lambda_, annualize=annualize)

    def _validate_params(self):
        if not 0 < self.lambda_ < 1:
            raise ValueError("lambda must be between 0 and 1")

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 收益率
        returns = np.log(close / close.shift(1))
        returns_sq = returns ** 2

        # EWMA方差
        ewma_var = returns_sq.ewm(alpha=1-self.lambda_, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var)

        if self.annualize:
            ewma_vol = ewma_vol * np.sqrt(252) * 100

        return ewma_vol


@indicator_registry.register('volatility')
class GARCH_Volatility(BaseIndicator):
    """
    GARCH-like Volatility (类GARCH波动率)

    简化的GARCH(1,1)模型波动率
    """

    _metadata = IndicatorMetadata(
        name='GARCH_Volatility',
        display_name='GARCH波动率',
        category='volatility',
        description='简化的GARCH模型，考虑波动率聚集效应',
        formula='σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}',
        required_columns=['close'],
        output_type='series',
        default_params={'omega': 0.00001, 'alpha': 0.1, 'beta': 0.85, 'annualize': True},
        complexity_level=3
    )

    def __init__(self, omega: float = 0.00001, alpha: float = 0.1,
                 beta: float = 0.85, annualize: bool = True):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.annualize = annualize
        super().__init__(omega=omega, alpha=alpha, beta=beta, annualize=annualize)

    def _validate_params(self):
        if self.alpha + self.beta >= 1:
            raise ValueError("alpha + beta must be less than 1 for stationarity")

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 收益率
        returns = np.log(close / close.shift(1))

        # 初始化方差（使用样本方差）
        initial_var = returns.iloc[:30].var()

        # GARCH(1,1)递归计算
        variance = pd.Series(index=close.index, dtype=float)
        variance.iloc[0] = initial_var

        for i in range(1, len(returns)):
            if pd.isna(returns.iloc[i-1]):
                variance.iloc[i] = variance.iloc[i-1] if not pd.isna(variance.iloc[i-1]) else initial_var
            else:
                variance.iloc[i] = (self.omega +
                                   self.alpha * returns.iloc[i-1]**2 +
                                   self.beta * variance.iloc[i-1])

        garch_vol = np.sqrt(variance)

        if self.annualize:
            garch_vol = garch_vol * np.sqrt(252) * 100

        return garch_vol
