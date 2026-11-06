"""
衍生品指标模块 (25个)

包含Greeks指标、隐含波动率、期权策略、期货指标和风险指标
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ============================================================================
# Greeks指标 (5个)
# ============================================================================

@indicator_registry.register('derivatives')
class Delta(BaseIndicator):
    """
    Delta (Delta值)

    期权价格相对于标的资产价格的一阶导数
    """

    _metadata = IndicatorMetadata(
        name='Delta',
        display_name='Delta值',
        category='derivatives',
        description='期权价格对标的资产价格的敏感度',
        formula='Call: Δ = N(d1), Put: Δ = N(d1) - 1',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'volatility': 0.2, 'option_type': 'call'},
        complexity_level=3
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, volatility: float = 0.2,
                 option_type: str = 'call'):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type.lower()
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, volatility=volatility,
                        option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.volatility, 'volatility')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']
        K = self.strike
        T = self.time_to_maturity
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        return pd.Series(delta, index=data.index)


@indicator_registry.register('derivatives')
class Gamma(BaseIndicator):
    """
    Gamma (Gamma值)

    期权价格相对于标的资产价格的二阶导数
    """

    _metadata = IndicatorMetadata(
        name='Gamma',
        display_name='Gamma值',
        category='derivatives',
        description='Delta对标的资产价格的敏感度',
        formula='Γ = N\'(d1) / (S * σ * √T)',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'volatility': 0.2},
        complexity_level=3
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, volatility: float = 0.2):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, volatility=volatility)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.volatility, 'volatility')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']
        K = self.strike
        T = self.time_to_maturity
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return pd.Series(gamma, index=data.index)


@indicator_registry.register('derivatives')
class Theta(BaseIndicator):
    """
    Theta (Theta值)

    期权价格相对于时间的导数
    """

    _metadata = IndicatorMetadata(
        name='Theta',
        display_name='Theta值',
        category='derivatives',
        description='期权价格对时间流逝的敏感度',
        formula='Call: Θ = -S*N\'(d1)*σ/(2√T) - r*K*e^(-rT)*N(d2)',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'volatility': 0.2, 'option_type': 'call'},
        complexity_level=3
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, volatility: float = 0.2,
                 option_type: str = 'call'):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type.lower()
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, volatility=volatility,
                        option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.volatility, 'volatility')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']
        K = self.strike
        T = self.time_to_maturity
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2))

        return pd.Series(theta, index=data.index)


@indicator_registry.register('derivatives')
class Vega(BaseIndicator):
    """
    Vega (Vega值)

    期权价格相对于波动率的导数
    """

    _metadata = IndicatorMetadata(
        name='Vega',
        display_name='Vega值',
        category='derivatives',
        description='期权价格对波动率的敏感度',
        formula='ν = S * N\'(d1) * √T',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'volatility': 0.2},
        complexity_level=3
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, volatility: float = 0.2):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, volatility=volatility)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.volatility, 'volatility')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']
        K = self.strike
        T = self.time_to_maturity
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        return pd.Series(vega, index=data.index)


@indicator_registry.register('derivatives')
class Rho(BaseIndicator):
    """
    Rho (Rho值)

    期权价格相对于无风险利率的导数
    """

    _metadata = IndicatorMetadata(
        name='Rho',
        display_name='Rho值',
        category='derivatives',
        description='期权价格对无风险利率的敏感度',
        formula='Call: ρ = K*T*e^(-rT)*N(d2)',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'volatility': 0.2, 'option_type': 'call'},
        complexity_level=3
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, volatility: float = 0.2,
                 option_type: str = 'call'):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type.lower()
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, volatility=volatility,
                        option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.volatility, 'volatility')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']
        K = self.strike
        T = self.time_to_maturity
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return pd.Series(rho, index=data.index)


# ============================================================================
# 隐含波动率 (5个)
# ============================================================================

@indicator_registry.register('derivatives')
class ImpliedVolatility(BaseIndicator):
    """
    Implied Volatility (隐含波动率)

    从期权价格反推的波动率
    """

    _metadata = IndicatorMetadata(
        name='ImpliedVolatility',
        display_name='隐含波动率',
        category='derivatives',
        description='Black-Scholes模型隐含波动率',
        formula='使用牛顿法求解BS公式得到的σ',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'option_price': 5.0, 'option_type': 'call'},
        complexity_level=4
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, option_price: float = 5.0,
                 option_type: str = 'call'):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.option_price = option_price
        self.option_type = option_type.lower()
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, option_price=option_price,
                        option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.option_price, 'option_price')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def _bs_price(self, S, K, T, r, sigma, option_type):
        """计算Black-Scholes期权价格"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']

        iv_values = []
        for spot in S:
            try:
                # 使用Brent方法求解隐含波动率
                iv = brentq(
                    lambda sigma: self._bs_price(spot, self.strike, self.time_to_maturity,
                                                 self.risk_free_rate, sigma, self.option_type) - self.option_price,
                    0.001, 5.0
                )
                iv_values.append(iv)
            except:
                iv_values.append(np.nan)

        return pd.Series(iv_values, index=data.index)


@indicator_registry.register('derivatives')
class IVSkew(BaseIndicator):
    """
    IV Skew (波动率偏斜)

    不同执行价格的隐含波动率差异
    """

    _metadata = IndicatorMetadata(
        name='IVSkew',
        display_name='波动率偏斜',
        category='derivatives',
        description='OTM看跌期权IV与OTM看涨期权IV的差异',
        formula='Skew = IV(Put, K_low) - IV(Call, K_high)',
        required_columns=['close'],
        output_type='series',
        default_params={'atm_volatility': 0.2, 'skew_factor': 0.1, 'strike_distance': 0.1},
        complexity_level=3
    )

    def __init__(self, atm_volatility: float = 0.2, skew_factor: float = 0.1,
                 strike_distance: float = 0.1):
        self.atm_volatility = atm_volatility
        self.skew_factor = skew_factor
        self.strike_distance = strike_distance
        super().__init__(atm_volatility=atm_volatility, skew_factor=skew_factor,
                        strike_distance=strike_distance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.atm_volatility, 'atm_volatility')
        IndicatorValidator.validate_range(self.strike_distance, 'strike_distance', 0.01, 0.5)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']

        # 简化模型：使用线性关系模拟波动率偏斜
        moneyness_low = 1 - self.strike_distance
        moneyness_high = 1 + self.strike_distance

        # OTM Put IV (低执行价)
        iv_put = self.atm_volatility + self.skew_factor * (1 - moneyness_low)

        # OTM Call IV (高执行价)
        iv_call = self.atm_volatility + self.skew_factor * (moneyness_high - 1)

        skew = iv_put - iv_call

        return pd.Series(skew, index=data.index)


@indicator_registry.register('derivatives')
class IVSmile(BaseIndicator):
    """
    IV Smile (波动率微笑)

    波动率曲线的微笑形状程度
    """

    _metadata = IndicatorMetadata(
        name='IVSmile',
        display_name='波动率微笑',
        category='derivatives',
        description='波动率曲线相对于ATM的凸性',
        formula='Smile = (IV_OTM + IV_ITM) / 2 - IV_ATM',
        required_columns=['close'],
        output_type='series',
        default_params={'atm_volatility': 0.2, 'smile_factor': 0.05},
        complexity_level=3
    )

    def __init__(self, atm_volatility: float = 0.2, smile_factor: float = 0.05):
        self.atm_volatility = atm_volatility
        self.smile_factor = smile_factor
        super().__init__(atm_volatility=atm_volatility, smile_factor=smile_factor)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.atm_volatility, 'atm_volatility')
        IndicatorValidator.validate_positive_float(self.smile_factor, 'smile_factor')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 模拟波动率微笑：OTM期权波动率高于ATM
        smile = pd.Series(self.smile_factor, index=data.index)

        return smile


@indicator_registry.register('derivatives')
class IVTermStructure(BaseIndicator):
    """
    IV Term Structure (波动率期限结构)

    不同到期日的隐含波动率差异
    """

    _metadata = IndicatorMetadata(
        name='IVTermStructure',
        display_name='波动率期限结构',
        category='derivatives',
        description='长期与短期隐含波动率的差异',
        formula='Term = IV_long - IV_short',
        required_columns=['close'],
        output_type='series',
        default_params={'short_term_vol': 0.2, 'long_term_vol': 0.18, 'period': 20},
        complexity_level=3
    )

    def __init__(self, short_term_vol: float = 0.2, long_term_vol: float = 0.18,
                 period: int = 20):
        self.short_term_vol = short_term_vol
        self.long_term_vol = long_term_vol
        self.period = period
        super().__init__(short_term_vol=short_term_vol, long_term_vol=long_term_vol,
                        period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.short_term_vol, 'short_term_vol')
        IndicatorValidator.validate_positive_float(self.long_term_vol, 'long_term_vol')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算实际波动率作为参考
        returns = np.log(data['close'] / data['close'].shift(1))
        realized_vol = returns.rolling(window=self.period).std() * np.sqrt(252)

        # 期限结构：长期波动率 - 短期波动率
        term_structure = self.long_term_vol - self.short_term_vol + realized_vol * 0.1

        return term_structure


@indicator_registry.register('derivatives')
class IVSurface(BaseIndicator):
    """
    IV Surface (波动率曲面)

    波动率曲面的综合指标
    """

    _metadata = IndicatorMetadata(
        name='IVSurface',
        display_name='波动率曲面',
        category='derivatives',
        description='波动率曲面的整体水平和形状',
        formula='Surface = f(Moneyness, TimeToMaturity)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'base_volatility': 0.2, 'period': 20},
        complexity_level=4
    )

    def __init__(self, base_volatility: float = 0.2, period: int = 20):
        self.base_volatility = base_volatility
        self.period = period
        super().__init__(base_volatility=base_volatility, period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.base_volatility, 'base_volatility')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 计算实际波动率
        returns = np.log(data['close'] / data['close'].shift(1))
        realized_vol = returns.rolling(window=self.period).std() * np.sqrt(252)

        # 模拟波动率曲面的关键点
        result = pd.DataFrame(index=data.index)
        result['atm_vol'] = self.base_volatility + realized_vol * 0.5
        result['otm_call_vol'] = result['atm_vol'] + 0.05
        result['otm_put_vol'] = result['atm_vol'] + 0.08
        result['skew'] = result['otm_put_vol'] - result['otm_call_vol']

        return result


# ============================================================================
# 期权策略指标 (5个)
# ============================================================================

@indicator_registry.register('derivatives')
class PutCallParity(BaseIndicator):
    """
    Put-Call Parity (看涨看跌平价)

    检验期权定价是否满足平价关系
    """

    _metadata = IndicatorMetadata(
        name='PutCallParity',
        display_name='看涨看跌平价',
        category='derivatives',
        description='期权平价关系偏离度',
        formula='C - P = S - K*e^(-rT)',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'time_to_maturity': 0.25, 'risk_free_rate': 0.05, 'call_price': 5.0, 'put_price': 3.0},
        complexity_level=2
    )

    def __init__(self, strike: float = 100.0, time_to_maturity: float = 0.25,
                 risk_free_rate: float = 0.05, call_price: float = 5.0,
                 put_price: float = 3.0):
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.call_price = call_price
        self.put_price = put_price
        super().__init__(strike=strike, time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate, call_price=call_price,
                        put_price=put_price)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.time_to_maturity, 'time_to_maturity')
        IndicatorValidator.validate_positive_float(self.call_price, 'call_price')
        IndicatorValidator.validate_positive_float(self.put_price, 'put_price')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']

        # 平价关系左侧
        left_side = self.call_price - self.put_price

        # 平价关系右侧
        right_side = S - self.strike * np.exp(-self.risk_free_rate * self.time_to_maturity)

        # 偏离度
        parity_deviation = left_side - right_side

        return pd.Series(parity_deviation, index=data.index)


@indicator_registry.register('derivatives')
class OptionMoneyness(BaseIndicator):
    """
    Option Moneyness (期权虚实度)

    期权相对于标的资产价格的虚实程度
    """

    _metadata = IndicatorMetadata(
        name='OptionMoneyness',
        display_name='期权虚实度',
        category='derivatives',
        description='期权价值状态的量化指标',
        formula='Moneyness = S / K',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0},
        complexity_level=1
    )

    def __init__(self, strike: float = 100.0):
        self.strike = strike
        super().__init__(strike=strike)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        moneyness = data['close'] / self.strike
        return moneyness


@indicator_registry.register('derivatives')
class TimeValue(BaseIndicator):
    """
    Time Value (时间价值)

    期权的时间价值部分
    """

    _metadata = IndicatorMetadata(
        name='TimeValue',
        display_name='时间价值',
        category='derivatives',
        description='期权价格中的时间价值组成',
        formula='TimeValue = OptionPrice - IntrinsicValue',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'option_price': 5.0, 'option_type': 'call'},
        complexity_level=2
    )

    def __init__(self, strike: float = 100.0, option_price: float = 5.0,
                 option_type: str = 'call'):
        self.strike = strike
        self.option_price = option_price
        self.option_type = option_type.lower()
        super().__init__(strike=strike, option_price=option_price, option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.option_price, 'option_price')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']

        # 计算内在价值
        if self.option_type == 'call':
            intrinsic_value = np.maximum(S - self.strike, 0)
        else:
            intrinsic_value = np.maximum(self.strike - S, 0)

        # 时间价值 = 期权价格 - 内在价值
        time_value = self.option_price - intrinsic_value

        return pd.Series(time_value, index=data.index)


@indicator_registry.register('derivatives')
class IntrinsicValue(BaseIndicator):
    """
    Intrinsic Value (内在价值)

    期权的内在价值部分
    """

    _metadata = IndicatorMetadata(
        name='IntrinsicValue',
        display_name='内在价值',
        category='derivatives',
        description='期权立即行权的价值',
        formula='Call: max(S-K, 0), Put: max(K-S, 0)',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'option_type': 'call'},
        complexity_level=1
    )

    def __init__(self, strike: float = 100.0, option_type: str = 'call'):
        self.strike = strike
        self.option_type = option_type.lower()
        super().__init__(strike=strike, option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        S = data['close']

        if self.option_type == 'call':
            intrinsic_value = np.maximum(S - self.strike, 0)
        else:
            intrinsic_value = np.maximum(self.strike - S, 0)

        return pd.Series(intrinsic_value, index=data.index)


@indicator_registry.register('derivatives')
class BreakevenPrice(BaseIndicator):
    """
    Breakeven Price (盈亏平衡价)

    期权策略的盈亏平衡点
    """

    _metadata = IndicatorMetadata(
        name='BreakevenPrice',
        display_name='盈亏平衡价',
        category='derivatives',
        description='期权策略开始盈利的标的价格',
        formula='Call: K + Premium, Put: K - Premium',
        required_columns=['close'],
        output_type='series',
        default_params={'strike': 100.0, 'premium': 5.0, 'option_type': 'call'},
        complexity_level=1
    )

    def __init__(self, strike: float = 100.0, premium: float = 5.0,
                 option_type: str = 'call'):
        self.strike = strike
        self.premium = premium
        self.option_type = option_type.lower()
        super().__init__(strike=strike, premium=premium, option_type=option_type)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.strike, 'strike')
        IndicatorValidator.validate_positive_float(self.premium, 'premium')
        IndicatorValidator.validate_in_list(self.option_type, 'option_type', ['call', 'put'])

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        if self.option_type == 'call':
            breakeven = self.strike + self.premium
        else:
            breakeven = self.strike - self.premium

        return pd.Series(breakeven, index=data.index)


# ============================================================================
# 期货指标 (5个)
# ============================================================================

@indicator_registry.register('derivatives')
class BasisSpread(BaseIndicator):
    """
    Basis Spread (基差)

    期货价格与现货价格的差值
    """

    _metadata = IndicatorMetadata(
        name='BasisSpread',
        display_name='基差',
        category='derivatives',
        description='期货与现货价格差异',
        formula='Basis = Futures - Spot',
        required_columns=['close'],
        output_type='series',
        default_params={'futures_price': 105.0},
        complexity_level=1
    )

    def __init__(self, futures_price: float = 105.0):
        self.futures_price = futures_price
        super().__init__(futures_price=futures_price)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.futures_price, 'futures_price')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        basis = self.futures_price - data['close']
        return pd.Series(basis, index=data.index)


@indicator_registry.register('derivatives')
class RollYield(BaseIndicator):
    """
    Roll Yield (展期收益)

    期货合约展期的收益或成本
    """

    _metadata = IndicatorMetadata(
        name='RollYield',
        display_name='展期收益',
        category='derivatives',
        description='期货合约滚动的收益率',
        formula='RollYield = (NearPrice - FarPrice) / FarPrice / DaysToRoll * 365',
        required_columns=['close'],
        output_type='series',
        default_params={'near_contract_price': 102.0, 'far_contract_price': 105.0, 'days_to_roll': 30},
        complexity_level=2
    )

    def __init__(self, near_contract_price: float = 102.0,
                 far_contract_price: float = 105.0, days_to_roll: int = 30):
        self.near_contract_price = near_contract_price
        self.far_contract_price = far_contract_price
        self.days_to_roll = days_to_roll
        super().__init__(near_contract_price=near_contract_price,
                        far_contract_price=far_contract_price,
                        days_to_roll=days_to_roll)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.near_contract_price, 'near_contract_price')
        IndicatorValidator.validate_positive_float(self.far_contract_price, 'far_contract_price')
        IndicatorValidator.validate_positive_int(self.days_to_roll, 'days_to_roll')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 年化展期收益率
        roll_yield = ((self.near_contract_price - self.far_contract_price) /
                     self.far_contract_price / self.days_to_roll * 365)

        return pd.Series(roll_yield, index=data.index)


@indicator_registry.register('derivatives')
class Contango(BaseIndicator):
    """
    Contango (正向市场)

    远期合约价格高于近期合约的程度
    """

    _metadata = IndicatorMetadata(
        name='Contango',
        display_name='正向市场',
        category='derivatives',
        description='期货曲线正向程度',
        formula='Contango = (FarPrice - NearPrice) / NearPrice',
        required_columns=['close'],
        output_type='series',
        default_params={'near_price': 100.0, 'far_price': 105.0},
        complexity_level=2
    )

    def __init__(self, near_price: float = 100.0, far_price: float = 105.0):
        self.near_price = near_price
        self.far_price = far_price
        super().__init__(near_price=near_price, far_price=far_price)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.near_price, 'near_price')
        IndicatorValidator.validate_positive_float(self.far_price, 'far_price')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        contango = (self.far_price - self.near_price) / self.near_price

        return pd.Series(contango, index=data.index)


@indicator_registry.register('derivatives')
class Backwardation(BaseIndicator):
    """
    Backwardation (反向市场)

    近期合约价格高于远期合约的程度
    """

    _metadata = IndicatorMetadata(
        name='Backwardation',
        display_name='反向市场',
        category='derivatives',
        description='期货曲线反向程度',
        formula='Backwardation = (NearPrice - FarPrice) / FarPrice',
        required_columns=['close'],
        output_type='series',
        default_params={'near_price': 105.0, 'far_price': 100.0},
        complexity_level=2
    )

    def __init__(self, near_price: float = 105.0, far_price: float = 100.0):
        self.near_price = near_price
        self.far_price = far_price
        super().__init__(near_price=near_price, far_price=far_price)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.near_price, 'near_price')
        IndicatorValidator.validate_positive_float(self.far_price, 'far_price')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        backwardation = (self.near_price - self.far_price) / self.far_price

        return pd.Series(backwardation, index=data.index)


@indicator_registry.register('derivatives')
class CalendarSpread(BaseIndicator):
    """
    Calendar Spread (日历价差)

    不同到期月份合约的价差
    """

    _metadata = IndicatorMetadata(
        name='CalendarSpread',
        display_name='日历价差',
        category='derivatives',
        description='跨期价差策略的价差值',
        formula='Spread = FarContract - NearContract',
        required_columns=['close'],
        output_type='series',
        default_params={'near_price': 100.0, 'far_price': 103.0, 'period': 20},
        complexity_level=2
    )

    def __init__(self, near_price: float = 100.0, far_price: float = 103.0,
                 period: int = 20):
        self.near_price = near_price
        self.far_price = far_price
        self.period = period
        super().__init__(near_price=near_price, far_price=far_price, period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.near_price, 'near_price')
        IndicatorValidator.validate_positive_float(self.far_price, 'far_price')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 日历价差
        spread = self.far_price - self.near_price

        # 计算移动平均作为参考
        spread_series = pd.Series(spread, index=data.index)
        ma_spread = spread_series.rolling(window=self.period).mean()

        return spread_series


# ============================================================================
# 风险指标 (5个)
# ============================================================================

@indicator_registry.register('derivatives')
class MaxPain(BaseIndicator):
    """
    Max Pain (最大痛点)

    期权到期时造成期权持有者最大损失的价格
    """

    _metadata = IndicatorMetadata(
        name='MaxPain',
        display_name='最大痛点',
        category='derivatives',
        description='期权到期时期权方总损失最大的价格点',
        formula='MaxPain = argmax(CallOI * max(S-K,0) + PutOI * max(K-S,0))',
        required_columns=['close'],
        output_type='series',
        default_params={'strike_prices': [95, 100, 105], 'call_oi': [100, 200, 150], 'put_oi': [150, 200, 100]},
        complexity_level=3
    )

    def __init__(self, strike_prices: List[float] = None,
                 call_oi: List[float] = None, put_oi: List[float] = None):
        self.strike_prices = strike_prices or [95, 100, 105]
        self.call_oi = call_oi or [100, 200, 150]
        self.put_oi = put_oi or [150, 200, 100]
        super().__init__(strike_prices=strike_prices, call_oi=call_oi, put_oi=put_oi)

    def _validate_params(self):
        if len(self.strike_prices) != len(self.call_oi) or len(self.strike_prices) != len(self.put_oi):
            raise ValueError("strike_prices, call_oi, and put_oi must have the same length")

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算最大痛点价格
        pain_values = []
        for S in self.strike_prices:
            total_pain = 0
            for i, K in enumerate(self.strike_prices):
                # Call损失
                call_loss = self.call_oi[i] * max(S - K, 0)
                # Put损失
                put_loss = self.put_oi[i] * max(K - S, 0)
                total_pain += call_loss + put_loss
            pain_values.append((S, total_pain))

        # 找到最大痛点
        max_pain_price = max(pain_values, key=lambda x: x[1])[0]

        return pd.Series(max_pain_price, index=data.index)


@indicator_registry.register('derivatives')
class OpenInterestRatio(BaseIndicator):
    """
    Open Interest Ratio (持仓量比率)

    看涨与看跌期权持仓量的比率
    """

    _metadata = IndicatorMetadata(
        name='OpenInterestRatio',
        display_name='持仓量比率',
        category='derivatives',
        description='看涨看跌期权持仓量对比',
        formula='OIRatio = CallOI / PutOI',
        required_columns=['close'],
        output_type='series',
        default_params={'call_oi': 10000.0, 'put_oi': 8000.0, 'period': 20},
        complexity_level=2
    )

    def __init__(self, call_oi: float = 10000.0, put_oi: float = 8000.0,
                 period: int = 20):
        self.call_oi = call_oi
        self.put_oi = put_oi
        self.period = period
        super().__init__(call_oi=call_oi, put_oi=put_oi, period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.call_oi, 'call_oi')
        IndicatorValidator.validate_positive_float(self.put_oi, 'put_oi')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        oi_ratio = self.call_oi / self.put_oi
        ratio_series = pd.Series(oi_ratio, index=data.index)

        # 计算移动平均
        ma_ratio = ratio_series.rolling(window=self.period).mean()

        return ratio_series


@indicator_registry.register('derivatives')
class VolumeOIRatio(BaseIndicator):
    """
    Volume/OI Ratio (成交持仓比)

    成交量与持仓量的比率
    """

    _metadata = IndicatorMetadata(
        name='VolumeOIRatio',
        display_name='成交持仓比',
        category='derivatives',
        description='成交量与持仓量的比率，衡量市场活跃度',
        formula='Ratio = Volume / OpenInterest',
        required_columns=['volume'],
        output_type='series',
        default_params={'open_interest': 50000.0},
        complexity_level=2
    )

    def __init__(self, open_interest: float = 50000.0):
        self.open_interest = open_interest
        super().__init__(open_interest=open_interest)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.open_interest, 'open_interest')

    @property
    def required_columns(self) -> List[str]:
        return ['volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        volume_oi_ratio = data['volume'] / self.open_interest

        return volume_oi_ratio


@indicator_registry.register('derivatives')
class PCRatio(BaseIndicator):
    """
    Put-Call Ratio (看跌看涨比率)

    看跌期权与看涨期权的成交量或持仓量比率
    """

    _metadata = IndicatorMetadata(
        name='PCRatio',
        display_name='看跌看涨比率',
        category='derivatives',
        description='看跌看涨期权的相对强度，市场情绪指标',
        formula='PCR = PutVolume / CallVolume',
        required_columns=['close'],
        output_type='series',
        default_params={'put_volume': 6000.0, 'call_volume': 5000.0, 'period': 20},
        complexity_level=2
    )

    def __init__(self, put_volume: float = 6000.0, call_volume: float = 5000.0,
                 period: int = 20):
        self.put_volume = put_volume
        self.call_volume = call_volume
        self.period = period
        super().__init__(put_volume=put_volume, call_volume=call_volume, period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.put_volume, 'put_volume')
        IndicatorValidator.validate_positive_float(self.call_volume, 'call_volume')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        pc_ratio = self.put_volume / self.call_volume
        ratio_series = pd.Series(pc_ratio, index=data.index)

        # 计算移动平均
        ma_ratio = ratio_series.rolling(window=self.period).mean()

        return ratio_series


@indicator_registry.register('derivatives')
class SkewIndex(BaseIndicator):
    """
    Skew Index (偏斜指数)

    衡量波动率偏斜程度的指数
    """

    _metadata = IndicatorMetadata(
        name='SkewIndex',
        display_name='偏斜指数',
        category='derivatives',
        description='波动率曲线的偏斜程度指数',
        formula='SkewIndex = (OTM_Put_IV - ATM_IV) / ATM_IV * 100',
        required_columns=['close'],
        output_type='series',
        default_params={'atm_vol': 0.2, 'otm_put_vol': 0.25, 'period': 20},
        complexity_level=3
    )

    def __init__(self, atm_vol: float = 0.2, otm_put_vol: float = 0.25,
                 period: int = 20):
        self.atm_vol = atm_vol
        self.otm_put_vol = otm_put_vol
        self.period = period
        super().__init__(atm_vol=atm_vol, otm_put_vol=otm_put_vol, period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.atm_vol, 'atm_vol')
        IndicatorValidator.validate_positive_float(self.otm_put_vol, 'otm_put_vol')
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算偏斜指数
        skew_index = (self.otm_put_vol - self.atm_vol) / self.atm_vol * 100

        # 添加实际波动率的影响
        returns = np.log(data['close'] / data['close'].shift(1))
        realized_vol = returns.rolling(window=self.period).std() * np.sqrt(252)

        # 调整后的偏斜指数
        adjusted_skew = skew_index + realized_vol * 10

        return adjusted_skew
