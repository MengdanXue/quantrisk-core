"""
跨品种分析指标 (15个)

分析不同资产类别之间的关系和传导效应
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 货币与商品关系 (5个) ====================

@indicator_registry.register('intermarket')
class DollarCommodityRelation(BaseIndicator):
    """
    美元-商品关系指标

    分析美元指数与商品价格的负相关关系
    """

    _metadata = IndicatorMetadata(
        name='DollarCommodityRelation',
        display_name='美元商品关系',
        category='intermarket',
        description='分析美元指数与商品价格的负相关关系',
        formula='Correlation(DXY, Commodity) over window',
        required_columns=['dollar_index', 'commodity_price'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['dollar_index', 'commodity_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算美元-商品相关性"""
        dollar = data['dollar_index']
        commodity = data['commodity_price']
        correlation = dollar.rolling(self.window).corr(commodity)
        return correlation


@indicator_registry.register('intermarket')
class GoldEquityRatio(BaseIndicator):
    """
    黄金/股票比率

    衡量避险情绪,比率上升表示风险厌恶
    """

    _metadata = IndicatorMetadata(
        name='GoldEquityRatio',
        display_name='黄金股票比率',
        category='intermarket',
        description='黄金价格与股票指数的比率,衡量避险情绪',
        formula='Gold Price / Equity Index',
        required_columns=['gold_price', 'equity_index'],
        output_type='series',
        default_params={'smooth': 5},
        complexity_level=1
    )

    def __init__(self, smooth: int = 5):
        self.smooth = smooth
        super().__init__(smooth=smooth)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.smooth, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['gold_price', 'equity_index']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算黄金/股票比率"""
        ratio = data['gold_price'] / data['equity_index']
        if self.smooth > 1:
            ratio = ratio.rolling(self.smooth).mean()
        return ratio


@indicator_registry.register('intermarket')
class CopperGoldRatio(BaseIndicator):
    """
    铜/黄金比率

    经济晴雨表,比率上升表示经济扩张预期
    """

    _metadata = IndicatorMetadata(
        name='CopperGoldRatio',
        display_name='铜金比率',
        category='intermarket',
        description='铜价与金价比率,反映经济周期预期',
        formula='Copper Price / Gold Price',
        required_columns=['copper_price', 'gold_price'],
        output_type='series',
        default_params={'smooth': 5},
        complexity_level=1
    )

    def __init__(self, smooth: int = 5):
        self.smooth = smooth
        super().__init__(smooth=smooth)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.smooth, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['copper_price', 'gold_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算铜/黄金比率"""
        ratio = data['copper_price'] / data['gold_price']
        if self.smooth > 1:
            ratio = ratio.rolling(self.smooth).mean()
        return ratio


@indicator_registry.register('intermarket')
class OilEquityCorrelation(BaseIndicator):
    """
    油价股市相关性

    分析原油价格与股市的动态相关关系
    """

    _metadata = IndicatorMetadata(
        name='OilEquityCorrelation',
        display_name='油价股市相关性',
        category='intermarket',
        description='原油价格与股票指数的滚动相关性',
        formula='Rolling Correlation(Oil, Equity)',
        required_columns=['oil_price', 'equity_index'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['oil_price', 'equity_index']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算油价与股市相关性"""
        oil_ret = data['oil_price'].pct_change()
        equity_ret = data['equity_index'].pct_change()
        correlation = oil_ret.rolling(self.window).corr(equity_ret)
        return correlation


@indicator_registry.register('intermarket')
class CurrencyCarryTrade(BaseIndicator):
    """
    货币套息交易指标

    基于利差的货币强弱信号
    """

    _metadata = IndicatorMetadata(
        name='CurrencyCarryTrade',
        display_name='套息交易信号',
        category='intermarket',
        description='基于利率差异的货币套息交易信号',
        formula='High Yield Rate - Low Yield Rate',
        required_columns=['rate_high', 'rate_low', 'fx_rate'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=2
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['rate_high', 'rate_low', 'fx_rate']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算套息交易信号"""
        carry = data['rate_high'] - data['rate_low']
        fx_momentum = data['fx_rate'].pct_change(self.window)
        signal = carry * 100 + fx_momentum * 100
        return signal


# ==================== 债券与股票关系 (5个) ====================

@indicator_registry.register('intermarket')
class BondEquityCorrelation(BaseIndicator):
    """
    债股相关性

    分析债券和股票的动态相关关系
    """

    _metadata = IndicatorMetadata(
        name='BondEquityCorrelation',
        display_name='债股相关性',
        category='intermarket',
        description='债券与股票收益率的滚动相关性',
        formula='Rolling Correlation(Bond Returns, Equity Returns)',
        required_columns=['bond_price', 'equity_price'],
        output_type='series',
        default_params={'window': 60},
        complexity_level=2
    )

    def __init__(self, window: int = 60):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=20)

    @property
    def required_columns(self) -> List[str]:
        return ['bond_price', 'equity_price']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算债股相关性"""
        bond_ret = data['bond_price'].pct_change()
        equity_ret = data['equity_price'].pct_change()
        correlation = bond_ret.rolling(self.window).corr(equity_ret)
        return correlation


@indicator_registry.register('intermarket')
class YieldCurveSlope(BaseIndicator):
    """
    收益率曲线斜率

    10年期与2年期利差,预测经济周期
    """

    _metadata = IndicatorMetadata(
        name='YieldCurveSlope',
        display_name='收益率曲线斜率',
        category='intermarket',
        description='长短期利差,负值预示经济衰退',
        formula='10Y Yield - 2Y Yield',
        required_columns=['yield_10y', 'yield_2y'],
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
        return ['yield_10y', 'yield_2y']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算收益率曲线斜率"""
        slope = data['yield_10y'] - data['yield_2y']
        return slope


@indicator_registry.register('intermarket')
class RealYield(BaseIndicator):
    """
    实际收益率

    名义收益率减去通胀预期
    """

    _metadata = IndicatorMetadata(
        name='RealYield',
        display_name='实际收益率',
        category='intermarket',
        description='名义收益率减去通胀预期',
        formula='Nominal Yield - Inflation Expectation',
        required_columns=['nominal_yield', 'inflation_expectation'],
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
        return ['nominal_yield', 'inflation_expectation']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算实际收益率"""
        real_yield = data['nominal_yield'] - data['inflation_expectation']
        return real_yield


@indicator_registry.register('intermarket')
class CreditSpread(BaseIndicator):
    """
    信用利差

    公司债与国债收益率之差,反映信用风险
    """

    _metadata = IndicatorMetadata(
        name='CreditSpread',
        display_name='信用利差',
        category='intermarket',
        description='公司债与国债收益率之差',
        formula='Corporate Yield - Treasury Yield',
        required_columns=['corporate_yield', 'treasury_yield'],
        output_type='series',
        default_params={'smooth': 5},
        complexity_level=1
    )

    def __init__(self, smooth: int = 5):
        self.smooth = smooth
        super().__init__(smooth=smooth)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.smooth, min_val=1)

    @property
    def required_columns(self) -> List[str]:
        return ['corporate_yield', 'treasury_yield']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算信用利差"""
        spread = data['corporate_yield'] - data['treasury_yield']
        if self.smooth > 1:
            spread = spread.rolling(self.smooth).mean()
        return spread


@indicator_registry.register('intermarket')
class TIPSSpread(BaseIndicator):
    """
    TIPS利差 (通胀预期)

    名义国债与通胀保值债券收益率之差
    """

    _metadata = IndicatorMetadata(
        name='TIPSSpread',
        display_name='TIPS利差',
        category='intermarket',
        description='名义国债与TIPS收益率之差,反映通胀预期',
        formula='Nominal Treasury Yield - TIPS Yield',
        required_columns=['treasury_yield', 'tips_yield'],
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
        return ['treasury_yield', 'tips_yield']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算TIPS利差"""
        spread = data['treasury_yield'] - data['tips_yield']
        return spread


# ==================== 风险情绪指标 (5个) ====================

@indicator_registry.register('intermarket')
class RiskOnOffIndicator(BaseIndicator):
    """
    风险偏好指标

    综合多资产表现判断市场风险偏好
    """

    _metadata = IndicatorMetadata(
        name='RiskOnOffIndicator',
        display_name='风险偏好指标',
        category='intermarket',
        description='基于多资产表现的风险偏好综合指标',
        formula='Weighted Score(Equity, Credit, FX, Commodity)',
        required_columns=['equity_ret', 'credit_ret', 'fx_ret', 'commodity_ret'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['equity_ret', 'credit_ret', 'fx_ret', 'commodity_ret']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算风险偏好指标"""
        weights = {'equity_ret': 0.4, 'credit_ret': 0.2, 'fx_ret': 0.2, 'commodity_ret': 0.2}
        score = sum(data[col].rolling(self.window).mean() * w for col, w in weights.items())
        normalized = (score - score.rolling(252).mean()) / score.rolling(252).std()
        return normalized


@indicator_registry.register('intermarket')
class SafeHavenDemand(BaseIndicator):
    """
    避险需求指标

    基于黄金、日元、瑞郎等避险资产表现
    """

    _metadata = IndicatorMetadata(
        name='SafeHavenDemand',
        display_name='避险需求',
        category='intermarket',
        description='基于避险资产表现的需求指标',
        formula='Average Performance(Gold, JPY, CHF, Treasuries)',
        required_columns=['gold_ret', 'jpy_ret', 'chf_ret', 'treasury_ret'],
        output_type='series',
        default_params={'window': 10},
        complexity_level=2
    )

    def __init__(self, window: int = 10):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=3)

    @property
    def required_columns(self) -> List[str]:
        return ['gold_ret', 'jpy_ret', 'chf_ret', 'treasury_ret']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算避险需求"""
        cols = ['gold_ret', 'jpy_ret', 'chf_ret', 'treasury_ret']
        avg_ret = data[cols].mean(axis=1)
        demand = avg_ret.rolling(self.window).sum()
        return demand


@indicator_registry.register('intermarket')
class LiquidityStress(BaseIndicator):
    """
    流动性压力指标

    综合多个流动性指标判断市场压力
    """

    _metadata = IndicatorMetadata(
        name='LiquidityStress',
        display_name='流动性压力',
        category='intermarket',
        description='综合流动性压力指标',
        formula='Composite(TED Spread, VIX, Credit Spread)',
        required_columns=['ted_spread', 'vix', 'credit_spread'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['ted_spread', 'vix', 'credit_spread']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算流动性压力"""
        # 标准化各指标
        ted_z = (data['ted_spread'] - data['ted_spread'].rolling(252).mean()) / data['ted_spread'].rolling(252).std()
        vix_z = (data['vix'] - data['vix'].rolling(252).mean()) / data['vix'].rolling(252).std()
        credit_z = (data['credit_spread'] - data['credit_spread'].rolling(252).mean()) / data['credit_spread'].rolling(252).std()

        stress = (ted_z + vix_z + credit_z) / 3
        return stress


@indicator_registry.register('intermarket')
class GlobalRiskAppetite(BaseIndicator):
    """
    全球风险偏好指数

    基于多国股市和汇率波动的综合指标
    """

    _metadata = IndicatorMetadata(
        name='GlobalRiskAppetite',
        display_name='全球风险偏好',
        category='intermarket',
        description='基于全球市场的风险偏好综合指数',
        formula='PCA First Component(Global Equity Returns)',
        required_columns=['us_ret', 'eu_ret', 'asia_ret', 'em_ret'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['us_ret', 'eu_ret', 'asia_ret', 'em_ret']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算全球风险偏好"""
        cols = ['us_ret', 'eu_ret', 'asia_ret', 'em_ret']
        avg_ret = data[cols].mean(axis=1)
        appetite = avg_ret.rolling(self.window).mean() / avg_ret.rolling(self.window).std()
        return appetite


@indicator_registry.register('intermarket')
class MarketRegimeIndicator(BaseIndicator):
    """
    市场状态指标

    判断当前市场处于风险偏好还是风险厌恶状态
    """

    _metadata = IndicatorMetadata(
        name='MarketRegimeIndicator',
        display_name='市场状态',
        category='intermarket',
        description='判断市场风险偏好/厌恶状态',
        formula='Regime Classification based on multi-asset signals',
        required_columns=['equity_ret', 'bond_ret', 'vix'],
        output_type='series',
        default_params={'window': 20, 'threshold': 1.0},
        complexity_level=3
    )

    def __init__(self, window: int = 20, threshold: float = 1.0):
        self.window = window
        self.threshold = threshold
        super().__init__(window=window, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_period(self.window, min_val=5)

    @property
    def required_columns(self) -> List[str]:
        return ['equity_ret', 'bond_ret', 'vix']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算市场状态"""
        equity_mom = data['equity_ret'].rolling(self.window).mean()
        bond_mom = data['bond_ret'].rolling(self.window).mean()
        vix_level = (data['vix'] - data['vix'].rolling(252).mean()) / data['vix'].rolling(252).std()

        # 综合评分: 股票正收益+债券负收益+低VIX = 风险偏好
        score = equity_mom * 100 - bond_mom * 50 - vix_level

        # 状态分类: 1=风险偏好, 0=中性, -1=风险厌恶
        regime = pd.Series(0, index=data.index)
        regime[score > self.threshold] = 1
        regime[score < -self.threshold] = -1

        return regime
