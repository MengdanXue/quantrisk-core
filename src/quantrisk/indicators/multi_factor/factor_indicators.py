"""
多因子组合指标 (25个)

包含价值、质量、成长、动量、波动、规模、流动性和组合因子
"""

from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from scipy import stats

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 价值因子 (5个) ====================

@indicator_registry.register('multi_factor')
class PBFactor(BaseIndicator):
    """
    市净率因子 (Price-to-Book Ratio)

    衡量股价相对于每股净资产的倍数,低PB通常表示价值被低估
    """

    _metadata = IndicatorMetadata(
        name='PBFactor',
        display_name='市净率因子',
        category='multi_factor',
        description='衡量股价相对于每股净资产的倍数,低PB通常表示价值被低估',
        formula='PB = 市值 / 净资产; Factor_Score = -1 * normalize(PB)',
        required_columns=['close', 'shares_outstanding', 'total_assets', 'total_liabilities'],
        output_type='series',
        default_params={'normalization': 'zscore'},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore'):
        """
        Args:
            normalization: 归一化方法 ('zscore', 'minmax', 'rank')
        """
        self.normalization = normalization
        super().__init__(normalization=normalization)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding', 'total_assets', 'total_liabilities']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算净资产 = 总资产 - 总负债
        book_value = data['total_assets'] - data['total_liabilities']

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 计算PB
        pb_ratio = self._safe_division(market_cap, book_value, fill_value=np.nan)

        # 归一化并反转 (低PB = 高分数)
        return -1 * self._normalize(pb_ratio)

    def _normalize(self, series: pd.Series) -> pd.Series:
        """归一化处理"""
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class PEFactor(BaseIndicator):
    """
    市盈率因子 (Price-to-Earnings Ratio)

    衡量股价相对于每股收益的倍数,低PE通常表示估值合理
    """

    _metadata = IndicatorMetadata(
        name='PEFactor',
        display_name='市盈率因子',
        category='multi_factor',
        description='衡量股价相对于每股收益的倍数,低PE通常表示估值合理',
        formula='PE = 市值 / 净利润; Factor_Score = -1 * normalize(PE)',
        required_columns=['close', 'shares_outstanding', 'net_income'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        """
        Args:
            normalization: 归一化方法
            ttm_periods: TTM(滚动12个月)周期数,默认4个季度
        """
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding', 'net_income']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM净利润
        ttm_net_income = data['net_income'].rolling(window=self.ttm_periods).sum()

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 计算PE
        pe_ratio = self._safe_division(market_cap, ttm_net_income, fill_value=np.nan)

        # 归一化并反转
        return -1 * self._normalize(pe_ratio)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class PSFactor(BaseIndicator):
    """
    市销率因子 (Price-to-Sales Ratio)

    衡量股价相对于每股销售收入的倍数,适合盈利不稳定的公司
    """

    _metadata = IndicatorMetadata(
        name='PSFactor',
        display_name='市销率因子',
        category='multi_factor',
        description='衡量股价相对于每股销售收入的倍数,适合盈利不稳定的公司',
        formula='PS = 市值 / 营业收入; Factor_Score = -1 * normalize(PS)',
        required_columns=['close', 'shares_outstanding', 'revenue'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding', 'revenue']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM收入
        ttm_revenue = data['revenue'].rolling(window=self.ttm_periods).sum()

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 计算PS
        ps_ratio = self._safe_division(market_cap, ttm_revenue, fill_value=np.nan)

        return -1 * self._normalize(ps_ratio)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class PCFFactor(BaseIndicator):
    """
    市现率因子 (Price-to-Cash-Flow Ratio)

    衡量股价相对于每股现金流的倍数,比PE更难被操纵
    """

    _metadata = IndicatorMetadata(
        name='PCFFactor',
        display_name='市现率因子',
        category='multi_factor',
        description='衡量股价相对于每股现金流的倍数,比PE更难被操纵',
        formula='PCF = 市值 / 经营现金流; Factor_Score = -1 * normalize(PCF)',
        required_columns=['close', 'shares_outstanding', 'operating_cash_flow'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding', 'operating_cash_flow']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM现金流
        ttm_cf = data['operating_cash_flow'].rolling(window=self.ttm_periods).sum()

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 计算PCF
        pcf_ratio = self._safe_division(market_cap, ttm_cf, fill_value=np.nan)

        return -1 * self._normalize(pcf_ratio)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class EVEBITDAFactor(BaseIndicator):
    """
    企业倍数因子 (EV/EBITDA)

    衡量企业价值相对于EBITDA的倍数,考虑负债的估值指标
    """

    _metadata = IndicatorMetadata(
        name='EVEBITDAFactor',
        display_name='企业倍数因子',
        category='multi_factor',
        description='衡量企业价值相对于EBITDA的倍数,考虑负债的估值指标',
        formula='EV/EBITDA = (市值 + 净负债) / EBITDA; Factor_Score = -1 * normalize(EV/EBITDA)',
        required_columns=['close', 'shares_outstanding', 'total_debt', 'cash', 'ebitda'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding', 'total_debt', 'cash', 'ebitda']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 计算企业价值 EV = 市值 + 总负债 - 现金
        enterprise_value = market_cap + data['total_debt'] - data['cash']

        # 计算TTM EBITDA
        ttm_ebitda = data['ebitda'].rolling(window=self.ttm_periods).sum()

        # 计算EV/EBITDA
        ev_ebitda = self._safe_division(enterprise_value, ttm_ebitda, fill_value=np.nan)

        return -1 * self._normalize(ev_ebitda)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 质量因子 (5个) ====================

@indicator_registry.register('multi_factor')
class ROEFactor(BaseIndicator):
    """
    净资产收益率因子 (Return on Equity)

    衡量公司利用股东权益创造利润的能力,高ROE表示盈利能力强
    """

    _metadata = IndicatorMetadata(
        name='ROEFactor',
        display_name='净资产收益率因子',
        category='multi_factor',
        description='衡量公司利用股东权益创造利润的能力,高ROE表示盈利能力强',
        formula='ROE = 净利润 / 股东权益; Factor_Score = normalize(ROE)',
        required_columns=['net_income', 'total_assets', 'total_liabilities'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['net_income', 'total_assets', 'total_liabilities']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM净利润
        ttm_net_income = data['net_income'].rolling(window=self.ttm_periods).sum()

        # 计算股东权益
        shareholders_equity = data['total_assets'] - data['total_liabilities']

        # 计算ROE
        roe = self._safe_division(ttm_net_income, shareholders_equity, fill_value=np.nan)

        return self._normalize(roe)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class ROAFactor(BaseIndicator):
    """
    总资产收益率因子 (Return on Assets)

    衡量公司利用总资产创造利润的能力,高ROA表示资产利用效率高
    """

    _metadata = IndicatorMetadata(
        name='ROAFactor',
        display_name='总资产收益率因子',
        category='multi_factor',
        description='衡量公司利用总资产创造利润的能力,高ROA表示资产利用效率高',
        formula='ROA = 净利润 / 总资产; Factor_Score = normalize(ROA)',
        required_columns=['net_income', 'total_assets'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['net_income', 'total_assets']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM净利润
        ttm_net_income = data['net_income'].rolling(window=self.ttm_periods).sum()

        # 计算ROA
        roa = self._safe_division(ttm_net_income, data['total_assets'], fill_value=np.nan)

        return self._normalize(roa)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class GrossProfitMarginFactor(BaseIndicator):
    """
    毛利率因子 (Gross Profit Margin)

    衡量公司产品的盈利能力,高毛利率表示定价能力强或成本控制好
    """

    _metadata = IndicatorMetadata(
        name='GrossProfitMarginFactor',
        display_name='毛利率因子',
        category='multi_factor',
        description='衡量公司产品的盈利能力,高毛利率表示定价能力强或成本控制好',
        formula='毛利率 = (收入 - 成本) / 收入; Factor_Score = normalize(毛利率)',
        required_columns=['revenue', 'cost_of_goods_sold'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['revenue', 'cost_of_goods_sold']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM收入和成本
        ttm_revenue = data['revenue'].rolling(window=self.ttm_periods).sum()
        ttm_cogs = data['cost_of_goods_sold'].rolling(window=self.ttm_periods).sum()

        # 计算毛利率
        gross_margin = self._safe_division(ttm_revenue - ttm_cogs, ttm_revenue, fill_value=np.nan)

        return self._normalize(gross_margin)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class NetProfitMarginFactor(BaseIndicator):
    """
    净利率因子 (Net Profit Margin)

    衡量公司整体盈利能力,高净利率表示成本控制和运营效率高
    """

    _metadata = IndicatorMetadata(
        name='NetProfitMarginFactor',
        display_name='净利率因子',
        category='multi_factor',
        description='衡量公司整体盈利能力,高净利率表示成本控制和运营效率高',
        formula='净利率 = 净利润 / 收入; Factor_Score = normalize(净利率)',
        required_columns=['net_income', 'revenue'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['net_income', 'revenue']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM
        ttm_net_income = data['net_income'].rolling(window=self.ttm_periods).sum()
        ttm_revenue = data['revenue'].rolling(window=self.ttm_periods).sum()

        # 计算净利率
        net_margin = self._safe_division(ttm_net_income, ttm_revenue, fill_value=np.nan)

        return self._normalize(net_margin)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class AssetTurnoverFactor(BaseIndicator):
    """
    资产周转率因子 (Asset Turnover)

    衡量公司利用资产创造收入的效率,高周转率表示资产使用效率高
    """

    _metadata = IndicatorMetadata(
        name='AssetTurnoverFactor',
        display_name='资产周转率因子',
        category='multi_factor',
        description='衡量公司利用资产创造收入的效率,高周转率表示资产使用效率高',
        formula='资产周转率 = 收入 / 总资产; Factor_Score = normalize(资产周转率)',
        required_columns=['revenue', 'total_assets'],
        output_type='series',
        default_params={'normalization': 'zscore', 'ttm_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', ttm_periods: int = 4):
        self.normalization = normalization
        self.ttm_periods = ttm_periods
        super().__init__(normalization=normalization, ttm_periods=ttm_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.ttm_periods, 'ttm_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['revenue', 'total_assets']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算TTM收入
        ttm_revenue = data['revenue'].rolling(window=self.ttm_periods).sum()

        # 计算资产周转率
        turnover = self._safe_division(ttm_revenue, data['total_assets'], fill_value=np.nan)

        return self._normalize(turnover)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 成长因子 (3个) ====================

@indicator_registry.register('multi_factor')
class RevenueGrowthFactor(BaseIndicator):
    """
    收入增长率因子 (Revenue Growth)

    衡量公司营收的增长速度,高增长表示市场份额扩大或需求增加
    """

    _metadata = IndicatorMetadata(
        name='RevenueGrowthFactor',
        display_name='收入增长率因子',
        category='multi_factor',
        description='衡量公司营收的增长速度,高增长表示市场份额扩大或需求增加',
        formula='收入增长率 = (当期收入 - 去年同期收入) / 去年同期收入; Factor_Score = normalize(增长率)',
        required_columns=['revenue'],
        output_type='series',
        default_params={'normalization': 'zscore', 'yoy_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', yoy_periods: int = 4):
        """
        Args:
            normalization: 归一化方法
            yoy_periods: 同比周期数(默认4个季度=1年)
        """
        self.normalization = normalization
        self.yoy_periods = yoy_periods
        super().__init__(normalization=normalization, yoy_periods=yoy_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.yoy_periods, 'yoy_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['revenue']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算同比增长率
        revenue_growth = data['revenue'].pct_change(periods=self.yoy_periods)

        return self._normalize(revenue_growth)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class ProfitGrowthFactor(BaseIndicator):
    """
    利润增长率因子 (Profit Growth)

    衡量公司净利润的增长速度,高增长表示盈利能力提升
    """

    _metadata = IndicatorMetadata(
        name='ProfitGrowthFactor',
        display_name='利润增长率因子',
        category='multi_factor',
        description='衡量公司净利润的增长速度,高增长表示盈利能力提升',
        formula='利润增长率 = (当期利润 - 去年同期利润) / 去年同期利润; Factor_Score = normalize(增长率)',
        required_columns=['net_income'],
        output_type='series',
        default_params={'normalization': 'zscore', 'yoy_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', yoy_periods: int = 4):
        self.normalization = normalization
        self.yoy_periods = yoy_periods
        super().__init__(normalization=normalization, yoy_periods=yoy_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.yoy_periods, 'yoy_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['net_income']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算同比增长率
        profit_growth = data['net_income'].pct_change(periods=self.yoy_periods)

        return self._normalize(profit_growth)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class EPSGrowthFactor(BaseIndicator):
    """
    每股收益增长率因子 (EPS Growth)

    衡量每股收益的增长速度,考虑股本变化的盈利增长指标
    """

    _metadata = IndicatorMetadata(
        name='EPSGrowthFactor',
        display_name='每股收益增长率因子',
        category='multi_factor',
        description='衡量每股收益的增长速度,考虑股本变化的盈利增长指标',
        formula='EPS增长率 = (当期EPS - 去年同期EPS) / 去年同期EPS; Factor_Score = normalize(增长率)',
        required_columns=['net_income', 'shares_outstanding'],
        output_type='series',
        default_params={'normalization': 'zscore', 'yoy_periods': 4},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', yoy_periods: int = 4):
        self.normalization = normalization
        self.yoy_periods = yoy_periods
        super().__init__(normalization=normalization, yoy_periods=yoy_periods)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.yoy_periods, 'yoy_periods')

    @property
    def required_columns(self) -> List[str]:
        return ['net_income', 'shares_outstanding']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算EPS
        eps = self._safe_division(data['net_income'], data['shares_outstanding'], fill_value=np.nan)

        # 计算同比增长率
        eps_growth = eps.pct_change(periods=self.yoy_periods)

        return self._normalize(eps_growth)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 动量因子 (3个) ====================

@indicator_registry.register('multi_factor')
class PriceMomentumFactor(BaseIndicator):
    """
    价格动量因子 (Price Momentum)

    衡量股价的趋势强度,高动量表示上涨趋势强劲
    """

    _metadata = IndicatorMetadata(
        name='PriceMomentumFactor',
        display_name='价格动量因子',
        category='multi_factor',
        description='衡量股价的趋势强度,高动量表示上涨趋势强劲',
        formula='动量 = (当前价格 - N日前价格) / N日前价格; Factor_Score = normalize(动量)',
        required_columns=['close'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 20},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', period: int = 20):
        """
        Args:
            normalization: 归一化方法
            period: 动量计算周期
        """
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        momentum = data['close'].pct_change(periods=self.period)

        return self._normalize(momentum)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class ReturnMomentumFactor(BaseIndicator):
    """
    收益动量因子 (Return Momentum)

    基于累计收益率的动量指标,常用于因子投资
    """

    _metadata = IndicatorMetadata(
        name='ReturnMomentumFactor',
        display_name='收益动量因子',
        category='multi_factor',
        description='基于累计收益率的动量指标,常用于因子投资',
        formula='收益动量 = Sum(日收益率, N日) / Std(日收益率, N日); Factor_Score = normalize(动量)',
        required_columns=['close'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 20},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', period: int = 20):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算日收益率
        daily_returns = data['close'].pct_change()

        # 计算累计收益
        cum_returns = daily_returns.rolling(window=self.period).sum()

        # 计算收益标准差
        returns_std = daily_returns.rolling(window=self.period).std()

        # 计算收益动量 (夏普比率风格)
        return_momentum = self._safe_division(cum_returns, returns_std, fill_value=np.nan)

        return self._normalize(return_momentum)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class RelativeStrengthFactor(BaseIndicator):
    """
    相对强度因子 (Relative Strength)

    衡量个股相对于市场的强度,高RS表示跑赢大盘
    """

    _metadata = IndicatorMetadata(
        name='RelativeStrengthFactor',
        display_name='相对强度因子',
        category='multi_factor',
        description='衡量个股相对于市场的强度,高RS表示跑赢大盘',
        formula='RS = 个股收益率 / 市场收益率; Factor_Score = normalize(RS)',
        required_columns=['close', 'market_index'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 20},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', period: int = 20):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'market_index']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算个股收益率
        stock_returns = data['close'].pct_change(periods=self.period)

        # 计算市场收益率
        market_returns = data['market_index'].pct_change(periods=self.period)

        # 计算相对强度
        relative_strength = self._safe_division(stock_returns, market_returns, fill_value=np.nan)

        return self._normalize(relative_strength)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 波动因子 (3个) ====================

@indicator_registry.register('multi_factor')
class BetaFactor(BaseIndicator):
    """
    Beta因子 (Market Beta)

    衡量个股相对于市场的系统性风险,低Beta表示防御性强
    """

    _metadata = IndicatorMetadata(
        name='BetaFactor',
        display_name='Beta因子',
        category='multi_factor',
        description='衡量个股相对于市场的系统性风险,低Beta表示防御性强',
        formula='Beta = Cov(股票收益, 市场收益) / Var(市场收益); Factor_Score = -1 * normalize(Beta)',
        required_columns=['close', 'market_index'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 60},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', period: int = 60):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'market_index']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        stock_returns = data['close'].pct_change()
        market_returns = data['market_index'].pct_change()

        # 计算滚动Beta
        market_variance = market_returns.rolling(window=self.period).var()
        beta = stock_returns.rolling(window=self.period).cov(market_returns).divide(
            market_variance.replace(0, np.nan)
        )

        # 低Beta = 高分数
        return -1 * self._normalize(beta)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class IdiosyncraticVolatilityFactor(BaseIndicator):
    """
    特质波动率因子 (Idiosyncratic Volatility)

    衡量股票非系统性风险,低特质波动表示更稳定
    """

    _metadata = IndicatorMetadata(
        name='IdiosyncraticVolatilityFactor',
        display_name='特质波动率因子',
        category='multi_factor',
        description='衡量股票非系统性风险,低特质波动表示更稳定',
        formula='特质波动率 = Std(残差收益); Factor_Score = -1 * normalize(特质波动率)',
        required_columns=['close', 'market_index'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 60},
        complexity_level=4
    )

    def __init__(self, normalization: str = 'zscore', period: int = 60):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'market_index']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        stock_returns = data['close'].pct_change()
        market_returns = data['market_index'].pct_change()

        # 计算滚动特质波动率
        market_variance = market_returns.rolling(window=self.period).var()
        beta = stock_returns.rolling(window=self.period).cov(market_returns).divide(
            market_variance.replace(0, np.nan)
        )
        alpha = (
            stock_returns.rolling(window=self.period).mean()
            - beta * market_returns.rolling(window=self.period).mean()
        )
        residuals = stock_returns - (alpha + beta * market_returns)
        idio_vol = residuals.rolling(window=self.period).std(ddof=0)

        # 低波动 = 高分数
        return -1 * self._normalize(idio_vol)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class DownsideVolatilityFactor(BaseIndicator):
    """
    下行波动率因子 (Downside Volatility)

    只考虑负收益的波动率,衡量下行风险
    """

    _metadata = IndicatorMetadata(
        name='DownsideVolatilityFactor',
        display_name='下行波动率因子',
        category='multi_factor',
        description='只考虑负收益的波动率,衡量下行风险',
        formula='下行波动率 = Std(Min(收益率, 0)); Factor_Score = -1 * normalize(下行波动率)',
        required_columns=['close'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 60},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', period: int = 60):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        returns = data['close'].pct_change()

        # 只保留负收益
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0

        # 计算下行波动率
        downside_vol = downside_returns.rolling(window=self.period).std()

        # 低下行波动 = 高分数
        return -1 * self._normalize(downside_vol)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 规模因子 (2个) ====================

@indicator_registry.register('multi_factor')
class MarketCapFactor(BaseIndicator):
    """
    市值因子 (Market Capitalization)

    总市值规模,小市值通常有更高的成长潜力
    """

    _metadata = IndicatorMetadata(
        name='MarketCapFactor',
        display_name='市值因子',
        category='multi_factor',
        description='总市值规模,小市值通常有更高的成长潜力',
        formula='市值 = 股价 * 总股本; Factor_Score = -1 * normalize(市值)',
        required_columns=['close', 'shares_outstanding'],
        output_type='series',
        default_params={'normalization': 'zscore'},
        complexity_level=1
    )

    def __init__(self, normalization: str = 'zscore'):
        self.normalization = normalization
        super().__init__(normalization=normalization)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'shares_outstanding']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算市值
        market_cap = data['close'] * data['shares_outstanding']

        # 小市值 = 高分数
        return -1 * self._normalize(market_cap)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class FloatMarketCapFactor(BaseIndicator):
    """
    流通市值因子 (Float Market Capitalization)

    流通股市值规模,更真实反映可交易规模
    """

    _metadata = IndicatorMetadata(
        name='FloatMarketCapFactor',
        display_name='流通市值因子',
        category='multi_factor',
        description='流通股市值规模,更真实反映可交易规模',
        formula='流通市值 = 股价 * 流通股本; Factor_Score = -1 * normalize(流通市值)',
        required_columns=['close', 'float_shares'],
        output_type='series',
        default_params={'normalization': 'zscore'},
        complexity_level=1
    )

    def __init__(self, normalization: str = 'zscore'):
        self.normalization = normalization
        super().__init__(normalization=normalization)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'float_shares']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算流通市值
        float_market_cap = data['close'] * data['float_shares']

        # 小流通市值 = 高分数
        return -1 * self._normalize(float_market_cap)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 流动性因子 (2个) ====================

@indicator_registry.register('multi_factor')
class TurnoverRateFactor(BaseIndicator):
    """
    换手率因子 (Turnover Rate)

    成交量相对于流通股本的比率,衡量流动性
    """

    _metadata = IndicatorMetadata(
        name='TurnoverRateFactor',
        display_name='换手率因子',
        category='multi_factor',
        description='成交量相对于流通股本的比率,衡量流动性',
        formula='换手率 = 成交量 / 流通股本; Factor_Score = normalize(平均换手率)',
        required_columns=['volume', 'float_shares'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 20},
        complexity_level=2
    )

    def __init__(self, normalization: str = 'zscore', period: int = 20):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['volume', 'float_shares']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算日换手率
        daily_turnover = self._safe_division(data['volume'], data['float_shares'], fill_value=np.nan)

        # 计算平均换手率
        avg_turnover = daily_turnover.rolling(window=self.period).mean()

        return self._normalize(avg_turnover)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


@indicator_registry.register('multi_factor')
class AmihudIlliquidityFactor(BaseIndicator):
    """
    Amihud非流动性因子 (Amihud Illiquidity)

    衡量单位成交量引起的价格变化,值越大流动性越差
    """

    _metadata = IndicatorMetadata(
        name='AmihudIlliquidityFactor',
        display_name='Amihud非流动性因子',
        category='multi_factor',
        description='衡量单位成交量引起的价格变化,值越大流动性越差',
        formula='Amihud = |收益率| / (成交额); Factor_Score = -1 * normalize(Amihud)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'normalization': 'zscore', 'period': 20},
        complexity_level=3
    )

    def __init__(self, normalization: str = 'zscore', period: int = 20):
        self.normalization = normalization
        self.period = period
        super().__init__(normalization=normalization, period=period)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 计算收益率
        returns = data['close'].pct_change()

        # 计算成交额 (价格 * 成交量)
        dollar_volume = data['close'] * data['volume']

        # 计算日度Amihud
        daily_amihud = self._safe_division(returns.abs(), dollar_volume, fill_value=np.nan)

        # 计算平均Amihud
        avg_amihud = daily_amihud.rolling(window=self.period).mean()

        # 低非流动性 = 高分数
        return -1 * self._normalize(avg_amihud)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if self.normalization == 'zscore':
            return (series - series.mean()) / series.std()
        elif self.normalization == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif self.normalization == 'rank':
            return series.rank(pct=True)
        return series


# ==================== 组合因子 (2个) ====================

@indicator_registry.register('multi_factor')
class MultiFactorScore(BaseIndicator):
    """
    多因子综合得分 (Multi-Factor Composite Score)

    综合多个因子的加权得分,用于选股排名
    """

    _metadata = IndicatorMetadata(
        name='MultiFactorScore',
        display_name='多因子综合得分',
        category='multi_factor',
        description='综合多个因子的加权得分,用于选股排名',
        formula='综合得分 = Σ(权重_i * 因子得分_i); 支持价值/质量/成长/动量/低波等因子',
        required_columns=['close'],  # 最小需求,实际需要更多列
        output_type='series',
        default_params={
            'normalization': 'zscore',
            'weights': {
                'value': 0.2,
                'quality': 0.2,
                'growth': 0.2,
                'momentum': 0.2,
                'low_volatility': 0.2
            }
        },
        complexity_level=5
    )

    def __init__(self, normalization: str = 'zscore', weights: Optional[Dict[str, float]] = None):
        """
        Args:
            normalization: 归一化方法
            weights: 各因子权重字典,默认等权
        """
        self.normalization = normalization
        self.weights = weights or {
            'value': 0.2,
            'quality': 0.2,
            'growth': 0.2,
            'momentum': 0.2,
            'low_volatility': 0.2
        }
        super().__init__(normalization=normalization, weights=weights)

    def _validate_params(self):
        if self.normalization not in ['zscore', 'minmax', 'rank']:
            raise ValueError("normalization must be 'zscore', 'minmax', or 'rank'")

        # 验证权重和为1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    @property
    def required_columns(self) -> List[str]:
        # 基础列,实际需要的列取决于启用的因子
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        # 初始化综合得分
        composite_score = pd.Series(0.0, index=data.index)

        # 价值因子 (需要 PB 数据)
        if 'value' in self.weights and self.weights['value'] > 0:
            if all(col in data.columns for col in ['shares_outstanding', 'total_assets', 'total_liabilities']):
                pb_factor = PBFactor(normalization=self.normalization)
                value_score = pb_factor.calculate(data)
                composite_score += self.weights['value'] * value_score.fillna(0)

        # 质量因子 (需要 ROE 数据)
        if 'quality' in self.weights and self.weights['quality'] > 0:
            if all(col in data.columns for col in ['net_income', 'total_assets', 'total_liabilities']):
                roe_factor = ROEFactor(normalization=self.normalization)
                quality_score = roe_factor.calculate(data)
                composite_score += self.weights['quality'] * quality_score.fillna(0)

        # 成长因子 (需要收入数据)
        if 'growth' in self.weights and self.weights['growth'] > 0:
            if 'revenue' in data.columns:
                revenue_growth_factor = RevenueGrowthFactor(normalization=self.normalization)
                growth_score = revenue_growth_factor.calculate(data)
                composite_score += self.weights['growth'] * growth_score.fillna(0)

        # 动量因子
        if 'momentum' in self.weights and self.weights['momentum'] > 0:
            momentum_factor = PriceMomentumFactor(normalization=self.normalization, period=20)
            momentum_score = momentum_factor.calculate(data)
            composite_score += self.weights['momentum'] * momentum_score.fillna(0)

        # 低波动因子
        if 'low_volatility' in self.weights and self.weights['low_volatility'] > 0:
            downside_vol_factor = DownsideVolatilityFactor(normalization=self.normalization, period=60)
            low_vol_score = downside_vol_factor.calculate(data)
            composite_score += self.weights['low_volatility'] * low_vol_score.fillna(0)

        return composite_score


@indicator_registry.register('multi_factor')
class FactorRotationSignal(BaseIndicator):
    """
    因子轮动信号 (Factor Rotation Signal)

    基于因子动量,动态选择表现最好的因子
    """

    _metadata = IndicatorMetadata(
        name='FactorRotationSignal',
        display_name='因子轮动信号',
        category='multi_factor',
        description='基于因子动量,动态选择表现最好的因子',
        formula='轮动信号 = argmax(因子收益率); 动态调整因子权重',
        required_columns=['close'],
        output_type='dataframe',
        default_params={
            'lookback_period': 60,
            'rebalance_period': 20,
            'top_n_factors': 3
        },
        complexity_level=5
    )

    def __init__(self, lookback_period: int = 60, rebalance_period: int = 20, top_n_factors: int = 3):
        """
        Args:
            lookback_period: 回看周期,用于评估因子表现
            rebalance_period: 再平衡周期
            top_n_factors: 选择top N个因子
        """
        self.lookback_period = lookback_period
        self.rebalance_period = rebalance_period
        self.top_n_factors = top_n_factors
        super().__init__(
            lookback_period=lookback_period,
            rebalance_period=rebalance_period,
            top_n_factors=top_n_factors
        )

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.lookback_period, 'lookback_period')
        IndicatorValidator.validate_positive_int(self.rebalance_period, 'rebalance_period')
        IndicatorValidator.validate_positive_int(self.top_n_factors, 'top_n_factors')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        # 计算多个因子的得分
        factors = {}

        # 动量因子
        if 'close' in data.columns:
            momentum = PriceMomentumFactor(period=20)
            factors['momentum'] = momentum.calculate(data)

        # 如果有市场指数,计算Beta因子
        if 'market_index' in data.columns:
            beta = BetaFactor(period=60)
            factors['low_beta'] = beta.calculate(data)

        # 下行波动因子
        downside_vol = DownsideVolatilityFactor(period=60)
        factors['low_volatility'] = downside_vol.calculate(data)

        # 转换为DataFrame
        factor_df = pd.DataFrame(factors)

        # 计算因子动量 (因子得分的变化率)
        factor_momentum = factor_df.pct_change(periods=self.lookback_period)

        # 每隔rebalance_period选择top N因子
        selected_factors = pd.DataFrame(0.0, index=data.index, columns=factor_df.columns)

        for i in range(self.lookback_period, len(data), self.rebalance_period):
            # 选择表现最好的top N因子
            current_momentum = factor_momentum.iloc[i]
            top_factors = current_momentum.nlargest(self.top_n_factors).index

            # 等权重分配
            weight = 1.0 / self.top_n_factors

            # 在再平衡周期内保持权重
            end_idx = min(i + self.rebalance_period, len(data))
            for factor in top_factors:
                selected_factors.loc[selected_factors.index[i:end_idx], factor] = weight

        # 计算加权综合信号
        weighted_signal = (factor_df * selected_factors).sum(axis=1)

        # 返回包含各因子权重和综合信号的DataFrame
        result = selected_factors.copy()
        result['composite_signal'] = weighted_signal

        return result
