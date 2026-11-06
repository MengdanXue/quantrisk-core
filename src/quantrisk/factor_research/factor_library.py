"""
因子库模块

提供常用量化因子的计算，包括动量、价值、质量、成长等类别。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class FactorCategory(Enum):
    """因子类别"""
    MOMENTUM = "momentum"      # 动量因子
    VALUE = "value"            # 价值因子
    QUALITY = "quality"        # 质量因子
    GROWTH = "growth"          # 成长因子
    VOLATILITY = "volatility"  # 波动率因子
    LIQUIDITY = "liquidity"    # 流动性因子
    SIZE = "size"              # 规模因子
    TECHNICAL = "technical"    # 技术因子


@dataclass
class Factor:
    """因子定义"""
    name: str
    category: FactorCategory
    description: str
    formula: str
    higher_is_better: bool = True
    lookback_days: int = 20
    compute_func: Optional[Callable] = None


@dataclass
class FactorData:
    """因子数据"""
    factor_name: str
    symbols: List[str]
    values: pd.DataFrame  # index=date, columns=symbols
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FactorLibrary:
    """因子库"""

    def __init__(self):
        self._factors: Dict[str, Factor] = {}
        self._register_default_factors()

    def _register_default_factors(self) -> None:
        """注册默认因子"""
        # 动量因子
        self.register_factor(Factor(
            name="momentum_20d",
            category=FactorCategory.MOMENTUM,
            description="20日动量",
            formula="close[t] / close[t-20] - 1",
            higher_is_better=True,
            lookback_days=20,
            compute_func=self._compute_momentum
        ))

        self.register_factor(Factor(
            name="momentum_60d",
            category=FactorCategory.MOMENTUM,
            description="60日动量",
            formula="close[t] / close[t-60] - 1",
            higher_is_better=True,
            lookback_days=60,
            compute_func=lambda df: self._compute_momentum(df, 60)
        ))

        self.register_factor(Factor(
            name="reversal_5d",
            category=FactorCategory.MOMENTUM,
            description="5日反转",
            formula="-(close[t] / close[t-5] - 1)",
            higher_is_better=True,
            lookback_days=5,
            compute_func=lambda df: -self._compute_momentum(df, 5)
        ))

        # 价值因子
        self.register_factor(Factor(
            name="pb_ratio",
            category=FactorCategory.VALUE,
            description="市净率倒数",
            formula="1 / (market_cap / book_value)",
            higher_is_better=True,
            lookback_days=1,
            compute_func=self._compute_pb_inverse
        ))

        self.register_factor(Factor(
            name="pe_ratio",
            category=FactorCategory.VALUE,
            description="市盈率倒数 (EP)",
            formula="earnings / market_cap",
            higher_is_better=True,
            lookback_days=1,
            compute_func=self._compute_ep
        ))

        self.register_factor(Factor(
            name="dividend_yield",
            category=FactorCategory.VALUE,
            description="股息率",
            formula="dividends / price",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_dividend_yield
        ))

        # 质量因子
        self.register_factor(Factor(
            name="roe",
            category=FactorCategory.QUALITY,
            description="净资产收益率",
            formula="net_income / equity",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_roe
        ))

        self.register_factor(Factor(
            name="roa",
            category=FactorCategory.QUALITY,
            description="资产收益率",
            formula="net_income / total_assets",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_roa
        ))

        self.register_factor(Factor(
            name="gross_margin",
            category=FactorCategory.QUALITY,
            description="毛利率",
            formula="(revenue - cogs) / revenue",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_gross_margin
        ))

        # 成长因子
        self.register_factor(Factor(
            name="revenue_growth",
            category=FactorCategory.GROWTH,
            description="营收增长率 (YoY)",
            formula="(revenue[t] - revenue[t-4]) / revenue[t-4]",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_revenue_growth
        ))

        self.register_factor(Factor(
            name="earnings_growth",
            category=FactorCategory.GROWTH,
            description="盈利增长率 (YoY)",
            formula="(earnings[t] - earnings[t-4]) / earnings[t-4]",
            higher_is_better=True,
            lookback_days=252,
            compute_func=self._compute_earnings_growth
        ))

        # 波动率因子
        self.register_factor(Factor(
            name="volatility_20d",
            category=FactorCategory.VOLATILITY,
            description="20日波动率",
            formula="std(returns, 20) * sqrt(252)",
            higher_is_better=False,
            lookback_days=20,
            compute_func=self._compute_volatility
        ))

        self.register_factor(Factor(
            name="idiosyncratic_vol",
            category=FactorCategory.VOLATILITY,
            description="特质波动率",
            formula="std(residual_returns, 60)",
            higher_is_better=False,
            lookback_days=60,
            compute_func=self._compute_idiosyncratic_vol
        ))

        # 流动性因子
        self.register_factor(Factor(
            name="turnover_20d",
            category=FactorCategory.LIQUIDITY,
            description="20日换手率均值",
            formula="mean(volume / shares_outstanding, 20)",
            higher_is_better=True,
            lookback_days=20,
            compute_func=self._compute_turnover
        ))

        self.register_factor(Factor(
            name="amihud_illiquidity",
            category=FactorCategory.LIQUIDITY,
            description="Amihud非流动性",
            formula="mean(|return| / dollar_volume, 20)",
            higher_is_better=False,
            lookback_days=20,
            compute_func=self._compute_amihud
        ))

        # 规模因子
        self.register_factor(Factor(
            name="log_market_cap",
            category=FactorCategory.SIZE,
            description="对数市值",
            formula="log(market_cap)",
            higher_is_better=False,
            lookback_days=1,
            compute_func=self._compute_log_market_cap
        ))

        # 技术因子
        self.register_factor(Factor(
            name="rsi_14d",
            category=FactorCategory.TECHNICAL,
            description="14日RSI",
            formula="100 - 100/(1 + avg_gain/avg_loss)",
            higher_is_better=True,
            lookback_days=14,
            compute_func=self._compute_rsi
        ))

        self.register_factor(Factor(
            name="ma_deviation_20d",
            category=FactorCategory.TECHNICAL,
            description="20日均线偏离",
            formula="(close - ma20) / ma20",
            higher_is_better=True,
            lookback_days=20,
            compute_func=self._compute_ma_deviation
        ))

    def register_factor(self, factor: Factor) -> None:
        """注册因子"""
        self._factors[factor.name] = factor

    def get_factor(self, name: str) -> Optional[Factor]:
        """获取因子定义"""
        return self._factors.get(name)

    def list_factors(
        self,
        category: Optional[FactorCategory] = None
    ) -> List[Factor]:
        """列出因子"""
        factors = list(self._factors.values())
        if category:
            factors = [f for f in factors if f.category == category]
        return factors

    def compute_factor(
        self,
        factor_name: str,
        data: pd.DataFrame,
        **kwargs
    ) -> FactorData:
        """
        计算因子值

        Args:
            factor_name: 因子名称
            data: 价格/财务数据 DataFrame
            **kwargs: 额外参数

        Returns:
            FactorData: 因子数据
        """
        factor = self.get_factor(factor_name)
        if not factor:
            raise ValueError(f"Unknown factor: {factor_name}")

        if not factor.compute_func:
            raise ValueError(f"Factor {factor_name} has no compute function")

        values = factor.compute_func(data, **kwargs)

        return FactorData(
            factor_name=factor_name,
            symbols=list(values.columns) if isinstance(values, pd.DataFrame) else [],
            values=values,
            metadata={
                "category": factor.category.value,
                "lookback_days": factor.lookback_days,
                "higher_is_better": factor.higher_is_better
            }
        )

    def compute_all_factors(
        self,
        data: pd.DataFrame,
        categories: Optional[List[FactorCategory]] = None
    ) -> Dict[str, FactorData]:
        """计算所有因子"""
        results = {}
        factors = self.list_factors()

        if categories:
            factors = [f for f in factors if f.category in categories]

        for factor in factors:
            try:
                results[factor.name] = self.compute_factor(factor.name, data)
            except Exception as e:
                print(f"Warning: Failed to compute {factor.name}: {e}")

        return results

    # 因子计算函数
    @staticmethod
    def _compute_momentum(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """计算动量因子"""
        if 'close' in data.columns:
            return data['close'].pct_change(lookback)
        return data.pct_change(lookback)

    @staticmethod
    def _compute_pb_inverse(data: pd.DataFrame) -> pd.DataFrame:
        """计算市净率倒数"""
        if 'pb' in data.columns:
            return 1 / data['pb'].replace(0, np.nan)
        if 'market_cap' in data.columns and 'book_value' in data.columns:
            pb = data['market_cap'] / data['book_value'].replace(0, np.nan)
            return 1 / pb
        raise ValueError("Missing pb or market_cap/book_value columns")

    @staticmethod
    def _compute_ep(data: pd.DataFrame) -> pd.DataFrame:
        """计算盈利收益率 (EP)"""
        if 'pe' in data.columns:
            return 1 / data['pe'].replace(0, np.nan)
        if 'earnings' in data.columns and 'market_cap' in data.columns:
            return data['earnings'] / data['market_cap'].replace(0, np.nan)
        raise ValueError("Missing pe or earnings/market_cap columns")

    @staticmethod
    def _compute_dividend_yield(data: pd.DataFrame) -> pd.DataFrame:
        """计算股息率"""
        if 'dividend_yield' in data.columns:
            return data['dividend_yield']
        if 'dividends' in data.columns and 'close' in data.columns:
            return data['dividends'] / data['close'].replace(0, np.nan)
        raise ValueError("Missing dividend_yield or dividends/close columns")

    @staticmethod
    def _compute_roe(data: pd.DataFrame) -> pd.DataFrame:
        """计算ROE"""
        if 'roe' in data.columns:
            return data['roe']
        if 'net_income' in data.columns and 'equity' in data.columns:
            return data['net_income'] / data['equity'].replace(0, np.nan)
        raise ValueError("Missing roe or net_income/equity columns")

    @staticmethod
    def _compute_roa(data: pd.DataFrame) -> pd.DataFrame:
        """计算ROA"""
        if 'roa' in data.columns:
            return data['roa']
        if 'net_income' in data.columns and 'total_assets' in data.columns:
            return data['net_income'] / data['total_assets'].replace(0, np.nan)
        raise ValueError("Missing roa or net_income/total_assets columns")

    @staticmethod
    def _compute_gross_margin(data: pd.DataFrame) -> pd.DataFrame:
        """计算毛利率"""
        if 'gross_margin' in data.columns:
            return data['gross_margin']
        if 'revenue' in data.columns and 'cogs' in data.columns:
            return (data['revenue'] - data['cogs']) / data['revenue'].replace(0, np.nan)
        raise ValueError("Missing gross_margin or revenue/cogs columns")

    @staticmethod
    def _compute_revenue_growth(data: pd.DataFrame) -> pd.DataFrame:
        """计算营收增长率"""
        if 'revenue' in data.columns:
            return data['revenue'].pct_change(4)  # YoY quarterly
        raise ValueError("Missing revenue column")

    @staticmethod
    def _compute_earnings_growth(data: pd.DataFrame) -> pd.DataFrame:
        """计算盈利增长率"""
        if 'earnings' in data.columns:
            prev_earnings = data['earnings'].shift(4).replace(0, np.nan)
            return (data['earnings'] - prev_earnings) / prev_earnings.abs()
        raise ValueError("Missing earnings column")

    @staticmethod
    def _compute_volatility(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """计算波动率"""
        if 'close' in data.columns:
            returns = data['close'].pct_change()
        else:
            returns = data.pct_change()
        return returns.rolling(lookback).std() * np.sqrt(252)

    @staticmethod
    def _compute_idiosyncratic_vol(
        data: pd.DataFrame,
        market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """计算特质波动率"""
        if 'close' in data.columns:
            returns = data['close'].pct_change()
        else:
            returns = data.pct_change()

        if market_returns is None:
            # 使用平均收益作为市场收益的近似
            market_returns = returns.mean(axis=1) if returns.ndim > 1 else returns

        # 简化处理: 直接使用残差波动率
        residuals = returns - market_returns.values.reshape(-1, 1) if returns.ndim > 1 else returns - market_returns
        return residuals.rolling(60).std() * np.sqrt(252)

    @staticmethod
    def _compute_turnover(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """计算换手率"""
        if 'turnover' in data.columns:
            return data['turnover'].rolling(lookback).mean()
        if 'volume' in data.columns and 'shares_outstanding' in data.columns:
            turnover = data['volume'] / data['shares_outstanding'].replace(0, np.nan)
            return turnover.rolling(lookback).mean()
        raise ValueError("Missing turnover or volume/shares_outstanding columns")

    @staticmethod
    def _compute_amihud(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """计算Amihud非流动性"""
        if 'close' in data.columns and 'volume' in data.columns:
            returns = data['close'].pct_change().abs()
            dollar_volume = data['close'] * data['volume']
            illiquidity = returns / dollar_volume.replace(0, np.nan)
            return illiquidity.rolling(lookback).mean() * 1e6  # 放大倍数
        raise ValueError("Missing close/volume columns")

    @staticmethod
    def _compute_log_market_cap(data: pd.DataFrame) -> pd.DataFrame:
        """计算对数市值"""
        if 'market_cap' in data.columns:
            return np.log(data['market_cap'].replace(0, np.nan))
        raise ValueError("Missing market_cap column")

    @staticmethod
    def _compute_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI"""
        if 'close' in data.columns:
            delta = data['close'].diff()
        else:
            delta = data.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_ma_deviation(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """计算均线偏离"""
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data

        ma = close.rolling(period).mean()
        return (close - ma) / ma


# 全局因子库实例
_factor_library_instance: Optional[FactorLibrary] = None


def get_factor_library() -> FactorLibrary:
    """获取因子库单例"""
    global _factor_library_instance
    if _factor_library_instance is None:
        _factor_library_instance = FactorLibrary()
    return _factor_library_instance
