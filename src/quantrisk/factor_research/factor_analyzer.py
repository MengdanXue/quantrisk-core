"""
因子分析器模块

提供因子IC/IR分析、因子衰减分析、因子表现评估等功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICAnalysis:
    """IC分析结果"""
    factor_name: str
    ic_mean: float              # IC均值
    ic_std: float               # IC标准差
    ir: float                   # 信息比率 (IC_mean / IC_std)
    ic_series: pd.Series        # IC时序
    t_stat: float               # t统计量
    p_value: float              # p值
    positive_ratio: float       # 正IC比例
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecayAnalysis:
    """因子衰减分析结果"""
    factor_name: str
    decay_periods: List[int]    # 衰减周期
    decay_ics: List[float]      # 各周期IC
    half_life: Optional[int]    # 半衰期
    decay_rate: float           # 衰减率


@dataclass
class FactorPerformance:
    """因子表现"""
    factor_name: str
    ic_analysis: ICAnalysis
    turnover: float             # 换手率
    long_return: float          # 多头收益
    short_return: float         # 空头收益
    long_short_return: float    # 多空收益
    sharpe_ratio: float         # 夏普比率
    max_drawdown: float         # 最大回撤
    win_rate: float             # 胜率


class FactorAnalyzer:
    """因子分析器"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = "spearman"
    ) -> ICAnalysis:
        """
        计算因子IC

        Args:
            factor_values: 因子值 (index=date, columns=symbols)
            forward_returns: 前向收益率 (index=date, columns=symbols)
            method: 相关性方法 ('spearman' 或 'pearson')

        Returns:
            ICAnalysis: IC分析结果
        """
        # 对齐日期
        common_dates = factor_values.index.intersection(forward_returns.index)
        common_symbols = factor_values.columns.intersection(forward_returns.columns)

        factor_aligned = factor_values.loc[common_dates, common_symbols]
        returns_aligned = forward_returns.loc[common_dates, common_symbols]

        # 逐日计算IC
        ic_series = []
        for date in common_dates:
            factor_row = factor_aligned.loc[date].dropna()
            return_row = returns_aligned.loc[date].dropna()

            common = factor_row.index.intersection(return_row.index)
            if len(common) < 10:
                continue

            if method == "spearman":
                corr, _ = stats.spearmanr(factor_row[common], return_row[common])
            else:
                corr, _ = stats.pearsonr(factor_row[common], return_row[common])

            if not np.isnan(corr):
                ic_series.append((date, corr))

        if not ic_series:
            raise ValueError("Unable to compute IC: insufficient data")

        ic_df = pd.Series(dict(ic_series))

        # 计算统计量
        ic_mean = ic_df.mean()
        ic_std = ic_df.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0

        # t检验
        t_stat, p_value = stats.ttest_1samp(ic_df.values, 0)
        positive_ratio = (ic_df > 0).mean()

        return ICAnalysis(
            factor_name=getattr(factor_values, 'name', 'unknown'),
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_series=ic_df,
            t_stat=t_stat,
            p_value=p_value,
            positive_ratio=positive_ratio
        )

    def calculate_decay(
        self,
        factor_values: pd.DataFrame,
        price_data: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20, 40, 60]
    ) -> DecayAnalysis:
        """
        计算因子衰减

        Args:
            factor_values: 因子值
            price_data: 价格数据
            periods: 检验周期列表

        Returns:
            DecayAnalysis: 衰减分析结果
        """
        decay_ics = []

        for period in periods:
            # 计算N日前向收益
            forward_returns = price_data.pct_change(period).shift(-period)

            try:
                ic_analysis = self.calculate_ic(factor_values, forward_returns)
                decay_ics.append(ic_analysis.ic_mean)
            except ValueError:
                decay_ics.append(0)

        # 估计半衰期
        half_life = self._estimate_half_life(periods, decay_ics)

        # 估计衰减率
        decay_rate = self._estimate_decay_rate(periods, decay_ics)

        return DecayAnalysis(
            factor_name=getattr(factor_values, 'name', 'unknown'),
            decay_periods=periods,
            decay_ics=decay_ics,
            half_life=half_life,
            decay_rate=decay_rate
        )

    def calculate_turnover(
        self,
        factor_values: pd.DataFrame,
        quantile: float = 0.2
    ) -> float:
        """
        计算因子换手率

        Args:
            factor_values: 因子值
            quantile: 分位数阈值

        Returns:
            float: 平均换手率
        """
        turnovers = []

        for i in range(1, len(factor_values)):
            prev_row = factor_values.iloc[i-1].dropna()
            curr_row = factor_values.iloc[i].dropna()

            common = prev_row.index.intersection(curr_row.index)
            if len(common) < 10:
                continue

            # 找出top/bottom分位的股票
            prev_threshold = prev_row[common].quantile(1 - quantile)
            curr_threshold = curr_row[common].quantile(1 - quantile)

            prev_top = set(prev_row[prev_row >= prev_threshold].index)
            curr_top = set(curr_row[curr_row >= curr_threshold].index)

            if len(prev_top) == 0:
                continue

            # 计算换手率
            turnover = 1 - len(prev_top & curr_top) / len(prev_top)
            turnovers.append(turnover)

        return np.mean(turnovers) if turnovers else 0

    def calculate_performance(
        self,
        factor_values: pd.DataFrame,
        price_data: pd.DataFrame,
        quantile: float = 0.2,
        holding_period: int = 1
    ) -> FactorPerformance:
        """
        计算因子表现

        Args:
            factor_values: 因子值
            price_data: 价格数据
            quantile: 分组分位数
            holding_period: 持仓周期

        Returns:
            FactorPerformance: 因子表现
        """
        # 计算前向收益
        forward_returns = price_data.pct_change(holding_period).shift(-holding_period)

        # IC分析
        ic_analysis = self.calculate_ic(factor_values, forward_returns)

        # 换手率
        turnover = self.calculate_turnover(factor_values, quantile)

        # 分组收益
        long_returns, short_returns = self._calculate_group_returns(
            factor_values, forward_returns, quantile
        )

        # 多空组合收益
        ls_returns = (long_returns - short_returns) / 2

        # 绩效指标
        long_return = long_returns.mean() * 252 / holding_period
        short_return = short_returns.mean() * 252 / holding_period
        long_short_return = ls_returns.mean() * 252 / holding_period

        sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(252 / holding_period) if ls_returns.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + ls_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (ls_returns > 0).mean()

        return FactorPerformance(
            factor_name=getattr(factor_values, 'name', 'unknown'),
            ic_analysis=ic_analysis,
            turnover=turnover,
            long_return=long_return,
            short_return=short_return,
            long_short_return=long_short_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate
        )

    def rank_factors(
        self,
        performances: List[FactorPerformance],
        metric: str = "ir"
    ) -> List[Tuple[str, float]]:
        """
        因子排名

        Args:
            performances: 因子表现列表
            metric: 排名指标 ('ir', 'sharpe', 'long_short_return')

        Returns:
            List[Tuple[str, float]]: 排名结果
        """
        results = []

        for perf in performances:
            if metric == "ir":
                value = perf.ic_analysis.ir
            elif metric == "sharpe":
                value = perf.sharpe_ratio
            elif metric == "long_short_return":
                value = perf.long_short_return
            else:
                value = perf.ic_analysis.ir

            results.append((perf.factor_name, value))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _estimate_half_life(
        self,
        periods: List[int],
        ics: List[float]
    ) -> Optional[int]:
        """估计因子半衰期"""
        if not ics or ics[0] == 0:
            return None

        initial_ic = abs(ics[0])
        half_ic = initial_ic / 2

        for i, (period, ic) in enumerate(zip(periods, ics)):
            if abs(ic) <= half_ic:
                return period

        return None

    def _estimate_decay_rate(
        self,
        periods: List[int],
        ics: List[float]
    ) -> float:
        """估计衰减率 (指数衰减)"""
        if len(periods) < 2 or not ics or ics[0] == 0:
            return 0

        # 使用对数线性回归估计衰减率
        valid_points = [(p, ic) for p, ic in zip(periods, ics) if ic > 0]
        if len(valid_points) < 2:
            return 0

        x = np.array([p for p, _ in valid_points])
        y = np.log(np.array([ic for _, ic in valid_points]))

        # 简单线性回归
        slope = np.polyfit(x, y, 1)[0]

        return -slope  # 衰减率为负斜率

    def _calculate_group_returns(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        quantile: float
    ) -> Tuple[pd.Series, pd.Series]:
        """计算分组收益"""
        long_returns = []
        short_returns = []

        common_dates = factor_values.index.intersection(forward_returns.index)

        for date in common_dates:
            factor_row = factor_values.loc[date].dropna()
            return_row = forward_returns.loc[date].dropna()

            common = factor_row.index.intersection(return_row.index)
            if len(common) < 10:
                continue

            # 分组
            top_threshold = factor_row[common].quantile(1 - quantile)
            bottom_threshold = factor_row[common].quantile(quantile)

            long_stocks = factor_row[factor_row >= top_threshold].index
            short_stocks = factor_row[factor_row <= bottom_threshold].index

            long_stocks = long_stocks.intersection(common)
            short_stocks = short_stocks.intersection(common)

            if len(long_stocks) > 0:
                long_ret = return_row[long_stocks].mean()
                long_returns.append((date, long_ret))

            if len(short_stocks) > 0:
                short_ret = return_row[short_stocks].mean()
                short_returns.append((date, short_ret))

        return pd.Series(dict(long_returns)), pd.Series(dict(short_returns))


# 单例实例
_factor_analyzer_instance: Optional[FactorAnalyzer] = None


def get_factor_analyzer() -> FactorAnalyzer:
    """获取因子分析器单例"""
    global _factor_analyzer_instance
    if _factor_analyzer_instance is None:
        _factor_analyzer_instance = FactorAnalyzer()
    return _factor_analyzer_instance
