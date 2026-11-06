"""
绩效分析模块
Performance Analytics

使用empyrical库进行专业的绩效分析：
- 风险调整收益指标
- 回撤分析
- 交易统计
- 基准对比

作者: TraderX Team
创建: 2025-12-23
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    logging.warning("empyrical not installed, using fallback implementations")

from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return_avg: float = 0.0
    daily_return_avg: float = 0.0

    # 风险指标
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # 尾部风险
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # 稳定性
    stability: float = 0.0
    tail_ratio: float = 0.0

    # Beta和Alpha
    beta: float = 0.0
    alpha: float = 0.0

    # 期间统计
    trading_days: int = 0
    positive_days: int = 0
    negative_days: int = 0
    win_rate: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0


@dataclass
class TradeStatistics:
    """交易统计"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # 盈亏
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # 平均
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0

    # 最大
    largest_profit: float = 0.0
    largest_loss: float = 0.0

    # 连续
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 持仓时间
    avg_holding_period: float = 0.0
    avg_winning_holding: float = 0.0
    avg_losing_holding: float = 0.0

    # MAE/MFE
    avg_mae: float = 0.0
    avg_mfe: float = 0.0


@dataclass
class DrawdownAnalysis:
    """回撤分析"""
    max_drawdown: float = 0.0
    max_drawdown_start: Optional[datetime] = None
    max_drawdown_end: Optional[datetime] = None
    max_drawdown_duration: int = 0
    recovery_time: Optional[int] = None

    avg_drawdown: float = 0.0
    avg_drawdown_duration: float = 0.0

    # Top N回撤
    top_drawdowns: List[Dict[str, Any]] = field(default_factory=list)

    # 回撤分布
    drawdown_percentiles: Dict[str, float] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    绩效分析器

    提供全面的策略绩效分析功能
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        benchmark_returns: Optional[pd.Series] = None
    ):
        """
        初始化绩效分析器

        Args:
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年交易日数
            benchmark_returns: 基准收益率序列
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.benchmark_returns = benchmark_returns

    def analyze(
        self,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
        trades: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        执行全面的绩效分析

        Args:
            returns: 日收益率序列
            positions: 持仓历史
            trades: 交易记录列表

        Returns:
            分析结果字典
        """
        metrics = self.calculate_metrics(returns)
        drawdown = self.analyze_drawdowns(returns)

        trade_stats = None
        if trades:
            trade_stats = self.calculate_trade_statistics(trades)

        return {
            'metrics': metrics,
            'drawdown': drawdown,
            'trade_statistics': trade_stats,
            'returns_distribution': self.analyze_returns_distribution(returns),
            'rolling_metrics': self.calculate_rolling_metrics(returns),
            'monthly_returns': self.get_monthly_returns(returns),
            'yearly_returns': self.get_yearly_returns(returns),
        }

    def calculate_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """
        计算绩效指标

        Args:
            returns: 日收益率序列

        Returns:
            PerformanceMetrics对象
        """
        metrics = PerformanceMetrics()

        if len(returns) < 2:
            return metrics

        # 清理数据
        returns = returns.dropna()

        # 基础统计
        metrics.trading_days = len(returns)
        metrics.positive_days = (returns > 0).sum()
        metrics.negative_days = (returns < 0).sum()
        metrics.win_rate = metrics.positive_days / metrics.trading_days

        # 收益指标
        if EMPYRICAL_AVAILABLE:
            metrics.total_return = ep.cum_returns_final(returns)
            metrics.annual_return = ep.annual_return(returns, period='daily')
            metrics.volatility = ep.annual_volatility(returns, period='daily')
            metrics.sharpe_ratio = ep.sharpe_ratio(
                returns,
                risk_free=self.risk_free_rate,
                period='daily'
            )
            metrics.sortino_ratio = ep.sortino_ratio(
                returns,
                required_return=0,
                period='daily'
            )
            metrics.max_drawdown = ep.max_drawdown(returns)
            metrics.calmar_ratio = ep.calmar_ratio(returns, period='daily')
            metrics.omega_ratio = ep.omega_ratio(returns, risk_free=0)
            metrics.stability = ep.stability_of_timeseries(returns)
            metrics.tail_ratio = ep.tail_ratio(returns)
            metrics.downside_volatility = ep.downside_risk(
                returns,
                required_return=0,
                period='daily'
            )
        else:
            # Fallback implementations
            metrics.total_return = (1 + returns).prod() - 1
            metrics.annual_return = (1 + metrics.total_return) ** (252 / len(returns)) - 1
            metrics.volatility = returns.std() * np.sqrt(252)

            excess_returns = returns - self.risk_free_rate / 252
            metrics.sharpe_ratio = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0 else 0
            )

            downside = returns[returns < 0]
            metrics.downside_volatility = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
            metrics.sortino_ratio = (
                (metrics.annual_return - self.risk_free_rate) / metrics.downside_volatility
                if metrics.downside_volatility > 0 else 0
            )

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            metrics.max_drawdown = drawdown.min()

            metrics.calmar_ratio = (
                metrics.annual_return / abs(metrics.max_drawdown)
                if metrics.max_drawdown != 0 else 0
            )

        # VaR/CVaR use the return convention: losses are negative values.
        metrics.var_95 = np.percentile(returns, 5)
        metrics.var_99 = np.percentile(returns, 1)
        metrics.cvar_95 = returns[returns <= metrics.var_95].mean()
        metrics.cvar_99 = returns[returns <= metrics.var_99].mean()

        # 高阶矩
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)

        # 日收益统计
        metrics.daily_return_avg = returns.mean()
        metrics.best_day = returns.max()
        metrics.worst_day = returns.min()

        # 月收益统计
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly) > 0:
            metrics.monthly_return_avg = monthly.mean()
            metrics.best_month = monthly.max()
            metrics.worst_month = monthly.min()

        # Beta和Alpha (如果有基准)
        if self.benchmark_returns is not None:
            aligned = pd.concat([returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                if EMPYRICAL_AVAILABLE:
                    metrics.beta = ep.beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    metrics.alpha = ep.alpha(
                        aligned.iloc[:, 0],
                        aligned.iloc[:, 1],
                        risk_free=self.risk_free_rate
                    )
                    # empyrical 可能没有 information_ratio 函数，使用 try-except
                    try:
                        metrics.information_ratio = ep.information_ratio(
                            aligned.iloc[:, 0],
                            aligned.iloc[:, 1]
                        )
                    except AttributeError:
                        # 手动计算 information_ratio
                        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                        tracking_error = excess.std()
                        metrics.information_ratio = (
                            excess.mean() / tracking_error * np.sqrt(252)
                            if tracking_error > 0 else 0
                        )
                else:
                    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    metrics.beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
                    metrics.alpha = metrics.annual_return - (
                        self.risk_free_rate +
                        metrics.beta * (aligned.iloc[:, 1].mean() * 252 - self.risk_free_rate)
                    )
                    # 手动计算 information_ratio
                    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                    tracking_error = excess.std()
                    metrics.information_ratio = (
                        excess.mean() / tracking_error * np.sqrt(252)
                        if tracking_error > 0 else 0
                    )

        return metrics

    def analyze_drawdowns(self, returns: pd.Series, top_n: int = 5) -> DrawdownAnalysis:
        """
        分析回撤

        Args:
            returns: 收益率序列
            top_n: 返回前N大回撤

        Returns:
            DrawdownAnalysis对象
        """
        analysis = DrawdownAnalysis()

        if len(returns) < 2:
            return analysis

        # 计算回撤序列
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # 最大回撤
        analysis.max_drawdown = drawdown.min()

        # 找到最大回撤的起止点
        max_dd_end_idx = drawdown.idxmin()
        max_dd_start_idx = cumulative[:max_dd_end_idx].idxmax()

        analysis.max_drawdown_start = max_dd_start_idx
        analysis.max_drawdown_end = max_dd_end_idx

        # 计算回撤持续时间
        if max_dd_start_idx and max_dd_end_idx:
            analysis.max_drawdown_duration = (max_dd_end_idx - max_dd_start_idx).days

        # 找到恢复时间
        after_trough = cumulative[max_dd_end_idx:]
        recovery_mask = after_trough >= running_max[max_dd_start_idx]
        if recovery_mask.any():
            recovery_date = after_trough[recovery_mask].index[0]
            analysis.recovery_time = (recovery_date - max_dd_end_idx).days

        # 识别所有回撤周期
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start_date = None
        for date, is_dd in in_drawdown.items():
            if is_dd and start_date is None:
                start_date = date
            elif not is_dd and start_date is not None:
                period_dd = drawdown[start_date:date].min()
                duration = (date - start_date).days
                drawdown_periods.append({
                    'start': start_date,
                    'end': date,
                    'drawdown': period_dd,
                    'duration': duration
                })
                start_date = None

        # 排序取Top N
        drawdown_periods.sort(key=lambda x: x['drawdown'])
        analysis.top_drawdowns = drawdown_periods[:top_n]

        # 平均回撤
        if drawdown_periods:
            analysis.avg_drawdown = np.mean([p['drawdown'] for p in drawdown_periods])
            analysis.avg_drawdown_duration = np.mean([p['duration'] for p in drawdown_periods])

        # 回撤分位数
        analysis.drawdown_percentiles = {
            '25%': np.percentile(drawdown[drawdown < 0], 25) if (drawdown < 0).any() else 0,
            '50%': np.percentile(drawdown[drawdown < 0], 50) if (drawdown < 0).any() else 0,
            '75%': np.percentile(drawdown[drawdown < 0], 75) if (drawdown < 0).any() else 0,
        }

        return analysis

    def calculate_trade_statistics(self, trades: List[Any]) -> TradeStatistics:
        """
        计算交易统计

        Args:
            trades: 交易记录列表

        Returns:
            TradeStatistics对象
        """
        stats = TradeStatistics()

        if not trades:
            return stats

        stats.total_trades = len(trades)

        # 提取盈亏
        pnls = [t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0) for t in trades]
        holding_periods = [
            t.holding_period if hasattr(t, 'holding_period')
            else t.get('holding_period', 1)
            for t in trades
        ]

        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        stats.winning_trades = len(winning)
        stats.losing_trades = len(losing)
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0

        # 盈亏金额
        stats.gross_profit = sum(winning)
        stats.gross_loss = abs(sum(losing))
        stats.net_profit = stats.gross_profit - stats.gross_loss
        stats.profit_factor = (
            stats.gross_profit / stats.gross_loss
            if stats.gross_loss > 0 else float('inf')
        )

        # 平均
        stats.avg_profit = np.mean(winning) if winning else 0
        stats.avg_loss = np.mean([abs(l) for l in losing]) if losing else 0
        stats.avg_trade = np.mean(pnls)
        stats.expectancy = stats.win_rate * stats.avg_profit - (1 - stats.win_rate) * stats.avg_loss

        # 最大
        stats.largest_profit = max(winning) if winning else 0
        stats.largest_loss = abs(min(losing)) if losing else 0

        # 连续统计
        stats.max_consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
        stats.max_consecutive_losses = self._max_consecutive(pnls, lambda x: x < 0)

        # 持仓时间
        stats.avg_holding_period = np.mean(holding_periods)

        winning_indices = [i for i, p in enumerate(pnls) if p > 0]
        losing_indices = [i for i, p in enumerate(pnls) if p < 0]

        stats.avg_winning_holding = np.mean([holding_periods[i] for i in winning_indices]) if winning_indices else 0
        stats.avg_losing_holding = np.mean([holding_periods[i] for i in losing_indices]) if losing_indices else 0

        # MAE/MFE
        maes = [t.mae if hasattr(t, 'mae') else t.get('mae', 0) for t in trades]
        mfes = [t.mfe if hasattr(t, 'mfe') else t.get('mfe', 0) for t in trades]

        stats.avg_mae = np.mean(maes) if maes else 0
        stats.avg_mfe = np.mean(mfes) if mfes else 0

        return stats

    def _max_consecutive(self, values: List[float], condition) -> int:
        """计算满足条件的最大连续次数"""
        max_count = 0
        current_count = 0

        for v in values:
            if condition(v):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def analyze_returns_distribution(self, returns: pd.Series) -> Dict[str, Any]:
        """分析收益率分布"""
        if len(returns) < 2:
            return {}

        returns = returns.dropna()

        # 正态性检验
        _, normality_p = stats.normaltest(returns) if len(returns) >= 20 else (0, 0)

        # 分位数
        percentiles = {
            '1%': np.percentile(returns, 1),
            '5%': np.percentile(returns, 5),
            '25%': np.percentile(returns, 25),
            '50%': np.percentile(returns, 50),
            '75%': np.percentile(returns, 75),
            '95%': np.percentile(returns, 95),
            '99%': np.percentile(returns, 99),
        }

        # 收益分布
        bins = np.linspace(returns.min(), returns.max(), 21)
        hist, edges = np.histogram(returns, bins=bins)

        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'normality_p_value': normality_p,
            'is_normal': normality_p > 0.05,
            'percentiles': percentiles,
            'histogram': {
                'counts': hist.tolist(),
                'edges': edges.tolist()
            }
        }

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """计算滚动指标"""
        if len(returns) < window:
            return pd.DataFrame()

        rolling_returns = returns.rolling(window)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        # 滚动夏普
        excess = returns - self.risk_free_rate / 252
        rolling_sharpe = (
            excess.rolling(window).mean() /
            excess.rolling(window).std() *
            np.sqrt(252)
        )

        # 滚动最大回撤
        def rolling_max_dd(x):
            cum = (1 + x).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max
            return dd.min()

        rolling_dd = returns.rolling(window).apply(rolling_max_dd, raw=False)

        return pd.DataFrame({
            'rolling_return': rolling_returns.apply(lambda x: (1 + x).prod() - 1),
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_max_drawdown': rolling_dd
        })

    def get_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """获取月度收益表"""
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # 转换为年-月表格
        monthly_table = monthly.groupby([monthly.index.year, monthly.index.month]).first()
        monthly_table = monthly_table.unstack()

        if hasattr(monthly_table, 'columns'):
            monthly_table.columns = [
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ][:len(monthly_table.columns)]

        return monthly_table

    def get_yearly_returns(self, returns: pd.Series) -> pd.Series:
        """获取年度收益"""
        return returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    def compare_to_benchmark(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """与基准对比"""
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ['Strategy', 'Benchmark']

        strategy_metrics = self.calculate_metrics(aligned['Strategy'])
        benchmark_metrics = self.calculate_metrics(aligned['Benchmark'])

        # 超额收益
        excess_returns = aligned['Strategy'] - aligned['Benchmark']

        return {
            'strategy': strategy_metrics,
            'benchmark': benchmark_metrics,
            'excess_return': strategy_metrics.total_return - benchmark_metrics.total_return,
            'excess_annual_return': strategy_metrics.annual_return - benchmark_metrics.annual_return,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'information_ratio': (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0 else 0
            ),
            'beta': strategy_metrics.beta,
            'alpha': strategy_metrics.alpha,
            'correlation': aligned['Strategy'].corr(aligned['Benchmark']),
            'up_capture': self._capture_ratio(aligned, up=True),
            'down_capture': self._capture_ratio(aligned, up=False),
        }

    def _capture_ratio(self, aligned: pd.DataFrame, up: bool = True) -> float:
        """计算上行/下行捕获率"""
        if up:
            mask = aligned['Benchmark'] > 0
        else:
            mask = aligned['Benchmark'] < 0

        if not mask.any():
            return 0

        strategy_filtered = aligned.loc[mask, 'Strategy']
        benchmark_filtered = aligned.loc[mask, 'Benchmark']

        strategy_return = (1 + strategy_filtered).prod() - 1
        benchmark_return = (1 + benchmark_filtered).prod() - 1

        return strategy_return / benchmark_return if benchmark_return != 0 else 0


# 便捷函数
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """计算夏普比率"""
    if EMPYRICAL_AVAILABLE:
        return ep.sharpe_ratio(returns, risk_free=risk_free_rate)
    else:
        excess = returns - risk_free_rate / 252
        return excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    if EMPYRICAL_AVAILABLE:
        return ep.max_drawdown(returns)
    else:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0
) -> float:
    """计算索提诺比率"""
    if EMPYRICAL_AVAILABLE:
        return ep.sortino_ratio(returns, required_return=target_return)
    else:
        downside = returns[returns < target_return]
        downside_std = downside.std() * np.sqrt(252)
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        return (annual_return - target_return) / downside_std if downside_std > 0 else 0


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates)

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(returns)

    print("=== Performance Metrics ===")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annual Return: {metrics.annual_return:.2%}")
    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"VaR 95%: {metrics.var_95:.2%}")
    print(f"CVaR 95%: {metrics.cvar_95:.2%}")
