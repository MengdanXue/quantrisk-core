"""
多资产组合回测

支持：
- 多资产同时回测
- 资产权重管理
- 组合再平衡
- 相关性分析
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np

from .engine import BacktestEngine, BacktestConfig, OrderSide, PositionSide


@dataclass
class PortfolioConfig:
    """组合回测配置"""
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly, quarterly
    rebalance_threshold: float = 0.05      # 偏离阈值触发再平衡
    max_single_weight: float = 0.3         # 单资产最大权重
    min_single_weight: float = 0.0         # 单资产最小权重
    risk_free_rate: float = 0.02


@dataclass
class AssetAllocation:
    """资产配置"""
    symbol: str
    target_weight: float
    current_weight: float = 0.0
    quantity: float = 0.0
    value: float = 0.0


@dataclass
class PortfolioResult:
    """组合回测结果"""
    config: PortfolioConfig
    metrics: Dict[str, float]
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    rebalance_dates: List[datetime]
    turnover: float
    asset_contributions: Dict[str, float]


class PortfolioBacktest:
    """
    多资产组合回测引擎

    支持特性：
    - 多资产同时交易
    - 动态权重调整
    - 多种再平衡策略
    - 组合风险分析
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.reset()

    def reset(self):
        """重置状态"""
        self.cash = self.config.initial_capital
        self.allocations: Dict[str, AssetAllocation] = {}
        self.equity_history: List[float] = []
        self.weights_history: List[Dict[str, float]] = []
        self.date_history: List[datetime] = []
        self.rebalance_dates: List[datetime] = []
        self.last_rebalance_date: Optional[datetime] = None
        self.turnover_total = 0.0

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_weights: Dict[str, float],
        signal_func: Optional[Callable] = None
    ) -> PortfolioResult:
        """
        运行组合回测

        Args:
            data_dict: {symbol: DataFrame} 多资产数据
            target_weights: {symbol: weight} 目标权重
            signal_func: 可选的动态权重调整函数

        Returns:
            PortfolioResult: 组合回测结果
        """
        self.reset()

        # 验证权重
        total_weight = sum(target_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # 归一化权重
            target_weights = {k: v / total_weight for k, v in target_weights.items()}

        # 初始化配置
        for symbol, weight in target_weights.items():
            self.allocations[symbol] = AssetAllocation(
                symbol=symbol,
                target_weight=weight
            )

        # 合并日期索引
        all_dates = sorted(set().union(*[set(df.index) for df in data_dict.values()]))

        # 初始建仓
        first_date = all_dates[0]
        prices = {sym: data_dict[sym].loc[first_date, 'close']
                  for sym in target_weights.keys()
                  if first_date in data_dict[sym].index}
        self._rebalance(prices, target_weights)
        self.last_rebalance_date = first_date

        # 逐日回测
        for date in all_dates:
            # 获取当日价格
            prices = {}
            for symbol in target_weights.keys():
                if date in data_dict[symbol].index:
                    prices[symbol] = data_dict[symbol].loc[date, 'close']

            if not prices:
                continue

            # 更新持仓市值
            self._update_values(prices)

            # 动态权重调整
            if signal_func:
                data_slices = {sym: data_dict[sym].loc[:date]
                               for sym in target_weights.keys()
                               if date in data_dict[sym].index}
                new_weights = signal_func(data_slices, self)
                if new_weights:
                    target_weights = new_weights
                    for sym, w in new_weights.items():
                        if sym in self.allocations:
                            self.allocations[sym].target_weight = w

            # 检查是否需要再平衡
            if self._should_rebalance(date):
                self._rebalance(prices, target_weights)
                self.rebalance_dates.append(date)
                self.last_rebalance_date = date

            # 记录历史
            equity = self._calculate_equity()
            self.equity_history.append(equity)
            self.date_history.append(date)

            current_weights = {sym: alloc.current_weight
                               for sym, alloc in self.allocations.items()}
            self.weights_history.append(current_weights)

        return self._generate_result(data_dict, target_weights)

    def _should_rebalance(self, current_date: datetime) -> bool:
        """检查是否需要再平衡"""
        if self.last_rebalance_date is None:
            return True

        # 时间触发
        freq = self.config.rebalance_frequency
        if freq == 'daily':
            return True
        elif freq == 'weekly':
            days_since = (current_date - self.last_rebalance_date).days
            if days_since >= 7:
                return True
        elif freq == 'monthly':
            if current_date.month != self.last_rebalance_date.month:
                return True
        elif freq == 'quarterly':
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (self.last_rebalance_date.month - 1) // 3
            if current_quarter != last_quarter:
                return True

        # 偏离触发
        for alloc in self.allocations.values():
            if abs(alloc.current_weight - alloc.target_weight) > self.config.rebalance_threshold:
                return True

        return False

    def _rebalance(self, prices: Dict[str, float], target_weights: Dict[str, float]):
        """执行再平衡"""
        total_equity = self._calculate_equity()

        turnover = 0.0
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            alloc = self.allocations[symbol]

            # 目标市值和数量
            target_value = total_equity * target_weight
            target_qty = target_value / price

            # 当前持仓
            current_qty = alloc.quantity
            delta_qty = target_qty - current_qty

            if abs(delta_qty) > 0.01:  # 最小交易阈值
                # 计算手续费
                trade_value = abs(delta_qty) * price
                commission = trade_value * self.config.commission_rate

                # 更新现金
                if delta_qty > 0:  # 买入
                    cost = delta_qty * price * (1 + self.config.slippage_rate) + commission
                    self.cash -= cost
                else:  # 卖出
                    proceeds = abs(delta_qty) * price * (1 - self.config.slippage_rate) - commission
                    self.cash += proceeds

                # 更新持仓
                alloc.quantity = target_qty
                alloc.value = target_qty * price

                # 累计换手率
                turnover += trade_value / total_equity

        self.turnover_total += turnover
        self._update_weights()

    def _update_values(self, prices: Dict[str, float]):
        """更新持仓市值"""
        for symbol, price in prices.items():
            if symbol in self.allocations:
                alloc = self.allocations[symbol]
                alloc.value = alloc.quantity * price

        self._update_weights()

    def _update_weights(self):
        """更新权重"""
        total_equity = self._calculate_equity()
        for alloc in self.allocations.values():
            alloc.current_weight = alloc.value / total_equity if total_equity > 0 else 0

    def _calculate_equity(self) -> float:
        """计算总权益"""
        total_value = sum(alloc.value for alloc in self.allocations.values())
        return self.cash + total_value

    def _generate_result(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_weights: Dict[str, float]
    ) -> PortfolioResult:
        """生成组合回测结果"""
        # 权益曲线
        equity_curve = pd.Series(self.equity_history, index=self.date_history)

        # 权重历史
        weights_df = pd.DataFrame(self.weights_history, index=self.date_history)

        # 资产收益率
        asset_returns = pd.DataFrame({
            sym: data_dict[sym]['close'].pct_change()
            for sym in target_weights.keys()
        })

        # 组合收益率
        portfolio_returns = equity_curve.pct_change().fillna(0)

        # 相关性矩阵
        correlation_matrix = asset_returns.corr()

        # 协方差矩阵
        covariance_matrix = asset_returns.cov() * 252  # 年化

        # 计算绩效指标
        metrics = self._calculate_metrics(equity_curve, portfolio_returns, asset_returns)

        # 资产贡献度
        asset_contributions = self._calculate_contributions(asset_returns, target_weights)

        return PortfolioResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            weights_history=weights_df,
            asset_returns=asset_returns,
            portfolio_returns=portfolio_returns,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            rebalance_dates=self.rebalance_dates,
            turnover=self.turnover_total,
            asset_contributions=asset_contributions
        )

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """计算组合绩效指标"""
        initial = self.config.initial_capital
        final = equity_curve.iloc[-1]
        total_return = (final - initial) / initial

        # 年化收益
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 波动率
        volatility = portfolio_returns.std() * np.sqrt(252)

        # 夏普比率
        excess_return = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # 信息比率 (相对于等权组合)
        equal_weight_returns = asset_returns.mean(axis=1)
        tracking_error = (portfolio_returns - equal_weight_returns).std() * np.sqrt(252)
        active_return = annual_return - equal_weight_returns.mean() * 252
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0

        # 组合Beta (相对于等权)
        if equal_weight_returns.var() > 0:
            beta = portfolio_returns.cov(equal_weight_returns) / equal_weight_returns.var()
        else:
            beta = 1.0

        # 组合Alpha
        alpha = annual_return - self.config.risk_free_rate - beta * (
            equal_weight_returns.mean() * 252 - self.config.risk_free_rate
        )

        # 分散化比率
        individual_vols = asset_returns.std() * np.sqrt(252)
        portfolio_vol = volatility
        diversification_ratio = (
            individual_vols.mean() / portfolio_vol if portfolio_vol > 0 else 1.0
        )

        return {
            'initial_capital': initial,
            'final_capital': final,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'diversification_ratio': diversification_ratio,
            'total_turnover': self.turnover_total,
            'rebalance_count': len(self.rebalance_dates),
        }

    def _calculate_contributions(
        self,
        asset_returns: pd.DataFrame,
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """计算各资产对组合收益的贡献"""
        contributions = {}
        total_return = 0.0

        for symbol, weight in target_weights.items():
            if symbol in asset_returns.columns:
                asset_total_return = (1 + asset_returns[symbol]).prod() - 1
                contribution = weight * asset_total_return
                contributions[symbol] = contribution
                total_return += contribution

        # 归一化为百分比贡献
        if total_return != 0:
            contributions = {k: v / total_return for k, v in contributions.items()}

        return contributions


class MeanVarianceOptimizer:
    """
    均值-方差优化器

    用于计算最优投资组合权重
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        returns: pd.DataFrame,
        target: str = 'sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        优化投资组合

        Args:
            returns: 资产收益率DataFrame
            target: 优化目标 ('sharpe', 'min_variance', 'max_return')
            constraints: 约束条件

        Returns:
            最优权重字典
        """
        from scipy.optimize import minimize

        n_assets = len(returns.columns)
        symbols = returns.columns.tolist()

        # 预期收益和协方差
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # 初始权重
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束：权重之和为1
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # 边界：每个权重在0-1之间
        bounds = tuple((0, 1) for _ in range(n_assets))

        if target == 'sharpe':
            # 最大化夏普比率
            def neg_sharpe(weights):
                port_return = np.dot(weights, mean_returns)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(port_return - self.risk_free_rate) / port_vol

            result = minimize(neg_sharpe, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints_list)

        elif target == 'min_variance':
            # 最小化方差
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            result = minimize(portfolio_variance, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints_list)

        elif target == 'max_return':
            # 最大化收益
            def neg_return(weights):
                return -np.dot(weights, mean_returns)

            result = minimize(neg_return, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints_list)

        else:
            raise ValueError(f"Unknown target: {target}")

        return dict(zip(symbols, result.x))

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        计算有效前沿

        Args:
            returns: 资产收益率
            n_points: 前沿上的点数

        Returns:
            DataFrame with columns ['return', 'volatility', 'sharpe']
        """
        from scipy.optimize import minimize

        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # 目标收益范围
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier = []

        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_ret}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            init_weights = np.array([1.0 / n_assets] * n_assets)

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            result = minimize(portfolio_variance, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            if result.success:
                vol = np.sqrt(result.fun)
                sharpe = (target_ret - self.risk_free_rate) / vol
                frontier.append({
                    'return': target_ret,
                    'volatility': vol,
                    'sharpe': sharpe
                })

        return pd.DataFrame(frontier)
