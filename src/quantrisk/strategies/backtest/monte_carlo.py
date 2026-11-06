"""
蒙特卡洛模拟

用于策略稳健性测试和风险估计
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings


@dataclass
class MonteCarloConfig:
    """蒙特卡洛配置"""
    n_simulations: int = 1000           # 模拟次数
    simulation_type: str = 'bootstrap'  # bootstrap, parametric, block_bootstrap
    block_size: int = 20                # 块大小（用于block_bootstrap）
    confidence_levels: List[float] = None  # 置信水平
    seed: Optional[int] = 42            # 随机种子
    parallel: bool = True               # 是否并行计算
    n_workers: int = 4                  # 并行工作进程数

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


@dataclass
class MonteCarloResult:
    """蒙特卡洛结果"""
    config: MonteCarloConfig
    simulated_returns: np.ndarray          # (n_simulations, n_days)
    simulated_equity: np.ndarray           # (n_simulations, n_days)
    final_equity_distribution: np.ndarray   # (n_simulations,)

    # 统计量
    mean_return: float
    median_return: float
    std_return: float

    # 置信区间
    return_confidence_intervals: Dict[float, tuple]
    drawdown_confidence_intervals: Dict[float, tuple]

    # 风险指标
    var_estimates: Dict[float, float]       # VaR at different confidence levels
    cvar_estimates: Dict[float, float]      # CVaR at different confidence levels
    probability_of_loss: float              # 亏损概率
    probability_of_ruin: float              # 破产概率（亏损>50%）

    # 路径分析
    best_case_equity: np.ndarray
    worst_case_equity: np.ndarray
    median_case_equity: np.ndarray


class MonteCarloSimulator:
    """
    蒙特卡洛模拟器

    支持多种模拟方法：
    - Bootstrap: 有放回抽样
    - Block Bootstrap: 块状有放回抽样（保持时间依赖性）
    - Parametric: 参数化模拟（假设正态分布）
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()
        if self.config.seed:
            np.random.seed(self.config.seed)

    def simulate(
        self,
        historical_returns: pd.Series,
        initial_capital: float = 1000000,
        strategy_func: Optional[Callable] = None
    ) -> MonteCarloResult:
        """
        运行蒙特卡洛模拟

        Args:
            historical_returns: 历史收益率序列
            initial_capital: 初始资金
            strategy_func: 可选的策略函数

        Returns:
            MonteCarloResult: 模拟结果
        """
        n_sims = self.config.n_simulations
        n_days = len(historical_returns)

        # 生成模拟路径
        if self.config.simulation_type == 'bootstrap':
            simulated_returns = self._bootstrap(historical_returns, n_sims)
        elif self.config.simulation_type == 'block_bootstrap':
            simulated_returns = self._block_bootstrap(historical_returns, n_sims)
        elif self.config.simulation_type == 'parametric':
            simulated_returns = self._parametric(historical_returns, n_sims)
        else:
            raise ValueError(f"Unknown simulation type: {self.config.simulation_type}")

        # 计算权益曲线
        simulated_equity = self._calculate_equity_curves(
            simulated_returns, initial_capital
        )

        # 最终权益分布
        final_equity = simulated_equity[:, -1]

        # 计算统计量
        total_returns = (final_equity - initial_capital) / initial_capital
        mean_return = np.mean(total_returns)
        median_return = np.median(total_returns)
        std_return = np.std(total_returns)

        # 置信区间
        return_ci = self._calculate_confidence_intervals(
            total_returns, self.config.confidence_levels
        )

        # 回撤分析
        drawdowns = self._calculate_max_drawdowns(simulated_equity)
        drawdown_ci = self._calculate_confidence_intervals(
            drawdowns, self.config.confidence_levels
        )

        # VaR和CVaR
        var_estimates = {}
        cvar_estimates = {}
        for level in self.config.confidence_levels:
            alpha = 1 - level
            var_estimates[level] = np.percentile(total_returns, alpha * 100)
            cvar_estimates[level] = total_returns[
                total_returns <= var_estimates[level]
            ].mean()

        # 风险概率
        prob_loss = (total_returns < 0).mean()
        prob_ruin = (total_returns < -0.5).mean()  # 亏损50%以上

        # 代表性路径
        best_idx = np.argmax(final_equity)
        worst_idx = np.argmin(final_equity)
        median_idx = np.argsort(final_equity)[len(final_equity) // 2]

        return MonteCarloResult(
            config=self.config,
            simulated_returns=simulated_returns,
            simulated_equity=simulated_equity,
            final_equity_distribution=final_equity,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            return_confidence_intervals=return_ci,
            drawdown_confidence_intervals=drawdown_ci,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            probability_of_loss=prob_loss,
            probability_of_ruin=prob_ruin,
            best_case_equity=simulated_equity[best_idx],
            worst_case_equity=simulated_equity[worst_idx],
            median_case_equity=simulated_equity[median_idx]
        )

    def _bootstrap(
        self,
        returns: pd.Series,
        n_simulations: int
    ) -> np.ndarray:
        """标准Bootstrap抽样"""
        n_days = len(returns)
        returns_array = returns.values

        # 有放回抽样
        indices = np.random.randint(0, n_days, size=(n_simulations, n_days))
        simulated = returns_array[indices]

        return simulated

    def _block_bootstrap(
        self,
        returns: pd.Series,
        n_simulations: int
    ) -> np.ndarray:
        """块状Bootstrap抽样（保持时间序列结构）"""
        n_days = len(returns)
        block_size = self.config.block_size
        returns_array = returns.values

        n_blocks = int(np.ceil(n_days / block_size))
        simulated = np.zeros((n_simulations, n_days))

        for sim in range(n_simulations):
            # 随机选择块的起始位置
            block_starts = np.random.randint(
                0, n_days - block_size + 1, size=n_blocks
            )

            # 组装模拟序列
            sim_returns = []
            for start in block_starts:
                sim_returns.extend(returns_array[start:start + block_size])

            simulated[sim] = sim_returns[:n_days]

        return simulated

    def _parametric(
        self,
        returns: pd.Series,
        n_simulations: int
    ) -> np.ndarray:
        """参数化模拟（正态分布）"""
        n_days = len(returns)

        # 估计参数
        mu = returns.mean()
        sigma = returns.std()

        # 生成正态分布随机数
        simulated = np.random.normal(mu, sigma, size=(n_simulations, n_days))

        return simulated

    def _calculate_equity_curves(
        self,
        simulated_returns: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """计算权益曲线"""
        # 复利计算
        growth_factors = 1 + simulated_returns
        cumulative_growth = np.cumprod(growth_factors, axis=1)
        equity_curves = initial_capital * cumulative_growth

        return equity_curves

    def _calculate_max_drawdowns(
        self,
        equity_curves: np.ndarray
    ) -> np.ndarray:
        """计算各模拟路径的最大回撤"""
        n_sims = equity_curves.shape[0]
        max_drawdowns = np.zeros(n_sims)

        for i in range(n_sims):
            equity = equity_curves[i]
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_drawdowns[i] = drawdown.min()

        return max_drawdowns

    def _calculate_confidence_intervals(
        self,
        values: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[float, tuple]:
        """计算置信区间"""
        intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower = np.percentile(values, alpha * 100)
            upper = np.percentile(values, (1 - alpha) * 100)
            intervals[level] = (lower, upper)
        return intervals

    def stress_test(
        self,
        historical_returns: pd.Series,
        stress_scenarios: List[Dict[str, Any]],
        initial_capital: float = 1000000
    ) -> Dict[str, Dict]:
        """
        压力测试

        Args:
            historical_returns: 历史收益率
            stress_scenarios: 压力场景列表，每个包含 {name, volatility_shock, return_shock}
            initial_capital: 初始资金

        Returns:
            各场景下的风险指标
        """
        results = {}

        for scenario in stress_scenarios:
            name = scenario.get('name', 'unnamed')
            vol_shock = scenario.get('volatility_shock', 1.0)  # 波动率乘数
            ret_shock = scenario.get('return_shock', 0.0)      # 收益率偏移

            # 修改收益率分布
            stressed_returns = historical_returns * vol_shock + ret_shock

            # 运行模拟
            mc_result = self.simulate(
                stressed_returns,
                initial_capital,
            )

            results[name] = {
                'mean_return': mc_result.mean_return,
                'var_95': mc_result.var_estimates.get(0.95, 0),
                'cvar_95': mc_result.cvar_estimates.get(0.95, 0),
                'probability_of_loss': mc_result.probability_of_loss,
                'probability_of_ruin': mc_result.probability_of_ruin,
                'worst_drawdown': mc_result.drawdown_confidence_intervals.get(0.99, (0, 0))[0]
            }

        return results


class GBMSimulator:
    """
    几何布朗运动模拟器

    用于期权定价和风险模拟
    """

    def __init__(self, seed: Optional[int] = 42):
        if seed:
            np.random.seed(seed)

    def simulate_paths(
        self,
        S0: float,              # 初始价格
        mu: float,              # 漂移率（年化）
        sigma: float,           # 波动率（年化）
        T: float,               # 时间长度（年）
        n_steps: int,           # 时间步数
        n_paths: int            # 模拟路径数
    ) -> np.ndarray:
        """
        模拟GBM价格路径

        dS = mu*S*dt + sigma*S*dW

        Returns:
            (n_paths, n_steps+1) 的价格矩阵
        """
        dt = T / n_steps

        # 生成随机增量
        Z = np.random.standard_normal((n_paths, n_steps))

        # 计算价格路径
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            )

        return paths

    def simulate_with_jumps(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        lambda_jump: float = 0.1,    # 跳跃强度
        mu_jump: float = -0.05,      # 跳跃均值
        sigma_jump: float = 0.1      # 跳跃标准差
    ) -> np.ndarray:
        """
        带跳跃的几何布朗运动（Merton Jump Diffusion）
        """
        dt = T / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, n_steps + 1):
            # 扩散部分
            Z = np.random.standard_normal(n_paths)
            diffusion = np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

            # 跳跃部分
            n_jumps = np.random.poisson(lambda_jump * dt, n_paths)
            jump_sizes = np.array([
                np.exp(np.sum(np.random.normal(mu_jump, sigma_jump, n)))
                if n > 0 else 1.0
                for n in n_jumps
            ])

            paths[:, t] = paths[:, t-1] * diffusion * jump_sizes

        return paths
