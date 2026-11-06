"""
回测引擎模块

专业级回测框架，包含：
- BacktestEngine: 核心回测引擎
- PortfolioBacktest: 多资产组合回测
- MonteCarloSimulator: 蒙特卡洛模拟
- WalkForwardOptimizer: Walk-Forward优化
- ReportGenerator: 回测报告生成器
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .portfolio import PortfolioBacktest, PortfolioResult
from .monte_carlo import MonteCarloSimulator, MonteCarloResult
from .walk_forward import WalkForwardOptimizer, WFOResult
from .report import ReportGenerator

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'PortfolioBacktest',
    'PortfolioResult',
    'MonteCarloSimulator',
    'MonteCarloResult',
    'WalkForwardOptimizer',
    'WFOResult',
    'ReportGenerator',
]
