"""
回测引擎模块

专业级回测框架，包含：
- run_portfolio / BaseStrategy.run: 唯一有效的回测路径（daily.py：T+1 开盘撮合、A 股规则、资金账本）
- BacktestEngine / PortfolioBacktest: 旧引擎，已停用（构造时抛出 LegacyEngineError）
- MonteCarloSimulator: 蒙特卡洛模拟
- WalkForwardOptimizer: Walk-Forward优化
- ReportGenerator: 回测报告生成器
"""

from .daily import AShareMarket, GenericMarket, Instrument, RiskBudget, run_daily, run_portfolio
from .engine import BacktestEngine, BacktestConfig, BacktestResult, LegacyEngineError
from .portfolio import PortfolioBacktest, PortfolioResult
from .monte_carlo import MonteCarloSimulator, MonteCarloResult
from .walk_forward import WalkForwardOptimizer, WFOResult
from .report import ReportGenerator

__all__ = [
    'run_portfolio',
    'run_daily',
    'Instrument',
    'RiskBudget',
    'AShareMarket',
    'GenericMarket',
    'LegacyEngineError',
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
