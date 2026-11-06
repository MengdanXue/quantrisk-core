"""
因子研究框架

提供量化因子库、IC/IR分析、因子组合优化等功能。

核心组件:
- FactorLibrary: 因子库 (动量、价值、质量、成长)
- FactorAnalyzer: 因子分析器 (IC/IR/Turnover)
- FactorCombiner: 因子组合器 (等权/IC加权/最优化)
"""

from .factor_library import (
    FactorLibrary,
    Factor,
    FactorCategory,
    FactorData,
)

from .factor_analyzer import (
    FactorAnalyzer,
    ICAnalysis,
    FactorPerformance,
    DecayAnalysis,
)

from .factor_combiner import (
    FactorCombiner,
    CombineMethod,
    CombinedFactor,
    OptimizationResult,
)

__all__ = [
    # 因子库
    'FactorLibrary',
    'Factor',
    'FactorCategory',
    'FactorData',
    # 分析器
    'FactorAnalyzer',
    'ICAnalysis',
    'FactorPerformance',
    'DecayAnalysis',
    # 组合器
    'FactorCombiner',
    'CombineMethod',
    'CombinedFactor',
    'OptimizationResult',
]
