"""
因子组合器模块

提供多因子组合、权重优化、因子正交化等功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class CombineMethod(Enum):
    """组合方法"""
    EQUAL_WEIGHT = "equal_weight"         # 等权
    IC_WEIGHT = "ic_weight"               # IC加权
    IR_WEIGHT = "ir_weight"               # IR加权
    MAX_IC_IR = "max_ic_ir"               # 最大化IC_IR
    MIN_CORRELATION = "min_correlation"   # 最小相关性


@dataclass
class CombinedFactor:
    """组合因子"""
    name: str
    weights: Dict[str, float]
    combined_values: pd.DataFrame
    method: CombineMethod
    ic: Optional[float] = None
    ir: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_weights: Dict[str, float]
    objective_value: float
    converged: bool
    iterations: int
    method: str
    constraints_satisfied: bool


class FactorCombiner:
    """因子组合器"""

    def __init__(self):
        self._factor_correlations: Optional[pd.DataFrame] = None
        self._factor_ics: Dict[str, float] = {}

    def combine_factors(
        self,
        factor_data: Dict[str, pd.DataFrame],
        method: CombineMethod = CombineMethod.EQUAL_WEIGHT,
        forward_returns: Optional[pd.DataFrame] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> CombinedFactor:
        """
        组合多个因子

        Args:
            factor_data: 因子数据字典 {因子名: DataFrame}
            method: 组合方法
            forward_returns: 前向收益 (IC/IR加权需要)
            custom_weights: 自定义权重

        Returns:
            CombinedFactor: 组合因子
        """
        if not factor_data:
            raise ValueError("No factor data provided")

        factor_names = list(factor_data.keys())

        # 计算权重
        if custom_weights:
            weights = custom_weights
        elif method == CombineMethod.EQUAL_WEIGHT:
            weights = self._equal_weights(factor_names)
        elif method == CombineMethod.IC_WEIGHT:
            weights = self._ic_weights(factor_data, forward_returns)
        elif method == CombineMethod.IR_WEIGHT:
            weights = self._ir_weights(factor_data, forward_returns)
        elif method == CombineMethod.MAX_IC_IR:
            weights = self._optimize_ic_ir(factor_data, forward_returns)
        elif method == CombineMethod.MIN_CORRELATION:
            weights = self._minimize_correlation(factor_data)
        else:
            weights = self._equal_weights(factor_names)

        # 标准化因子值
        standardized = self._standardize_factors(factor_data)

        # 加权组合
        combined_values = self._weighted_combine(standardized, weights)

        # 计算组合因子的IC/IR
        ic, ir = None, None
        if forward_returns is not None:
            ic, ir = self._calculate_combined_ic_ir(combined_values, forward_returns)

        return CombinedFactor(
            name=f"combined_{method.value}",
            weights=weights,
            combined_values=combined_values,
            method=method,
            ic=ic,
            ir=ir
        )

    def orthogonalize_factors(
        self,
        factor_data: Dict[str, pd.DataFrame],
        method: str = "gram_schmidt"
    ) -> Dict[str, pd.DataFrame]:
        """
        正交化因子

        Args:
            factor_data: 因子数据
            method: 正交化方法 ('gram_schmidt', 'symmetric')

        Returns:
            Dict[str, pd.DataFrame]: 正交化后的因子
        """
        if method == "gram_schmidt":
            return self._gram_schmidt_orthogonalize(factor_data)
        elif method == "symmetric":
            return self._symmetric_orthogonalize(factor_data)
        else:
            return factor_data

    def optimize_weights(
        self,
        factor_data: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        objective: str = "max_ir",
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        优化因子权重

        Args:
            factor_data: 因子数据
            forward_returns: 前向收益
            objective: 优化目标 ('max_ir', 'max_ic', 'min_vol')
            constraints: 约束条件

        Returns:
            OptimizationResult: 优化结果
        """
        factor_names = list(factor_data.keys())
        n_factors = len(factor_names)

        # 初始权重
        x0 = np.ones(n_factors) / n_factors

        # 目标函数
        def objective_func(weights: np.ndarray) -> float:
            weight_dict = dict(zip(factor_names, weights))
            standardized = self._standardize_factors(factor_data)
            combined = self._weighted_combine(standardized, weight_dict)
            ic, ir = self._calculate_combined_ic_ir(combined, forward_returns)

            if objective == "max_ir":
                return -ir if ir else 1e6
            elif objective == "max_ic":
                return -ic if ic else 1e6
            else:
                return -ir if ir else 1e6

        # 约束
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1

        # 边界
        bounds = [(0, 1) for _ in range(n_factors)]

        # 优化
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )

        optimal_weights = dict(zip(factor_names, result.x))

        return OptimizationResult(
            optimal_weights=optimal_weights,
            objective_value=-result.fun,
            converged=result.success,
            iterations=result.nit,
            method="SLSQP",
            constraints_satisfied=abs(sum(result.x) - 1) < 1e-6
        )

    def calculate_factor_correlation(
        self,
        factor_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        factor_names = list(factor_data.keys())

        # 构建因子面板
        panels = []
        for name, data in factor_data.items():
            stacked = data.stack()
            stacked.name = name
            panels.append(stacked)

        factor_panel = pd.concat(panels, axis=1).dropna()

        return factor_panel.corr()

    def _equal_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """等权重"""
        weight = 1.0 / len(factor_names)
        return {name: weight for name in factor_names}

    def _ic_weights(
        self,
        factor_data: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """IC加权"""
        if forward_returns is None:
            return self._equal_weights(list(factor_data.keys()))

        ics = {}
        for name, data in factor_data.items():
            ic, _ = self._calculate_factor_ic(data, forward_returns)
            ics[name] = max(ic, 0)  # 只使用正IC

        total_ic = sum(ics.values())
        if total_ic == 0:
            return self._equal_weights(list(factor_data.keys()))

        return {name: ic / total_ic for name, ic in ics.items()}

    def _ir_weights(
        self,
        factor_data: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """IR加权"""
        if forward_returns is None:
            return self._equal_weights(list(factor_data.keys()))

        irs = {}
        for name, data in factor_data.items():
            _, ir = self._calculate_factor_ic(data, forward_returns)
            irs[name] = max(ir, 0)

        total_ir = sum(irs.values())
        if total_ir == 0:
            return self._equal_weights(list(factor_data.keys()))

        return {name: ir / total_ir for name, ir in irs.items()}

    def _optimize_ic_ir(
        self,
        factor_data: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """最大化IC*IR"""
        if forward_returns is None:
            return self._equal_weights(list(factor_data.keys()))

        result = self.optimize_weights(
            factor_data, forward_returns, objective="max_ir"
        )
        return result.optimal_weights

    def _minimize_correlation(
        self,
        factor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """最小相关性加权"""
        corr_matrix = self.calculate_factor_correlation(factor_data)
        factor_names = list(factor_data.keys())

        # 计算每个因子与其他因子的平均相关性
        avg_corrs = {}
        for name in factor_names:
            other_corrs = [corr_matrix.loc[name, other]
                          for other in factor_names if other != name]
            avg_corrs[name] = np.mean(np.abs(other_corrs))

        # 相关性越低，权重越高
        inv_corrs = {name: 1 / (corr + 0.1) for name, corr in avg_corrs.items()}
        total = sum(inv_corrs.values())

        return {name: v / total for name, v in inv_corrs.items()}

    def _standardize_factors(
        self,
        factor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """标准化因子 (截面标准化)"""
        standardized = {}

        for name, data in factor_data.items():
            # 截面标准化
            mean = data.mean(axis=1)
            std = data.std(axis=1)

            standardized[name] = data.sub(mean, axis=0).div(std.replace(0, 1), axis=0)

        return standardized

    def _weighted_combine(
        self,
        factor_data: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """加权组合因子"""
        combined = None

        for name, data in factor_data.items():
            weight = weights.get(name, 0)
            weighted = data * weight

            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)

        return combined

    def _calculate_factor_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> Tuple[float, float]:
        """计算单因子IC/IR"""
        from scipy import stats

        common_dates = factor_values.index.intersection(forward_returns.index)
        common_symbols = factor_values.columns.intersection(forward_returns.columns)

        ic_series = []

        for date in common_dates:
            factor_row = factor_values.loc[date, common_symbols].dropna()
            return_row = forward_returns.loc[date, common_symbols].dropna()

            common = factor_row.index.intersection(return_row.index)
            if len(common) < 10:
                continue

            corr, _ = stats.spearmanr(factor_row[common], return_row[common])
            if not np.isnan(corr):
                ic_series.append(corr)

        if not ic_series:
            return 0, 0

        ic_mean = np.mean(ic_series)
        ic_std = np.std(ic_series)
        ir = ic_mean / ic_std if ic_std > 0 else 0

        return ic_mean, ir

    def _calculate_combined_ic_ir(
        self,
        combined_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> Tuple[float, float]:
        """计算组合因子的IC/IR"""
        return self._calculate_factor_ic(combined_values, forward_returns)

    def _gram_schmidt_orthogonalize(
        self,
        factor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Gram-Schmidt正交化"""
        factor_names = list(factor_data.keys())
        orthogonalized = {}

        for i, name in enumerate(factor_names):
            data = factor_data[name].copy()

            # 减去之前因子的投影
            for prev_name in factor_names[:i]:
                prev_data = orthogonalized[prev_name]

                # 计算投影系数
                common_dates = data.index.intersection(prev_data.index)
                common_symbols = data.columns.intersection(prev_data.columns)

                if len(common_dates) == 0 or len(common_symbols) == 0:
                    continue

                proj_coef = self._calculate_projection_coefficient(
                    data.loc[common_dates, common_symbols],
                    prev_data.loc[common_dates, common_symbols]
                )

                # 减去投影
                data = data - proj_coef * prev_data

            orthogonalized[name] = data

        return orthogonalized

    def _symmetric_orthogonalize(
        self,
        factor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """对称正交化"""
        # 构建因子矩阵
        factor_names = list(factor_data.keys())
        stacked = []

        for name in factor_names:
            s = factor_data[name].stack()
            s.name = name
            stacked.append(s)

        factor_matrix = pd.concat(stacked, axis=1).dropna()

        # 计算协方差矩阵
        cov = factor_matrix.cov()

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 计算白化矩阵
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        whitening_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T

        # 应用白化
        orthogonalized_matrix = factor_matrix @ whitening_matrix

        # 重构DataFrame
        orthogonalized = {}
        for i, name in enumerate(factor_names):
            values = orthogonalized_matrix.iloc[:, i]
            unstacked = values.unstack()
            orthogonalized[name] = unstacked

        return orthogonalized

    def _calculate_projection_coefficient(
        self,
        factor1: pd.DataFrame,
        factor2: pd.DataFrame
    ) -> float:
        """计算投影系数"""
        # 展平为向量
        v1 = factor1.values.flatten()
        v2 = factor2.values.flatten()

        # 去除NaN
        mask = ~(np.isnan(v1) | np.isnan(v2))
        v1 = v1[mask]
        v2 = v2[mask]

        if len(v1) == 0:
            return 0

        # 计算投影系数
        return np.dot(v1, v2) / (np.dot(v2, v2) + 1e-10)


# 单例实例
_factor_combiner_instance: Optional[FactorCombiner] = None


def get_factor_combiner() -> FactorCombiner:
    """获取因子组合器单例"""
    global _factor_combiner_instance
    if _factor_combiner_instance is None:
        _factor_combiner_instance = FactorCombiner()
    return _factor_combiner_instance
