"""
单证券 Walk-Forward 工具（旧版，按比例切分、无验证集与隔离期）。
组合级样本外研究请使用 quantrisk.research.walk_forward.WalkForwardStudy。

Walk-Forward优化

用于策略参数的稳健性优化，避免过拟合
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools
import warnings


@dataclass
class WFOConfig:
    """Walk-Forward优化配置"""
    n_splits: int = 5                    # 分割数
    train_ratio: float = 0.7             # 训练集比例
    optimization_target: str = 'sharpe'  # 优化目标
    anchored: bool = False               # 是否使用锚定方式（累积训练集）
    min_train_size: int = 252            # 最小训练集大小（天）
    step_size: Optional[int] = None      # 步进大小（None表示自动）
    parallel: bool = True
    n_workers: int = 4


@dataclass
class WFOResult:
    """Walk-Forward优化结果"""
    config: WFOConfig
    best_params_per_fold: List[Dict[str, Any]]
    in_sample_metrics: List[Dict[str, float]]
    out_sample_metrics: List[Dict[str, float]]
    combined_equity: pd.Series
    combined_metrics: Dict[str, float]
    parameter_stability: Dict[str, float]   # 参数稳定性得分
    robustness_ratio: float                 # 样本外/样本内比率
    fold_details: List[Dict]


class WalkForwardOptimizer:
    """
    Walk-Forward优化器

    特性：
    - 滚动窗口优化
    - 锚定窗口优化
    - 参数稳定性分析
    - 过拟合检测
    """

    def __init__(self, config: Optional[WFOConfig] = None):
        self.config = config or WFOConfig()

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        metric_func: Optional[Callable] = None
    ) -> WFOResult:
        """
        运行Walk-Forward优化

        Args:
            data: OHLCV数据
            strategy_class: 策略类
            param_grid: 参数网格
            metric_func: 自定义评估函数

        Returns:
            WFOResult: 优化结果
        """
        # 生成时间分割
        splits = self._generate_splits(data)

        best_params_list = []
        in_sample_metrics_list = []
        out_sample_metrics_list = []
        fold_details = []
        out_sample_equities = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # 在训练集上优化参数
            best_params, best_is_metrics = self._optimize_on_fold(
                train_data, strategy_class, param_grid, metric_func
            )
            best_params_list.append(best_params)
            in_sample_metrics_list.append(best_is_metrics)

            # 在测试集上验证
            os_result = self._evaluate_on_fold(
                test_data, strategy_class, best_params
            )
            out_sample_metrics_list.append(os_result['metrics'])
            out_sample_equities.append(os_result['equity'])

            fold_details.append({
                'fold': fold_idx,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_size': len(train_data),
                'test_size': len(test_data),
                'best_params': best_params,
                'in_sample_sharpe': best_is_metrics.get('sharpe_ratio', 0),
                'out_sample_sharpe': os_result['metrics'].get('sharpe_ratio', 0),
            })

        # 合并样本外权益曲线
        combined_equity = pd.concat(out_sample_equities)
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep='first')]
        combined_equity = combined_equity.sort_index()

        # 计算综合指标
        combined_metrics = self._calculate_combined_metrics(combined_equity)

        # 参数稳定性分析
        param_stability = self._analyze_parameter_stability(best_params_list)

        # 稳健性比率
        avg_is = np.mean([m.get('sharpe_ratio', 0) for m in in_sample_metrics_list])
        avg_os = np.mean([m.get('sharpe_ratio', 0) for m in out_sample_metrics_list])
        robustness_ratio = avg_os / avg_is if avg_is != 0 else 0

        return WFOResult(
            config=self.config,
            best_params_per_fold=best_params_list,
            in_sample_metrics=in_sample_metrics_list,
            out_sample_metrics=out_sample_metrics_list,
            combined_equity=combined_equity,
            combined_metrics=combined_metrics,
            parameter_stability=param_stability,
            robustness_ratio=robustness_ratio,
            fold_details=fold_details
        )

    def _generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成训练/测试分割"""
        n = len(data)
        n_splits = self.config.n_splits
        train_ratio = self.config.train_ratio

        splits = []

        if self.config.anchored:
            # 锚定方式：训练集从开头开始，逐渐扩大
            step = (n - self.config.min_train_size) // n_splits

            for i in range(n_splits):
                train_end = self.config.min_train_size + i * step
                test_start = train_end
                test_end = min(test_start + step, n)

                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)

                if len(train_idx) >= self.config.min_train_size and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
        else:
            # 滚动窗口方式
            fold_size = n // n_splits
            train_size = int(fold_size * train_ratio * n_splits / (n_splits - 1))

            if self.config.step_size:
                step = self.config.step_size
            else:
                step = (n - train_size) // (n_splits - 1)

            for i in range(n_splits):
                train_start = i * step
                train_end = train_start + train_size
                test_start = train_end
                test_end = min(test_start + step, n)

                if test_end > n:
                    break

                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)

                if len(train_idx) >= self.config.min_train_size and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        return splits

    def _optimize_on_fold(
        self,
        train_data: pd.DataFrame,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        metric_func: Optional[Callable]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """在单个fold上优化参数"""
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_metric = float('-inf')
        best_params = None
        best_metrics = None

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                # 创建策略实例
                strategy = strategy_class(params)

                # 运行回测
                result = strategy.run(train_data)

                # 获取目标指标
                if metric_func:
                    metric_value = metric_func(result)
                else:
                    metric_value = result.metrics.get(self.config.optimization_target, 0)

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_metrics = result.metrics

            except Exception as e:
                # 不再静默跳过：失败的参数组合会让选择结果失真，必须显式报错
                raise RuntimeError(f"参数组合 {params} 回测失败") from e

        return best_params or {}, best_metrics or {}

    def _evaluate_on_fold(
        self,
        test_data: pd.DataFrame,
        strategy_class,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """在测试集上评估"""
        try:
            strategy = strategy_class(params)
            result = strategy.run(test_data)

            return {
                'metrics': result.metrics,
                'equity': result.equity_curve
            }
        except Exception as e:
            raise RuntimeError("测试段评估失败") from e

    def _calculate_combined_metrics(
        self,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """计算综合指标"""
        if equity_curve.empty:
            return {}

        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return {}

        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

        # 年化
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': annual_return / abs(max_dd) if max_dd != 0 else 0
        }

    def _analyze_parameter_stability(
        self,
        params_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """分析参数稳定性"""
        if not params_list:
            return {}

        # 收集每个参数的值
        param_values = {}
        for params in params_list:
            for key, value in params.items():
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)

        # 计算稳定性得分（变异系数的倒数）
        stability = {}
        for param, values in param_values.items():
            try:
                values = np.array(values, dtype=float)
                mean_val = np.mean(values)
                std_val = np.std(values)

                if mean_val != 0:
                    cv = std_val / abs(mean_val)  # 变异系数
                    stability[param] = 1 / (1 + cv)  # 转换为0-1的稳定性得分
                else:
                    stability[param] = 1.0 if std_val == 0 else 0.0

            except (ValueError, TypeError):
                # 非数值参数，检查一致性
                unique_values = len(set(str(v) for v in values))
                stability[param] = 1.0 / unique_values

        return stability


class CombinatoricsPurgedKFold:
    """
    组合清洗K折交叉验证

    专门用于时间序列数据，避免前瞻偏差
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,      # 清洗间隔（天）
        embargo_gap: int = 5     # 禁用间隔（天）
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成交叉验证分割

        Args:
            data: 时间序列数据

        Yields:
            (train_indices, test_indices) 元组
        """
        n = len(data)
        fold_size = n // self.n_splits
        indices = np.arange(n)

        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]

            # 训练集：排除测试集及其前后的清洗/禁用区间
            purge_start = max(0, test_start - self.purge_gap)
            embargo_end = min(n, test_end + self.embargo_gap)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits
