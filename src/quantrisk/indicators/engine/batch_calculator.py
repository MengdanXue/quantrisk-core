"""
批量指标计算引擎

支持高性能并行计算多个指标
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import logging

from ..registry.indicator_registry import indicator_registry
from ..base.indicator_base import BaseIndicator

# Import the categories referenced by IndicatorPresets so their decorator-based
# registrations are available even when callers import ``indicators.engine``
# directly.
from .. import momentum, statistical, trend, volatility, volume  # noqa: F401,E402

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """指标配置"""
    name: str
    params: Dict[str, Any] = None
    output_key: str = None  # 自定义输出键名

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.output_key is None:
            self.output_key = self.name


@dataclass
class CalculationResult:
    """计算结果"""
    indicator_name: str
    success: bool
    result: Optional[Union[pd.Series, pd.DataFrame]] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class BatchCalculator:
    """
    批量指标计算器

    支持并行计算多个指标，提供性能优化和错误处理
    """

    def __init__(self, max_workers: int = 4, cache_enabled: bool = True):
        """
        初始化批量计算器

        Args:
            max_workers: 最大并行工作线程数
            cache_enabled: 是否启用缓存
        """
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        self._indicator_instances: Dict[str, BaseIndicator] = {}

    def _get_cache_key(self, indicator_name: str, params: Dict, data_hash: str) -> str:
        """生成缓存键"""
        param_str = str(sorted(params.items()))
        return f"{indicator_name}_{param_str}_{data_hash}"

    def _hash_data(self, data: pd.DataFrame) -> str:
        """生成数据哈希"""
        return str(hash(tuple(data.values.tobytes())))

    def _get_or_create_indicator(self, config: IndicatorConfig) -> BaseIndicator:
        """获取或创建指标实例"""
        cache_key = f"{config.name}_{str(sorted(config.params.items()))}"

        if cache_key not in self._indicator_instances:
            self._indicator_instances[cache_key] = indicator_registry.create_indicator(
                config.name, **config.params
            )

        return self._indicator_instances[cache_key]

    def calculate_single(
        self,
        data: pd.DataFrame,
        config: IndicatorConfig
    ) -> CalculationResult:
        """
        计算单个指标

        Args:
            data: OHLCV数据
            config: 指标配置

        Returns:
            计算结果
        """
        start_time = time.time()

        try:
            indicator = self._get_or_create_indicator(config)

            # 检查缓存
            if self.cache_enabled:
                data_hash = self._hash_data(data)
                cache_key = self._get_cache_key(config.name, config.params, data_hash)

                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    return CalculationResult(
                        indicator_name=config.output_key,
                        success=True,
                        result=cached_result,
                        execution_time=time.time() - start_time
                    )

            # 计算指标
            result = indicator.calculate(data)

            # 存入缓存
            if self.cache_enabled:
                self._cache[cache_key] = result

            return CalculationResult(
                indicator_name=config.output_key,
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Error calculating {config.name}: {str(e)}")
            return CalculationResult(
                indicator_name=config.output_key,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def calculate_batch(
        self,
        data: pd.DataFrame,
        configs: List[IndicatorConfig],
        parallel: bool = True
    ) -> Dict[str, CalculationResult]:
        """
        批量计算多个指标

        Args:
            data: OHLCV数据
            configs: 指标配置列表
            parallel: 是否并行计算

        Returns:
            指标名 -> 计算结果 的字典
        """
        results = {}

        if parallel and len(configs) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self.calculate_single, data, config): config
                    for config in configs
                }

                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results[result.indicator_name] = result
                    except Exception as e:
                        results[config.output_key] = CalculationResult(
                            indicator_name=config.output_key,
                            success=False,
                            error=str(e)
                        )
        else:
            for config in configs:
                result = self.calculate_single(data, config)
                results[result.indicator_name] = result

        return results

    def calculate_to_dataframe(
        self,
        data: pd.DataFrame,
        configs: List[IndicatorConfig],
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        批量计算并合并为单一DataFrame

        Args:
            data: OHLCV数据
            configs: 指标配置列表
            parallel: 是否并行计算

        Returns:
            包含所有指标结果的DataFrame
        """
        results = self.calculate_batch(data, configs, parallel)

        output = data.copy()

        for name, calc_result in results.items():
            if calc_result.success and calc_result.result is not None:
                if isinstance(calc_result.result, pd.Series):
                    output[name] = calc_result.result
                elif isinstance(calc_result.result, pd.DataFrame):
                    for col in calc_result.result.columns:
                        output[f"{name}_{col}"] = calc_result.result[col]

        return output

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        self._indicator_instances.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'cached_results': len(self._cache),
            'cached_instances': len(self._indicator_instances)
        }


class IndicatorPipeline:
    """
    指标计算流水线

    支持定义复杂的指标计算流程
    """

    def __init__(self, calculator: BatchCalculator = None):
        """
        初始化流水线

        Args:
            calculator: 批量计算器实例
        """
        self.calculator = calculator or BatchCalculator()
        self._stages: List[List[IndicatorConfig]] = []

    def add_stage(self, configs: List[IndicatorConfig]) -> 'IndicatorPipeline':
        """
        添加计算阶段

        同一阶段的指标会并行计算，不同阶段按顺序执行

        Args:
            configs: 该阶段的指标配置列表

        Returns:
            self，支持链式调用
        """
        self._stages.append(configs)
        return self

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行流水线

        Args:
            data: 输入数据

        Returns:
            包含所有计算结果的DataFrame
        """
        result = data.copy()

        for stage_idx, stage in enumerate(self._stages):
            logger.info(f"Executing pipeline stage {stage_idx + 1}/{len(self._stages)}")

            stage_result = self.calculator.calculate_to_dataframe(
                result, stage, parallel=True
            )

            # 合并结果
            for col in stage_result.columns:
                if col not in result.columns:
                    result[col] = stage_result[col]

        return result

    def clear(self) -> 'IndicatorPipeline':
        """清除所有阶段"""
        self._stages.clear()
        return self


class IndicatorPresets:
    """
    预设指标组合

    提供常用的指标组合配置
    """

    @staticmethod
    def trend_following() -> List[IndicatorConfig]:
        """趋势跟踪策略指标组合"""
        return [
            IndicatorConfig('SMA', {'period': 20}, 'SMA_20'),
            IndicatorConfig('SMA', {'period': 50}, 'SMA_50'),
            IndicatorConfig('SMA', {'period': 200}, 'SMA_200'),
            IndicatorConfig('EMA', {'period': 12}, 'EMA_12'),
            IndicatorConfig('EMA', {'period': 26}, 'EMA_26'),
            IndicatorConfig('MACD', {}, 'MACD'),
            IndicatorConfig('ADX', {'period': 14}, 'ADX'),
            IndicatorConfig('Supertrend', {}, 'Supertrend'),
        ]

    @staticmethod
    def momentum_trading() -> List[IndicatorConfig]:
        """动量交易策略指标组合"""
        return [
            IndicatorConfig('RSI', {'period': 14}, 'RSI_14'),
            IndicatorConfig('RSI', {'period': 7}, 'RSI_7'),
            IndicatorConfig('StochasticOscillator', {}, 'Stoch'),
            IndicatorConfig('CCI', {'period': 20}, 'CCI'),
            IndicatorConfig('WilliamsR', {'period': 14}, 'WilliamsR'),
            IndicatorConfig('ROC', {'period': 10}, 'ROC'),
            IndicatorConfig('Momentum', {'period': 10}, 'MOM'),
        ]

    @staticmethod
    def volatility_analysis() -> List[IndicatorConfig]:
        """波动率分析指标组合"""
        return [
            IndicatorConfig('BollingerBands', {}, 'BB'),
            IndicatorConfig('ATR', {'period': 14}, 'ATR_14'),
            IndicatorConfig('ATR', {'period': 7}, 'ATR_7'),
            IndicatorConfig('HistoricalVolatility', {'period': 20}, 'HV_20'),
            IndicatorConfig('KeltnerChannel', {}, 'KC'),
            IndicatorConfig('DonchianChannel', {}, 'DC'),
        ]

    @staticmethod
    def volume_analysis() -> List[IndicatorConfig]:
        """成交量分析指标组合"""
        return [
            IndicatorConfig('OBV', {}, 'OBV'),
            IndicatorConfig('VolumeMA', {'period': 20}, 'Vol_MA'),
            IndicatorConfig('VWAP', {'period': 20}, 'VWAP'),
            IndicatorConfig('ChaikinMoneyFlow', {'period': 20}, 'CMF'),
            IndicatorConfig('MoneyFlowIndex', {'period': 14}, 'MFI'),
            IndicatorConfig('ForceIndex', {'period': 13}, 'FI'),
        ]

    @staticmethod
    def risk_metrics() -> List[IndicatorConfig]:
        """风险指标组合"""
        return [
            IndicatorConfig('MaxDrawdown', {'period': 252}, 'MDD'),
            IndicatorConfig('Sharpe', {'period': 252}, 'Sharpe'),
            IndicatorConfig('Sortino', {'period': 252}, 'Sortino'),
            IndicatorConfig('VaR', {'period': 252}, 'VaR'),
            IndicatorConfig('CVaR', {'period': 252}, 'CVaR'),
            IndicatorConfig('HistoricalVolatility', {'period': 20}, 'Vol'),
        ]

    @staticmethod
    def comprehensive() -> List[IndicatorConfig]:
        """综合分析指标组合"""
        configs = []
        configs.extend(IndicatorPresets.trend_following())
        configs.extend(IndicatorPresets.momentum_trading())
        configs.extend(IndicatorPresets.volatility_analysis())
        return configs


# 创建全局计算器实例
batch_calculator = BatchCalculator()
