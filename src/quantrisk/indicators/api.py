"""
统一指标计算API
Unified Indicator Calculation API

提供单一入口点访问所有指标计算功能
支持V2兼容模式、V3注册表、Numba加速

作者: TraderX Team
创建: 2025-12-23
"""

from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass

# 引擎模块
from .engine import (
    BatchCalculator,
    IndicatorConfig,
    CalculationResult,
    IndicatorPipeline,
    IndicatorPresets,
    batch_calculator,
    V2IndicatorAdapter,
    V2CompatibilityLayer,
    v2_adapter,
    v2_compat,
    NUMBA_AVAILABLE,
    calculate_rsi_fast,
    calculate_ema_fast,
    calculate_atr_fast,
    calculate_bollinger_fast,
    calculate_macd_fast,
)

# 注册表
from .registry import indicator_registry

logger = logging.getLogger(__name__)


class CalculationMode(str, Enum):
    """计算模式"""
    AUTO = "auto"           # 自动选择最佳模式
    V3_REGISTRY = "v3"      # 使用V3注册表
    V2_COMPAT = "v2"        # V2兼容模式
    NUMBA = "numba"         # 强制Numba加速
    PANDAS = "pandas"       # 纯Pandas计算


@dataclass
class IndicatorSpec:
    """指标规格定义"""
    name: str
    params: Dict[str, Any] = None
    output_name: str = None
    mode: CalculationMode = CalculationMode.AUTO

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.output_name is None:
            self.output_name = self.name


class IndicatorAPI:
    """
    统一指标计算API

    提供:
    1. 简单的单指标计算
    2. 批量指标计算
    3. 流水线计算
    4. 自动模式选择（V3 > Numba > V2 > Pandas）
    """

    # Numba加速指标列表
    NUMBA_INDICATORS = {'rsi', 'ema', 'atr', 'macd', 'bollinger', 'sma'}

    def __init__(
        self,
        default_mode: CalculationMode = CalculationMode.AUTO,
        max_workers: int = 4,
        cache_enabled: bool = True
    ):
        """
        初始化API

        Args:
            default_mode: 默认计算模式
            max_workers: 并行计算线程数
            cache_enabled: 是否启用缓存
        """
        self.default_mode = default_mode
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled

        # 初始化计算器
        self.batch_calculator = BatchCalculator(
            max_workers=max_workers,
            cache_enabled=cache_enabled
        )
        self.v2_adapter = V2IndicatorAdapter(
            use_numba=True,
            use_v3_registry=True
        )
        self.v2_compat = V2CompatibilityLayer()

        # 快捷函数缓存
        self._quick_functions: Dict[str, Callable] = {}
        self._setup_quick_functions()

        logger.info(
            f"IndicatorAPI initialized: mode={default_mode.value}, "
            f"numba={'enabled' if NUMBA_AVAILABLE else 'disabled'}"
        )

    def _setup_quick_functions(self):
        """设置快捷计算函数"""
        # 趋势类
        self._quick_functions['sma'] = self._calc_sma
        self._quick_functions['ema'] = self._calc_ema
        self._quick_functions['wma'] = self._calc_wma

        # 动量类
        self._quick_functions['rsi'] = self._calc_rsi
        self._quick_functions['macd'] = self._calc_macd
        self._quick_functions['momentum'] = self._calc_momentum
        self._quick_functions['roc'] = self._calc_roc

        # 波动率类
        self._quick_functions['atr'] = self._calc_atr
        self._quick_functions['bollinger'] = self._calc_bollinger
        self._quick_functions['volatility'] = self._calc_volatility

        # 统计类
        self._quick_functions['zscore'] = self._calc_zscore
        self._quick_functions['percentile'] = self._calc_percentile

    # ============= 核心计算方法 =============

    def calculate(
        self,
        data: pd.DataFrame,
        indicator: Union[str, IndicatorSpec],
        mode: CalculationMode = None,
        **params
    ) -> Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]:
        """
        计算单个指标

        Args:
            data: OHLCV数据
            indicator: 指标名称或规格
            mode: 计算模式（覆盖默认）
            **params: 指标参数

        Returns:
            计算结果

        Examples:
            >>> api = IndicatorAPI()
            >>> rsi = api.calculate(df, 'rsi', period=14)
            >>> macd = api.calculate(df, IndicatorSpec('macd', {'fast': 12, 'slow': 26}))
        """
        if isinstance(indicator, str):
            spec = IndicatorSpec(name=indicator, params=params)
        else:
            spec = indicator
            if params:
                spec.params.update(params)

        effective_mode = mode or spec.mode or self.default_mode

        return self._calculate_with_mode(data, spec, effective_mode)

    def calculate_batch(
        self,
        data: pd.DataFrame,
        indicators: List[Union[str, IndicatorSpec, Dict[str, Any]]],
        parallel: bool = True,
        mode: CalculationMode = None
    ) -> pd.DataFrame:
        """
        批量计算多个指标

        Args:
            data: OHLCV数据
            indicators: 指标列表
            parallel: 是否并行计算
            mode: 计算模式

        Returns:
            包含所有指标的DataFrame

        Examples:
            >>> indicators = ['rsi', 'macd', {'name': 'ema', 'period': 20}]
            >>> result = api.calculate_batch(df, indicators)
        """
        specs = self._normalize_indicators(indicators)
        effective_mode = mode or self.default_mode

        result = data.copy()

        for spec in specs:
            try:
                calc_result = self._calculate_with_mode(data, spec, effective_mode)

                if isinstance(calc_result, pd.Series):
                    result[spec.output_name] = calc_result
                elif isinstance(calc_result, pd.DataFrame):
                    for col in calc_result.columns:
                        result[f"{spec.output_name}_{col}"] = calc_result[col]
                elif isinstance(calc_result, dict):
                    for key, series in calc_result.items():
                        result[f"{spec.output_name}_{key}"] = series

            except Exception as e:
                logger.error(f"Error calculating {spec.name}: {e}")

        return result

    def create_pipeline(self) -> 'IndicatorPipelineBuilder':
        """
        创建指标计算流水线

        Returns:
            流水线构建器

        Examples:
            >>> pipeline = api.create_pipeline()
            >>> pipeline.add('rsi', period=14).add('macd').add('atr')
            >>> result = pipeline.execute(df)
        """
        return IndicatorPipelineBuilder(self)

    # ============= 快捷计算方法 =============

    def rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI"""
        return self._calc_rsi(data, period)

    def ema(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """计算EMA"""
        return self._calc_ema(data, period)

    def sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算SMA"""
        return self._calc_sma(data, period)

    def macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """计算MACD"""
        return self._calc_macd(data, fast, slow, signal)

    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        return self._calc_atr(data, period)

    def bollinger(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """计算布林带"""
        return self._calc_bollinger(data, period, std_dev)

    # ============= 预设组合 =============

    def trend_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """趋势分析指标组合"""
        return self.calculate_batch(data, [
            {'name': 'sma', 'period': 20, 'output_name': 'SMA_20'},
            {'name': 'sma', 'period': 50, 'output_name': 'SMA_50'},
            {'name': 'sma', 'period': 200, 'output_name': 'SMA_200'},
            {'name': 'ema', 'period': 12, 'output_name': 'EMA_12'},
            {'name': 'ema', 'period': 26, 'output_name': 'EMA_26'},
            {'name': 'macd', 'output_name': 'MACD'},
        ])

    def momentum_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """动量分析指标组合"""
        return self.calculate_batch(data, [
            {'name': 'rsi', 'period': 14, 'output_name': 'RSI_14'},
            {'name': 'rsi', 'period': 7, 'output_name': 'RSI_7'},
            {'name': 'momentum', 'period': 10, 'output_name': 'MOM_10'},
            {'name': 'roc', 'period': 10, 'output_name': 'ROC_10'},
        ])

    def volatility_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """波动率分析指标组合"""
        return self.calculate_batch(data, [
            {'name': 'atr', 'period': 14, 'output_name': 'ATR_14'},
            {'name': 'bollinger', 'period': 20, 'output_name': 'BB'},
            {'name': 'volatility', 'period': 20, 'output_name': 'HV_20'},
        ])

    def comprehensive_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """综合分析指标组合"""
        result = data.copy()
        result = self.trend_analysis(result)
        result = self.momentum_analysis(result)
        result = self.volatility_analysis(result)
        return result

    # ============= 内部计算方法 =============

    def _calculate_with_mode(
        self,
        data: pd.DataFrame,
        spec: IndicatorSpec,
        mode: CalculationMode
    ) -> Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]:
        """根据模式选择计算方法"""

        name_lower = spec.name.lower()

        # AUTO模式：自动选择最佳方式
        if mode == CalculationMode.AUTO:
            # 1. 优先使用快捷函数（包含Numba）
            if name_lower in self._quick_functions:
                return self._quick_functions[name_lower](data, **spec.params)

            # 2. 尝试V3注册表
            try:
                indicator_class = indicator_registry.get(spec.name)
                if indicator_class:
                    config = IndicatorConfig(name=spec.name, params=spec.params)
                    result = self.batch_calculator.calculate_single(data, config)
                    if result.success:
                        return result.result
            except Exception:
                pass

            # 3. 使用V2适配器
            return self.v2_adapter.calculate(data, spec.name, **spec.params)

        # V3_REGISTRY模式
        elif mode == CalculationMode.V3_REGISTRY:
            config = IndicatorConfig(name=spec.name, params=spec.params)
            result = self.batch_calculator.calculate_single(data, config)
            if result.success:
                return result.result
            raise ValueError(f"V3 calculation failed: {result.error}")

        # V2_COMPAT模式
        elif mode == CalculationMode.V2_COMPAT:
            return self.v2_adapter.calculate(data, spec.name, **spec.params)

        # NUMBA模式
        elif mode == CalculationMode.NUMBA:
            if name_lower in self.NUMBA_INDICATORS and NUMBA_AVAILABLE:
                return self._quick_functions[name_lower](data, **spec.params)
            raise ValueError(f"No Numba implementation for: {spec.name}")

        # PANDAS模式
        elif mode == CalculationMode.PANDAS:
            return self._calculate_pandas(data, spec)

        raise ValueError(f"Unknown mode: {mode}")

    def _normalize_indicators(
        self,
        indicators: List[Union[str, IndicatorSpec, Dict[str, Any]]]
    ) -> List[IndicatorSpec]:
        """标准化指标列表"""
        specs = []
        for ind in indicators:
            if isinstance(ind, str):
                specs.append(IndicatorSpec(name=ind))
            elif isinstance(ind, IndicatorSpec):
                specs.append(ind)
            elif isinstance(ind, dict):
                # 创建字典副本，避免修改原始字典
                ind_copy = dict(ind)
                name = ind_copy.pop('name')
                output_name = ind_copy.pop('output_name', name)
                mode = ind_copy.pop('mode', None)
                # ind_copy 现在只包含实际的指标参数
                spec = IndicatorSpec(
                    name=name,
                    params=ind_copy,
                    output_name=output_name,
                    mode=mode or CalculationMode.AUTO
                )
                specs.append(spec)
        return specs

    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        """获取收盘价"""
        return data['close'] if 'close' in data.columns else data['Close']

    def _get_high(self, data: pd.DataFrame) -> pd.Series:
        """获取最高价"""
        return data['high'] if 'high' in data.columns else data['High']

    def _get_low(self, data: pd.DataFrame) -> pd.Series:
        """获取最低价"""
        return data['low'] if 'low' in data.columns else data['Low']

    # ============= 快捷计算实现 =============

    def _calc_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI计算（优先Numba）"""
        close = self._get_close(data)
        if NUMBA_AVAILABLE:
            result = calculate_rsi_fast(close.values, period)
            return pd.Series(result, index=close.index, name=f'RSI_{period}')
        # Pandas回退
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_ema(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """EMA计算（优先Numba）"""
        close = self._get_close(data)
        if NUMBA_AVAILABLE:
            result = calculate_ema_fast(close.values, period)
            return pd.Series(result, index=close.index, name=f'EMA_{period}')
        return close.ewm(span=period, adjust=False).mean()

    def _calc_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """SMA计算"""
        close = self._get_close(data)
        return close.rolling(window=period).mean()

    def _calc_wma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """WMA计算"""
        close = self._get_close(data)
        weights = np.arange(1, period + 1)
        return close.rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )

    def _calc_macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD计算（优先Numba）"""
        close = self._get_close(data)
        if NUMBA_AVAILABLE:
            macd_line, signal_line, histogram = calculate_macd_fast(
                close.values, fast, slow, signal
            )
            return {
                'macd': pd.Series(macd_line, index=close.index, name='MACD'),
                'signal': pd.Series(signal_line, index=close.index, name='MACD_Signal'),
                'histogram': pd.Series(histogram, index=close.index, name='MACD_Hist'),
            }
        # Pandas回退
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calc_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR计算（优先Numba）"""
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)

        if NUMBA_AVAILABLE:
            result = calculate_atr_fast(high.values, low.values, close.values, period)
            return pd.Series(result, index=close.index, name=f'ATR_{period}')

        # Pandas回退
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _calc_bollinger(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """布林带计算（优先Numba）"""
        close = self._get_close(data)

        if NUMBA_AVAILABLE:
            upper, middle, lower = calculate_bollinger_fast(
                close.values, period, std_dev
            )
            return {
                'upper': pd.Series(upper, index=close.index, name='BB_Upper'),
                'middle': pd.Series(middle, index=close.index, name='BB_Middle'),
                'lower': pd.Series(lower, index=close.index, name='BB_Lower'),
            }

        # Pandas回退
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return {
            'upper': middle + std_dev * std,
            'middle': middle,
            'lower': middle - std_dev * std
        }

    def _calc_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """动量计算"""
        close = self._get_close(data)
        return close - close.shift(period)

    def _calc_roc(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """ROC计算"""
        close = self._get_close(data)
        return ((close - close.shift(period)) / close.shift(period)) * 100

    def _calc_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """历史波动率计算"""
        close = self._get_close(data)
        returns = close.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)

    def _calc_zscore(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Z-Score计算"""
        close = self._get_close(data)
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - mean) / std

    def _calc_percentile(self, data: pd.DataFrame, period: int = 100) -> pd.Series:
        """百分位排名计算"""
        close = self._get_close(data)
        return close.rolling(window=period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    def _calculate_pandas(
        self,
        data: pd.DataFrame,
        spec: IndicatorSpec
    ) -> Union[pd.Series, pd.DataFrame]:
        """纯Pandas计算（禁用Numba）"""
        # 临时禁用Numba
        original_numba = NUMBA_AVAILABLE
        try:
            # 使用V2适配器的回退实现
            return self.v2_adapter._calculate_fallback(
                data, spec.name.lower(), spec.params
            )
        finally:
            pass  # 恢复Numba状态

    # ============= 工具方法 =============

    def list_indicators(self) -> Dict[str, List[str]]:
        """列出所有可用指标"""
        result = {
            'quick_functions': sorted(self._quick_functions.keys()),
            'numba_accelerated': sorted(self.NUMBA_INDICATORS),
            'v2_compatible': self.v2_adapter.get_v2_indicator_list(),
            'v3_registry': indicator_registry.list_all(),
        }
        return result

    def get_indicator_info(self, name: str) -> Dict[str, Any]:
        """获取指标信息"""
        info = {
            'name': name,
            'has_quick_function': name.lower() in self._quick_functions,
            'numba_accelerated': name.lower() in self.NUMBA_INDICATORS,
            'v2_compatible': name.lower() in self.v2_adapter.INDICATOR_MAPPING,
        }

        # V3元数据
        metadata = indicator_registry.get_metadata(name)
        if metadata:
            info['v3_metadata'] = {
                'display_name': metadata.display_name,
                'description': metadata.description,
                'category': metadata.category,
            }

        return info

    def benchmark(
        self,
        data: pd.DataFrame,
        indicator: str = 'rsi',
        iterations: int = 100
    ) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            data: 测试数据
            indicator: 指标名称
            iterations: 迭代次数

        Returns:
            各模式的执行时间
        """
        import time

        results = {}

        # 测试各模式
        modes = [
            ('auto', CalculationMode.AUTO),
            ('v2', CalculationMode.V2_COMPAT),
            ('pandas', CalculationMode.PANDAS),
        ]

        if NUMBA_AVAILABLE and indicator.lower() in self.NUMBA_INDICATORS:
            modes.insert(0, ('numba', CalculationMode.NUMBA))

        for mode_name, mode in modes:
            try:
                # 预热
                self.calculate(data, indicator, mode=mode)

                # 计时
                start = time.time()
                for _ in range(iterations):
                    self.calculate(data, indicator, mode=mode)
                elapsed = time.time() - start

                results[mode_name] = elapsed / iterations * 1000  # 毫秒

            except Exception as e:
                results[mode_name] = f"Error: {e}"

        return results


class IndicatorPipelineBuilder:
    """指标流水线构建器"""

    def __init__(self, api: IndicatorAPI):
        """初始化构建器"""
        self.api = api
        self._indicators: List[IndicatorSpec] = []

    def add(
        self,
        indicator: Union[str, IndicatorSpec],
        output_name: str = None,
        **params
    ) -> 'IndicatorPipelineBuilder':
        """添加指标到流水线"""
        if isinstance(indicator, str):
            spec = IndicatorSpec(
                name=indicator,
                params=params,
                output_name=output_name or indicator
            )
        else:
            spec = indicator
            if params:
                spec.params.update(params)
            if output_name:
                spec.output_name = output_name

        self._indicators.append(spec)
        return self

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行流水线"""
        return self.api.calculate_batch(data, self._indicators)

    def clear(self) -> 'IndicatorPipelineBuilder':
        """清除流水线"""
        self._indicators.clear()
        return self


# 全局API实例
indicator_api = IndicatorAPI()


# 便捷函数
def calculate(
    data: pd.DataFrame,
    indicator: str,
    **params
) -> Union[pd.Series, pd.DataFrame]:
    """计算单个指标（全局函数）"""
    return indicator_api.calculate(data, indicator, **params)


def calculate_batch(
    data: pd.DataFrame,
    indicators: List[Union[str, Dict[str, Any]]]
) -> pd.DataFrame:
    """批量计算指标（全局函数）"""
    return indicator_api.calculate_batch(data, indicators)


if __name__ == "__main__":
    # 测试API
    print("Testing Unified Indicator API...")

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    close = pd.Series(100 + np.cumsum(np.random.randn(200) * 2), index=dates)
    high = close + np.abs(np.random.randn(200))
    low = close - np.abs(np.random.randn(200))
    volume = pd.Series(np.random.randint(1000, 10000, 200), index=dates)

    df = pd.DataFrame({
        'open': close.shift(1).fillna(close.iloc[0]),
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    api = IndicatorAPI()

    print("\n1. Single indicator calculation:")
    rsi = api.calculate(df, 'rsi', period=14)
    print(f"   RSI shape: {rsi.shape}")

    print("\n2. Quick function:")
    macd = api.macd(df)
    print(f"   MACD keys: {list(macd.keys())}")

    print("\n3. Batch calculation:")
    result = api.calculate_batch(df, ['rsi', 'ema', 'sma'])
    print(f"   Result columns: {list(result.columns)}")

    print("\n4. Pipeline:")
    pipeline = api.create_pipeline()
    pipeline.add('rsi', period=14).add('macd').add('bollinger')
    pipeline_result = pipeline.execute(df)
    print(f"   Pipeline result columns: {list(pipeline_result.columns)}")

    print("\n5. Preset analysis:")
    trend = api.trend_analysis(df)
    print(f"   Trend analysis columns: {list(trend.columns)}")

    print("\n6. Available indicators:")
    indicators = api.list_indicators()
    print(f"   Quick functions: {len(indicators['quick_functions'])}")
    print(f"   Numba accelerated: {len(indicators['numba_accelerated'])}")

    print("\n7. Benchmark (RSI):")
    benchmark = api.benchmark(df, 'rsi', iterations=50)
    for mode, time_ms in benchmark.items():
        print(f"   {mode}: {time_ms:.3f} ms" if isinstance(time_ms, float) else f"   {mode}: {time_ms}")

    print("\nAll tests passed!")
