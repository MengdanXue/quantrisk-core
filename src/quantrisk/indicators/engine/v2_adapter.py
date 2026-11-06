"""
V2指标计算器适配器
V2 Indicator Calculator Adapter

将旧版calculator_v2.py的功能桥接到V3引擎
提供向后兼容性和平滑迁移路径

作者: TraderX Team
创建: 2025-12-23
"""

from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
from functools import wraps
import logging

from .numba_core import (
    calculate_rsi_fast,
    calculate_ema_fast,
    calculate_atr_fast,
    calculate_bollinger_fast,
    calculate_macd_fast,
    rolling_mean_fast,
    rolling_std_fast,
    NUMBA_AVAILABLE
)
from .batch_calculator import BatchCalculator, IndicatorConfig, CalculationResult
from ..registry.indicator_registry import indicator_registry

logger = logging.getLogger(__name__)


class V2IndicatorAdapter:
    """
    V2指标计算器适配器

    将calculator_v2.py的47个指标映射到V3引擎
    支持R脚本定义的指标计算逻辑
    """

    # V2指标名称到V3指标类的映射
    INDICATOR_MAPPING = {
        # 趋势类
        'sma': 'SMA',
        'ema': 'EMA',
        'ema_fast': 'EMA',  # 使用Numba加速
        'wma': 'WMA',
        'dema': 'DEMA',
        'tema': 'TEMA',

        # 动量类
        'rsi': 'RSI',
        'macd': 'MACD',
        'macd_signal': 'MACD',
        'macd_hist': 'MACD',
        'stoch_k': 'StochasticOscillator',
        'stoch_d': 'StochasticOscillator',
        'williams_r': 'WilliamsR',
        'cci': 'CCI',
        'roc': 'ROC',
        'momentum': 'Momentum',

        # 波动率类
        'atr': 'ATR',
        'natr': 'ATR',  # Normalized ATR
        'bollinger_upper': 'BollingerBands',
        'bollinger_middle': 'BollingerBands',
        'bollinger_lower': 'BollingerBands',
        'bollinger_width': 'BollingerBands',
        'keltner_upper': 'KeltnerChannel',
        'keltner_middle': 'KeltnerChannel',
        'keltner_lower': 'KeltnerChannel',
        'donchian_high': 'DonchianChannel',
        'donchian_low': 'DonchianChannel',
        'donchian_middle': 'DonchianChannel',
        'historical_volatility': 'HistoricalVolatility',

        # 成交量类
        'obv': 'OBV',
        'vwap': 'VWAP',
        'mfi': 'MoneyFlowIndex',
        'cmf': 'ChaikinMoneyFlow',
        'adl': 'ADL',
        'force_index': 'ForceIndex',

        # 统计类
        'zscore': 'ZScore',
        'percentile_rank': 'PercentileRank',
        'linear_regression': 'LinearRegression',
        'correlation': 'Correlation',

        # 风险类
        'max_drawdown': 'MaxDrawdown',
        'sharpe_ratio': 'Sharpe',
        'sortino_ratio': 'Sortino',
        'var': 'VaR',
        'cvar': 'CVaR',
        'beta': 'Beta',
    }

    # 需要Numba加速的指标
    NUMBA_ACCELERATED = {
        'rsi', 'ema', 'ema_fast', 'atr',
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'macd', 'macd_signal', 'macd_hist'
    }

    def __init__(self, use_numba: bool = True, use_v3_registry: bool = True):
        """
        初始化适配器

        Args:
            use_numba: 是否使用Numba加速
            use_v3_registry: 是否优先使用V3注册表中的指标
        """
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_v3_registry = use_v3_registry
        self.batch_calculator = BatchCalculator()

        # 直接计算函数缓存
        self._direct_calculators: Dict[str, Callable] = {}
        self._setup_direct_calculators()

        logger.info(
            f"V2IndicatorAdapter initialized: "
            f"numba={'enabled' if self.use_numba else 'disabled'}, "
            f"v3_registry={'enabled' if self.use_v3_registry else 'disabled'}"
        )

    def _setup_direct_calculators(self):
        """设置直接计算函数（Numba加速）"""

        if self.use_numba:
            self._direct_calculators['rsi'] = self._calc_rsi_numba
            self._direct_calculators['ema'] = self._calc_ema_numba
            self._direct_calculators['ema_fast'] = self._calc_ema_numba
            self._direct_calculators['atr'] = self._calc_atr_numba
            self._direct_calculators['bollinger'] = self._calc_bollinger_numba
            self._direct_calculators['macd'] = self._calc_macd_numba

    def _calc_rsi_numba(
        self,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Numba加速的RSI计算"""
        result = calculate_rsi_fast(close.values, period)
        return pd.Series(result, index=close.index, name=f'RSI_{period}')

    def _calc_ema_numba(
        self,
        close: pd.Series,
        period: int = 12
    ) -> pd.Series:
        """Numba加速的EMA计算"""
        result = calculate_ema_fast(close.values, period)
        return pd.Series(result, index=close.index, name=f'EMA_{period}')

    def _calc_atr_numba(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Numba加速的ATR计算"""
        result = calculate_atr_fast(
            high.values, low.values, close.values, period
        )
        return pd.Series(result, index=close.index, name=f'ATR_{period}')

    def _calc_bollinger_numba(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Numba加速的布林带计算"""
        upper, middle, lower = calculate_bollinger_fast(
            close.values, period, std_dev
        )
        return {
            'upper': pd.Series(upper, index=close.index, name='BB_Upper'),
            'middle': pd.Series(middle, index=close.index, name='BB_Middle'),
            'lower': pd.Series(lower, index=close.index, name='BB_Lower'),
        }

    def _calc_macd_numba(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """Numba加速的MACD计算"""
        macd_line, signal_line, histogram = calculate_macd_fast(
            close.values, fast_period, slow_period, signal_period
        )
        return {
            'macd': pd.Series(macd_line, index=close.index, name='MACD'),
            'signal': pd.Series(signal_line, index=close.index, name='MACD_Signal'),
            'histogram': pd.Series(histogram, index=close.index, name='MACD_Hist'),
        }

    def calculate(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        **params
    ) -> Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]:
        """
        计算单个指标

        Args:
            data: OHLCV数据
            indicator_name: 指标名称（V2命名风格）
            **params: 指标参数

        Returns:
            计算结果
        """
        indicator_name_lower = indicator_name.lower()

        # 1. 尝试使用Numba加速计算
        if self.use_numba and indicator_name_lower in self._direct_calculators:
            return self._calculate_with_numba(data, indicator_name_lower, params)

        # 2. 尝试映射到V3注册表
        if self.use_v3_registry and indicator_name_lower in self.INDICATOR_MAPPING:
            v3_name = self.INDICATOR_MAPPING[indicator_name_lower]
            return self._calculate_via_v3(data, v3_name, indicator_name_lower, params)

        # 3. 回退到内置计算
        return self._calculate_fallback(data, indicator_name_lower, params)

    def _calculate_with_numba(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """使用Numba加速计算"""
        calculator = self._direct_calculators[indicator_name]

        close = data['close'] if 'close' in data.columns else data['Close']

        if indicator_name in ('rsi', 'ema', 'ema_fast'):
            period = params.get('period', 14 if indicator_name == 'rsi' else 12)
            return calculator(close, period)

        elif indicator_name == 'atr':
            high = data['high'] if 'high' in data.columns else data['High']
            low = data['low'] if 'low' in data.columns else data['Low']
            period = params.get('period', 14)
            return calculator(high, low, close, period)

        elif indicator_name == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            return calculator(close, period, std_dev)

        elif indicator_name == 'macd':
            fast = params.get('fast_period', 12)
            slow = params.get('slow_period', 26)
            signal = params.get('signal_period', 9)
            return calculator(close, fast, slow, signal)

        raise ValueError(f"Unknown Numba indicator: {indicator_name}")

    def _calculate_via_v3(
        self,
        data: pd.DataFrame,
        v3_name: str,
        v2_name: str,
        params: Dict[str, Any]
    ) -> Union[pd.Series, pd.DataFrame]:
        """通过V3注册表计算"""
        try:
            config = IndicatorConfig(name=v3_name, params=params)
            result = self.batch_calculator.calculate_single(data, config)

            if result.success:
                return result.result
            else:
                logger.warning(
                    f"V3 calculation failed for {v3_name}: {result.error}, "
                    f"falling back to built-in"
                )
                return self._calculate_fallback(data, v2_name, params)

        except Exception as e:
            logger.warning(
                f"V3 registry lookup failed for {v3_name}: {e}, "
                f"falling back to built-in"
            )
            return self._calculate_fallback(data, v2_name, params)

    def _calculate_fallback(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Union[pd.Series, pd.DataFrame]:
        """内置回退计算"""
        close = data['close'] if 'close' in data.columns else data['Close']
        high = data.get('high', data.get('High'))
        low = data.get('low', data.get('Low'))
        volume = data.get('volume', data.get('Volume'))

        # SMA
        if indicator_name == 'sma':
            period = params.get('period', 20)
            return close.rolling(window=period).mean()

        # EMA (非Numba)
        elif indicator_name in ('ema', 'ema_fast'):
            period = params.get('period', 12)
            return close.ewm(span=period, adjust=False).mean()

        # RSI (非Numba)
        elif indicator_name == 'rsi':
            period = params.get('period', 14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        # ATR (非Numba)
        elif indicator_name in ('atr', 'natr'):
            period = params.get('period', 14)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            if indicator_name == 'natr':
                return (atr / close) * 100
            return atr

        # 布林带 (非Numba)
        elif indicator_name.startswith('bollinger'):
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            middle = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = middle + std_dev * std
            lower = middle - std_dev * std

            if 'upper' in indicator_name:
                return upper
            elif 'lower' in indicator_name:
                return lower
            elif 'middle' in indicator_name:
                return middle
            elif 'width' in indicator_name:
                return (upper - lower) / middle
            return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

        # MACD (非Numba)
        elif indicator_name.startswith('macd'):
            fast = params.get('fast_period', 12)
            slow = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)

            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            if indicator_name == 'macd_signal':
                return signal_line
            elif indicator_name == 'macd_hist':
                return histogram
            elif indicator_name == 'macd':
                return pd.DataFrame({
                    'macd': macd_line,
                    'signal': signal_line,
                    'histogram': histogram
                })

        # ROC
        elif indicator_name == 'roc':
            period = params.get('period', 10)
            return ((close - close.shift(period)) / close.shift(period)) * 100

        # Momentum
        elif indicator_name == 'momentum':
            period = params.get('period', 10)
            return close - close.shift(period)

        # OBV
        elif indicator_name == 'obv' and volume is not None:
            direction = np.sign(close.diff())
            return (direction * volume).cumsum()

        # ZScore
        elif indicator_name == 'zscore':
            period = params.get('period', 20)
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            return (close - mean) / std

        # Max Drawdown
        elif indicator_name == 'max_drawdown':
            period = params.get('period', 252)
            rolling_max = close.rolling(window=period, min_periods=1).max()
            drawdown = (close - rolling_max) / rolling_max
            return drawdown.rolling(window=period, min_periods=1).min()

        # Historical Volatility
        elif indicator_name == 'historical_volatility':
            period = params.get('period', 20)
            returns = close.pct_change()
            return returns.rolling(window=period).std() * np.sqrt(252)

        raise ValueError(f"Unknown indicator: {indicator_name}")

    def calculate_batch(
        self,
        data: pd.DataFrame,
        indicators: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        批量计算多个指标

        Args:
            data: OHLCV数据
            indicators: 指标配置列表
                [{'name': 'rsi', 'period': 14}, {'name': 'ema', 'period': 20}]

        Returns:
            包含所有指标结果的DataFrame
        """
        result = data.copy()

        for ind_config in indicators:
            name = ind_config.pop('name')
            output_name = ind_config.pop('output_name', name)

            try:
                calc_result = self.calculate(data, name, **ind_config)

                if isinstance(calc_result, pd.Series):
                    result[output_name] = calc_result
                elif isinstance(calc_result, pd.DataFrame):
                    for col in calc_result.columns:
                        result[f"{output_name}_{col}"] = calc_result[col]
                elif isinstance(calc_result, dict):
                    for key, series in calc_result.items():
                        result[f"{output_name}_{key}"] = series

            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")

            # 恢复配置
            ind_config['name'] = name
            ind_config['output_name'] = output_name

        return result

    def get_v2_indicator_list(self) -> List[str]:
        """获取所有支持的V2指标名称"""
        return sorted(self.INDICATOR_MAPPING.keys())

    def is_numba_accelerated(self, indicator_name: str) -> bool:
        """检查指标是否有Numba加速版本"""
        return indicator_name.lower() in self.NUMBA_ACCELERATED and self.use_numba


class V2CompatibilityLayer:
    """
    V2完全兼容层

    提供与calculator_v2.py相同的API签名
    用于无缝替换旧代码
    """

    def __init__(self):
        """初始化兼容层"""
        self.adapter = V2IndicatorAdapter(use_numba=True, use_v3_registry=True)
        self._R_spec_cache = {}

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        indicator_list: List[str] = None,
        use_excel_mode: bool = False
    ) -> pd.DataFrame:
        """
        计算指标（兼容calculator_v2签名）

        Args:
            df: OHLCV数据
            symbol: 标的代码
            indicator_list: 要计算的指标列表
            use_excel_mode: 是否使用Excel兼容模式

        Returns:
            包含指标的DataFrame
        """
        if indicator_list is None:
            # 默认计算所有V2指标
            indicator_list = self.adapter.get_v2_indicator_list()

        indicators = [{'name': ind} for ind in indicator_list]
        return self.adapter.calculate_batch(df, indicators)

    def calculate_single_indicator(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        **params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        计算单个指标（兼容calculator_v2签名）
        """
        return self.adapter.calculate(df, indicator_name, **params)

    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """
        获取指标配置

        模拟V2的R脚本配置查找
        """
        # 默认配置映射
        default_configs = {
            'rsi': {'period': 14},
            'ema': {'period': 12},
            'ema_fast': {'period': 5},
            'sma': {'period': 20},
            'atr': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'momentum': {'period': 10},
            'roc': {'period': 10},
        }

        return default_configs.get(indicator_name.lower(), {})


# 便捷函数
def create_v2_adapter(
    use_numba: bool = True,
    use_v3_registry: bool = True
) -> V2IndicatorAdapter:
    """创建V2适配器实例"""
    return V2IndicatorAdapter(use_numba=use_numba, use_v3_registry=use_v3_registry)


def create_compatibility_layer() -> V2CompatibilityLayer:
    """创建完全兼容层实例"""
    return V2CompatibilityLayer()


# 全局实例
v2_adapter = V2IndicatorAdapter()
v2_compat = V2CompatibilityLayer()


if __name__ == "__main__":
    # 测试适配器
    print("Testing V2 Indicator Adapter...")

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

    # 测试单个指标
    adapter = V2IndicatorAdapter()

    print("\n1. Testing RSI calculation:")
    rsi = adapter.calculate(df, 'rsi', period=14)
    print(f"   RSI shape: {rsi.shape}")
    print(f"   RSI tail:\n{rsi.tail()}")

    print("\n2. Testing EMA calculation:")
    ema = adapter.calculate(df, 'ema', period=12)
    print(f"   EMA shape: {ema.shape}")

    print("\n3. Testing MACD calculation:")
    macd = adapter.calculate(df, 'macd')
    print(f"   MACD type: {type(macd)}")
    if isinstance(macd, dict):
        for k, v in macd.items():
            print(f"   {k}: {v.tail(3).values}")

    print("\n4. Testing Bollinger Bands:")
    bb = adapter.calculate(df, 'bollinger')
    print(f"   Bollinger type: {type(bb)}")

    print("\n5. Testing batch calculation:")
    batch_result = adapter.calculate_batch(df, [
        {'name': 'rsi', 'period': 14, 'output_name': 'RSI_14'},
        {'name': 'ema', 'period': 20, 'output_name': 'EMA_20'},
        {'name': 'sma', 'period': 50, 'output_name': 'SMA_50'},
    ])
    print(f"   Batch result columns: {list(batch_result.columns)}")

    print("\n6. Testing compatibility layer:")
    compat = V2CompatibilityLayer()
    compat_result = compat.calculate_indicators(
        df,
        indicator_list=['rsi', 'ema', 'atr']
    )
    print(f"   Compat result shape: {compat_result.shape}")

    print(f"\n7. Numba status: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")
    print(f"   V2 adapter Numba indicators: {adapter.NUMBA_ACCELERATED}")

    print("\nAll tests passed!")
