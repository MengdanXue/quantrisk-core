"""
Numba JIT加速核心计算
Numba JIT Accelerated Core Calculations

使用Numba加速关键性能瓶颈的指标计算：
- RSI (高频使用)
- EMA (基础依赖)
- ATR (循环依赖)
- Bollinger Bands (滚动窗口)
- MACD (复杂计算)

作者: TraderX Team
创建: 2025-12-23
"""
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 尝试导入numba，如果不可用则使用纯Python实现
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logger.info("Numba JIT acceleration enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using pure Python implementations")

    # 创建一个空的装饰器作为fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


# ==================== RSI ====================

@jit(nopython=True, cache=True)
def _rsi_core_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Numba加速的RSI核心计算

    使用Wilder平滑方法计算RSI

    Args:
        prices: 价格数组
        period: RSI周期

    Returns:
        RSI值数组
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    # 计算价格变化
    deltas = np.diff(prices)

    # 分离涨跌
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # 初始平均值（简单平均）
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # 计算初始RSI
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder平滑法计算后续RSI
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    高性能RSI计算

    Args:
        prices: 价格数组
        period: RSI周期

    Returns:
        RSI值数组
    """
    if NUMBA_AVAILABLE:
        return _rsi_core_numba(prices.astype(np.float64), period)
    else:
        return _rsi_pure_python(prices, period)


def _rsi_pure_python(prices: np.ndarray, period: int) -> np.ndarray:
    """纯Python RSI实现（Numba不可用时的fallback）"""
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


# ==================== EMA ====================

@jit(nopython=True, cache=True)
def _ema_core_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Numba加速的EMA计算

    Args:
        prices: 价格数组
        period: EMA周期

    Returns:
        EMA值数组
    """
    n = len(prices)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    # 计算平滑因子
    alpha = 2.0 / (period + 1)

    # 初始值使用SMA
    ema[period - 1] = np.mean(prices[:period])

    # 计算EMA
    for i in range(period, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """
    高性能EMA计算

    Args:
        prices: 价格数组
        period: EMA周期

    Returns:
        EMA值数组
    """
    if NUMBA_AVAILABLE:
        return _ema_core_numba(prices.astype(np.float64), period)
    else:
        return _ema_pure_python(prices, period)


def _ema_pure_python(prices: np.ndarray, period: int) -> np.ndarray:
    """纯Python EMA实现"""
    n = len(prices)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(prices[:period])

    for i in range(period, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


# ==================== ATR ====================

@jit(nopython=True, cache=True)
def _atr_core_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Numba加速的ATR计算

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR周期

    Returns:
        ATR值数组
    """
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)

    if n < period + 1:
        return atr

    # 计算True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # 初始ATR（SMA）
    atr[period] = np.mean(tr[1:period + 1])

    # Wilder平滑
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calculate_atr_fast(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    高性能ATR计算

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR周期

    Returns:
        ATR值数组
    """
    if NUMBA_AVAILABLE:
        return _atr_core_numba(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            period
        )
    else:
        return _atr_pure_python(high, low, close, period)


def _atr_pure_python(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """纯Python ATR实现"""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)

    if n < period + 1:
        return atr

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ==================== Bollinger Bands ====================

@jit(nopython=True, cache=True)
def _bollinger_core_numba(
    prices: np.ndarray,
    period: int,
    std_dev: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba加速的布林带计算

    Args:
        prices: 价格数组
        period: 周期
        std_dev: 标准差倍数

    Returns:
        (upper, middle, lower) 布林带三条线
    """
    n = len(prices)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    if n < period:
        return upper, middle, lower

    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)

        middle[i] = mean
        upper[i] = mean + std_dev * std
        lower[i] = mean - std_dev * std

    return upper, middle, lower


def calculate_bollinger_fast(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    高性能布林带计算

    Args:
        prices: 价格数组
        period: 周期
        std_dev: 标准差倍数

    Returns:
        (upper, middle, lower) 布林带三条线
    """
    if NUMBA_AVAILABLE:
        return _bollinger_core_numba(prices.astype(np.float64), period, std_dev)
    else:
        return _bollinger_pure_python(prices, period, std_dev)


def _bollinger_pure_python(
    prices: np.ndarray,
    period: int,
    std_dev: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯Python布林带实现"""
    n = len(prices)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    if n < period:
        return upper, middle, lower

    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)

        middle[i] = mean
        upper[i] = mean + std_dev * std
        lower[i] = mean - std_dev * std

    return upper, middle, lower


# ==================== MACD ====================

@jit(nopython=True, cache=True)
def _macd_core_numba(
    prices: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba加速的MACD计算

    Args:
        prices: 价格数组
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期

    Returns:
        (macd_line, signal_line, histogram)
    """
    n = len(prices)
    macd_line = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    histogram = np.full(n, np.nan)

    if n < slow_period:
        return macd_line, signal_line, histogram

    # 计算EMA
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)

    fast_ema = np.full(n, np.nan)
    slow_ema = np.full(n, np.nan)

    # 初始化快线EMA
    fast_ema[fast_period - 1] = np.mean(prices[:fast_period])
    for i in range(fast_period, n):
        fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i - 1]

    # 初始化慢线EMA
    slow_ema[slow_period - 1] = np.mean(prices[:slow_period])
    for i in range(slow_period, n):
        slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i - 1]

    # 计算MACD线
    for i in range(slow_period - 1, n):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # 计算信号线
    # 找到第一个有效的MACD值
    first_valid = slow_period - 1
    while first_valid < n and np.isnan(macd_line[first_valid]):
        first_valid += 1

    if first_valid + signal_period <= n:
        # 初始信号线
        signal_start = first_valid + signal_period - 1
        valid_macd = macd_line[first_valid:signal_start + 1]
        valid_macd = valid_macd[~np.isnan(valid_macd)]
        if len(valid_macd) >= signal_period:
            signal_line[signal_start] = np.mean(valid_macd[-signal_period:])

            for i in range(signal_start + 1, n):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = signal_alpha * macd_line[i] + (1 - signal_alpha) * signal_line[i - 1]

    # 计算柱状图
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


def calculate_macd_fast(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    高性能MACD计算

    Args:
        prices: 价格数组
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期

    Returns:
        (macd_line, signal_line, histogram)
    """
    if NUMBA_AVAILABLE:
        return _macd_core_numba(
            prices.astype(np.float64),
            fast_period, slow_period, signal_period
        )
    else:
        return _macd_pure_python(prices, fast_period, slow_period, signal_period)


def _macd_pure_python(
    prices: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯Python MACD实现"""
    fast_ema = _ema_pure_python(prices, fast_period)
    slow_ema = _ema_pure_python(prices, slow_period)

    macd_line = fast_ema - slow_ema

    # 信号线
    valid_macd = macd_line[~np.isnan(macd_line)]
    if len(valid_macd) >= signal_period:
        signal_ema = _ema_pure_python(valid_macd, signal_period)
        signal_line = np.full(len(prices), np.nan)

        # 放回原位置
        valid_indices = np.where(~np.isnan(macd_line))[0]
        for i, idx in enumerate(valid_indices):
            if i < len(signal_ema):
                signal_line[idx] = signal_ema[i]
    else:
        signal_line = np.full(len(prices), np.nan)

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ==================== Rolling Statistics ====================

@jit(nopython=True, cache=True)
def _rolling_mean_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滚动均值计算
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        result[i] = np.mean(window_data)

    return result


def rolling_mean_fast(data: np.ndarray, window: int) -> np.ndarray:
    """高性能滚动均值"""
    if NUMBA_AVAILABLE:
        return _rolling_mean_numba(data.astype(np.float64), window)
    else:
        import pandas as pd
        return pd.Series(data).rolling(window).mean().values


def rolling_std_fast(data: np.ndarray, window: int) -> np.ndarray:
    """高性能滚动标准差"""
    if NUMBA_AVAILABLE:
        return _rolling_std_numba(data.astype(np.float64), window)
    else:
        import pandas as pd
        return pd.Series(data).rolling(window).std().values


@jit(nopython=True, cache=True, parallel=True)
def _rolling_std_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滚动标准差计算

    使用样本标准差(ddof=1)以匹配pandas行为
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in prange(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        mean = np.mean(window_data)
        # 使用样本方差 (ddof=1): 除以 (n-1) 而非 n
        var = np.sum((window_data - mean) ** 2) / (window - 1)
        result[i] = np.sqrt(var)

    return result


@jit(nopython=True, cache=True)
def _rolling_percentile_numba(
    data: np.ndarray,
    window: int,
    percentile: float
) -> np.ndarray:
    """
    Numba加速的滚动百分位计算
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        sorted_data = np.sort(window_data)
        idx = int(percentile * (window - 1))
        result[i] = sorted_data[idx]

    return result


def calculate_rolling_std_fast(data: np.ndarray, window: int) -> np.ndarray:
    """高性能滚动标准差"""
    if NUMBA_AVAILABLE:
        return _rolling_std_numba(data.astype(np.float64), window)
    else:
        import pandas as pd
        return pd.Series(data).rolling(window).std().values


def calculate_rolling_percentile_fast(
    data: np.ndarray,
    window: int,
    percentile: float
) -> np.ndarray:
    """高性能滚动百分位"""
    if NUMBA_AVAILABLE:
        return _rolling_percentile_numba(data.astype(np.float64), window, percentile)
    else:
        import pandas as pd
        return pd.Series(data).rolling(window).quantile(percentile).values


# ==================== 性能基准测试 ====================

def benchmark_numba():
    """运行性能基准测试"""
    import time

    print("=" * 60)
    print("Numba JIT Performance Benchmark")
    print("=" * 60)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()

    # 生成测试数据
    np.random.seed(42)
    n = 10000
    prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.02, n))
    high = prices * (1 + np.random.uniform(0, 0.02, n))
    low = prices * (1 - np.random.uniform(0, 0.02, n))

    tests = [
        ("RSI (14)", lambda: calculate_rsi_fast(prices, 14)),
        ("EMA (20)", lambda: calculate_ema_fast(prices, 20)),
        ("ATR (14)", lambda: calculate_atr_fast(high, low, prices, 14)),
        ("Bollinger (20)", lambda: calculate_bollinger_fast(prices, 20, 2.0)),
        ("MACD (12,26,9)", lambda: calculate_macd_fast(prices, 12, 26, 9)),
        ("Rolling Std (20)", lambda: calculate_rolling_std_fast(prices, 20)),
    ]

    for name, func in tests:
        # 预热
        func()

        # 计时
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        elapsed = (time.perf_counter() - start) / iterations * 1000

        print(f"{name:20s}: {elapsed:8.3f} ms per call")

    print()
    print("=" * 60)


if __name__ == "__main__":
    benchmark_numba()
