"""quantrisk 指标与 TA-Lib 的数值对照表（tests/test_validation.py 与 docs/VALIDATION.md 共用）。

只收录已确认一致的项；口径不同的指标在 CONVENTION_CASES 中对照它们自己声明的公式。

每一项：(指标名, 输出列或 None, TA-Lib 计算函数, 构造参数)。
TA-Lib 函数接收 DataFrame，返回与我方输出对齐的 Series。
"""
import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


def _s(df, values):
    return pd.Series(values, index=df.index)


def _ohlc(df):
    return df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()


CASES = [
    # 趋势
    ("SMA", None, lambda d: _s(d, talib.SMA(d["close"].to_numpy(), 20)), {}),
    ("EMA", None, lambda d: _s(d, talib.EMA(d["close"].to_numpy(), 20)), {}),
    ("WMA", None, lambda d: _s(d, talib.WMA(d["close"].to_numpy(), 20)), {}),
    ("DEMA", None, lambda d: _s(d, talib.DEMA(d["close"].to_numpy(), 20)), {}),
    ("TEMA", None, lambda d: _s(d, talib.TEMA(d["close"].to_numpy(), 20)), {}),
    ("TRIMA", None, lambda d: _s(d, talib.TRIMA(d["close"].to_numpy(), 20)), {}),
    ("KAMA", None, lambda d: _s(d, talib.KAMA(d["close"].to_numpy(), 10)), {}),
    ("T3", None, lambda d: _s(d, talib.T3(d["close"].to_numpy(), 20, 0.7)), {}),
    ("ADX", "ADX", lambda d: _s(d, talib.ADX(*_ohlc(d), 14)), {}),
    ("DI_Plus", None, lambda d: _s(d, talib.PLUS_DI(*_ohlc(d), 14)), {}),
    ("DI_Minus", None, lambda d: _s(d, talib.MINUS_DI(*_ohlc(d), 14)), {}),
    ("Aroon", "Aroon_Up", lambda d: _s(d, talib.AROON(d["high"].to_numpy(), d["low"].to_numpy(), 25)[1]), {}),
    ("Aroon", "Aroon_Down", lambda d: _s(d, talib.AROON(d["high"].to_numpy(), d["low"].to_numpy(), 25)[0]), {}),
    ("AroonOscillator", None, lambda d: _s(d, talib.AROONOSC(d["high"].to_numpy(), d["low"].to_numpy(), 25)), {}),
    ("TRIX", "TRIX", lambda d: _s(d, talib.TRIX(d["close"].to_numpy(), 15)), {}),
    ("LinearRegressionLine", "LR_Value", lambda d: _s(d, talib.LINEARREG(d["close"].to_numpy(), 20)), {}),
    ("LinearRegressionLine", "LR_Slope", lambda d: _s(d, talib.LINEARREG_SLOPE(d["close"].to_numpy(), 20)), {}),
    # 动量
    ("CCI", None, lambda d: _s(d, talib.CCI(*_ohlc(d), 20)), {}),
    ("MACD", "MACD", lambda d: _s(d, talib.MACD(d["close"].to_numpy(), 12, 26, 9)[0]), {}),
    ("MACD", "MACD_Signal", lambda d: _s(d, talib.MACD(d["close"].to_numpy(), 12, 26, 9)[1]), {}),
    ("MACD", "MACD_Histogram", lambda d: _s(d, talib.MACD(d["close"].to_numpy(), 12, 26, 9)[2]), {}),
    ("Momentum", None, lambda d: _s(d, talib.MOM(d["close"].to_numpy(), 10)), {}),
    ("ROC", None, lambda d: _s(d, talib.ROC(d["close"].to_numpy(), 10)), {}),
    ("RSI", None, lambda d: _s(d, talib.RSI(d["close"].to_numpy(), 14)), {}),
    ("StochasticOscillator", "Stoch_K", lambda d: _s(d, talib.STOCH(*_ohlc(d), 14, 3, 0, 3, 0)[0]), {}),
    ("StochasticOscillator", "Stoch_D", lambda d: _s(d, talib.STOCH(*_ohlc(d), 14, 3, 0, 3, 0)[1]), {}),
    ("StochasticOscillator", "Fast_K", lambda d: _s(d, talib.STOCHF(*_ohlc(d), 14, 3, 0)[0]), {}),
    ("UltimateOscillator", None, lambda d: _s(d, talib.ULTOSC(*_ohlc(d), 7, 14, 28)), {}),
    ("WilliamsR", None, lambda d: _s(d, talib.WILLR(*_ohlc(d), 14)), {}),
    ("LinearRegressionSlope", None, lambda d: _s(d, talib.LINEARREG_SLOPE(d["close"].to_numpy(), 14)), {}),
    ("PPOMACD", "PPO", lambda d: _s(d, talib.PPO(d["close"].to_numpy(), 12, 26, matype=1)), {}),
    # 波动率
    ("TrueRange", None, lambda d: _s(d, talib.TRANGE(*_ohlc(d))), {}),
    ("BollingerBands", "BB_Middle", lambda d: _s(d, talib.BBANDS(d["close"].to_numpy(), 20, 2, 2, 0)[1]), {}),
    # 成交量
    ("AccumulationDistribution", None, lambda d: _s(d, talib.AD(*_ohlc(d), d["volume"].to_numpy())), {}),
    ("ChaikinOscillator", None, lambda d: _s(d, talib.ADOSC(*_ohlc(d), d["volume"].to_numpy(), 3, 10)), {}),
    ("MoneyFlowIndex", None, lambda d: _s(d, talib.MFI(*_ohlc(d), d["volume"].to_numpy(), 14)), {}),
    # 统计
    # 周期
]


# 口径不同的指标：(指标名, 输出列, 按该指标声明公式的独立实现, 与 TA-Lib 的差异说明)
CONVENTION_CASES = [
    ("StandardDeviation", "StdDev", lambda d: d["close"].rolling(20).std(ddof=1), "样本标准差（ddof=1），TA-Lib 为总体标准差"),
    ("Variance", None, lambda d: d["close"].rolling(20).var(ddof=1), "样本方差（ddof=1），TA-Lib 为总体方差"),
    ("BollingerBands", "BB_Upper", lambda d: d["close"].rolling(20).mean() + 2 * d["close"].rolling(20).std(ddof=1), "标准差用 ddof=1，TA-Lib 用总体标准差"),
    ("BollingerBands", "BB_Lower", lambda d: d["close"].rolling(20).mean() - 2 * d["close"].rolling(20).std(ddof=1), "同上"),
    ("APO", None, lambda d: d["close"].ewm(span=12, adjust=False).mean() - d["close"].ewm(span=26, adjust=False).mean(), "用 EMA，TA-Lib 默认用 SMA（matype=0）"),
    ("CMO", None, lambda d: _cmo_simple(d["close"], 14), "Chande 原始定义（简单求和），TA-Lib 用 Wilder 平滑"),
    ("ATR", None, lambda d: _true_range(d).ewm(span=14, adjust=False).mean(), "EMA(span=n) 平滑，TA-Lib 用 Wilder 平滑（alpha=1/n）"),
    ("OBV", None, lambda d: (np.sign(d["close"].diff()).fillna(0) * d["volume"]).cumsum(), "从 0 起算，TA-Lib 从第一根成交量起算（差一个常数）"),
]


def _true_range(d):
    prev = d["close"].shift(1)
    return pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)


def _cmo_simple(close, n):
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).sum()
    down = (-diff.clip(upper=0)).rolling(n).sum()
    return 100 * (up - down) / (up + down)


def compare(ours: pd.Series, ref: pd.Series, tail: int):
    """比较最后 tail 根K线（跳过各自的预热期），返回 (最大相对误差, 比较点数)。"""
    a, b = ours.iloc[-tail:].astype(float), ref.iloc[-tail:].astype(float)
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("inf"), 0
    a, b = a[mask], b[mask]
    scale = max(float(np.abs(b).max()), 1e-12)
    return float(np.abs(a - b).max() / scale), int(mask.sum())
