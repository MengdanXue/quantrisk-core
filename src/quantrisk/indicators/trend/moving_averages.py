"""
移动平均线系列指标 (20个)

包含各种移动平均线变体，从简单到自适应
"""

from typing import List, Union
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 基础移动平均线 ====================

@indicator_registry.register('trend')
class SMA(BaseIndicator):
    """
    Simple Moving Average (简单移动平均线)

    计算指定周期内价格的算术平均值
    """

    _metadata = IndicatorMetadata(
        name='SMA',
        display_name='简单移动平均线',
        category='trend',
        description='计算指定周期内收盘价的算术平均值',
        formula='SMA = Sum(Close, n) / n',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=1
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        return data['close'].rolling(window=self.period).mean()


@indicator_registry.register('trend')
class EMA(BaseIndicator):
    """
    Exponential Moving Average (指数移动平均线)

    对近期价格赋予更高权重的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='EMA',
        display_name='指数移动平均线',
        category='trend',
        description='对近期价格赋予更高权重的移动平均线',
        formula='EMA = α * Close + (1 - α) * EMA_prev, α = 2 / (n + 1)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=1
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        return data['close'].ewm(span=self.period, adjust=False).mean()


@indicator_registry.register('trend')
class WMA(BaseIndicator):
    """
    Weighted Moving Average (加权移动平均线)

    对近期价格赋予线性递增权重
    """

    _metadata = IndicatorMetadata(
        name='WMA',
        display_name='加权移动平均线',
        category='trend',
        description='对近期价格赋予线性递增权重的移动平均线',
        formula='WMA = Sum(Weight_i * Close_i) / Sum(Weight_i)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=1
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        weights = np.arange(1, self.period + 1)

        def weighted_mean(x):
            return np.sum(weights * x) / np.sum(weights)

        return data['close'].rolling(window=self.period).apply(weighted_mean, raw=True)


@indicator_registry.register('trend')
class DEMA(BaseIndicator):
    """
    Double Exponential Moving Average (双重指数移动平均线)

    减少滞后的EMA变体
    """

    _metadata = IndicatorMetadata(
        name='DEMA',
        display_name='双重指数移动平均线',
        category='trend',
        description='通过两次EMA计算减少滞后',
        formula='DEMA = 2 * EMA(Close, n) - EMA(EMA(Close, n), n)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        ema1 = data['close'].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        return 2 * ema1 - ema2


@indicator_registry.register('trend')
class TEMA(BaseIndicator):
    """
    Triple Exponential Moving Average (三重指数移动平均线)

    进一步减少滞后的EMA变体
    """

    _metadata = IndicatorMetadata(
        name='TEMA',
        display_name='三重指数移动平均线',
        category='trend',
        description='通过三次EMA计算进一步减少滞后',
        formula='TEMA = 3*EMA1 - 3*EMA2 + EMA3',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        ema1 = data['close'].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3


# ==================== 自适应移动平均线 ====================

@indicator_registry.register('trend')
class KAMA(BaseIndicator):
    """
    Kaufman Adaptive Moving Average (考夫曼自适应移动平均线)

    根据市场效率自动调整平滑参数
    """

    _metadata = IndicatorMetadata(
        name='KAMA',
        display_name='考夫曼自适应移动平均线',
        category='trend',
        description='根据价格效率比率自动调整平滑系数的自适应移动平均线',
        formula='KAMA = KAMA_prev + SC * (Price - KAMA_prev)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 10, 'fast_period': 2, 'slow_period': 30},
        complexity_level=3
    )

    def __init__(self, period: int = 10, fast_period: int = 2, slow_period: int = 30):
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(period=period, fast_period=fast_period, slow_period=slow_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.fast_period, 'fast_period')
        IndicatorValidator.validate_positive_int(self.slow_period, 'slow_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 计算效率比率 (ER)
        direction = abs(close - close.shift(self.period))
        volatility = close.diff().abs().rolling(self.period).sum()
        er = self._safe_division(direction, volatility)

        # 计算平滑常数 (SC)
        fast_sc = 2 / (self.fast_period + 1)
        slow_sc = 2 / (self.slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # 计算 KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[self.period - 1] = close.iloc[self.period - 1]

        for i in range(self.period, len(close)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])

        return kama


@indicator_registry.register('trend')
class HMA(BaseIndicator):
    """
    Hull Moving Average (赫尔移动平均线)

    超低滞后的移动平均线，由Alan Hull开发
    """

    _metadata = IndicatorMetadata(
        name='HMA',
        display_name='赫尔移动平均线',
        category='trend',
        description='使用加权移动平均和平方根周期减少滞后',
        formula='HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        half_period = int(self.period / 2)
        sqrt_period = int(np.sqrt(self.period))

        # WMA(n/2)
        wma_half = self._wma(close, half_period)

        # WMA(n)
        wma_full = self._wma(close, self.period)

        # 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # WMA(sqrt(n))
        return self._wma(raw_hma, sqrt_period)

    def _wma(self, series: pd.Series, period: int) -> pd.Series:
        """计算加权移动平均"""
        weights = np.arange(1, period + 1)

        def weighted_mean(x):
            return np.sum(weights * x) / np.sum(weights)

        return series.rolling(window=period).apply(weighted_mean, raw=True)


@indicator_registry.register('trend')
class ALMA(BaseIndicator):
    """
    Arnaud Legoux Moving Average (阿诺德莱古克斯移动平均线)

    使用高斯分布权重的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='ALMA',
        display_name='阿诺德莱古克斯移动平均线',
        category='trend',
        description='使用高斯分布权重，可调节偏移和平滑度',
        formula='ALMA = Sum(Gaussian_Weight * Close) / Sum(Gaussian_Weight)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'offset': 0.85, 'sigma': 6},
        complexity_level=3
    )

    def __init__(self, period: int = 20, offset: float = 0.85, sigma: float = 6):
        self.period = period
        self.offset = offset
        self.sigma = sigma
        super().__init__(period=period, offset=offset, sigma=sigma)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_range(self.offset, 'offset', 0, 1)
        IndicatorValidator.validate_positive_float(self.sigma, 'sigma')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 计算高斯权重
        m = np.floor(self.offset * (self.period - 1))
        s = self.period / self.sigma

        weights = np.array([
            np.exp(-((i - m) ** 2) / (2 * s * s))
            for i in range(self.period)
        ])
        weights = weights / weights.sum()

        def alma_calc(x):
            return np.sum(weights * x)

        return close.rolling(window=self.period).apply(alma_calc, raw=True)


@indicator_registry.register('trend')
class T3(BaseIndicator):
    """
    Tillson T3 Moving Average (蒂尔森T3移动平均线)

    超平滑的移动平均线，滞后极低
    """

    _metadata = IndicatorMetadata(
        name='T3',
        display_name='蒂尔森T3移动平均线',
        category='trend',
        description='通过多次指数平滑实现超平滑低滞后',
        formula='T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'vfactor': 0.7},
        complexity_level=3
    )

    def __init__(self, period: int = 20, vfactor: float = 0.7):
        self.period = period
        self.vfactor = vfactor
        super().__init__(period=period, vfactor=vfactor)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_range(self.vfactor, 'vfactor', 0, 1)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        v = self.vfactor

        # 计算系数
        c1 = -v ** 3
        c2 = 3 * v ** 2 + 3 * v ** 3
        c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
        c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2

        # 六次EMA
        e1 = close.ewm(span=self.period, adjust=False).mean()
        e2 = e1.ewm(span=self.period, adjust=False).mean()
        e3 = e2.ewm(span=self.period, adjust=False).mean()
        e4 = e3.ewm(span=self.period, adjust=False).mean()
        e5 = e4.ewm(span=self.period, adjust=False).mean()
        e6 = e5.ewm(span=self.period, adjust=False).mean()

        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


@indicator_registry.register('trend')
class TRIMA(BaseIndicator):
    """
    Triangular Moving Average (三角移动平均线)

    中间权重最高的对称权重移动平均线
    """

    _metadata = IndicatorMetadata(
        name='TRIMA',
        display_name='三角移动平均线',
        category='trend',
        description='使用三角形权重分布的移动平均线',
        formula='TRIMA = SMA(SMA(Close, ceil(n/2)), floor(n/2)+1)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 三角移动平均 = SMA(SMA)，两层窗口合计覆盖 period 根K线
        if self.period % 2 == 0:
            first, second = self.period // 2, self.period // 2 + 1
        else:
            first = second = (self.period + 1) // 2
        sma1 = close.rolling(window=first).mean()
        return sma1.rolling(window=second).mean()


@indicator_registry.register('trend')
class ZLEMA(BaseIndicator):
    """
    Zero Lag Exponential Moving Average (零滞后指数移动平均线)

    通过去除滞后因子实现零延迟
    """

    _metadata = IndicatorMetadata(
        name='ZLEMA',
        display_name='零滞后指数移动平均线',
        category='trend',
        description='通过移除滞后实现零延迟的EMA',
        formula='ZLEMA = EMA(2*Close - Close[lag], n)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 计算滞后
        lag = int((self.period - 1) / 2)

        # 去滞后价格
        ema_data = 2 * close - close.shift(lag)

        return ema_data.ewm(span=self.period, adjust=False).mean()


@indicator_registry.register('trend')
class SMMA(BaseIndicator):
    """
    Smoothed Moving Average (平滑移动平均线)

    也称为修正移动平均线(MMA)或运行移动平均线(RMA)
    """

    _metadata = IndicatorMetadata(
        name='SMMA',
        display_name='平滑移动平均线',
        category='trend',
        description='使用前一个SMMA值进行平滑的移动平均线',
        formula='SMMA = (Sum(Close, n) - SMMA_prev + Close) / n',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # SMMA等同于alpha=1/n的EMA
        return close.ewm(alpha=1/self.period, adjust=False).mean()


@indicator_registry.register('trend')
class LSMA(BaseIndicator):
    """
    Least Squares Moving Average (最小二乘移动平均线)

    也称为端点移动平均线或线性回归值
    """

    _metadata = IndicatorMetadata(
        name='LSMA',
        display_name='最小二乘移动平均线',
        category='trend',
        description='使用线性回归拟合的移动平均线',
        formula='LSMA = 线性回归在当前点的值',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=3
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        def linreg(x):
            n = len(x)
            x_idx = np.arange(n)
            slope, intercept = np.polyfit(x_idx, x, 1)
            return intercept + slope * (n - 1)

        return close.rolling(window=self.period).apply(linreg, raw=True)


@indicator_registry.register('trend')
class VWMA(BaseIndicator):
    """
    Volume Weighted Moving Average (成交量加权移动平均线)

    使用成交量作为权重的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='VWMA',
        display_name='成交量加权移动平均线',
        category='trend',
        description='以成交量为权重的移动平均线',
        formula='VWMA = Sum(Close * Volume, n) / Sum(Volume, n)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        pv = data['close'] * data['volume']
        return pv.rolling(window=self.period).sum() / data['volume'].rolling(window=self.period).sum()


@indicator_registry.register('trend')
class FRAMA(BaseIndicator):
    """
    Fractal Adaptive Moving Average (分形自适应移动平均线)

    基于分形几何自动调整的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='FRAMA',
        display_name='分形自适应移动平均线',
        category='trend',
        description='使用分形维度自动调整平滑系数',
        formula='FRAMA = α * Close + (1 - α) * FRAMA_prev',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=4
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        if self.period % 2 != 0:
            self.period += 1  # 确保是偶数

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        high = data['high']
        low = data['low']

        half = self.period // 2

        # 计算分形维度
        def calc_frama(idx):
            if idx < self.period:
                return np.nan

            # 第一半的高低差
            h1 = high[idx - self.period:idx - half].max()
            l1 = low[idx - self.period:idx - half].min()
            n1 = (h1 - l1) / half

            # 第二半的高低差
            h2 = high[idx - half:idx].max()
            l2 = low[idx - half:idx].min()
            n2 = (h2 - l2) / half

            # 整个周期的高低差
            h3 = high[idx - self.period:idx].max()
            l3 = low[idx - self.period:idx].min()
            n3 = (h3 - l3) / self.period

            if n1 + n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                d = 1

            alpha = np.exp(-4.6 * (d - 1))
            alpha = min(max(alpha, 0.01), 1)  # 限制在0.01-1之间

            return alpha

        # 计算FRAMA
        frama = pd.Series(index=close.index, dtype=float)
        frama.iloc[self.period - 1] = close.iloc[self.period - 1]

        for i in range(self.period, len(close)):
            alpha = calc_frama(i)
            if np.isnan(alpha):
                alpha = 0.5
            frama.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * frama.iloc[i - 1]

        return frama


@indicator_registry.register('trend')
class VIDYA(BaseIndicator):
    """
    Variable Index Dynamic Average (可变指数动态平均线)

    由Tushar Chande开发的自适应移动平均线
    """

    _metadata = IndicatorMetadata(
        name='VIDYA',
        display_name='可变指数动态平均线',
        category='trend',
        description='使用Chande动量振荡器调整平滑系数',
        formula='VIDYA = α * |CMO| * Close + (1 - α * |CMO|) * VIDYA_prev',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'cmo_period': 9},
        complexity_level=3
    )

    def __init__(self, period: int = 20, cmo_period: int = 9):
        self.period = period
        self.cmo_period = cmo_period
        super().__init__(period=period, cmo_period=cmo_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.cmo_period, 'cmo_period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 计算CMO
        change = close.diff()
        gain = change.where(change > 0, 0).rolling(self.cmo_period).sum()
        loss = (-change.where(change < 0, 0)).rolling(self.cmo_period).sum()
        cmo = abs((gain - loss) / (gain + loss))

        # EMA的alpha
        alpha = 2 / (self.period + 1)

        # 计算VIDYA
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[self.cmo_period - 1] = close.iloc[self.cmo_period - 1]

        for i in range(self.cmo_period, len(close)):
            sc = alpha * cmo.iloc[i]
            if np.isnan(sc):
                sc = alpha
            vidya.iloc[i] = sc * close.iloc[i] + (1 - sc) * vidya.iloc[i - 1]

        return vidya


@indicator_registry.register('trend')
class JMA(BaseIndicator):
    """
    Jurik Moving Average (尤里克移动平均线)

    Mark Jurik开发的超平滑低滞后移动平均线（简化版）
    """

    _metadata = IndicatorMetadata(
        name='JMA',
        display_name='尤里克移动平均线',
        category='trend',
        description='超平滑低滞后的专业级移动平均线',
        formula='JMA = 自适应平滑算法',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'phase': 0},
        complexity_level=4
    )

    def __init__(self, period: int = 20, phase: int = 0):
        self.period = period
        self.phase = phase
        super().__init__(period=period, phase=phase)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_range(self.phase, 'phase', -100, 100)

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 简化版JMA实现（类似KAMA但使用不同的平滑因子）
        phase_ratio = (self.phase / 100 + 1.5) / 2.5
        beta = 0.45 * (self.period - 1) / (0.45 * (self.period - 1) + 2)

        alpha = beta ** phase_ratio

        jma = pd.Series(index=close.index, dtype=float)
        jma.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            jma.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * jma.iloc[i - 1]

        return jma


@indicator_registry.register('trend')
class EVWMA(BaseIndicator):
    """
    Elastic Volume Weighted Moving Average (弹性成交量加权移动平均线)

    动态调整成交量权重的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='EVWMA',
        display_name='弹性成交量加权移动平均线',
        category='trend',
        description='根据累积成交量动态调整的VWMA',
        formula='EVWMA = (Sum(Volume, n) - Volume) * EVWMA_prev + Volume * Close) / Sum(Volume, n)',
        required_columns=['close', 'volume'],
        output_type='series',
        default_params={'period': 20},
        complexity_level=3
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        vol_sum = volume.rolling(window=self.period).sum()

        evwma = pd.Series(index=close.index, dtype=float)
        evwma.iloc[self.period - 1] = close.iloc[:self.period].mean()

        for i in range(self.period, len(close)):
            if vol_sum.iloc[i] > 0:
                evwma.iloc[i] = ((vol_sum.iloc[i] - volume.iloc[i]) * evwma.iloc[i - 1] +
                                volume.iloc[i] * close.iloc[i]) / vol_sum.iloc[i]
            else:
                evwma.iloc[i] = evwma.iloc[i - 1]

        return evwma


@indicator_registry.register('trend')
class McGinleyDynamic(BaseIndicator):
    """
    McGinley Dynamic Indicator (麦金利动态指标)

    John R. McGinley开发的自动调整速度的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='McGinleyDynamic',
        display_name='麦金利动态指标',
        category='trend',
        description='自动调整平滑因子以减少whipsaws的移动平均线',
        formula='MD = MD_prev + (Close - MD_prev) / (k * n * (Close/MD_prev)^4)',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 20, 'k': 0.6},
        complexity_level=3
    )

    def __init__(self, period: int = 20, k: float = 0.6):
        self.period = period
        self.k = k
        super().__init__(period=period, k=k)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.k, 'k')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        md = pd.Series(index=close.index, dtype=float)
        md.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            prev_md = md.iloc[i - 1]
            if prev_md != 0 and not np.isnan(prev_md):
                ratio = close.iloc[i] / prev_md
                md.iloc[i] = prev_md + (close.iloc[i] - prev_md) / \
                            (self.k * self.period * (ratio ** 4))
            else:
                md.iloc[i] = close.iloc[i]

        return md


@indicator_registry.register('trend')
class SWMA(BaseIndicator):
    """
    Symmetrically Weighted Moving Average (对称加权移动平均线)

    使用对称权重分布的移动平均线
    """

    _metadata = IndicatorMetadata(
        name='SWMA',
        display_name='对称加权移动平均线',
        category='trend',
        description='使用对称权重分布(1,2,2,1)的移动平均线',
        formula='SWMA = (C[0] + 2*C[1] + 2*C[2] + C[3]) / 6',
        required_columns=['close'],
        output_type='series',
        default_params={'period': 4},
        complexity_level=1
    )

    def __init__(self, period: int = 4):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']

        # 默认使用4周期对称权重 [1, 2, 2, 1]
        if self.period == 4:
            weights = np.array([1, 2, 2, 1])
        else:
            # 生成对称权重
            half = self.period // 2
            if self.period % 2 == 0:
                weights = np.concatenate([
                    np.arange(1, half + 1),
                    np.arange(half, 0, -1)
                ])
            else:
                weights = np.concatenate([
                    np.arange(1, half + 1),
                    [half + 1],
                    np.arange(half, 0, -1)
                ])

        def swma_calc(x):
            return np.sum(weights * x) / np.sum(weights)

        return close.rolling(window=len(weights)).apply(swma_calc, raw=True)
