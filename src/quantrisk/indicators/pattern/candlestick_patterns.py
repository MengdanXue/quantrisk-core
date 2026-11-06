"""
K线形态识别指标 (15个)

包含经典的单根和双根K线形态识别
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 单根K线形态 ====================

@indicator_registry.register('pattern')
class Hammer(BaseIndicator):
    """
    锤子线形态

    特征:
    - 小实体在上部
    - 下影线长度至少是实体的2倍
    - 上影线很短或无
    - 出现在下跌趋势末端为看涨信号
    """

    _metadata = IndicatorMetadata(
        name='Hammer',
        display_name='锤子线',
        category='pattern',
        description='底部反转信号,小实体在上部,长下影线,上影线很短',
        formula='下影线 >= 2 * 实体 AND 上影线 < 0.3 * 实体',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 2.0, 'shadow_ratio': 0.3},
        complexity_level=2
    )

    def __init__(self, body_ratio: float = 2.0, shadow_ratio: float = 0.3):
        """
        Args:
            body_ratio: 下影线与实体的最小比例
            shadow_ratio: 上影线与实体的最大比例
        """
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
        super().__init__(body_ratio=body_ratio, shadow_ratio=shadow_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_ratio, 'body_ratio')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        # 锤子线条件
        hammer = (
            (lower_shadow >= self.body_ratio * body) &  # 长下影线
            (upper_shadow <= self.shadow_ratio * body) &  # 短上影线
            (body > 0)  # 有实体
        )

        return hammer.astype(int)


@indicator_registry.register('pattern')
class InvertedHammer(BaseIndicator):
    """
    倒锤子线形态

    特征:
    - 小实体在下部
    - 上影线长度至少是实体的2倍
    - 下影线很短或无
    - 出现在下跌趋势末端为看涨信号
    """

    _metadata = IndicatorMetadata(
        name='InvertedHammer',
        display_name='倒锤子线',
        category='pattern',
        description='底部反转信号,小实体在下部,长上影线,下影线很短',
        formula='上影线 >= 2 * 实体 AND 下影线 < 0.3 * 实体',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 2.0, 'shadow_ratio': 0.3},
        complexity_level=2
    )

    def __init__(self, body_ratio: float = 2.0, shadow_ratio: float = 0.3):
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
        super().__init__(body_ratio=body_ratio, shadow_ratio=shadow_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_ratio, 'body_ratio')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        inverted_hammer = (
            (upper_shadow >= self.body_ratio * body) &  # 长上影线
            (lower_shadow <= self.shadow_ratio * body) &  # 短下影线
            (body > 0)
        )

        return inverted_hammer.astype(int)


@indicator_registry.register('pattern')
class HangingMan(BaseIndicator):
    """
    上吊线形态

    特征:
    - 与锤子线形态相同,但出现在上涨趋势顶部
    - 小实体在上部
    - 长下影线
    - 看跌反转信号
    """

    _metadata = IndicatorMetadata(
        name='HangingMan',
        display_name='上吊线',
        category='pattern',
        description='顶部反转信号,小实体在上部,长下影线',
        formula='下影线 >= 2 * 实体 AND 上影线 < 0.3 * 实体 AND 出现在上涨趋势',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 2.0, 'shadow_ratio': 0.3, 'trend_period': 10},
        complexity_level=2
    )

    def __init__(self, body_ratio: float = 2.0, shadow_ratio: float = 0.3, trend_period: int = 10):
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
        self.trend_period = trend_period
        super().__init__(body_ratio=body_ratio, shadow_ratio=shadow_ratio, trend_period=trend_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_ratio, 'body_ratio')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')
        IndicatorValidator.validate_positive_int(self.trend_period, 'trend_period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        # 上涨趋势判断
        uptrend = data['close'] > data['close'].shift(self.trend_period)

        hanging_man = (
            (lower_shadow >= self.body_ratio * body) &
            (upper_shadow <= self.shadow_ratio * body) &
            (body > 0) &
            uptrend
        )

        return hanging_man.astype(int)


@indicator_registry.register('pattern')
class ShootingStar(BaseIndicator):
    """
    流星线形态

    特征:
    - 小实体在下部
    - 长上影线
    - 出现在上涨趋势顶部
    - 看跌反转信号
    """

    _metadata = IndicatorMetadata(
        name='ShootingStar',
        display_name='流星线',
        category='pattern',
        description='顶部反转信号,小实体在下部,长上影线',
        formula='上影线 >= 2 * 实体 AND 下影线 < 0.3 * 实体 AND 出现在上涨趋势',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 2.0, 'shadow_ratio': 0.3, 'trend_period': 10},
        complexity_level=2
    )

    def __init__(self, body_ratio: float = 2.0, shadow_ratio: float = 0.3, trend_period: int = 10):
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
        self.trend_period = trend_period
        super().__init__(body_ratio=body_ratio, shadow_ratio=shadow_ratio, trend_period=trend_period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_ratio, 'body_ratio')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')
        IndicatorValidator.validate_positive_int(self.trend_period, 'trend_period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        uptrend = data['close'] > data['close'].shift(self.trend_period)

        shooting_star = (
            (upper_shadow >= self.body_ratio * body) &
            (lower_shadow <= self.shadow_ratio * body) &
            (body > 0) &
            uptrend
        )

        return shooting_star.astype(int)


@indicator_registry.register('pattern')
class Doji(BaseIndicator):
    """
    十字星形态

    特征:
    - 开盘价和收盘价几乎相等
    - 上下影线都存在
    - 表示市场犹豫不决
    """

    _metadata = IndicatorMetadata(
        name='Doji',
        display_name='十字星',
        category='pattern',
        description='开盘价收盘价相近,表示市场犹豫',
        formula='abs(Close - Open) <= 0.1% * (High - Low)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_threshold': 0.001},
        complexity_level=1
    )

    def __init__(self, body_threshold: float = 0.001):
        """
        Args:
            body_threshold: 实体与价格范围的最大比例
        """
        self.body_threshold = body_threshold
        super().__init__(body_threshold=body_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_threshold, 'body_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        price_range = data['high'] - data['low']

        doji = (body <= self.body_threshold * price_range) & (price_range > 0)

        return doji.astype(int)


@indicator_registry.register('pattern')
class Dragonfly(BaseIndicator):
    """
    蜻蜓十字星形态

    特征:
    - 开盘价、收盘价和最高价几乎相等
    - 长下影线
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='Dragonfly',
        display_name='蜻蜓十字星',
        category='pattern',
        description='底部反转,开收盘价在最高价附近,长下影线',
        formula='Close ≈ Open ≈ High AND 下影线长',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_threshold': 0.001, 'shadow_ratio': 2.0},
        complexity_level=2
    )

    def __init__(self, body_threshold: float = 0.001, shadow_ratio: float = 2.0):
        self.body_threshold = body_threshold
        self.shadow_ratio = shadow_ratio
        super().__init__(body_threshold=body_threshold, shadow_ratio=shadow_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_threshold, 'body_threshold')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        price_range = data['high'] - data['low']
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        dragonfly = (
            (body <= self.body_threshold * price_range) &
            (upper_shadow <= self.body_threshold * price_range) &
            (lower_shadow >= self.shadow_ratio * body) &
            (price_range > 0)
        )

        return dragonfly.astype(int)


@indicator_registry.register('pattern')
class Gravestone(BaseIndicator):
    """
    墓碑十字星形态

    特征:
    - 开盘价、收盘价和最低价几乎相等
    - 长上影线
    - 顶部反转信号
    """

    _metadata = IndicatorMetadata(
        name='Gravestone',
        display_name='墓碑十字星',
        category='pattern',
        description='顶部反转,开收盘价在最低价附近,长上影线',
        formula='Close ≈ Open ≈ Low AND 上影线长',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_threshold': 0.001, 'shadow_ratio': 2.0},
        complexity_level=2
    )

    def __init__(self, body_threshold: float = 0.001, shadow_ratio: float = 2.0):
        self.body_threshold = body_threshold
        self.shadow_ratio = shadow_ratio
        super().__init__(body_threshold=body_threshold, shadow_ratio=shadow_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_threshold, 'body_threshold')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        price_range = data['high'] - data['low']
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        gravestone = (
            (body <= self.body_threshold * price_range) &
            (lower_shadow <= self.body_threshold * price_range) &
            (upper_shadow >= self.shadow_ratio * body) &
            (price_range > 0)
        )

        return gravestone.astype(int)


@indicator_registry.register('pattern')
class MarubozuWhite(BaseIndicator):
    """
    白色大阳线(光头光脚阳线)

    特征:
    - 长阳线
    - 无上下影线或影线很短
    - 强烈看涨信号
    """

    _metadata = IndicatorMetadata(
        name='MarubozuWhite',
        display_name='白色大阳线',
        category='pattern',
        description='长阳线无影线,强烈看涨信号',
        formula='Close > Open AND 实体 > 0.9 * (High - Low)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 0.9},
        complexity_level=1
    )

    def __init__(self, body_ratio: float = 0.9):
        """
        Args:
            body_ratio: 实体占价格范围的最小比例
        """
        self.body_ratio = body_ratio
        super().__init__(body_ratio=body_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.body_ratio, 'body_ratio', 0.5, 1.0)

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = data['close'] - data['open']
        price_range = data['high'] - data['low']

        marubozu_white = (
            (body > 0) &
            (body >= self.body_ratio * price_range) &
            (price_range > 0)
        )

        return marubozu_white.astype(int)


@indicator_registry.register('pattern')
class MarubozuBlack(BaseIndicator):
    """
    黑色大阴线(光头光脚阴线)

    特征:
    - 长阴线
    - 无上下影线或影线很短
    - 强烈看跌信号
    """

    _metadata = IndicatorMetadata(
        name='MarubozuBlack',
        display_name='黑色大阴线',
        category='pattern',
        description='长阴线无影线,强烈看跌信号',
        formula='Open > Close AND 实体 > 0.9 * (High - Low)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 0.9},
        complexity_level=1
    )

    def __init__(self, body_ratio: float = 0.9):
        self.body_ratio = body_ratio
        super().__init__(body_ratio=body_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.body_ratio, 'body_ratio', 0.5, 1.0)

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = data['open'] - data['close']
        price_range = data['high'] - data['low']

        marubozu_black = (
            (body > 0) &
            (body >= self.body_ratio * price_range) &
            (price_range > 0)
        )

        return marubozu_black.astype(int)


@indicator_registry.register('pattern')
class SpinningTop(BaseIndicator):
    """
    陀螺线形态

    特征:
    - 小实体
    - 上下影线较长且相近
    - 表示市场犹豫
    """

    _metadata = IndicatorMetadata(
        name='SpinningTop',
        display_name='陀螺线',
        category='pattern',
        description='小实体,上下影线长且相近,表示犹豫',
        formula='实体小 AND 上下影线长',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'body_ratio': 0.3, 'shadow_ratio': 1.5},
        complexity_level=2
    )

    def __init__(self, body_ratio: float = 0.3, shadow_ratio: float = 1.5):
        """
        Args:
            body_ratio: 实体占价格范围的最大比例
            shadow_ratio: 影线与实体的最小比例
        """
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
        super().__init__(body_ratio=body_ratio, shadow_ratio=shadow_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.body_ratio, 'body_ratio')
        IndicatorValidator.validate_positive_float(self.shadow_ratio, 'shadow_ratio')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        body = abs(data['close'] - data['open'])
        price_range = data['high'] - data['low']
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        spinning_top = (
            (body <= self.body_ratio * price_range) &
            (upper_shadow >= self.shadow_ratio * body) &
            (lower_shadow >= self.shadow_ratio * body) &
            (price_range > 0)
        )

        return spinning_top.astype(int)


# ==================== 双K线形态 ====================

@indicator_registry.register('pattern')
class BullishEngulfing(BaseIndicator):
    """
    看涨吞没形态

    特征:
    - 前一根为阴线
    - 后一根为阳线且完全吞没前一根
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='BullishEngulfing',
        display_name='看涨吞没',
        category='pattern',
        description='阳线完全吞没前一根阴线,底部反转信号',
        formula='前阴后阳 AND Close[0] > Open[-1] AND Open[0] < Close[-1]',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        prev_bearish = data['close'].shift(1) < data['open'].shift(1)
        curr_bullish = data['close'] > data['open']

        engulfing = (
            prev_bearish &
            curr_bullish &
            (data['close'] > data['open'].shift(1)) &
            (data['open'] < data['close'].shift(1))
        )

        return engulfing.astype(int)


@indicator_registry.register('pattern')
class BearishEngulfing(BaseIndicator):
    """
    看跌吞没形态

    特征:
    - 前一根为阳线
    - 后一根为阴线且完全吞没前一根
    - 顶部反转信号
    """

    _metadata = IndicatorMetadata(
        name='BearishEngulfing',
        display_name='看跌吞没',
        category='pattern',
        description='阴线完全吞没前一根阳线,顶部反转信号',
        formula='前阳后阴 AND Open[0] > Close[-1] AND Close[0] < Open[-1]',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={},
        complexity_level=2
    )

    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        prev_bullish = data['close'].shift(1) > data['open'].shift(1)
        curr_bearish = data['close'] < data['open']

        engulfing = (
            prev_bullish &
            curr_bearish &
            (data['open'] > data['close'].shift(1)) &
            (data['close'] < data['open'].shift(1))
        )

        return engulfing.astype(int)


@indicator_registry.register('pattern')
class Piercing(BaseIndicator):
    """
    刺透形态

    特征:
    - 前一根为阴线
    - 后一根为阳线,开盘低于前一根最低价
    - 收盘价穿透前一根实体的50%以上
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='Piercing',
        display_name='刺透形态',
        category='pattern',
        description='阳线穿透前阴线实体50%以上,底部反转',
        formula='前阴后阳 AND Open[0] < Low[-1] AND Close[0] > (Open[-1] + Close[-1]) / 2',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'penetration_ratio': 0.5},
        complexity_level=2
    )

    def __init__(self, penetration_ratio: float = 0.5):
        """
        Args:
            penetration_ratio: 穿透比例阈值
        """
        self.penetration_ratio = penetration_ratio
        super().__init__(penetration_ratio=penetration_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.penetration_ratio, 'penetration_ratio', 0.0, 1.0)

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        prev_bearish = data['close'].shift(1) < data['open'].shift(1)
        curr_bullish = data['close'] > data['open']

        prev_midpoint = (data['open'].shift(1) + data['close'].shift(1)) / 2

        piercing = (
            prev_bearish &
            curr_bullish &
            (data['open'] < data['low'].shift(1)) &
            (data['close'] > prev_midpoint)
        )

        return piercing.astype(int)


@indicator_registry.register('pattern')
class DarkCloudCover(BaseIndicator):
    """
    乌云盖顶形态

    特征:
    - 前一根为阳线
    - 后一根为阴线,开盘高于前一根最高价
    - 收盘价深入前一根实体的50%以上
    - 顶部反转信号
    """

    _metadata = IndicatorMetadata(
        name='DarkCloudCover',
        display_name='乌云盖顶',
        category='pattern',
        description='阴线深入前阳线实体50%以上,顶部反转',
        formula='前阳后阴 AND Open[0] > High[-1] AND Close[0] < (Open[-1] + Close[-1]) / 2',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'penetration_ratio': 0.5},
        complexity_level=2
    )

    def __init__(self, penetration_ratio: float = 0.5):
        self.penetration_ratio = penetration_ratio
        super().__init__(penetration_ratio=penetration_ratio)

    def _validate_params(self):
        IndicatorValidator.validate_range(self.penetration_ratio, 'penetration_ratio', 0.0, 1.0)

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        prev_bullish = data['close'].shift(1) > data['open'].shift(1)
        curr_bearish = data['close'] < data['open']

        prev_midpoint = (data['open'].shift(1) + data['close'].shift(1)) / 2

        dark_cloud = (
            prev_bullish &
            curr_bearish &
            (data['open'] > data['high'].shift(1)) &
            (data['close'] < prev_midpoint)
        )

        return dark_cloud.astype(int)


@indicator_registry.register('pattern')
class TweezerBottom(BaseIndicator):
    """
    镊子底形态

    特征:
    - 两根K线最低价相同或非常接近
    - 第一根通常为阴线,第二根为阳线
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='TweezerBottom',
        display_name='镊子底',
        category='pattern',
        description='两根K线最低价相近,底部反转信号',
        formula='abs(Low[0] - Low[-1]) <= threshold',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='series',
        default_params={'price_tolerance': 0.002},
        complexity_level=2
    )

    def __init__(self, price_tolerance: float = 0.002):
        """
        Args:
            price_tolerance: 价格容差比例
        """
        self.price_tolerance = price_tolerance
        super().__init__(price_tolerance=price_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_float(self.price_tolerance, 'price_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        low_diff = abs(data['low'] - data['low'].shift(1))
        avg_price = (data['low'] + data['low'].shift(1)) / 2

        prev_bearish = data['close'].shift(1) < data['open'].shift(1)
        curr_bullish = data['close'] > data['open']

        tweezer_bottom = (
            (low_diff <= self.price_tolerance * avg_price) &
            prev_bearish &
            curr_bullish
        )

        return tweezer_bottom.astype(int)
