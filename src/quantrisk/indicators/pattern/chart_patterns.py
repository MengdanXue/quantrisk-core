"""
图表形态识别指标 (15个)

包含经典的价格形态识别,如头肩顶、双底、三角形等
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ==================== 反转形态 ====================

@indicator_registry.register('pattern')
class HeadAndShoulders(BaseIndicator):
    """
    头肩顶形态

    特征:
    - 三个峰,中间峰(头)最高
    - 两侧峰(肩)高度相近
    - 颈线由两个低点连接
    - 顶部反转信号
    """

    _metadata = IndicatorMetadata(
        name='HeadAndShoulders',
        display_name='头肩顶',
        category='pattern',
        description='三个峰,中间最高,顶部反转信号',
        formula='识别三个峰值,Head > Shoulders AND 颈线突破',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'shoulder_tolerance': 0.03},
        complexity_level=4
    )

    def __init__(self, window: int = 20, shoulder_tolerance: float = 0.03):
        """
        Args:
            window: 形态识别窗口
            shoulder_tolerance: 左右肩高度容差
        """
        self.window = window
        self.shoulder_tolerance = shoulder_tolerance
        super().__init__(window=window, shoulder_tolerance=shoulder_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.shoulder_tolerance, 'shoulder_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        # 寻找峰值
        highs = data['high'].values
        peaks, _ = find_peaks(highs, distance=self.window // 4)

        # 寻找谷值
        lows = data['low'].values
        troughs, _ = find_peaks(-lows, distance=self.window // 4)

        # 检查头肩顶形态
        for i in range(len(data) - self.window):
            window_peaks = peaks[(peaks >= i) & (peaks < i + self.window)]

            if len(window_peaks) >= 3:
                # 找到三个最高的峰
                peak_heights = highs[window_peaks]
                top_3_indices = np.argsort(peak_heights)[-3:]
                sorted_peaks = window_peaks[top_3_indices]
                sorted_peaks = np.sort(sorted_peaks)

                left_shoulder = sorted_peaks[0]
                head = sorted_peaks[1]
                right_shoulder = sorted_peaks[2]

                # 检查头是否最高
                if (highs[head] > highs[left_shoulder] and
                    highs[head] > highs[right_shoulder]):

                    # 检查两肩高度是否相近
                    shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder])
                    avg_shoulder = (highs[left_shoulder] + highs[right_shoulder]) / 2

                    if shoulder_diff <= self.shoulder_tolerance * avg_shoulder:
                        result.iloc[right_shoulder] = 1

        return result


@indicator_registry.register('pattern')
class InverseHeadAndShoulders(BaseIndicator):
    """
    倒头肩底形态

    特征:
    - 三个谷,中间谷(头)最低
    - 两侧谷(肩)深度相近
    - 颈线由两个高点连接
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='InverseHeadAndShoulders',
        display_name='倒头肩底',
        category='pattern',
        description='三个谷,中间最低,底部反转信号',
        formula='识别三个谷值,Head < Shoulders AND 颈线突破',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'shoulder_tolerance': 0.03},
        complexity_level=4
    )

    def __init__(self, window: int = 20, shoulder_tolerance: float = 0.03):
        self.window = window
        self.shoulder_tolerance = shoulder_tolerance
        super().__init__(window=window, shoulder_tolerance=shoulder_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.shoulder_tolerance, 'shoulder_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        lows = data['low'].values
        troughs, _ = find_peaks(-lows, distance=self.window // 4)

        for i in range(len(data) - self.window):
            window_troughs = troughs[(troughs >= i) & (troughs < i + self.window)]

            if len(window_troughs) >= 3:
                trough_depths = lows[window_troughs]
                bottom_3_indices = np.argsort(trough_depths)[:3]
                sorted_troughs = window_troughs[bottom_3_indices]
                sorted_troughs = np.sort(sorted_troughs)

                left_shoulder = sorted_troughs[0]
                head = sorted_troughs[1]
                right_shoulder = sorted_troughs[2]

                if (lows[head] < lows[left_shoulder] and
                    lows[head] < lows[right_shoulder]):

                    shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder])
                    avg_shoulder = (lows[left_shoulder] + lows[right_shoulder]) / 2

                    if shoulder_diff <= self.shoulder_tolerance * avg_shoulder:
                        result.iloc[right_shoulder] = 1

        return result


@indicator_registry.register('pattern')
class DoubleTop(BaseIndicator):
    """
    双顶形态(M顶)

    特征:
    - 两个相近高度的峰
    - 中间有一个明显回调
    - 顶部反转信号
    """

    _metadata = IndicatorMetadata(
        name='DoubleTop',
        display_name='双顶(M顶)',
        category='pattern',
        description='两个相近高度的峰,顶部反转信号',
        formula='识别两个峰值,高度相近',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'peak_tolerance': 0.02, 'min_pullback': 0.05},
        complexity_level=3
    )

    def __init__(self, window: int = 20, peak_tolerance: float = 0.02, min_pullback: float = 0.05):
        """
        Args:
            window: 形态识别窗口
            peak_tolerance: 峰值高度容差
            min_pullback: 最小回调幅度
        """
        self.window = window
        self.peak_tolerance = peak_tolerance
        self.min_pullback = min_pullback
        super().__init__(window=window, peak_tolerance=peak_tolerance, min_pullback=min_pullback)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.peak_tolerance, 'peak_tolerance')
        IndicatorValidator.validate_positive_float(self.min_pullback, 'min_pullback')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        highs = data['high'].values
        lows = data['low'].values
        peaks, _ = find_peaks(highs, distance=self.window // 4)

        for i in range(1, len(peaks)):
            peak1_idx = peaks[i - 1]
            peak2_idx = peaks[i]

            peak1_high = highs[peak1_idx]
            peak2_high = highs[peak2_idx]

            # 检查峰值高度是否相近
            height_diff = abs(peak1_high - peak2_high)
            avg_peak = (peak1_high + peak2_high) / 2

            if height_diff <= self.peak_tolerance * avg_peak:
                # 检查中间的回调
                middle_low = lows[peak1_idx:peak2_idx + 1].min()
                pullback = (avg_peak - middle_low) / avg_peak

                if pullback >= self.min_pullback:
                    result.iloc[peak2_idx] = 1

        return result


@indicator_registry.register('pattern')
class DoubleBottom(BaseIndicator):
    """
    双底形态(W底)

    特征:
    - 两个相近深度的谷
    - 中间有一个明显反弹
    - 底部反转信号
    """

    _metadata = IndicatorMetadata(
        name='DoubleBottom',
        display_name='双底(W底)',
        category='pattern',
        description='两个相近深度的谷,底部反转信号',
        formula='识别两个谷值,深度相近',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'trough_tolerance': 0.02, 'min_bounce': 0.05},
        complexity_level=3
    )

    def __init__(self, window: int = 20, trough_tolerance: float = 0.02, min_bounce: float = 0.05):
        self.window = window
        self.trough_tolerance = trough_tolerance
        self.min_bounce = min_bounce
        super().__init__(window=window, trough_tolerance=trough_tolerance, min_bounce=min_bounce)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.trough_tolerance, 'trough_tolerance')
        IndicatorValidator.validate_positive_float(self.min_bounce, 'min_bounce')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        highs = data['high'].values
        lows = data['low'].values
        troughs, _ = find_peaks(-lows, distance=self.window // 4)

        for i in range(1, len(troughs)):
            trough1_idx = troughs[i - 1]
            trough2_idx = troughs[i]

            trough1_low = lows[trough1_idx]
            trough2_low = lows[trough2_idx]

            depth_diff = abs(trough1_low - trough2_low)
            avg_trough = (trough1_low + trough2_low) / 2

            if depth_diff <= self.trough_tolerance * avg_trough:
                middle_high = highs[trough1_idx:trough2_idx + 1].max()
                bounce = (middle_high - avg_trough) / avg_trough

                if bounce >= self.min_bounce:
                    result.iloc[trough2_idx] = 1

        return result


@indicator_registry.register('pattern')
class TripleTop(BaseIndicator):
    """
    三重顶形态

    特征:
    - 三个相近高度的峰
    - 顶部反转信号
    - 比双顶更强的反转信号
    """

    _metadata = IndicatorMetadata(
        name='TripleTop',
        display_name='三重顶',
        category='pattern',
        description='三个相近高度的峰,强顶部反转信号',
        formula='识别三个峰值,高度相近',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 30, 'peak_tolerance': 0.02},
        complexity_level=3
    )

    def __init__(self, window: int = 30, peak_tolerance: float = 0.02):
        self.window = window
        self.peak_tolerance = peak_tolerance
        super().__init__(window=window, peak_tolerance=peak_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.peak_tolerance, 'peak_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        highs = data['high'].values
        peaks, _ = find_peaks(highs, distance=self.window // 5)

        for i in range(2, len(peaks)):
            peak1_idx = peaks[i - 2]
            peak2_idx = peaks[i - 1]
            peak3_idx = peaks[i]

            peak1_high = highs[peak1_idx]
            peak2_high = highs[peak2_idx]
            peak3_high = highs[peak3_idx]

            avg_peak = (peak1_high + peak2_high + peak3_high) / 3
            max_diff = max(abs(peak1_high - avg_peak),
                          abs(peak2_high - avg_peak),
                          abs(peak3_high - avg_peak))

            if max_diff <= self.peak_tolerance * avg_peak:
                result.iloc[peak3_idx] = 1

        return result


@indicator_registry.register('pattern')
class TripleBottom(BaseIndicator):
    """
    三重底形态

    特征:
    - 三个相近深度的谷
    - 底部反转信号
    - 比双底更强的反转信号
    """

    _metadata = IndicatorMetadata(
        name='TripleBottom',
        display_name='三重底',
        category='pattern',
        description='三个相近深度的谷,强底部反转信号',
        formula='识别三个谷值,深度相近',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 30, 'trough_tolerance': 0.02},
        complexity_level=3
    )

    def __init__(self, window: int = 30, trough_tolerance: float = 0.02):
        self.window = window
        self.trough_tolerance = trough_tolerance
        super().__init__(window=window, trough_tolerance=trough_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.trough_tolerance, 'trough_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        lows = data['low'].values
        troughs, _ = find_peaks(-lows, distance=self.window // 5)

        for i in range(2, len(troughs)):
            trough1_idx = troughs[i - 2]
            trough2_idx = troughs[i - 1]
            trough3_idx = troughs[i]

            trough1_low = lows[trough1_idx]
            trough2_low = lows[trough2_idx]
            trough3_low = lows[trough3_idx]

            avg_trough = (trough1_low + trough2_low + trough3_low) / 3
            max_diff = max(abs(trough1_low - avg_trough),
                          abs(trough2_low - avg_trough),
                          abs(trough3_low - avg_trough))

            if max_diff <= self.trough_tolerance * avg_trough:
                result.iloc[trough3_idx] = 1

        return result


@indicator_registry.register('pattern')
class RisingWedge(BaseIndicator):
    """
    上升楔形

    特征:
    - 价格在两条收敛的上升趋势线之间运动
    - 上轨和下轨都向上倾斜
    - 通常为看跌反转信号
    """

    _metadata = IndicatorMetadata(
        name='RisingWedge',
        display_name='上升楔形',
        category='pattern',
        description='两条收敛的上升趋势线,看跌反转',
        formula='上轨斜率 > 0 AND 下轨斜率 > 0 AND 上下轨收敛',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'min_touches': 4},
        complexity_level=4
    )

    def __init__(self, window: int = 20, min_touches: int = 4):
        """
        Args:
            window: 形态识别窗口
            min_touches: 最小触碰次数
        """
        self.window = window
        self.min_touches = min_touches
        super().__init__(window=window, min_touches=min_touches)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.min_touches, 'min_touches')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            # 简化版:检查高点和低点的线性趋势
            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            # 拟合上轨和下轨
            upper_slope = np.polyfit(x, highs, 1)[0]
            lower_slope = np.polyfit(x, lows, 1)[0]

            # 上升楔形:两条线都向上,且上轨斜率小于下轨
            if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope * 1.5:
                result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class FallingWedge(BaseIndicator):
    """
    下降楔形

    特征:
    - 价格在两条收敛的下降趋势线之间运动
    - 上轨和下轨都向下倾斜
    - 通常为看涨反转信号
    """

    _metadata = IndicatorMetadata(
        name='FallingWedge',
        display_name='下降楔形',
        category='pattern',
        description='两条收敛的下降趋势线,看涨反转',
        formula='上轨斜率 < 0 AND 下轨斜率 < 0 AND 上下轨收敛',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'min_touches': 4},
        complexity_level=4
    )

    def __init__(self, window: int = 20, min_touches: int = 4):
        self.window = window
        self.min_touches = min_touches
        super().__init__(window=window, min_touches=min_touches)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_int(self.min_touches, 'min_touches')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            upper_slope = np.polyfit(x, highs, 1)[0]
            lower_slope = np.polyfit(x, lows, 1)[0]

            # 下降楔形:两条线都向下,且下轨斜率大于上轨
            if upper_slope < 0 and lower_slope < 0 and abs(lower_slope) < abs(upper_slope) * 1.5:
                result.iloc[i] = 1

        return result


# ==================== 持续形态 ====================

@indicator_registry.register('pattern')
class AscendingTriangle(BaseIndicator):
    """
    上升三角形

    特征:
    - 水平阻力线(高点连线)
    - 上升支撑线(低点连线)
    - 看涨持续形态
    """

    _metadata = IndicatorMetadata(
        name='AscendingTriangle',
        display_name='上升三角形',
        category='pattern',
        description='水平阻力+上升支撑,看涨持续',
        formula='高点水平 AND 低点上升',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'horizontal_tolerance': 0.02},
        complexity_level=3
    )

    def __init__(self, window: int = 20, horizontal_tolerance: float = 0.02):
        """
        Args:
            window: 形态识别窗口
            horizontal_tolerance: 水平线容差
        """
        self.window = window
        self.horizontal_tolerance = horizontal_tolerance
        super().__init__(window=window, horizontal_tolerance=horizontal_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.horizontal_tolerance, 'horizontal_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            # 检查高点是否水平
            upper_slope = np.polyfit(x, highs, 1)[0]
            avg_high = np.mean(highs)

            # 检查低点是否上升
            lower_slope = np.polyfit(x, lows, 1)[0]

            if (abs(upper_slope) <= self.horizontal_tolerance * avg_high / self.window and
                lower_slope > 0):
                result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class DescendingTriangle(BaseIndicator):
    """
    下降三角形

    特征:
    - 水平支撑线(低点连线)
    - 下降阻力线(高点连线)
    - 看跌持续形态
    """

    _metadata = IndicatorMetadata(
        name='DescendingTriangle',
        display_name='下降三角形',
        category='pattern',
        description='水平支撑+下降阻力,看跌持续',
        formula='低点水平 AND 高点下降',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'horizontal_tolerance': 0.02},
        complexity_level=3
    )

    def __init__(self, window: int = 20, horizontal_tolerance: float = 0.02):
        self.window = window
        self.horizontal_tolerance = horizontal_tolerance
        super().__init__(window=window, horizontal_tolerance=horizontal_tolerance)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.horizontal_tolerance, 'horizontal_tolerance')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            # 检查低点是否水平
            lower_slope = np.polyfit(x, lows, 1)[0]
            avg_low = np.mean(lows)

            # 检查高点是否下降
            upper_slope = np.polyfit(x, highs, 1)[0]

            if (abs(lower_slope) <= self.horizontal_tolerance * avg_low / self.window and
                upper_slope < 0):
                result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class SymmetricalTriangle(BaseIndicator):
    """
    对称三角形

    特征:
    - 上轨下降
    - 下轨上升
    - 两线收敛
    - 通常为持续形态
    """

    _metadata = IndicatorMetadata(
        name='SymmetricalTriangle',
        display_name='对称三角形',
        category='pattern',
        description='上轨下降下轨上升,收敛形态',
        formula='上轨斜率 < 0 AND 下轨斜率 > 0',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20},
        complexity_level=3
    )

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            upper_slope = np.polyfit(x, highs, 1)[0]
            lower_slope = np.polyfit(x, lows, 1)[0]

            # 对称三角形:上轨下降,下轨上升
            if upper_slope < 0 and lower_slope > 0:
                # 检查斜率是否相近(对称)
                if abs(abs(upper_slope) - abs(lower_slope)) < abs(upper_slope) * 0.5:
                    result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class BullFlag(BaseIndicator):
    """
    看涨旗形

    特征:
    - 强势上涨后的小幅下降通道
    - 旗杆:急速上涨
    - 旗面:平行下降通道
    - 看涨持续形态
    """

    _metadata = IndicatorMetadata(
        name='BullFlag',
        display_name='看涨旗形',
        category='pattern',
        description='急速上涨后的下降通道,看涨持续',
        formula='前期强势上涨 + 小幅回调整理',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'pole_period': 10, 'flag_period': 10, 'min_pole_gain': 0.05},
        complexity_level=3
    )

    def __init__(self, pole_period: int = 10, flag_period: int = 10, min_pole_gain: float = 0.05):
        """
        Args:
            pole_period: 旗杆周期
            flag_period: 旗面周期
            min_pole_gain: 旗杆最小涨幅
        """
        self.pole_period = pole_period
        self.flag_period = flag_period
        self.min_pole_gain = min_pole_gain
        super().__init__(pole_period=pole_period, flag_period=flag_period, min_pole_gain=min_pole_gain)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.pole_period, 'pole_period')
        IndicatorValidator.validate_positive_int(self.flag_period, 'flag_period')
        IndicatorValidator.validate_positive_float(self.min_pole_gain, 'min_pole_gain')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)
        total_period = self.pole_period + self.flag_period

        for i in range(total_period, len(data)):
            # 旗杆:前期急速上涨
            pole_start = i - total_period
            pole_end = i - self.flag_period
            pole_gain = (data['close'].iloc[pole_end] - data['close'].iloc[pole_start]) / data['close'].iloc[pole_start]

            if pole_gain >= self.min_pole_gain:
                # 旗面:小幅下降
                flag_data = data.iloc[pole_end:i]
                flag_slope = np.polyfit(range(len(flag_data)), flag_data['close'].values, 1)[0]

                # 旗面应该向下或水平
                if flag_slope <= 0:
                    result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class BearFlag(BaseIndicator):
    """
    看跌旗形

    特征:
    - 强势下跌后的小幅上升通道
    - 旗杆:急速下跌
    - 旗面:平行上升通道
    - 看跌持续形态
    """

    _metadata = IndicatorMetadata(
        name='BearFlag',
        display_name='看跌旗形',
        category='pattern',
        description='急速下跌后的上升通道,看跌持续',
        formula='前期强势下跌 + 小幅反弹整理',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'pole_period': 10, 'flag_period': 10, 'min_pole_loss': 0.05},
        complexity_level=3
    )

    def __init__(self, pole_period: int = 10, flag_period: int = 10, min_pole_loss: float = 0.05):
        self.pole_period = pole_period
        self.flag_period = flag_period
        self.min_pole_loss = min_pole_loss
        super().__init__(pole_period=pole_period, flag_period=flag_period, min_pole_loss=min_pole_loss)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.pole_period, 'pole_period')
        IndicatorValidator.validate_positive_int(self.flag_period, 'flag_period')
        IndicatorValidator.validate_positive_float(self.min_pole_loss, 'min_pole_loss')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)
        total_period = self.pole_period + self.flag_period

        for i in range(total_period, len(data)):
            pole_start = i - total_period
            pole_end = i - self.flag_period
            pole_loss = (data['close'].iloc[pole_start] - data['close'].iloc[pole_end]) / data['close'].iloc[pole_start]

            if pole_loss >= self.min_pole_loss:
                flag_data = data.iloc[pole_end:i]
                flag_slope = np.polyfit(range(len(flag_data)), flag_data['close'].values, 1)[0]

                # 旗面应该向上或水平
                if flag_slope >= 0:
                    result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class Pennant(BaseIndicator):
    """
    三角旗形

    特征:
    - 强势趋势后的小型对称三角形
    - 旗杆:急速移动
    - 旗面:收敛的三角形
    - 持续形态
    """

    _metadata = IndicatorMetadata(
        name='Pennant',
        display_name='三角旗形',
        category='pattern',
        description='急速移动后的小型三角形整理',
        formula='前期强势移动 + 收敛三角形',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'pole_period': 10, 'pennant_period': 10, 'min_pole_move': 0.05},
        complexity_level=3
    )

    def __init__(self, pole_period: int = 10, pennant_period: int = 10, min_pole_move: float = 0.05):
        self.pole_period = pole_period
        self.pennant_period = pennant_period
        self.min_pole_move = min_pole_move
        super().__init__(pole_period=pole_period, pennant_period=pennant_period, min_pole_move=min_pole_move)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.pole_period, 'pole_period')
        IndicatorValidator.validate_positive_int(self.pennant_period, 'pennant_period')
        IndicatorValidator.validate_positive_float(self.min_pole_move, 'min_pole_move')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)
        total_period = self.pole_period + self.pennant_period

        for i in range(total_period, len(data)):
            pole_start = i - total_period
            pole_end = i - self.pennant_period

            # 检查旗杆的强势移动
            pole_move = abs(data['close'].iloc[pole_end] - data['close'].iloc[pole_start]) / data['close'].iloc[pole_start]

            if pole_move >= self.min_pole_move:
                # 检查三角旗面
                pennant_data = data.iloc[pole_end:i]
                highs = pennant_data['high'].values
                lows = pennant_data['low'].values
                x = np.arange(len(highs))

                if len(x) > 2:
                    upper_slope = np.polyfit(x, highs, 1)[0]
                    lower_slope = np.polyfit(x, lows, 1)[0]

                    # 收敛的三角形
                    if upper_slope < 0 and lower_slope > 0:
                        result.iloc[i] = 1

        return result


@indicator_registry.register('pattern')
class Rectangle(BaseIndicator):
    """
    矩形整理

    特征:
    - 价格在水平的支撑和阻力之间震荡
    - 上轨和下轨都接近水平
    - 可能是持续或反转形态
    """

    _metadata = IndicatorMetadata(
        name='Rectangle',
        display_name='矩形整理',
        category='pattern',
        description='价格在水平支撑阻力间震荡',
        formula='高点水平 AND 低点水平',
        required_columns=['high', 'low', 'close'],
        output_type='series',
        default_params={'window': 20, 'horizontal_tolerance': 0.02, 'min_bounces': 4},
        complexity_level=3
    )

    def __init__(self, window: int = 20, horizontal_tolerance: float = 0.02, min_bounces: int = 4):
        """
        Args:
            window: 形态识别窗口
            horizontal_tolerance: 水平线容差
            min_bounces: 最小反弹次数
        """
        self.window = window
        self.horizontal_tolerance = horizontal_tolerance
        self.min_bounces = min_bounces
        super().__init__(window=window, horizontal_tolerance=horizontal_tolerance, min_bounces=min_bounces)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.window, 'window')
        IndicatorValidator.validate_positive_float(self.horizontal_tolerance, 'horizontal_tolerance')
        IndicatorValidator.validate_positive_int(self.min_bounces, 'min_bounces')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        result = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            window_data = data.iloc[i - self.window:i]

            highs = window_data['high'].values
            lows = window_data['low'].values
            x = np.arange(len(highs))

            # 检查上轨和下轨是否都接近水平
            upper_slope = np.polyfit(x, highs, 1)[0]
            lower_slope = np.polyfit(x, lows, 1)[0]

            avg_high = np.mean(highs)
            avg_low = np.mean(lows)

            if (abs(upper_slope) <= self.horizontal_tolerance * avg_high / self.window and
                abs(lower_slope) <= self.horizontal_tolerance * avg_low / self.window):

                # 检查价格区间
                price_range = avg_high - avg_low
                if price_range > 0.02 * avg_low:  # 至少有2%的波动空间
                    result.iloc[i] = 1

        return result
