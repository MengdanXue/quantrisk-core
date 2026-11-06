"""
形态识别指标模块

包含K线形态和图表形态识别指标
"""

from .candlestick_patterns import (
    # 单根K线形态
    Hammer,
    InvertedHammer,
    HangingMan,
    ShootingStar,
    Doji,
    Dragonfly,
    Gravestone,
    MarubozuWhite,
    MarubozuBlack,
    SpinningTop,

    # 双K线形态
    BullishEngulfing,
    BearishEngulfing,
    Piercing,
    DarkCloudCover,
    TweezerBottom,
)

from .chart_patterns import (
    # 反转形态
    HeadAndShoulders,
    InverseHeadAndShoulders,
    DoubleTop,
    DoubleBottom,
    TripleTop,
    TripleBottom,
    RisingWedge,
    FallingWedge,

    # 持续形态
    AscendingTriangle,
    DescendingTriangle,
    SymmetricalTriangle,
    BullFlag,
    BearFlag,
    Pennant,
    Rectangle,
)

__all__ = [
    # 单根K线形态
    'Hammer',
    'InvertedHammer',
    'HangingMan',
    'ShootingStar',
    'Doji',
    'Dragonfly',
    'Gravestone',
    'MarubozuWhite',
    'MarubozuBlack',
    'SpinningTop',

    # 双K线形态
    'BullishEngulfing',
    'BearishEngulfing',
    'Piercing',
    'DarkCloudCover',
    'TweezerBottom',

    # 反转形态
    'HeadAndShoulders',
    'InverseHeadAndShoulders',
    'DoubleTop',
    'DoubleBottom',
    'TripleTop',
    'TripleBottom',
    'RisingWedge',
    'FallingWedge',

    # 持续形态
    'AscendingTriangle',
    'DescendingTriangle',
    'SymmetricalTriangle',
    'BullFlag',
    'BearFlag',
    'Pennant',
    'Rectangle',
]
