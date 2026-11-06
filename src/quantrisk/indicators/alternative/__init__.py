"""
另类数据指标模块

包含基于社交媒体、新闻、链上数据、资金流向和其他另类数据源的20个指标
"""

from .alternative_indicators import (
    # 社交媒体指标 (4个)
    SocialSentiment,
    MentionVolume,
    InfluencerActivity,
    ViralityScore,

    # 新闻情绪 (4个)
    NewsSentiment,
    HeadlineImpact,
    MediaCoverage,
    EventSurprise,

    # 链上数据 (4个)
    WhaleActivity,
    ExchangeFlow,
    ActiveAddresses,
    NetworkHashRate,

    # 资金流向 (4个)
    InstitutionalFlow,
    RetailFlow,
    ETFFlow,
    CrossBorderFlow,

    # 另类信号 (4个)
    GoogleTrends,
    SatelliteData,
    WeatherImpact,
    GeopoliticalRisk,
)

__all__ = [
    # 社交媒体指标
    'SocialSentiment',
    'MentionVolume',
    'InfluencerActivity',
    'ViralityScore',

    # 新闻情绪
    'NewsSentiment',
    'HeadlineImpact',
    'MediaCoverage',
    'EventSurprise',

    # 链上数据
    'WhaleActivity',
    'ExchangeFlow',
    'ActiveAddresses',
    'NetworkHashRate',

    # 资金流向
    'InstitutionalFlow',
    'RetailFlow',
    'ETFFlow',
    'CrossBorderFlow',

    # 另类信号
    'GoogleTrends',
    'SatelliteData',
    'WeatherImpact',
    'GeopoliticalRisk',
]
