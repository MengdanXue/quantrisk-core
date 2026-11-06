"""
机器学习特征指标模块

提供20个专为机器学习设计的特征工程指标：
- 时间序列特征：LagFeatures, RollingFeatures, DiffFeatures, AutocorrelationFeature, PartialAutocorrelation
- 统计特征：DistributionFeatures, OutlierScore, ChangePointDetection, TrendDecomposition, SeasonalDecomposition
- 技术特征组合：IndicatorClusters, TechnicalPCA, FeatureInteraction, NonlinearTransform, PolynomialFeatures
- 标签生成：TrendLabeler, VolatilityRegime, ReturnBuckets, TripleBarrier, MetaLabeling
"""

from .ml_feature_indicators import (
    # 时间序列特征
    LagFeatures,
    RollingFeatures,
    DiffFeatures,
    AutocorrelationFeature,
    PartialAutocorrelation,

    # 统计特征
    DistributionFeatures,
    OutlierScore,
    ChangePointDetection,
    TrendDecomposition,
    SeasonalDecomposition,

    # 技术特征组合
    IndicatorClusters,
    TechnicalPCA,
    FeatureInteraction,
    NonlinearTransform,
    PolynomialFeatures,

    # 标签生成
    TrendLabeler,
    VolatilityRegime,
    ReturnBuckets,
    TripleBarrier,
    MetaLabeling,
)

__all__ = [
    # 时间序列特征 (5个)
    'LagFeatures',
    'RollingFeatures',
    'DiffFeatures',
    'AutocorrelationFeature',
    'PartialAutocorrelation',

    # 统计特征 (5个)
    'DistributionFeatures',
    'OutlierScore',
    'ChangePointDetection',
    'TrendDecomposition',
    'SeasonalDecomposition',

    # 技术特征组合 (5个)
    'IndicatorClusters',
    'TechnicalPCA',
    'FeatureInteraction',
    'NonlinearTransform',
    'PolynomialFeatures',

    # 标签生成 (5个)
    'TrendLabeler',
    'VolatilityRegime',
    'ReturnBuckets',
    'TripleBarrier',
    'MetaLabeling',
]


# 指标分类映射
ML_FEATURE_CATEGORIES = {
    'time_series': [
        'LagFeatures',
        'RollingFeatures',
        'DiffFeatures',
        'AutocorrelationFeature',
        'PartialAutocorrelation',
    ],
    'statistical': [
        'DistributionFeatures',
        'OutlierScore',
        'ChangePointDetection',
        'TrendDecomposition',
        'SeasonalDecomposition',
    ],
    'technical_combination': [
        'IndicatorClusters',
        'TechnicalPCA',
        'FeatureInteraction',
        'NonlinearTransform',
        'PolynomialFeatures',
    ],
    'label_generation': [
        'TrendLabeler',
        'VolatilityRegime',
        'ReturnBuckets',
        'TripleBarrier',
        'MetaLabeling',
    ],
}


def get_ml_feature_indicators(category: str = None):
    """
    获取机器学习特征指标列表

    Args:
        category: 类别名称，可选值：
            - 'time_series': 时间序列特征
            - 'statistical': 统计特征
            - 'technical_combination': 技术特征组合
            - 'label_generation': 标签生成
            - None: 返回所有指标

    Returns:
        指标名称列表
    """
    if category is None:
        return __all__

    if category not in ML_FEATURE_CATEGORIES:
        raise ValueError(
            f"Invalid category: {category}. "
            f"Must be one of {list(ML_FEATURE_CATEGORIES.keys())}"
        )

    return ML_FEATURE_CATEGORIES[category]


def get_category_info():
    """
    获取所有类别的信息

    Returns:
        包含类别和数量的字典
    """
    return {
        category: {
            'count': len(indicators),
            'indicators': indicators
        }
        for category, indicators in ML_FEATURE_CATEGORIES.items()
    }
