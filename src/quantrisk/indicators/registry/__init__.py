"""
指标注册模块
提供指标的注册、发现和管理功能
"""

from .indicator_registry import IndicatorRegistry, indicator_registry

__all__ = [
    'IndicatorRegistry',
    'indicator_registry',
]
