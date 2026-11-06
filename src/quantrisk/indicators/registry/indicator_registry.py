"""
指标注册表
全局指标注册和管理系统
"""

from typing import Dict, List, Type, Optional, Any
import logging

from ..base.indicator_base import BaseIndicator, IndicatorMetadata

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """
    指标注册表

    支持按类别注册和查询指标，提供全局单例访问
    """

    # 指标类别列表
    CATEGORIES = [
        'trend',           # 趋势类
        'momentum',        # 动量类
        'volatility',      # 波动率
        'volume',          # 成交量
        'statistical',     # 统计类
        'multi_factor',    # 多因子
        'sentiment',       # 情绪类
        'microstructure',  # 市场微观结构
        'pattern',         # 形态识别
        'derivatives',     # 衍生品
        'ml_features',     # 机器学习特征
        'cross_market',    # 跨市场分析
        'custom',          # 自定义
        'advanced_stats',  # 高级统计
        'cycle',           # 周期分析
        'alternative',     # 另类数据
        'intermarket',     # 跨品种分析
        'execution',       # 执行质量
    ]

    def __init__(self):
        """初始化注册表"""
        self._indicators: Dict[str, Dict[str, Type[BaseIndicator]]] = {
            category: {} for category in self.CATEGORIES
        }
        self._metadata_cache: Dict[str, IndicatorMetadata] = {}

    def register(self, category: str):
        """
        装饰器：注册指标到指定类别

        Usage:
            @indicator_registry.register('trend')
            class MyIndicator(BaseIndicator):
                ...

        Args:
            category: 指标类别

        Returns:
            装饰器函数
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {self.CATEGORIES}")

        def decorator(indicator_class: Type[BaseIndicator]) -> Type[BaseIndicator]:
            name = indicator_class.__name__
            self._indicators[category][name] = indicator_class

            # 缓存元数据
            if hasattr(indicator_class, '_metadata') and indicator_class._metadata:
                self._metadata_cache[name] = indicator_class._metadata

            logger.debug(f"Registered indicator: {name} in category: {category}")
            return indicator_class

        return decorator

    def get(self, name: str) -> Optional[Type[BaseIndicator]]:
        """
        获取指标类

        Args:
            name: 指标名称

        Returns:
            指标类，如果不存在返回None
        """
        for category_indicators in self._indicators.values():
            if name in category_indicators:
                return category_indicators[name]
        return None

    def get_by_category(self, category: str) -> Dict[str, Type[BaseIndicator]]:
        """
        获取某类别的所有指标

        Args:
            category: 类别名称

        Returns:
            指标字典 {名称: 类}
        """
        return self._indicators.get(category, {})

    def get_all(self) -> Dict[str, Dict[str, Type[BaseIndicator]]]:
        """
        获取所有指标

        Returns:
            完整的指标注册表
        """
        return self._indicators

    def list_all(self) -> List[str]:
        """
        列出所有已注册的指标名称

        Returns:
            指标名称列表
        """
        all_names = []
        for category_indicators in self._indicators.values():
            all_names.extend(category_indicators.keys())
        return sorted(all_names)

    def list_by_category(self, category: str) -> List[str]:
        """
        列出某类别的所有指标名称

        Args:
            category: 类别名称

        Returns:
            指标名称列表
        """
        return sorted(self._indicators.get(category, {}).keys())

    def search(self, keyword: str) -> List[Dict[str, str]]:
        """
        搜索指标

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的指标列表，包含名称和类别
        """
        results = []
        keyword_lower = keyword.lower()

        for category, indicators in self._indicators.items():
            for name, indicator_class in indicators.items():
                # 搜索名称
                if keyword_lower in name.lower():
                    results.append({
                        'name': name,
                        'category': category,
                        'display_name': getattr(indicator_class, '_metadata', None)
                                       and indicator_class._metadata.display_name or name
                    })
                    continue

                # 搜索元数据
                if hasattr(indicator_class, '_metadata') and indicator_class._metadata:
                    metadata = indicator_class._metadata
                    if (keyword_lower in metadata.display_name.lower() or
                        keyword_lower in metadata.description.lower()):
                        results.append({
                            'name': name,
                            'category': category,
                            'display_name': metadata.display_name
                        })

        return results

    def get_metadata(self, name: str) -> Optional[IndicatorMetadata]:
        """
        获取指标元数据

        Args:
            name: 指标名称

        Returns:
            元数据，如果不存在返回None
        """
        return self._metadata_cache.get(name)

    def get_category_stats(self) -> Dict[str, int]:
        """
        获取各类别的指标数量统计

        Returns:
            {类别: 数量}
        """
        return {
            category: len(indicators)
            for category, indicators in self._indicators.items()
        }

    def get_total_count(self) -> int:
        """
        获取总指标数量

        Returns:
            总数量
        """
        return sum(len(indicators) for indicators in self._indicators.values())

    def create_indicator(self, name: str, **params) -> BaseIndicator:
        """
        创建指标实例

        Args:
            name: 指标名称
            **params: 指标参数

        Returns:
            指标实例

        Raises:
            ValueError: 指标不存在
        """
        indicator_class = self.get(name)
        if indicator_class is None:
            raise ValueError(f"Indicator not found: {name}")

        return indicator_class(**params)


# 全局单例
indicator_registry = IndicatorRegistry()
