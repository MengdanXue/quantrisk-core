"""
指标基类
所有技术指标的抽象基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np

from .indicator_exceptions import (
    InvalidParameterError,
    InsufficientDataError,
    CalculationError,
    MissingColumnError
)


@dataclass
class IndicatorMetadata:
    """指标元数据"""
    name: str
    display_name: str = ""
    category: str = "custom"
    description: str = ""
    formula: str = ""
    required_columns: List[str] = None  # 旧版字段
    output_type: str = 'series'  # 'series', 'dataframe', 'dict'
    default_params: Dict[str, Any] = None
    complexity_level: int = 2  # 1-5
    version: str = "1.0.0"
    # 扩展字段 (用于更详细的指标定义)
    parameters: Dict[str, Any] = None  # 参数描述
    required_inputs: List[str] = None  # 输入数据列 (替代 required_columns)

    def __post_init__(self):
        # 兼容性处理：如果只设置了required_inputs，则同步到required_columns
        if self.required_inputs and not self.required_columns:
            self.required_columns = self.required_inputs
        elif self.required_columns and not self.required_inputs:
            self.required_inputs = self.required_columns
        # 默认值处理
        if self.required_columns is None:
            self.required_columns = ['close']
        if self.default_params is None:
            self.default_params = {}
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有元数据字段的字典
        """
        return {
            'name': self.name,
            'display_name': self.display_name,
            'category': self.category,
            'description': self.description,
            'formula': self.formula,
            'required_columns': self.required_columns,
            'output_type': self.output_type,
            'default_params': self.default_params,
            'complexity_level': self.complexity_level,
            'version': self.version,
            'parameters': self.parameters,
            'required_inputs': self.required_inputs,
        }


class BaseIndicator(ABC):
    """
    指标基类

    所有技术指标必须继承此类并实现抽象方法
    """

    # 类属性：指标元数据（子类必须定义）
    _metadata: Optional[IndicatorMetadata] = None

    def __init__(self, **params):
        """
        初始化指标

        Args:
            **params: 指标参数
        """
        self.params = params
        self._validate_params()

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, Dict]:
        """
        计算指标值

        Args:
            data: OHLCV数据，必须包含required_columns指定的列

        Returns:
            指标计算结果，类型由output_type决定

        Raises:
            InsufficientDataError: 数据不足
            CalculationError: 计算错误
        """
        pass

    @abstractmethod
    def _validate_params(self):
        """
        验证参数

        Raises:
            InvalidParameterError: 参数无效
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        需要的数据列

        Returns:
            列名列表，如 ['close', 'volume']
        """
        pass

    @property
    def name(self) -> str:
        """指标名称"""
        if self._metadata:
            return self._metadata.name
        return self.__class__.__name__

    @property
    def category(self) -> str:
        """指标类别"""
        if self._metadata:
            return self._metadata.category
        return "unknown"

    @property
    def display_name(self) -> str:
        """显示名称"""
        if self._metadata:
            return self._metadata.display_name
        return self.name

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        """
        获取指标元数据

        Returns:
            IndicatorMetadata对象
        """
        if cls._metadata is None:
            raise NotImplementedError(
                f"{cls.__name__} must define _metadata class attribute"
            )
        return cls._metadata

    def validate_data(self, data: pd.DataFrame):
        """
        验证输入数据

        Args:
            data: OHLCV数据

        Raises:
            MissingColumnError: 缺少必需列
        """
        # 检查必需列
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise MissingColumnError(
                f"Missing required columns: {missing_columns}"
            )

        # 注意：不再检查数据量，让滚动计算自然返回NaN
        # 这样用户可以处理短数据而不会抛异常

    def _get_min_periods(self) -> int:
        """
        获取最小数据周期数

        Returns:
            最小周期数
        """
        # 默认实现：查找params中的period参数
        if 'period' in self.params:
            return self.params['period']
        elif 'window' in self.params:
            return self.params['window']
        elif 'length' in self.params:
            return self.params['length']
        else:
            return 1

    def _safe_division(self, numerator: Union[pd.Series, np.ndarray],
                      denominator: Union[pd.Series, np.ndarray],
                      fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
        """
        安全除法（避免除零）

        Args:
            numerator: 分子
            denominator: 分母
            fill_value: 除零时的填充值

        Returns:
            除法结果
        """
        if isinstance(numerator, pd.Series):
            result = numerator.divide(denominator).replace([np.inf, -np.inf], np.nan)
            return result.fillna(fill_value)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(numerator, denominator)
                result[~np.isfinite(result)] = fill_value
                return result

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"

    def __str__(self) -> str:
        return self.display_name
