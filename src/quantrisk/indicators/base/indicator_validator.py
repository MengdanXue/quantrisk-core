"""
指标参数验证器
"""

from typing import Any, Dict, List, Type
from .indicator_base import BaseIndicator
from .indicator_exceptions import InvalidParameterError


class IndicatorValidator:
    """指标参数验证器"""

    @staticmethod
    def validate_positive_int(value: Any, param_name: str):
        """验证正整数"""
        if not isinstance(value, int) or value <= 0:
            raise InvalidParameterError(
                f"{param_name} must be a positive integer, got {value}"
            )

    @staticmethod
    def validate_positive_float(value: Any, param_name: str):
        """验证正浮点数"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise InvalidParameterError(
                f"{param_name} must be a positive number, got {value}"
            )

    @staticmethod
    def validate_range(value: Any, param_name: str, min_val: float, max_val: float):
        """验证范围"""
        if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
            raise InvalidParameterError(
                f"{param_name} must be between {min_val} and {max_val}, got {value}"
            )

    @staticmethod
    def validate_period(value: Any, min_val: int = 1, max_val: int = None):
        """Validate the legacy period contract used by extended indicators."""
        if not isinstance(value, int) or value < min_val or (
            max_val is not None and value > max_val
        ):
            upper = f" and at most {max_val}" if max_val is not None else ""
            raise InvalidParameterError(
                f"period must be an integer of at least {min_val}{upper}, got {value}"
            )

    @staticmethod
    def validate_in_range(
        value: Any,
        min_val: float,
        max_val: float,
        param_name: str,
    ):
        """Compatibility wrapper for indicators using the older argument order."""
        IndicatorValidator.validate_range(value, param_name, min_val, max_val)

    @staticmethod
    def validate_in_list(value: Any, param_name: str, valid_values: List[Any]):
        """验证是否在有效值列表中"""
        if value not in valid_values:
            raise InvalidParameterError(
                f"{param_name} must be one of {valid_values}, got {value}"
            )

    @staticmethod
    def validate_type(value: Any, param_name: str, expected_type: Type):
        """验证类型"""
        if not isinstance(value, expected_type):
            raise InvalidParameterError(
                f"{param_name} must be of type {expected_type.__name__}, got {type(value).__name__}"
            )

    @classmethod
    def validate_indicator_params(cls, indicator: BaseIndicator, params: Dict[str, Any]):
        """
        验证指标参数

        Args:
            indicator: 指标实例
            params: 参数字典
        """
        # 调用指标自己的验证方法
        indicator._validate_params()
