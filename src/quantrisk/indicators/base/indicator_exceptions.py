"""
指标计算异常类
"""


class IndicatorError(Exception):
    """指标基础异常"""
    pass


class InvalidParameterError(IndicatorError):
    """无效参数异常"""
    pass


class InsufficientDataError(IndicatorError):
    """数据不足异常"""
    pass


class CalculationError(IndicatorError):
    """计算错误异常"""
    pass


class MissingColumnError(IndicatorError):
    """缺少必要列异常"""
    pass


# 别名，保持向后兼容
IndicatorCalculationError = CalculationError
