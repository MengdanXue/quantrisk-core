"""
高级配对交易模块 (Advanced Pairs Trading Module)

提供专业的统计套利工具:
- 协整检验 (Cointegration Test)
- 半衰期计算 (Half-Life Calculation)
- 配对选择 (Pair Selection)
- 动态对冲比率 (Dynamic Hedge Ratio)
- 卡尔曼滤波 (Kalman Filter)

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

class SpreadType(str, Enum):
    """价差类型"""
    RATIO = "ratio"             # 比率价差
    DIFFERENCE = "difference"   # 差值价差
    LOG_RATIO = "log_ratio"     # 对数比率
    RESIDUAL = "residual"       # 回归残差


class SignalType(str, Enum):
    """信号类型"""
    LONG_SPREAD = "long_spread"     # 做多价差
    SHORT_SPREAD = "short_spread"   # 做空价差
    CLOSE = "close"                 # 平仓
    HOLD = "hold"                   # 持有


@dataclass
class CointegrationResult:
    """协整检验结果"""
    asset1: str
    asset2: str
    statistic: float            # 检验统计量
    p_value: float              # P值
    critical_values: Dict[str, float]  # 临界值
    is_cointegrated: bool       # 是否协整
    hedge_ratio: float          # 对冲比率
    half_life: float            # 半衰期
    mean_spread: float          # 平均价差
    std_spread: float           # 价差标准差

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset1': self.asset1,
            'asset2': self.asset2,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_cointegrated': self.is_cointegrated,
            'hedge_ratio': self.hedge_ratio,
            'half_life': self.half_life,
            'mean_spread': self.mean_spread,
            'std_spread': self.std_spread,
        }


@dataclass
class PairSignal:
    """配对交易信号"""
    timestamp: datetime
    asset1: str
    asset2: str
    signal_type: SignalType
    zscore: float
    spread: float
    hedge_ratio: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairPosition:
    """配对持仓"""
    asset1: str
    asset2: str
    asset1_qty: float           # 资产1数量（正=多，负=空）
    asset2_qty: float           # 资产2数量
    entry_spread: float         # 入场价差
    entry_zscore: float         # 入场Z-Score
    entry_time: datetime
    current_spread: float = 0.0
    current_zscore: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PairsTradingConfig:
    """配对交易配置"""
    # 协整检验参数
    cointegration_window: int = 252     # 协整检验窗口
    cointegration_pvalue: float = 0.05  # 协整P值阈值

    # 信号参数
    entry_zscore: float = 2.0           # 入场Z-Score
    exit_zscore: float = 0.5            # 出场Z-Score
    stop_loss_zscore: float = 3.5       # 止损Z-Score
    zscore_window: int = 60             # Z-Score计算窗口

    # 半衰期约束
    min_half_life: int = 5              # 最小半衰期（天）
    max_half_life: int = 60             # 最大半衰期（天）

    # 价差类型
    spread_type: SpreadType = SpreadType.RESIDUAL

    # 风险参数
    max_position_value: float = 100000  # 最大持仓价值
    max_pairs: int = 10                 # 最大配对数


# ==================== 协整检验 ====================

class CointegrationTest:
    """协整检验器"""

    @staticmethod
    def engle_granger(
        y1: pd.Series,
        y2: pd.Series,
        significance: float = 0.05,
    ) -> CointegrationResult:
        """
        Engle-Granger两步法协整检验

        Args:
            y1: 第一个时间序列
            y2: 第二个时间序列
            significance: 显著性水平

        Returns:
            协整检验结果
        """
        # 对齐数据
        y1 = y1.dropna()
        y2 = y2.dropna()
        common_idx = y1.index.intersection(y2.index)
        y1 = y1.loc[common_idx]
        y2 = y2.loc[common_idx]

        if len(y1) < 30:
            return CointegrationResult(
                asset1=y1.name or "asset1",
                asset2=y2.name or "asset2",
                statistic=0,
                p_value=1.0,
                critical_values={},
                is_cointegrated=False,
                hedge_ratio=1.0,
                half_life=float('inf'),
                mean_spread=0,
                std_spread=0,
            )

        # Step 1: OLS回归获取对冲比率
        X = np.column_stack([np.ones(len(y2)), y2.values])
        y = y1.values
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            intercept, hedge_ratio = coeffs
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"OLS regression failed for hedge ratio calculation: {e}")
            hedge_ratio = 1.0
            intercept = 0

        # 计算残差（价差）
        spread = y1 - hedge_ratio * y2

        # Step 2: ADF检验残差
        statistic, p_value, crit_values = CointegrationTest._adf_test(spread)

        # 计算半衰期
        half_life = HalfLife.calculate(spread)

        # 判断是否协整
        is_cointegrated = p_value < significance

        return CointegrationResult(
            asset1=y1.name or "asset1",
            asset2=y2.name or "asset2",
            statistic=statistic,
            p_value=p_value,
            critical_values=crit_values,
            is_cointegrated=is_cointegrated,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            mean_spread=spread.mean(),
            std_spread=spread.std(),
        )

    @staticmethod
    def _adf_test(series: pd.Series) -> Tuple[float, float, Dict[str, float]]:
        """
        ADF单位根检验

        Returns:
            (统计量, p值, 临界值字典)
        """
        series = series.dropna()
        n = len(series)

        if n < 20:
            return 0, 1.0, {}

        # 计算差分和滞后
        y = series.values
        dy = np.diff(y)
        y_lag = y[:-1]

        # 回归: dy = rho * y_lag + epsilon
        X = y_lag.reshape(-1, 1)
        try:
            rho = np.linalg.lstsq(X, dy, rcond=None)[0][0]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"ADF test regression failed: {e}")
            return 0, 1.0, {}

        # 计算残差标准误
        residuals = dy - rho * y_lag
        se = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(np.sum(y_lag**2))

        # ADF统计量
        t_stat = rho / se if se > 0 else 0

        # 使用MacKinnon临界值的近似
        # p值近似
        if t_stat < -3.43:
            p_value = 0.01
        elif t_stat < -2.86:
            p_value = 0.05
        elif t_stat < -2.57:
            p_value = 0.10
        else:
            p_value = min(1.0, 0.10 + (t_stat + 2.57) * 0.3)

        crit_values = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57,
        }

        return t_stat, p_value, crit_values

    @staticmethod
    def johansen(
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> Dict[str, Any]:
        """
        Johansen协整检验（简化版本）

        用于检验多个时间序列之间的协整关系

        Args:
            data: 包含多个资产价格的DataFrame
            det_order: 决定性趋势阶数
            k_ar_diff: VAR差分阶数

        Returns:
            检验结果字典
        """
        n = len(data)
        k = len(data.columns)

        if n < 50 or k < 2:
            return {
                'trace_stat': [],
                'critical_values': [],
                'num_cointegrating': 0,
            }

        # 简化的Johansen检验
        # 计算差分和滞后
        dy = data.diff().dropna()
        y_lag = data.shift(1).dropna()

        # 对齐
        common_idx = dy.index.intersection(y_lag.index)
        dy = dy.loc[common_idx]
        y_lag = y_lag.loc[common_idx]

        # 计算协方差矩阵
        try:
            cov_dy = dy.cov()
            cov_y_lag = y_lag.cov()

            # 特征值分解
            matrix = np.linalg.solve(cov_dy.values, cov_y_lag.values)
            eigenvalues = np.real(np.linalg.eigvals(matrix))
            eigenvalues = np.sort(eigenvalues)[::-1]

            # 迹统计量
            trace_stats = []
            for i in range(k):
                trace = -n * np.sum(np.log(1 - eigenvalues[i:]))
                trace_stats.append(trace)

            # 临界值（近似）
            critical_values_5pct = [15.41, 3.76][:k]  # 简化

            num_coint = sum(
                1 for i, t in enumerate(trace_stats)
                if i < len(critical_values_5pct) and t > critical_values_5pct[i]
            )

        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Johansen test calculation failed: {e}")
            return {
                'trace_stat': [],
                'critical_values': [],
                'num_cointegrating': 0,
            }

        return {
            'trace_stat': trace_stats,
            'critical_values': critical_values_5pct,
            'num_cointegrating': num_coint,
        }


# ==================== 半衰期计算 ====================

class HalfLife:
    """半衰期计算器"""

    @staticmethod
    def calculate(spread: pd.Series) -> float:
        """
        计算均值回归半衰期

        使用AR(1)模型: spread_t = rho * spread_{t-1} + epsilon
        半衰期 = -ln(2) / ln(rho)

        Args:
            spread: 价差序列

        Returns:
            半衰期（交易日数）
        """
        spread = spread.dropna()
        if len(spread) < 10:
            return float('inf')

        # AR(1)回归
        y = spread.values[1:]
        x = spread.values[:-1].reshape(-1, 1)

        try:
            rho = np.linalg.lstsq(x, y, rcond=None)[0][0]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Half-life AR(1) regression failed: {e}")
            return float('inf')

        # 确保rho在有效范围内
        if rho <= 0 or rho >= 1:
            return float('inf')

        # 半衰期
        half_life = -np.log(2) / np.log(rho)

        return max(0, half_life)

    @staticmethod
    def ou_half_life(spread: pd.Series, dt: float = 1.0) -> float:
        """
        使用Ornstein-Uhlenbeck模型估计半衰期

        Args:
            spread: 价差序列
            dt: 时间步长（默认1天）

        Returns:
            半衰期
        """
        spread = spread.dropna()
        if len(spread) < 20:
            return float('inf')

        # OU过程参数估计
        dy = spread.diff().dropna()
        y_lag = spread.shift(1).dropna()

        # 对齐
        common_idx = dy.index.intersection(y_lag.index)
        dy = dy.loc[common_idx]
        y_lag = y_lag.loc[common_idx]

        # 回归: dy = a + b*y + epsilon
        X = np.column_stack([np.ones(len(y_lag)), y_lag.values])
        try:
            coeffs = np.linalg.lstsq(X, dy.values, rcond=None)[0]
            b = coeffs[1]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"OU half-life regression failed: {e}")
            return float('inf')

        # theta = -b/dt
        theta = -b / dt

        if theta <= 0:
            return float('inf')

        # 半衰期
        half_life = np.log(2) / theta

        return max(0, half_life)


# ==================== 价差计算 ====================

class SpreadCalculator:
    """价差计算器"""

    def __init__(self, spread_type: SpreadType = SpreadType.RESIDUAL):
        self.spread_type = spread_type
        self._hedge_ratio: float = 1.0
        self._intercept: float = 0.0

    def fit(self, y1: pd.Series, y2: pd.Series) -> 'SpreadCalculator':
        """
        拟合对冲比率

        Args:
            y1: 第一个资产价格
            y2: 第二个资产价格

        Returns:
            self
        """
        if self.spread_type == SpreadType.RESIDUAL:
            # OLS回归
            X = np.column_stack([np.ones(len(y2)), y2.values])
            y = y1.values
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                self._intercept, self._hedge_ratio = coeffs
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"Hedge ratio fitting failed: {e}")
                self._hedge_ratio = y1.mean() / y2.mean() if y2.mean() != 0 else 1.0
                self._intercept = 0
        elif self.spread_type == SpreadType.RATIO:
            self._hedge_ratio = 1.0
        elif self.spread_type == SpreadType.LOG_RATIO:
            self._hedge_ratio = 1.0
        else:  # DIFFERENCE
            self._hedge_ratio = y1.std() / y2.std()

        return self

    def calculate(self, y1: pd.Series, y2: pd.Series) -> pd.Series:
        """
        计算价差

        Args:
            y1: 第一个资产价格
            y2: 第二个资产价格

        Returns:
            价差序列
        """
        if self.spread_type == SpreadType.RATIO:
            return y1 / y2
        elif self.spread_type == SpreadType.LOG_RATIO:
            return np.log(y1) - np.log(y2)
        elif self.spread_type == SpreadType.DIFFERENCE:
            return y1 - self._hedge_ratio * y2
        else:  # RESIDUAL
            return y1 - self._intercept - self._hedge_ratio * y2

    def zscore(
        self,
        y1: pd.Series,
        y2: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """
        计算Z-Score

        Args:
            y1: 第一个资产价格
            y2: 第二个资产价格
            window: 滚动窗口

        Returns:
            Z-Score序列
        """
        spread = self.calculate(y1, y2)
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        return (spread - mean) / std

    @property
    def hedge_ratio(self) -> float:
        return self._hedge_ratio


# ==================== 配对选择 ====================

class PairSelection:
    """配对选择器"""

    def __init__(self, config: Optional[PairsTradingConfig] = None):
        self.config = config or PairsTradingConfig()
        self._pairs: List[CointegrationResult] = []

    def find_pairs(
        self,
        data: pd.DataFrame,
        min_correlation: float = 0.7,
    ) -> List[CointegrationResult]:
        """
        在给定数据中寻找协整配对

        Args:
            data: 包含多个资产价格的DataFrame（每列一个资产）
            min_correlation: 最小相关性阈值

        Returns:
            协整配对列表
        """
        symbols = data.columns.tolist()
        n = len(symbols)
        pairs = []

        # 计算相关矩阵
        corr_matrix = data.corr()

        # 遍历所有可能的配对
        for i in range(n):
            for j in range(i + 1, n):
                sym1, sym2 = symbols[i], symbols[j]

                # 先检查相关性
                correlation = corr_matrix.loc[sym1, sym2]
                if abs(correlation) < min_correlation:
                    continue

                # 协整检验
                y1 = data[sym1].dropna()
                y2 = data[sym2].dropna()

                result = CointegrationTest.engle_granger(
                    y1, y2,
                    significance=self.config.cointegration_pvalue,
                )
                result.asset1 = sym1
                result.asset2 = sym2

                if not result.is_cointegrated:
                    continue

                # 检查半衰期
                if not (self.config.min_half_life <= result.half_life <= self.config.max_half_life):
                    continue

                pairs.append(result)

        # 按p值排序
        pairs.sort(key=lambda x: x.p_value)

        # 限制配对数量
        self._pairs = pairs[:self.config.max_pairs]

        return self._pairs

    def rank_pairs(
        self,
        pairs: List[CointegrationResult],
    ) -> List[Tuple[CointegrationResult, float]]:
        """
        对配对进行评分排序

        评分标准:
        - 协整强度（p值越小越好）
        - 半衰期合适程度
        - 历史夏普比

        Args:
            pairs: 配对列表

        Returns:
            (配对, 评分) 列表
        """
        ranked = []

        for pair in pairs:
            score = 0.0

            # P值得分（越小越好）
            p_score = (1 - pair.p_value) * 30

            # 半衰期得分（20-40天最优）
            optimal_hl = 30
            hl_deviation = abs(pair.half_life - optimal_hl) / optimal_hl
            hl_score = max(0, 30 * (1 - hl_deviation))

            # 标准差得分（适中为好）
            std_score = 20 if 0.01 < pair.std_spread < 0.1 else 10

            score = p_score + hl_score + std_score
            ranked.append((pair, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# ==================== 卡尔曼滤波对冲比率 ====================

class KalmanHedgeRatio:
    """
    卡尔曼滤波动态对冲比率估计

    状态方程: beta_t = beta_{t-1} + w_t
    观测方程: y1_t = beta_t * y2_t + v_t
    """

    def __init__(
        self,
        delta: float = 0.0001,      # 状态转移噪声方差
        ve: float = 0.001,          # 观测噪声方差
    ):
        self.delta = delta
        self.ve = ve
        self.beta = 0.0
        self.P = 1.0  # 估计误差协方差

    def update(self, y1: float, y2: float) -> float:
        """
        更新对冲比率

        Args:
            y1: 资产1价格
            y2: 资产2价格

        Returns:
            更新后的对冲比率
        """
        # 预测步骤
        beta_pred = self.beta
        P_pred = self.P + self.delta

        # 更新步骤
        y_pred = beta_pred * y2
        y_err = y1 - y_pred

        # 卡尔曼增益
        S = P_pred * y2 * y2 + self.ve
        K = P_pred * y2 / S

        # 状态更新
        self.beta = beta_pred + K * y_err
        self.P = (1 - K * y2) * P_pred

        return self.beta

    def filter(self, y1: pd.Series, y2: pd.Series) -> pd.Series:
        """
        对整个序列进行滤波

        Args:
            y1: 资产1价格序列
            y2: 资产2价格序列

        Returns:
            动态对冲比率序列
        """
        betas = []
        self.beta = y1.iloc[0] / y2.iloc[0] if y2.iloc[0] != 0 else 1.0
        self.P = 1.0

        for i in range(len(y1)):
            beta = self.update(y1.iloc[i], y2.iloc[i])
            betas.append(beta)

        return pd.Series(betas, index=y1.index)


# ==================== 配对交易策略 ====================

class AdvancedPairsTradingStrategy:
    """
    高级配对交易策略

    特性:
    - 自动配对选择
    - 动态对冲比率（可选卡尔曼滤波）
    - 多配对管理
    - 风险控制
    """

    def __init__(self, config: Optional[PairsTradingConfig] = None):
        self.config = config or PairsTradingConfig()
        self.pair_selector = PairSelection(self.config)
        self.spread_calculators: Dict[str, SpreadCalculator] = {}
        self.kalman_filters: Dict[str, KalmanHedgeRatio] = {}
        self.positions: Dict[str, PairPosition] = {}
        self.signals: List[PairSignal] = []

    def initialize(
        self,
        data: pd.DataFrame,
        use_kalman: bool = False,
    ) -> List[CointegrationResult]:
        """
        初始化策略，寻找配对

        Args:
            data: 历史价格数据
            use_kalman: 是否使用卡尔曼滤波

        Returns:
            找到的配对列表
        """
        # 寻找配对
        pairs = self.pair_selector.find_pairs(data)

        # 为每个配对初始化计算器
        for pair in pairs:
            key = f"{pair.asset1}_{pair.asset2}"

            # 价差计算器
            calc = SpreadCalculator(self.config.spread_type)
            calc.fit(data[pair.asset1], data[pair.asset2])
            self.spread_calculators[key] = calc

            # 卡尔曼滤波器
            if use_kalman:
                kf = KalmanHedgeRatio()
                kf.filter(data[pair.asset1], data[pair.asset2])
                self.kalman_filters[key] = kf

        logger.info(f"找到 {len(pairs)} 个协整配对")
        return pairs

    def generate_signal(
        self,
        pair: CointegrationResult,
        y1: float,
        y2: float,
        timestamp: datetime,
    ) -> PairSignal:
        """
        生成单个配对的信号

        Args:
            pair: 配对信息
            y1: 资产1当前价格
            y2: 资产2当前价格
            timestamp: 时间戳

        Returns:
            交易信号
        """
        key = f"{pair.asset1}_{pair.asset2}"
        calc = self.spread_calculators.get(key)

        if calc is None:
            return PairSignal(
                timestamp=timestamp,
                asset1=pair.asset1,
                asset2=pair.asset2,
                signal_type=SignalType.HOLD,
                zscore=0,
                spread=0,
                hedge_ratio=pair.hedge_ratio,
            )

        # 计算价差
        spread = y1 - calc.hedge_ratio * y2

        # 计算Z-Score（使用配对的历史均值和标准差）
        zscore = (spread - pair.mean_spread) / pair.std_spread if pair.std_spread > 0 else 0

        # 确定信号
        if abs(zscore) > self.config.stop_loss_zscore:
            # 超过止损，强制平仓
            signal_type = SignalType.CLOSE
        elif zscore > self.config.entry_zscore:
            # Z-Score过高，做空价差
            signal_type = SignalType.SHORT_SPREAD
        elif zscore < -self.config.entry_zscore:
            # Z-Score过低，做多价差
            signal_type = SignalType.LONG_SPREAD
        elif abs(zscore) < self.config.exit_zscore:
            # 回归均值，平仓
            if key in self.positions:
                signal_type = SignalType.CLOSE
            else:
                signal_type = SignalType.HOLD
        else:
            signal_type = SignalType.HOLD

        signal = PairSignal(
            timestamp=timestamp,
            asset1=pair.asset1,
            asset2=pair.asset2,
            signal_type=signal_type,
            zscore=zscore,
            spread=spread,
            hedge_ratio=calc.hedge_ratio,
            confidence=self._calculate_confidence(zscore, pair.half_life),
        )

        self.signals.append(signal)
        return signal

    def _calculate_confidence(self, zscore: float, half_life: float) -> float:
        """计算信号置信度"""
        # 基于Z-Score的置信度
        z_conf = min(1.0, abs(zscore) / 3.0)

        # 基于半衰期的置信度
        optimal_hl = 30
        hl_conf = max(0, 1 - abs(half_life - optimal_hl) / optimal_hl)

        return 0.7 * z_conf + 0.3 * hl_conf

    def update_position(
        self,
        signal: PairSignal,
        capital: float = 100000,
    ) -> Optional[Dict[str, float]]:
        """
        根据信号更新持仓

        Args:
            signal: 交易信号
            capital: 可用资金

        Returns:
            交易指令 {'asset1': qty1, 'asset2': qty2} 或 None
        """
        key = f"{signal.asset1}_{signal.asset2}"

        if signal.signal_type == SignalType.HOLD:
            return None

        if signal.signal_type == SignalType.CLOSE:
            if key in self.positions:
                pos = self.positions.pop(key)
                return {
                    signal.asset1: -pos.asset1_qty,
                    signal.asset2: -pos.asset2_qty,
                }
            return None

        # 计算持仓大小
        position_value = min(capital * 0.1, self.config.max_position_value)

        if signal.signal_type == SignalType.LONG_SPREAD:
            # 做多价差: 买asset1, 卖asset2
            # 假设signal包含当前价格信息，这里简化处理
            qty1 = position_value / 2  # 简化
            qty2 = -position_value / 2 * signal.hedge_ratio

        else:  # SHORT_SPREAD
            # 做空价差: 卖asset1, 买asset2
            qty1 = -position_value / 2
            qty2 = position_value / 2 * signal.hedge_ratio

        # 记录持仓
        self.positions[key] = PairPosition(
            asset1=signal.asset1,
            asset2=signal.asset2,
            asset1_qty=qty1,
            asset2_qty=qty2,
            entry_spread=signal.spread,
            entry_zscore=signal.zscore,
            entry_time=signal.timestamp,
        )

        return {signal.asset1: qty1, signal.asset2: qty2}

    def get_portfolio_exposure(self) -> Dict[str, float]:
        """获取组合各资产敞口"""
        exposure = {}
        for pos in self.positions.values():
            exposure[pos.asset1] = exposure.get(pos.asset1, 0) + pos.asset1_qty
            exposure[pos.asset2] = exposure.get(pos.asset2, 0) + pos.asset2_qty
        return exposure

    def get_active_pairs(self) -> List[str]:
        """获取活跃配对"""
        return list(self.positions.keys())
