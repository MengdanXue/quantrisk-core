"""
策略基类模块

定义所有交易策略的基础架构，包括：
- 信号枚举
- 策略基类
- 策略结果数据类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


class Signal(Enum):
    """交易信号枚举"""
    STRONG_BUY = 2      # 强烈买入
    BUY = 1             # 买入
    HOLD = 0            # 持有/观望
    SELL = -1           # 卖出
    STRONG_SELL = -2    # 强烈卖出


@dataclass
class TradeSignal:
    """单个交易信号"""
    timestamp: datetime
    symbol: str
    signal: Signal
    strength: float  # 信号强度 0-1
    price: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """策略运行结果"""
    strategy_name: str
    symbol: str
    signals: pd.Series  # 信号序列
    positions: pd.Series  # 持仓序列
    returns: pd.Series  # 收益序列
    equity_curve: pd.Series  # 权益曲线
    trades: List[Dict]  # 交易记录
    metrics: Dict[str, float]  # 绩效指标
    params: Dict[str, Any]  # 策略参数
    generated_at: datetime = field(default_factory=datetime.now)
    ledger: Optional[pd.DataFrame] = None   # 逐日账本：cash / frozen / position / close / market_value / equity
    orders: List[Dict] = field(default_factory=list)   # 订单及其最终状态、拒绝原因
    fills: List[Dict] = field(default_factory=list)    # 成交明细（含分项费用）
    journal: List[Dict] = field(default_factory=list)  # 资金流水
    data_issues: Dict[str, Any] = field(default_factory=dict)  # 被剔除的无效行情等


class BaseStrategy(ABC):
    """
    策略基类

    所有量化策略都应该继承此类并实现以下方法：
    - default_params(): 返回默认参数
    - generate_signals(): 生成交易信号

    可选覆盖：
    - validate_params(): 验证参数
    - preprocess_data(): 数据预处理
    - calculate_position_size(): 计算仓位
    """

    # 策略分类
    category: str = "base"

    # 策略版本
    version: str = "1.0.0"

    def __init__(self, params: Dict = None):
        """
        初始化策略

        Args:
            params: 策略参数，如果为None则使用默认参数
        """
        self._default_params = self.default_params()
        self.params = {**self._default_params, **(params or {})}
        self.validate_params()
        self.name = self.__class__.__name__

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """
        返回策略的默认参数

        Returns:
            参数字典
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据市场数据生成交易信号

        Args:
            data: OHLCV数据，包含 open, high, low, close, volume 列
                  索引为日期时间

        Returns:
            信号序列，值为 Signal 枚举值或整数：
            2=强烈买入, 1=买入, 0=持有, -1=卖出, -2=强烈卖出
        """
        pass

    def validate_params(self) -> None:
        """
        验证策略参数

        如果参数无效，应抛出 ValueError
        """
        # 子类可以覆盖此方法进行参数验证
        pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        Args:
            data: 原始OHLCV数据

        Returns:
            处理后的数据
        """
        # 确保数据按时间排序
        data = data.sort_index()

        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close']
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 不补值：缺失或无效的K线直接剔除（视为当日无行情），避免向后填充把未来数据带入历史，
        # 也避免向前填充制造出并不存在的可成交行情。被剔除的日期记录在 data.attrs['dropped_rows']。
        price = data[required_columns]
        valid = price.notna().all(axis=1) & (price > 0).all(axis=1)
        dropped = [str(d) for d in data.index[~valid]]
        data = data[valid].copy()
        data.attrs['dropped_rows'] = dropped

        return data

    def calculate_position_size(
        self,
        signal: Signal,
        price: float,
        equity: float,
        volatility: float = None,
        **kwargs
    ) -> float:
        """
        计算建议仓位大小

        Args:
            signal: 交易信号
            price: 当前价格
            equity: 当前权益
            volatility: 波动率（可选）

        Returns:
            建议仓位（股数或合约数）
        """
        if signal == Signal.HOLD:
            return 0

        # 默认使用固定比例仓位
        position_pct = self.params.get('position_pct', 0.1)

        # 根据信号强度调整
        if signal in (Signal.STRONG_BUY, Signal.STRONG_SELL):
            position_pct *= 1.5
        elif signal in (Signal.BUY, Signal.SELL):
            position_pct *= 1.0

        # 根据波动率调整（如果提供）
        if volatility and volatility > 0:
            target_volatility = self.params.get('target_volatility', 0.15)
            vol_scalar = target_volatility / volatility
            position_pct *= min(vol_scalar, 2.0)  # 最多放大2倍

        # 计算仓位
        position_value = equity * position_pct
        position_size = position_value / price

        return position_size

    def calculate_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        atr: float = None,
        **kwargs
    ) -> float:
        """
        计算止损价格

        Args:
            entry_price: 入场价格
            signal: 交易信号（决定多空方向）
            atr: ATR值（可选）

        Returns:
            止损价格
        """
        stop_pct = self.params.get('stop_loss_pct', 0.02)

        if atr:
            # 基于ATR的止损
            atr_multiplier = self.params.get('atr_stop_multiplier', 2.0)
            stop_distance = atr * atr_multiplier
        else:
            # 固定百分比止损
            stop_distance = entry_price * stop_pct

        if signal in (Signal.BUY, Signal.STRONG_BUY):
            return entry_price - stop_distance
        elif signal in (Signal.SELL, Signal.STRONG_SELL):
            return entry_price + stop_distance
        else:
            return entry_price

    def calculate_take_profit(
        self,
        entry_price: float,
        signal: Signal,
        atr: float = None,
        **kwargs
    ) -> float:
        """
        计算止盈价格

        Args:
            entry_price: 入场价格
            signal: 交易信号
            atr: ATR值（可选）

        Returns:
            止盈价格
        """
        profit_pct = self.params.get('take_profit_pct', 0.06)
        risk_reward = self.params.get('risk_reward_ratio', 3.0)

        if atr:
            atr_multiplier = self.params.get('atr_stop_multiplier', 2.0)
            profit_distance = atr * atr_multiplier * risk_reward
        else:
            profit_distance = entry_price * profit_pct

        if signal in (Signal.BUY, Signal.STRONG_BUY):
            return entry_price + profit_distance
        elif signal in (Signal.SELL, Signal.STRONG_SELL):
            return entry_price - profit_distance
        else:
            return entry_price

    def run(
        self,
        data: pd.DataFrame,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        market=None,
        bars: Optional[pd.DataFrame] = None,
        calendar: Optional[pd.DatetimeIndex] = None,
        corporate_actions=(),
        symbol: Optional[str] = None,
    ) -> StrategyResult:
        """
        运行策略并返回结果

        成交规则：T 日收盘产生信号，T+1 交易日开盘价撮合（见 strategies/backtest/daily.py）。

        Args:
            data: 用于生成信号的 OHLCV 数据（A 股建议用后复权价，见 quantrisk.data.ashare）
            initial_capital: 初始资金
            commission: 佣金率（仅在未指定 market 时使用）
            slippage: 滑点（仅在未指定 market 时使用）
            market: 市场规则，默认 GenericMarket(commission, slippage)；A 股用 AShareMarket
            bars: 实际成交与估值用的价格（A 股为不复权价）；默认与 data 相同
            calendar: 交易日历；默认为 bars 的日期
            corporate_actions: 除权除息事件（CorporateAction 列表）
            symbol: 证券代码

        Returns:
            策略运行结果
        """
        from .backtest.daily import GenericMarket, run_daily

        symbol = symbol or data.attrs.get('symbol', 'UNKNOWN')
        # 撮合用未经剔除的原始行情：无效K线在撮合层按“当日无行情”处理，日期本身不能消失
        exec_bars = data.sort_index() if bars is None else bars
        data = self.preprocess_data(data.copy())
        signals = self.generate_signals(data)
        market = market or GenericMarket(commission=commission, slippage=slippage)

        out = run_daily(self, signals, exec_bars, market, initial_capital, symbol=symbol,
                        calendar=calendar, corporate_actions=corporate_actions)
        ledger = out['ledger']
        equity_curve = ledger['equity'].rename('equity')
        returns = equity_curve.pct_change()
        returns.iloc[0] = equity_curve.iloc[0] / initial_capital - 1   # 首日成本也计入收益
        trades = out['trades']
        metrics = self._calculate_metrics(returns, equity_curve, trades, initial_capital)

        return StrategyResult(
            strategy_name=self.name,
            symbol=symbol,
            signals=signals,
            positions=ledger['position'],
            returns=returns,
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            params=self.params,
            ledger=ledger,
            orders=out['orders'],
            fills=out['fills'],
            journal=out['journal'],
            data_issues={'dropped_signal_rows': data.attrs.get('dropped_rows', []),
                         'invalid_execution_days': out['invalid_days']},
        )

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Dict],
        initial_capital: Optional[float] = None,
    ) -> Dict[str, float]:
        """计算绩效指标（收益相对初始资金计算；交易盈亏已含买卖全部费用）"""
        # 基础收益指标
        base = initial_capital if initial_capital else equity_curve.iloc[0]
        total_return = (equity_curve.iloc[-1] / base) - 1

        # 计算年化收益
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
        else:
            annual_return = 0

        # 波动率和夏普比率
        daily_returns = returns.dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (daily_returns.mean() * 252 - 0.03) / (daily_returns.std() * np.sqrt(252))
        else:
            volatility = 0
            sharpe_ratio = 0

        # 最大回撤（以初始资金为起点）
        cummax = equity_curve.cummax().clip(lower=base)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # 交易统计
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(wins) / len(trades) if trades else 0

            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            avg_holding = np.mean([t['holding_days'] for t in trades])  # 自然日
        else:
            win_rate = 0
            profit_factor = 0
            avg_holding = 0

        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino比率
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() * 252 - 0.03) / (downside_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = 0

        return {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'volatility': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'sortino_ratio': round(sortino_ratio, 4),
            'calmar_ratio': round(calmar_ratio, 4),
            'max_drawdown': round(max_drawdown, 4),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 4),
            'total_trades': len(trades),
            'avg_holding_days': round(avg_holding, 1),
        }

    def get_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            'name': self.name,
            'category': self.category,
            'version': self.version,
            'description': self.__doc__.strip() if self.__doc__ else '',
            'params': self.params,
            'default_params': self._default_params,
        }

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"
