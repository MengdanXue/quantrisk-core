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

        # 处理缺失值
        data = data.ffill().bfill()

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
    ) -> StrategyResult:
        """
        运行策略并返回结果

        Args:
            data: OHLCV数据
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点

        Returns:
            策略运行结果
        """
        # 预处理数据
        data = self.preprocess_data(data.copy())

        # 生成信号
        signals = self.generate_signals(data)

        # 模拟交易
        positions, returns, trades = self._simulate_trading(
            data, signals, initial_capital, commission, slippage
        )

        # 计算权益曲线
        equity_curve = (1 + returns).cumprod() * initial_capital

        # 计算绩效指标
        metrics = self._calculate_metrics(returns, equity_curve, trades)

        return StrategyResult(
            strategy_name=self.name,
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            signals=signals,
            positions=positions,
            returns=returns,
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            params=self.params,
        )

    def _simulate_trading(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float,
        commission: float,
        slippage: float,
    ) -> Tuple[pd.Series, pd.Series, List[Dict]]:
        """
        模拟交易执行

        Returns:
            (positions, returns, trades)
        """
        positions = pd.Series(0.0, index=data.index)
        returns = pd.Series(0.0, index=data.index)
        trades = []

        cash = initial_capital
        position = 0.0
        entry_price = 0.0
        entry_date = None

        for i, (date, row) in enumerate(data.iterrows()):
            price = row['close']
            signal = signals.get(date, 0)

            # 将信号转换为整数（如果是枚举）
            if isinstance(signal, Signal):
                signal = signal.value

            # 开多仓
            if signal > 0 and position == 0:
                exec_price = price * (1 + slippage)
                position_size = self.calculate_position_size(
                    Signal.BUY, exec_price, cash
                )
                cost = position_size * exec_price * (1 + commission)

                if cost <= cash and position_size > 0:
                    cash -= cost
                    position = position_size
                    entry_price = exec_price
                    entry_date = date

            # 平多仓
            elif signal < 0 and position > 0:
                exec_price = price * (1 - slippage)
                revenue = position * exec_price * (1 - commission)

                pnl = revenue - (position * entry_price)
                pnl_pct = (exec_price - entry_price) / entry_price

                trades.append({
                    'entry_date': str(entry_date),
                    'exit_date': str(date),
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exec_price,
                    'quantity': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_days': (date - entry_date).days if hasattr(date, 'days') else 1,
                })

                cash += revenue
                position = 0

            # 记录持仓和收益
            # 收益 = 本bar期末权益 / 上一bar期末权益 - 1（不能用成交后的持仓回溯上一bar，否则会产生前视偏差）
            equity = cash + position * price
            positions.iloc[i] = position
            if i > 0:
                returns.iloc[i] = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            prev_equity = equity

        return positions, returns, trades

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Dict],
    ) -> Dict[str, float]:
        """计算绩效指标"""
        # 基础收益指标
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

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

        # 最大回撤
        cummax = equity_curve.cummax()
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

            avg_holding = np.mean([t.get('holding_days', 1) for t in trades])
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
