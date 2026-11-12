"""
旧版回测引擎（已停用，见 BacktestEngine.__init__）

提供事件驱动的回测框架，支持：
- 逐bar模拟
- 滑点和手续费
- 保证金交易
- 多空双向交易
- 详细交易记录
"""

from dataclasses import dataclass, field


class LegacyEngineError(RuntimeError):
    """调用了已停用的旧回测引擎。"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled: bool = False
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    holding_period: int  # 持仓天数
    mae: float = 0.0     # Maximum Adverse Excursion (最大不利偏离)
    mfe: float = 0.0     # Maximum Favorable Excursion (最大有利偏离)


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001       # 手续费率
    slippage_rate: float = 0.0005        # 滑点率
    margin_rate: float = 1.0             # 保证金比例 (1.0=全额, 0.1=10倍杠杆)
    max_position_size: float = 0.2       # 单个持仓最大占比
    allow_short: bool = True             # 是否允许做空
    allow_fractional: bool = True        # 是否允许小数股
    risk_free_rate: float = 0.02         # 无风险利率
    benchmark: Optional[str] = None      # 基准代码


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    metrics: Dict[str, float]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions_history: pd.DataFrame
    trades: List[Trade]
    orders: List[Order]
    daily_returns: pd.Series
    monthly_returns: pd.DataFrame
    yearly_returns: pd.Series

    # 风险指标
    var_95: float = 0.0
    cvar_95: float = 0.0
    var_99: float = 0.0

    # 回撤分析
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    avg_drawdown_duration: int = 0

    # 交易分析
    best_trade: Optional[Trade] = None
    worst_trade: Optional[Trade] = None
    avg_trade_duration: float = 0.0


class BacktestEngine:
    """
    事件驱动回测引擎

    支持特性：
    - 逐bar模拟，精确复现历史行情
    - 完整的订单管理系统
    - 滑点和手续费模拟
    - 保证金和杠杆交易
    - 多空双向交易
    - 详细的绩效分析
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        # 已停用：该引擎按当根价格成交、支持保证金与做空，且不使用 daily.py 的 A 股规则与资金账本，
        # 其结果与 BaseStrategy.run / run_portfolio 相互矛盾，不能作为有效回测结果。
        raise LegacyEngineError(
            "BacktestEngine 已停用：请使用 BaseStrategy.run()（单证券）或 "
            "quantrisk.strategies.backtest.daily.run_portfolio()（多证券）")
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """重置引擎状态"""
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_history: List[float] = []
        self.date_history: List[datetime] = []
        self.positions_history: List[Dict[str, Any]] = []  # 持仓历史快照
        self.current_bar: Optional[pd.Series] = None
        self.current_time: Optional[datetime] = None
        self._order_id_counter = 0
        self._trade_id_counter = 0

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, 'BacktestEngine'], None],
        warmup_period: int = 0
    ) -> BacktestResult:
        """
        运行回测

        Args:
            data: OHLCV数据
            strategy_func: 策略函数，签名为 func(data_slice, engine)
            warmup_period: 预热期（跳过的bar数）

        Returns:
            BacktestResult: 回测结果
        """
        self.reset()

        for i in range(warmup_period, len(data)):
            # 设置当前bar
            self.current_bar = data.iloc[i]
            self.current_time = data.index[i]

            # 更新持仓市值
            self._update_positions()

            # 处理挂单
            self._process_pending_orders()

            # 检查止损止盈
            self._check_stops()

            # 执行策略逻辑
            data_slice = data.iloc[:i+1]
            strategy_func(data_slice, self)

            # 记录权益
            equity = self._calculate_equity()
            self.equity_history.append(equity)
            self.date_history.append(self.current_time)

            # 记录持仓快照
            self._record_positions_snapshot()

        return self._generate_result()

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        提交订单

        Args:
            symbol: 标的代码
            side: 买卖方向
            quantity: 数量
            order_type: 订单类型
            limit_price: 限价
            stop_price: 止损价

        Returns:
            Order: 订单对象
        """
        self._order_id_counter += 1
        order = Order(
            order_id=f"ORD-{self._order_id_counter:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=self.current_time
        )

        if order_type == OrderType.MARKET:
            self._execute_order(order)
        else:
            self.orders.append(order)

        return order

    def close_position(self, symbol: str, quantity: Optional[float] = None):
        """
        平仓

        Args:
            symbol: 标的代码
            quantity: 平仓数量（None表示全部平仓）
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        close_qty = quantity or pos.quantity

        side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        self.submit_order(symbol, side, close_qty)

    def set_stop_loss(self, symbol: str, stop_price: float):
        """设置止损"""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_price

    def set_take_profit(self, symbol: str, take_price: float):
        """设置止盈"""
        if symbol in self.positions:
            self.positions[symbol].take_profit = take_price

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)

    def get_equity(self) -> float:
        """获取当前权益"""
        return self._calculate_equity()

    def get_cash(self) -> float:
        """获取可用资金"""
        return self.cash

    def _execute_order(self, order: Order):
        """执行订单"""
        if self.current_bar is None:
            return

        # 计算成交价（含滑点）
        base_price = self.current_bar['close']
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.config.slippage_rate)
        else:
            fill_price = base_price * (1 - self.config.slippage_rate)

        # 计算手续费
        commission = fill_price * order.quantity * self.config.commission_rate

        # 检查资金是否充足
        required_capital = fill_price * order.quantity * self.config.margin_rate + commission
        if order.side == OrderSide.BUY and required_capital > self.cash:
            # 资金不足，调整数量
            max_qty = (self.cash - commission) / (fill_price * self.config.margin_rate)
            if max_qty <= 0:
                return
            order.quantity = max_qty if self.config.allow_fractional else int(max_qty)

        # 更新订单状态
        order.filled = True
        order.filled_price = fill_price
        order.filled_time = self.current_time
        order.commission = commission

        # 更新持仓
        symbol = order.symbol
        if symbol in self.positions:
            self._update_existing_position(order)
        else:
            self._open_new_position(order)

        # 更新现金
        if order.side == OrderSide.BUY:
            self.cash -= fill_price * order.quantity + commission
        else:
            self.cash += fill_price * order.quantity - commission

    def _open_new_position(self, order: Order):
        """开新仓"""
        side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            side=side,
            quantity=order.quantity,
            entry_price=order.filled_price,
            entry_time=order.filled_time,
            current_price=order.filled_price
        )

    def _update_existing_position(self, order: Order):
        """更新现有持仓"""
        pos = self.positions[order.symbol]

        # 判断是加仓还是平仓
        is_same_direction = (
            (pos.side == PositionSide.LONG and order.side == OrderSide.BUY) or
            (pos.side == PositionSide.SHORT and order.side == OrderSide.SELL)
        )

        if is_same_direction:
            # 加仓 - 计算平均成本
            total_value = pos.entry_price * pos.quantity + order.filled_price * order.quantity
            pos.quantity += order.quantity
            pos.entry_price = total_value / pos.quantity
        else:
            # 平仓
            close_qty = min(order.quantity, pos.quantity)

            # 计算盈亏
            if pos.side == PositionSide.LONG:
                pnl = (order.filled_price - pos.entry_price) * close_qty
            else:
                pnl = (pos.entry_price - order.filled_price) * close_qty

            pnl_pct = pnl / (pos.entry_price * close_qty)

            # 记录交易
            self._trade_id_counter += 1
            trade = Trade(
                trade_id=f"TRD-{self._trade_id_counter:06d}",
                symbol=order.symbol,
                side=pos.side.value,
                quantity=close_qty,
                entry_price=pos.entry_price,
                exit_price=order.filled_price,
                entry_time=pos.entry_time,
                exit_time=order.filled_time,
                pnl=pnl - order.commission,
                pnl_pct=pnl_pct,
                commission=order.commission,
                holding_period=(order.filled_time - pos.entry_time).days
            )
            self.trades.append(trade)

            # 更新持仓
            pos.quantity -= close_qty
            pos.realized_pnl += pnl

            if pos.quantity <= 0:
                del self.positions[order.symbol]

    def _update_positions(self):
        """更新持仓市值"""
        if self.current_bar is None:
            return

        current_price = self.current_bar['close']
        for symbol, pos in self.positions.items():
            pos.current_price = current_price
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity

    def _record_positions_snapshot(self):
        """记录当前持仓快照"""
        snapshot = {
            'timestamp': self.current_time,
            'cash': self.cash,
            'positions': {},
            'total_value': self._calculate_equity(),
        }

        for symbol, pos in self.positions.items():
            snapshot['positions'][symbol] = {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'market_value': pos.quantity * pos.current_price,
            }

        self.positions_history.append(snapshot)

    def _build_positions_dataframe(self) -> pd.DataFrame:
        """将持仓历史快照转换为 DataFrame"""
        if not self.positions_history:
            return pd.DataFrame()

        # 收集所有出现过的 symbol
        all_symbols = set()
        for snapshot in self.positions_history:
            all_symbols.update(snapshot['positions'].keys())

        if not all_symbols:
            # 没有持仓，返回基本信息
            records = [
                {
                    'timestamp': s['timestamp'],
                    'cash': s['cash'],
                    'total_value': s['total_value'],
                }
                for s in self.positions_history
            ]
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df

        # 构建扁平化记录
        records = []
        for snapshot in self.positions_history:
            record = {
                'timestamp': snapshot['timestamp'],
                'cash': snapshot['cash'],
                'total_value': snapshot['total_value'],
            }

            # 为每个 symbol 添加列
            for symbol in all_symbols:
                pos_data = snapshot['positions'].get(symbol, {})
                record[f'{symbol}_quantity'] = pos_data.get('quantity', 0)
                record[f'{symbol}_entry_price'] = pos_data.get('entry_price', 0)
                record[f'{symbol}_current_price'] = pos_data.get('current_price', 0)
                record[f'{symbol}_unrealized_pnl'] = pos_data.get('unrealized_pnl', 0)
                record[f'{symbol}_market_value'] = pos_data.get('market_value', 0)

            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df

    def _process_pending_orders(self):
        """处理挂单"""
        if self.current_bar is None:
            return

        pending_orders = [o for o in self.orders if not o.filled]

        for order in pending_orders:
            should_execute = False

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY:
                    should_execute = self.current_bar['low'] <= order.limit_price
                else:
                    should_execute = self.current_bar['high'] >= order.limit_price

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY:
                    should_execute = self.current_bar['high'] >= order.stop_price
                else:
                    should_execute = self.current_bar['low'] <= order.stop_price

            if should_execute:
                self._execute_order(order)

    def _check_stops(self):
        """检查止损止盈"""
        if self.current_bar is None:
            return

        symbols_to_close = []
        current_price = self.current_bar['close']

        for symbol, pos in self.positions.items():
            should_close = False

            if pos.side == PositionSide.LONG:
                if pos.stop_loss and current_price <= pos.stop_loss:
                    should_close = True
                if pos.take_profit and current_price >= pos.take_profit:
                    should_close = True
            else:  # SHORT
                if pos.stop_loss and current_price >= pos.stop_loss:
                    should_close = True
                if pos.take_profit and current_price <= pos.take_profit:
                    should_close = True

            if should_close:
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            self.close_position(symbol)

    def _calculate_equity(self) -> float:
        """计算总权益"""
        total_position_value = sum(
            pos.current_price * pos.quantity
            for pos in self.positions.values()
        )
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        return self.cash + total_position_value + total_unrealized_pnl

    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        equity_curve = pd.Series(self.equity_history, index=self.date_history)
        daily_returns = equity_curve.pct_change().fillna(0)

        # 计算绩效指标
        metrics = self._calculate_metrics(equity_curve, daily_returns)

        # 计算回撤
        drawdown_curve = self._calculate_drawdown(equity_curve)

        # 月度收益
        monthly_returns = daily_returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        ).to_frame('return')

        # 年度收益
        yearly_returns = daily_returns.resample('YE').apply(
            lambda x: (1 + x).prod() - 1
        )

        # VaR计算 (VaR应该是损失分布的百分位数)
        # 🔧 2026-01-01修复: 修正VaR计算公式
        losses = -daily_returns
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        cvar_95 = losses[losses >= var_95].mean()

        # 回撤分析
        max_dd, max_dd_duration, avg_dd, avg_dd_duration = self._analyze_drawdowns(drawdown_curve)

        # 最佳/最差交易
        best_trade = max(self.trades, key=lambda t: t.pnl) if self.trades else None
        worst_trade = min(self.trades, key=lambda t: t.pnl) if self.trades else None

        # 平均持仓时间
        avg_duration = np.mean([t.holding_period for t in self.trades]) if self.trades else 0

        # 构建持仓历史 DataFrame
        positions_df = self._build_positions_dataframe()

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            positions_history=positions_df,
            trades=self.trades,
            orders=self.orders,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            avg_drawdown_duration=avg_dd_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_duration
        )

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series
    ) -> Dict[str, float]:
        """计算绩效指标"""
        initial = self.config.initial_capital
        final = equity_curve.iloc[-1]
        total_return = (final - initial) / initial

        # 年化收益率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 波动率
        volatility = daily_returns.std() * np.sqrt(252)

        # 夏普比率
        excess_return = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # 索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 交易统计
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 盈亏比
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 平均盈亏
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        return {
            'initial_capital': initial,
            'final_capital': final,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0,
            'expectancy': win_rate * avg_win - (1 - win_rate) * avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit - gross_loss,
            'total_commission': sum(t.commission for t in self.trades),
        }

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """计算回撤序列"""
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown

    def _analyze_drawdowns(self, drawdown_curve: pd.Series):
        """分析回撤"""
        max_dd = drawdown_curve.min()

        # 回撤周期分析
        in_drawdown = drawdown_curve < 0
        drawdown_periods = []
        current_start = None

        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and current_start is None:
                current_start = date
            elif not is_dd and current_start is not None:
                duration = (date - current_start).days
                drawdown_periods.append(duration)
                current_start = None

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_dd_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        # 平均回撤
        avg_dd = drawdown_curve[drawdown_curve < 0].mean() if (drawdown_curve < 0).any() else 0

        return max_dd, max_dd_duration, avg_dd, avg_dd_duration
