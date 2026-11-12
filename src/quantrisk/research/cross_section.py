"""
横截面选股与换手约束

每隔 rebalance_days 个交易日，在当日按时点可交易的股票中，用截至当日的复权价计算因子并排名，选出前 top_n 只
作为目标持仓；两次调仓之间目标不变。目标翻译成逐只股票的信号后交给 portfolio_run.run_strategy_portfolio，
撮合、账本、风险预算与单因子策略完全相同（本模块不做任何成交或记账）。

因子（只用调仓日及以前的数据）：
- momentum：close[t-skip] / close[t-skip-lookback] - 1，降序；
- low_vol：过去 lookback 个交易日日收益率的标准差，升序（波动越低排名越前）。

换手约束（两层，都作用在目标持仓上）：
1. 保留缓冲 buffer：已在目标中的股票，排名不差于 top_n + buffer 就继续持有；
2. 主动换出上限 max_turnover：每次调仓因排名下降而主动换出的股票数 ≤ floor(max_turnover × top_n)，
   超出的部分按排名从差到好依次换出，其余延后（记为 deferred）。建仓（目标为空）不受限制。
被动换出——股票不再满足按时点股票池资格（调出、退市、上市未满期）——不计入主动换手额度，单独记录。
调仓日停牌、或历史不足以计算因子的已持有股票继续保留，不因缺数据被换出。

信号语义（目标状态）：目标内的股票每天给买入信号（引擎只在空仓时下单，因此买入失败会在之后的交易日重试），
曾经进入目标、后来被移出的股票每天给卖出信号（引擎只在有持仓时下单）。已知限制：部分成交或被风险预算缩单后
不会补仓；持仓按建仓时的等权金额买入，之后不按目标权重再平衡。
"""

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..strategies.backtest.daily import valid_bar_mask
from ..strategies.base import BaseStrategy, Signal
from .walk_forward import WalkForwardStudy, segment_returns

FACTORS = ("momentum", "low_vol")


@dataclass(frozen=True)
class CrossSectionParams:
    factor: str = "momentum"
    lookback: int = 120
    skip: int = 0
    top_n: int = 10
    rebalance_days: int = 21
    buffer: int = 0
    max_turnover: Optional[float] = None      # 每次调仓主动换出比例上限（相对 top_n）；None 表示不限制

    def __post_init__(self):
        if self.factor not in FACTORS:
            raise ValueError(f"未知因子 {self.factor}，可选 {FACTORS}")
        if self.lookback < 2 or self.skip < 0 or self.top_n < 1 or self.rebalance_days < 1 or self.buffer < 0:
            raise ValueError(f"参数非法: {self}")
        if self.max_turnover is not None and not (0 <= self.max_turnover <= 1):
            raise ValueError("max_turnover 必须在 [0, 1] 内")

    @property
    def max_voluntary_drops(self) -> Optional[int]:
        return None if self.max_turnover is None else math.floor(self.max_turnover * self.top_n + 1e-9)

    @property
    def history_needed(self) -> int:
        return self.lookback + self.skip + 1


def factor_scores(close: pd.DataFrame, params: CrossSectionParams) -> pd.DataFrame:
    """close：交易日 × 股票的复权收盘价（缺失为 NaN）。返回分数，越大越优先；每个值只依赖当日及以前的数据。"""
    if params.factor == "momentum":
        return close.shift(params.skip) / close.shift(params.skip + params.lookback) - 1
    r = close.pct_change(fill_method=None)
    vol = r.rolling(params.lookback, min_periods=max(2, int(params.lookback * 0.8))).std()
    return -vol


def _aligned(datasets, calendar: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close, tradable = {}, {}
    for ds in datasets:
        close[ds.symbol] = ds.signal_data["close"].reindex(calendar)
        b = ds.bars
        ok = valid_bar_mask(b)
        if "volume" in b:
            ok &= b["volume"].fillna(0) > 0
        tradable[ds.symbol] = ok.reindex(calendar, fill_value=False)
    return pd.DataFrame(close), pd.DataFrame(tradable).astype(bool)


def cross_sectional_signals(datasets, params: CrossSectionParams, universe=None,
                            trade_start=None) -> Tuple[Dict[str, pd.Series], List[dict]]:
    """
    返回（每只股票的信号序列, 调仓记录）。信号只覆盖 trade_start 及以后的交易日；之前的数据只用于计算因子。
    """
    calendar = pd.DatetimeIndex(datasets[0].calendar)
    close, tradable = _aligned(datasets, calendar)
    scores = factor_scores(close, params)
    symbols = sorted(close.columns)
    if universe is None:
        eligible = pd.DataFrame(True, index=calendar, columns=symbols)
    else:
        eligible = pd.DataFrame({s: universe.eligibility(s, calendar) for s in symbols})
    days = calendar if trade_start is None else calendar[calendar >= pd.Timestamp(trade_start)]
    rebalance = set(days[::params.rebalance_days])
    limit = params.max_voluntary_drops
    keep_zone = params.top_n + params.buffer

    target: set = set()
    ever: set = set()
    log: List[dict] = []
    sig = {s: pd.Series(Signal.HOLD.value, index=days, dtype=int) for s in symbols}
    for d in days:
        if d in rebalance:
            elig = eligible.loc[d]
            row = scores.loc[d]
            ranked = sorted((s for s in symbols if elig[s] and tradable.at[d, s] and pd.notna(row[s])),
                            key=lambda s: (-row[s], s))
            rank = {s: i + 1 for i, s in enumerate(ranked)}
            forced = sorted(s for s in target if not elig[s])
            unranked_kept = sorted(s for s in target if elig[s] and s not in rank)
            voluntary = sorted((s for s in target if s in rank and rank[s] > keep_zone), key=lambda s: -rank[s])
            formation = not target
            n_drop = len(voluntary) if (limit is None or formation) else min(len(voluntary), limit)
            dropped, deferred = voluntary[:n_drop], voluntary[n_drop:]
            new_target = target - set(forced) - set(dropped)
            slots = max(params.top_n - len(new_target), 0)
            added = [s for s in ranked if s not in new_target][:slots]
            new_target |= set(added)
            log.append({"date": str(d.date()), "candidates": len(ranked), "target": sorted(new_target),
                        "added": added, "dropped_voluntary": dropped, "dropped_forced": forced,
                        "deferred_by_turnover_limit": deferred, "kept_without_rank": unranked_kept,
                        "voluntary_limit": limit, "formation": formation,
                        "name_turnover": len(added) / params.top_n,
                        "ranks": {s: rank.get(s) for s in sorted(new_target | set(dropped) | set(forced))}})
            target = new_target
            ever |= target
        for s in ever:
            sig[s].loc[d] = Signal.BUY.value if s in target else Signal.SELL.value
    return sig, log


class PrecomputedSignals(BaseStrategy):
    """把预先算好的信号交给组合入口；仓位按 position_pct（与其他策略相同的 calculate_position_size）。"""

    def __init__(self, signals: pd.Series, position_pct: float):
        self._signals = signals
        super().__init__({"position_pct": position_pct})

    def default_params(self):
        return {"position_pct": 0.1}

    def generate_signals(self, data):
        return self._signals.reindex(data.index).fillna(Signal.HOLD.value).astype(int)


def run_cross_section(datasets, params: CrossSectionParams, initial_capital: float, budget, market_kwargs=None,
                      universe=None, trade_start=None, gross_target: float = 0.95, allow_unusable: bool = False) -> dict:
    """按目标等权（gross_target / top_n）运行横截面组合；返回 run_strategy_portfolio 的结果，并附调仓记录。"""
    from ..strategies.backtest.portfolio_run import run_strategy_portfolio
    sig, log = cross_sectional_signals(datasets, params, universe, trade_start)
    pct = gross_target / params.top_n
    out = run_strategy_portfolio(lambda s: PrecomputedSignals(sig[s], pct), datasets, initial_capital,
                                 budget=budget, market_kwargs=market_kwargs, universe=universe,
                                 trade_start=trade_start, allow_unusable=allow_unusable)
    out["rebalance_log"] = log
    out["cross_section_params"] = asdict(params)
    out["turnover_audit"] = turnover_audit(out, log, params)
    return out


def turnover_audit(out: dict, log: List[dict], params: CrossSectionParams) -> dict:
    """核对约束是否落实：主动换出 ≤ 上限；每笔卖单都对应目标移出或股票池调出；按调仓周期统计实际成交换手。"""
    limit = params.max_voluntary_drops
    violations = [r["date"] for r in log if not r["formation"] and limit is not None
                  and len(r["dropped_voluntary"]) > limit]
    removed: Dict[str, List[str]] = {}
    for r in log:
        for s in r["dropped_voluntary"] + r["dropped_forced"]:
            removed.setdefault(s, []).append(r["date"])
    unexplained = []
    for o in out["orders"]:
        if o["side"] != "SELL" or o["signal"] in ("UNIVERSE_EXIT", "RISK_TRIM"):
            continue
        if not any(d <= o["signal_date"] for d in removed.get(o["symbol"], [])):
            unexplained.append(o["order_id"])
    fills = pd.DataFrame(out["fills"])
    led = out["ledger"]
    per_period = []
    dates = [pd.Timestamp(r["date"]) for r in log] + [led.index[-1] + pd.Timedelta(days=1)]
    for a, b in zip(dates[:-1], dates[1:]):
        f = fills[(pd.to_datetime(fills["date"]) > a) & (pd.to_datetime(fills["date"]) <= b)] if len(fills) else fills
        eq = led["equity"].loc[:a].iloc[-1] if (led.index <= a).any() else out["initial_capital"]
        sells = float((f.loc[f["side"] == "SELL", "quantity"] * f.loc[f["side"] == "SELL", "price"]).sum()) if len(f) else 0.0
        buys = float((f.loc[f["side"] == "BUY", "quantity"] * f.loc[f["side"] == "BUY", "price"]).sum()) if len(f) else 0.0
        per_period.append({"rebalance": str(a.date()), "buy_value": buys, "sell_value": sells,
                           "one_way_turnover": min(buys, sells) / eq if eq else 0.0})
    vol = [len(r["dropped_voluntary"]) for r in log if not r["formation"]]
    return {"voluntary_limit": limit, "limit_violations": violations, "unexplained_sell_orders": unexplained,
            "voluntary_drops_total": int(sum(vol)), "voluntary_drops_max": int(max(vol, default=0)),
            "forced_drops_total": int(sum(len(r["dropped_forced"]) for r in log)),
            "deferred_total": int(sum(len(r["deferred_by_turnover_limit"]) for r in log)),
            "rebalances": len(log),
            "mean_one_way_turnover_per_rebalance": float(np.mean([p["one_way_turnover"] for p in per_period[1:]]))
            if len(per_period) > 1 else 0.0,
            "per_rebalance": per_period}


class CrossSectionStudy(WalkForwardStudy):
    """
    横截面选股的 Walk-Forward：折的划分、按段加载数据、训练→验证选参、测试段只跑一次、实验登记都沿用
    WalkForwardStudy；只把“单段回测”换成 run_cross_section。param_grid 的键为 CrossSectionParams 的字段。
    config.warmup_days 必须覆盖最长的因子历史（lookback + skip + 1）。
    """

    def __init__(self, config, universe=None, gross_target: float = 0.95):
        super().__init__(config, strategy_factory=None, universe=universe)
        self.gross_target = gross_target
        self.audits: Dict[Tuple, dict] = {}
        need = max(CrossSectionParams(**dict(zip(config.param_grid, v))).history_needed
                   for v in itertools.product(*config.param_grid.values()))
        if config.warmup_days < need:
            raise ValueError(f"warmup_days={config.warmup_days} 不足以计算因子（至少 {need} 个交易日）")

    def run_segment(self, seg, params: dict, strategy_factory=None) -> dict:
        if strategy_factory is not None:
            return super().run_segment(seg, params, strategy_factory)
        p = CrossSectionParams(**params)
        out = run_cross_section(self.datasets(seg), p, self.cfg.initial_capital, self.budget,
                                market_kwargs=self.cfg.market, universe=self.universe_for(seg), trade_start=seg.start,
                                gross_target=self.gross_target, allow_unusable=self._allow_unusable(seg))
        out["returns"] = segment_returns(out["ledger"], self.cfg.initial_capital)
        self.audits[(seg.start, seg.end, tuple(sorted(params.items())))] = {
            "turnover_audit": {k: v for k, v in out["turnover_audit"].items() if k != "per_rebalance"},
            "rebalance_log": out["rebalance_log"]}
        return out
