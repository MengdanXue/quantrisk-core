"""
组合级 Walk-Forward 样本外研究

每个折（fold）由时间上严格先后、互不重叠的三段组成：训练 → 验证 → 测试，相邻两段之间留 embargo 个交易日。
- 参数只在训练段和验证段上选择：按训练段目标函数取前 top_k 组，再在验证段上选出最终参数；
- 测试段只运行一次，使用已选定的参数，结果不参与任何选择；
- 每一段的数据只加载到该段最后一个交易日（之后的数据不进入内存），指标预热只向前取 warmup_days 个交易日；
- 所有折的测试段按配置固定（由起止年份决定），报告中全部列出，不提供挑选测试区间的选项；
- 每次运行追加到实验登记表（JSONL），记录代码版本、配置哈希、数据快照哈希、各折选择过程与结果，
  并统计同一测试区间此前已被评估的次数。

样本外结果只说明在这些历史区间上的表现，不构成对未来收益的任何保证。
"""

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.ashare import AShareDataset, DataQualityError, load_dataset, point_in_time_usable
from ..data.universe import PointInTimeUniverse
from ..strategies.backtest.daily import RiskBudget
from ..strategies.backtest.portfolio_run import _git_version, run_strategy_portfolio
from ..strategies.base import BaseStrategy, Signal


@dataclass(frozen=True)
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp

    def as_dict(self):
        return {"start": str(self.start.date()), "end": str(self.end.date())}


@dataclass(frozen=True)
class Fold:
    train: Segment
    validation: Segment
    test: Segment


def annual_folds(calendar: pd.DatetimeIndex, test_years: List[int], train_years: int = 2,
                 embargo_days: int = 5) -> List[Fold]:
    """测试段为整个日历年；验证段为前一年；训练段为再往前 train_years 年。段间各留 embargo_days 个交易日。"""
    cal = pd.DatetimeIndex(calendar)

    def year_days(y0, y1):
        return cal[(cal.year >= y0) & (cal.year <= y1)]

    folds = []
    for y in test_years:
        test = year_days(y, y)
        val = year_days(y - 1, y - 1)
        train = year_days(y - 1 - train_years, y - 2)
        if len(test) == 0 or len(val) <= embargo_days or len(train) <= embargo_days:
            raise ValueError(f"{y} 年的折缺少足够的交易日")
        folds.append(Fold(Segment(train[0], train[-1 - embargo_days]),
                          Segment(val[0], val[-1 - embargo_days]),
                          Segment(test[0], test[-1])))
    check_folds(folds, cal, embargo_days)
    return folds


def check_folds(folds: List[Fold], calendar: pd.DatetimeIndex, embargo_days: int):
    """时间隔离检查：训练 < 验证 < 测试，段间至少 embargo_days 个交易日；各折测试段递增且互不重叠。"""
    pos = {d: i for i, d in enumerate(calendar)}
    prev_test_end = None
    for f in folds:
        for a, b in ((f.train, f.validation), (f.validation, f.test)):
            if not (a.start <= a.end and b.start <= b.end):
                raise ValueError("段的起止日期颠倒")
            if pos[b.start] - pos[a.end] <= embargo_days:
                raise ValueError(f"{a.end.date()} 与 {b.start.date()} 之间的隔离期不足 {embargo_days} 个交易日")
        if prev_test_end is not None and f.test.start <= prev_test_end:
            raise ValueError("测试段重叠或未按时间递增")
        prev_test_end = f.test.end


def segment_returns(ledger: pd.DataFrame, capital: float) -> pd.Series:
    r = ledger["equity"].pct_change()
    r.iloc[0] = ledger["equity"].iloc[0] / capital - 1
    return r


def sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd * math.sqrt(252)) if sd and sd > 0 else 0.0


def performance(returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    eq = (1 + returns).cumprod()
    out = {"total_return": float(eq.iloc[-1] - 1), "annualized_volatility": float(returns.std() * math.sqrt(252)),
           "sharpe": sharpe(returns), "max_drawdown": float((eq / eq.cummax().clip(lower=1) - 1).min()),
           "days": int(len(returns))}
    if benchmark is not None:
        b = benchmark.reindex(returns.index).fillna(0.0)
        active = returns - b
        beq = (1 + b).cumprod()
        out.update({"benchmark_return": float(beq.iloc[-1] - 1),
                    "excess_return": out["total_return"] - float(beq.iloc[-1] - 1),
                    "tracking_error": float(active.std() * math.sqrt(252)),
                    "information_ratio": float(active.mean() / active.std() * math.sqrt(252)) if active.std() > 0 else 0.0,
                    "beta": float(np.cov(returns, b)[0, 1] / np.var(b, ddof=1)) if np.var(b, ddof=1) > 0 else 0.0})
    return out


class _PointInTimeQualityUniverse:
    """在按时点股票池上叠加按时点数据可用性：资格 = 股票池资格 ∧ 当日数据可用。其余属性沿用原股票池。"""

    def __init__(self, base: PointInTimeUniverse, masks: Dict[str, pd.Series]):
        self._base, self._masks = base, masks

    def eligibility(self, symbol: str, calendar: pd.DatetimeIndex) -> pd.Series:
        elig = self._base.eligibility(symbol, calendar)
        mask = self._masks.get(symbol)
        if mask is None:
            return elig
        return elig & mask.reindex(elig.index, fill_value=True).astype(bool)

    def __getattr__(self, name):
        return getattr(self._base, name)


class EqualWeightBuyAndHold(BaseStrategy):
    """基准：交易段第一天收盘买入，持有到段末。"""

    def default_params(self):
        return {"position_pct": 0.05, "start": None}

    def generate_signals(self, d):
        s = pd.Series(Signal.HOLD.value, index=d.index)
        start = self.params["start"]
        first = s.index[s.index >= start] if start is not None else s.index
        if len(first):
            s.loc[first[0]] = Signal.BUY.value
        return s


@dataclass
class StudyConfig:
    snapshot: str
    strategy: str
    param_grid: Dict[str, List]
    test_years: List[int]
    train_years: int = 2
    embargo_days: int = 5
    warmup_days: int = 120
    top_k: int = 3
    initial_capital: float = 10_000_000
    position_pct: float = 0.1
    max_weight: float = 0.1
    max_gross: float = 0.95
    trim_to_limits: bool = False
    seasoning_days: int = 60
    # 数据质量：point_in_time = 按时点判定（事件之前可用，事件后 warmup_days 个交易日内不可用）；
    # whole_segment = 修复前的做法（整段内任一问题即整段排除，含段内之后才发生的事件），仅用于复现与对比
    quality_mode: str = "point_in_time"
    knowledge_lag_days: int = 0
    market: Dict = field(default_factory=lambda: {"slippage": 0.0005, "commission_rate": 0.00025,
                                                  "min_commission": 5.0})

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


class WalkForwardStudy:
    def __init__(self, config: StudyConfig, strategy_factory: Callable[[dict], BaseStrategy],
                 universe: Optional[PointInTimeUniverse] = None):
        self.cfg = config
        self.factory = strategy_factory
        self.snapshot = Path(config.snapshot)
        self.universe = universe or PointInTimeUniverse.from_research_snapshot(self.snapshot, config.seasoning_days)
        cal = pd.to_datetime(pd.read_csv(self.snapshot / "calendar.csv")["date"])
        self.calendar = pd.DatetimeIndex(cal).sort_values()
        self.budget = RiskBudget(config.max_weight, config.max_gross, config.trim_to_limits)
        self.folds = annual_folds(self.calendar, config.test_years, config.train_years, config.embargo_days)
        if config.quality_mode not in ("point_in_time", "whole_segment"):
            raise ValueError(f"未知的 quality_mode: {config.quality_mode}")
        self._cache: Dict[Tuple, List[AShareDataset]] = {}
        self._pit: Dict[Tuple, Dict[str, pd.Series]] = {}
        self.excluded: List[dict] = []

    # ---------- 数据：每段只加载到段末 ----------
    def _key(self, seg: Segment) -> Tuple:
        i = self.calendar.get_loc(seg.start)
        return self.calendar[max(0, i - self.cfg.warmup_days)], seg.end

    def datasets(self, seg: Segment) -> List[AShareDataset]:
        key = self._key(seg)
        load_start = key[0]
        if key not in self._cache:
            out, masks = [], {}
            for sym in self.universe.all_symbols():
                try:
                    ds = load_dataset(self.snapshot, sym, str(load_start.date()), str(seg.end.date()))
                except DataQualityError as exc:
                    self.excluded.append({"segment": seg.as_dict(), "symbol": sym, "reason": f"拒绝加载: {exc}"})
                    continue
                if ds.bars.empty or ds.bars.index[-1] < seg.start:
                    self.excluded.append({"segment": seg.as_dict(), "symbol": sym, "reason": "段内无行情（未上市或已无数据）"})
                    continue
                if ds.bars.index[-1] > seg.end:
                    raise AssertionError("加载了段末之后的数据")
                if not ds.quality["usable"]:
                    events = ds.quality.get("issue_events", [])
                    if self.cfg.quality_mode == "whole_segment" or not events:
                        # 修复前的做法：整段排除（含段内之后才发生的事件，存在前视）
                        self.excluded.append({"segment": seg.as_dict(), "symbol": sym,
                                              "reason": "; ".join(ds.quality["issues"])})
                        continue
                    # 按时点：只在事件发生后的污染窗口内不可用，事件之前照常参与
                    mask = point_in_time_usable(ds, self.cfg.warmup_days, self.cfg.knowledge_lag_days)
                    masks[sym] = mask["usable"]
                    bad = mask.index[~mask["usable"]]
                    self.excluded.append({"segment": seg.as_dict(), "symbol": sym, "mode": "point_in_time",
                                          "unusable_from": str(bad[0].date()) if len(bad) else None,
                                          "unusable_to": str(bad[-1].date()) if len(bad) else None,
                                          "event_dates": sorted({e["date"] for e in events}),
                                          "reason": "; ".join(e["detail"] for e in events)})
                out.append(ds)
            self._cache[key] = out
            self._pit[key] = masks
        return self._cache[key]

    def universe_for(self, seg: Segment):
        """该段使用的股票池：按时点股票池资格 ∧ 按时点数据可用性。"""
        self.datasets(seg)
        masks = self._pit.get(self._key(seg), {})
        return _PointInTimeQualityUniverse(self.universe, masks) if masks else self.universe

    def _allow_unusable(self, seg: Segment) -> bool:
        return bool(self._pit.get(self._key(seg)))

    def run_segment(self, seg: Segment, params: dict, strategy_factory=None) -> dict:
        factory = strategy_factory or (lambda s: self.factory({**params, "position_pct": self.cfg.position_pct}))
        out = run_strategy_portfolio(factory, self.datasets(seg), self.cfg.initial_capital, budget=self.budget,
                                     market_kwargs=self.cfg.market, universe=self.universe_for(seg),
                                     trade_start=seg.start, allow_unusable=self._allow_unusable(seg))
        out["returns"] = segment_returns(out["ledger"], self.cfg.initial_capital)
        return out

    def benchmark_returns(self, seg: Segment) -> pd.Series:
        idx = pd.read_csv(self.snapshot / "benchmark_index_daily.csv", parse_dates=["date"]).set_index("date")["close"]
        idx = idx.loc[:seg.end]
        r = idx.pct_change()
        return r.loc[seg.start:seg.end]

    def equal_weight_returns(self, seg: Segment) -> pd.Series:
        dss = self.datasets(seg)
        uni = self.universe_for(seg)
        n = max(1, sum(bool(uni.eligibility(d.symbol, pd.DatetimeIndex([seg.start])).iloc[0]) for d in dss))
        w = min(1.0, 0.98 / n)
        out = run_strategy_portfolio(lambda s: EqualWeightBuyAndHold({"position_pct": w, "start": seg.start}), dss,
                                     self.cfg.initial_capital, budget=RiskBudget(max_weight=min(1.0, w * 1.05)),
                                     market_kwargs=self.cfg.market, universe=uni, trade_start=seg.start,
                                     allow_unusable=self._allow_unusable(seg))
        return segment_returns(out["ledger"], self.cfg.initial_capital)

    # ---------- 研究流程 ----------
    def run(self, registry_path=None, report_dir=None) -> dict:
        grid = [dict(zip(self.cfg.param_grid, v)) for v in itertools.product(*self.cfg.param_grid.values())]
        folds_out, oos = [], []
        for f in self.folds:
            train_scores = []
            for p in grid:
                r = self.run_segment(f.train, p)
                train_scores.append({"params": p, "train_sharpe": sharpe(r["returns"]),
                                     "train_return": float((1 + r["returns"]).prod() - 1)})
            ranked = sorted(train_scores, key=lambda x: -x["train_sharpe"])[: self.cfg.top_k]
            for c in ranked:
                r = self.run_segment(f.validation, c["params"])
                c["validation_sharpe"] = sharpe(r["returns"])
                c["validation_return"] = float((1 + r["returns"]).prod() - 1)
            chosen = max(ranked, key=lambda x: x["validation_sharpe"])["params"]
            test = self.run_segment(f.test, chosen)                       # 测试段只运行一次
            bench = self.benchmark_returns(f.test)
            ew = self.equal_weight_returns(f.test)
            perf = performance(test["returns"], bench)
            perf["excess_vs_equal_weight"] = perf["total_return"] - float((1 + ew).prod() - 1)
            m = test["metrics"]
            perf.update({k: m[k] for k in ("turnover", "gross_exposure_mean", "gross_exposure_max",
                                           "max_single_weight", "fees_total", "fills", "orders_not_fully_filled")})
            perf["risk_events"] = len(test["risk_events"])
            perf["risk_events_severe"] = sum(e.get("severity") == "严重" for e in test["risk_events"])
            perf["universe_size_at_test_start"] = len([s for s in self.universe.all_symbols()
                                                       if self.universe.eligibility(s, pd.DatetimeIndex([f.test.start])).iloc[0]])
            perf["benchmark_volatility"] = float(bench.std() * math.sqrt(252))
            folds_out.append({"train": f.train.as_dict(), "validation": f.validation.as_dict(), "test": f.test.as_dict(),
                              "train_candidates": train_scores, "validation_candidates": ranked,
                              "chosen_params": chosen, "test_performance": perf,
                              "equal_weight_return": float((1 + ew).prod() - 1)})
            oos.append(test["returns"].rename(None))
            if report_dir is not None:
                d = Path(report_dir) / f"test_{f.test.start.year}"
                d.mkdir(parents=True, exist_ok=True)
                test["ledger"].to_csv(d / "equity_daily.csv")
                test["positions"].to_csv(d / "positions_daily.csv")
                pd.DataFrame(test["orders"]).to_csv(d / "orders.csv", index=False)
                pd.DataFrame(test["fills"]).drop(columns=["fees"], errors="ignore").to_csv(d / "fills.csv", index=False)
                pd.DataFrame(test["risk_events"]).to_csv(d / "risk_events.csv", index=False)
                pd.DataFrame(test["risk_log"]).to_csv(d / "risk_log.csv", index=False)
        oos_returns = pd.concat(oos)
        bench_all = pd.concat([self.benchmark_returns(f.test) for f in self.folds])
        result = {"config": asdict(self.cfg), "config_hash": self.cfg.config_hash(), "folds": folds_out,
                  "oos_performance": performance(oos_returns, bench_all),
                  "excluded": self.excluded, "universe_limitations": self.universe.limitations,
                  "disclaimer": "样本外结果只描述这些历史区间上的表现，不构成对未来收益的保证。"}
        if registry_path is not None:
            result["registry"] = self._register(Path(registry_path), result)
        if report_dir is not None:
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            oos_returns.to_csv(Path(report_dir) / "oos_daily_returns.csv", header=["return"])
            (Path(report_dir) / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                                         encoding="utf-8")
        return result

    def _register(self, path: Path, result: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        windows = [f["test"] for f in result["folds"]]
        prior = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if any(w in e["test_windows"] for w in windows):
                    prior.append(e["run_id"])
        manifest_sha = hashlib.sha256((self.snapshot / "manifest.json").read_bytes()).hexdigest()
        entry = {"run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{result['config_hash'][:8]}",
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "config_hash": result["config_hash"], "config": result["config"],
                 "code_version": _git_version(Path(__file__).resolve().parents[3]),
                 "snapshot_manifest_sha256": manifest_sha, "test_windows": windows,
                 "chosen_params": [f["chosen_params"] for f in result["folds"]],
                 "test_performance": [f["test_performance"] for f in result["folds"]],
                 "oos_performance": result["oos_performance"],
                 "prior_runs_on_same_test_windows": prior}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return {"run_id": entry["run_id"], "prior_runs_on_same_test_windows": len(prior)}
