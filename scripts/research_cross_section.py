"""
横截面选股研究：换手约束、两类因子的 Walk-Forward、多重检验与稳健性。

数据：本地研究快照 data/snapshots/research_sse50（上证 50 当前成分股按纳入日期入池，含幸存者偏差，见输出的
universe_limitations）。风险预算：单票 12%、总仓位 95%，不开自动减仓。

1. Walk-Forward：momentum（48 组）与 low_vol（24 组）分别按 2 年训练 → 1 年验证 → 1 年测试选参，测试年 2021–2024；
   每个测试年的账本由 scripts/verify_portfolio.py 独立核算；选中参数在测试年上另跑一次成本加倍的压力测试（只报告，不参与选择）。
2. 试验矩阵：上述 72 组 + M3 的 6 组均线交叉，在 2019–2024 连续回测，得到逐日收益矩阵。
3. 多重检验：矩阵上全部 78 组的 Deflated Sharpe Ratio（以最优者为对象）与 CSCV 回测过拟合概率 PBO。
4. 换手约束：同一组参数在“无约束 / 保留缓冲 / 换出上限 / 两者”之间的配对比较。

    python scripts/research_cross_section.py               # 全部（约 13 分钟）
    python scripts/research_cross_section.py --stats-only  # 只从已保存的试验矩阵重算多重检验，不重跑回测、不写实验登记
    python scripts/research_cross_section.py --knowledge-lag 1   # 敏感性：事件晚 1 个交易日才被知悉
    输出：reports/research_cross_section/；实验登记追加到 reports/experiments/registry.jsonl

结果只描述这些历史区间与这个（有偏的）股票池上的表现，不构成对未来收益的保证。
"""
import itertools
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from quantrisk.research.cross_section import CrossSectionParams, CrossSectionStudy, run_cross_section  # noqa: E402
from quantrisk.research.overfitting import (deflated_sharpe_ratio, pbo_cscv, per_period_sharpe,  # noqa: E402
                                            sampling_sharpe_variance)
from quantrisk.research.walk_forward import Segment, StudyConfig, performance  # noqa: E402
from quantrisk.strategies import get_strategy  # noqa: E402
from scripts.verify_portfolio import verify  # noqa: E402

# --knowledge-lag N：敏感性分析，假设配股等事件在除权日之后第 N 个交易日才被知悉（默认 0）；N>0 时输出到单独目录
KNOWLEDGE_LAG = int(sys.argv[sys.argv.index("--knowledge-lag") + 1]) if "--knowledge-lag" in sys.argv else 0
OUT = ROOT / ("reports/research_cross_section" + (f"_lag{KNOWLEDGE_LAG}" if KNOWLEDGE_LAG else ""))
REGISTRY = ROOT / "reports/experiments/registry.jsonl"
SNAPSHOT = ROOT / "data/snapshots/research_sse50"
COMMON = dict(rebalance_days=[21], top_n=[8, 12], buffer=[0, 4], max_turnover=[None, 0.25])
FAMILIES = {
    "momentum": {"factor": ["momentum"], "lookback": [60, 120, 250], "skip": [0, 20], **COMMON},
    "low_vol": {"factor": ["low_vol"], "lookback": [60, 120, 250], **COMMON},
}
MA_GRID = {"fast_period": [5, 10, 20], "slow_period": [30, 60]}
FULL = ("2019-01-01", "2024-12-31")


def config(grid):
    return StudyConfig(snapshot=str(SNAPSHOT), strategy="cross_section", param_grid=grid,
                       test_years=[2021, 2022, 2023, 2024], train_years=2, embargo_days=5, warmup_days=280, top_k=3,
                       initial_capital=10_000_000, position_pct=0.0, max_weight=0.12, max_gross=0.95,
                       knowledge_lag_days=KNOWLEDGE_LAG)


def expand(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def label(p):
    if "fast_period" in p:
        return f"ma_cross_{p['fast_period']}_{p['slow_period']}"
    mt = "none" if p["max_turnover"] is None else p["max_turnover"]
    return f"{p['factor']}_L{p['lookback']}_S{p.get('skip', 0)}_N{p['top_n']}_B{p['buffer']}_T{mt}"


def walk_forward(name, grid):
    study = CrossSectionStudy(config(grid))
    out_dir = OUT / f"wf_{name}"
    result = study.run(registry_path=REGISTRY, report_dir=out_dir)
    cfg = study.cfg
    doubled = {**cfg.market, "slippage": cfg.market["slippage"] * 2, "commission_rate": cfg.market["commission_rate"] * 2,
               "min_commission": cfg.market["min_commission"] * 2}
    rows = []
    for f, fold in zip(result["folds"], study.folds):
        year = f["test"]["start"][:4]
        d = out_dir / f"test_{year}"
        (d / "run_manifest.json").write_text(json.dumps({"params": {"initial_capital": cfg.initial_capital}}),
                                             encoding="utf-8")
        v = verify(cfg.snapshot, d, cfg.initial_capital, cfg.market["slippage"], cfg.market["commission_rate"],
                   cfg.market["min_commission"], cfg.max_weight, cfg.max_gross)
        audit = study.audits[(fold.test.start, fold.test.end, tuple(sorted(f["chosen_params"].items())))]
        (d / "rebalance_log.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        stress = run_cross_section(study.datasets(fold.test), CrossSectionParams(**f["chosen_params"]), cfg.initial_capital,
                                   study.budget, market_kwargs=doubled, universe=study.universe_for(fold.test),
                                   trade_start=fold.test.start, allow_unusable=study._allow_unusable(fold.test))
        p = f["test_performance"]
        rows.append({"family": name, "test_year": int(year), "chosen": label(f["chosen_params"]),
                     "return": p["total_return"], "benchmark": p["benchmark_return"], "excess": p["excess_return"],
                     "vs_equal_weight": p["excess_vs_equal_weight"], "max_drawdown": p["max_drawdown"],
                     "sharpe": p["sharpe"], "turnover": p["turnover"], "fees": p["fees_total"],
                     "return_cost_x2": stress["metrics"]["total_return"],
                     "voluntary_drops": audit["turnover_audit"]["voluntary_drops_total"],
                     "deferred": audit["turnover_audit"]["deferred_total"],
                     "forced_drops": audit["turnover_audit"]["forced_drops_total"],
                     "limit_violations": len(audit["turnover_audit"]["limit_violations"]),
                     "unexplained_sells": len(audit["turnover_audit"]["unexplained_sell_orders"]),
                     "fills_checked": v["fills_checked"], "max_equity_diff": v["max_abs_equity_diff"],
                     "verify_problems": len(v["problems"])})
    return study, result, rows


def trial_matrix(study):
    cal = study.calendar
    seg = Segment(cal[cal >= FULL[0]][0], cal[cal <= FULL[1]][-1])
    returns, stats = {}, []
    trials = [p for g in FAMILIES.values() for p in expand(g)] + expand(MA_GRID)
    for p in trials:
        if "fast_period" in p:
            out = study.run_segment(seg, p, strategy_factory=lambda s, p=p: get_strategy("ma_cross", {**p, "position_pct": 0.1}))
            audit = {}
        else:
            out = study.run_segment(seg, p)
            audit = out["turnover_audit"]
        r = out["returns"]
        returns[label(p)] = r
        m = out["metrics"]
        stats.append({"trial": label(p), **{k: v for k, v in p.items()}, "sharpe": per_period_sharpe(r) * math.sqrt(252),
                      "total_return": m["total_return"], "max_drawdown": m["max_drawdown"],
                      "annualized_turnover": m["annualized_turnover"], "fees_total": m["fees_total"],
                      "fills": m["fills"], "voluntary_drops": audit.get("voluntary_drops_total"),
                      "deferred": audit.get("deferred_total"), "limit_violations": len(audit.get("limit_violations", []))})
    excluded = sorted({e["symbol"] for e in study.excluded if e["segment"] == seg.as_dict() and "mode" not in e})
    pit = [e for e in study.excluded if e["segment"] == seg.as_dict() and e.get("mode") == "point_in_time"]
    return seg, pd.DataFrame(returns), pd.DataFrame(stats), excluded, pit


def turnover_pairs(stats):
    cs = stats[stats["trial"].str.match("momentum|low_vol")].copy()
    cs["max_turnover"] = cs["max_turnover"].astype(object).where(cs["max_turnover"].notna(), None)
    key = ["factor", "lookback", "skip", "top_n"]
    cs["skip"] = cs["skip"].fillna(0)
    base = cs[(cs["buffer"] == 0) & cs["max_turnover"].isna()].set_index(key)
    out = {}
    for name, sel in {"buffer_4": (cs["buffer"] == 4) & cs["max_turnover"].isna(),
                      "max_turnover_0.25": (cs["buffer"] == 0) & cs["max_turnover"].notna(),
                      "both": (cs["buffer"] == 4) & cs["max_turnover"].notna()}.items():
        alt = cs[sel].set_index(key)
        j = base.join(alt, lsuffix="_base", rsuffix="_alt", how="inner")
        d_turn = j["annualized_turnover_alt"] / j["annualized_turnover_base"] - 1
        out[name] = {"pairs": int(len(j)), "turnover_change_median": float(d_turn.median()),
                     "turnover_reduced_in": int((d_turn < 0).sum()),
                     "fees_change_median": float((j["fees_total_alt"] / j["fees_total_base"] - 1).median()),
                     "sharpe_change_median": float((j["sharpe_alt"] - j["sharpe_base"]).median()),
                     "sharpe_improved_in": int((j["sharpe_alt"] > j["sharpe_base"]).sum())}
    return out


def multiple_testing(matrix: pd.DataFrame) -> dict:
    """
    DSR 用两种口径估计“纯运气的最大夏普”：跨试验夏普方差（原文做法；本研究的试验共用股票池与行情、高度相关，会低估运气）
    与零假设下的抽样方差 1/(T-1)（保守）。两者都报告；结论以保守口径为准。
    """
    out = {}
    var0 = sampling_sharpe_variance(len(matrix))
    for fam, cols in {"all": list(matrix.columns), "momentum": [c for c in matrix if c.startswith("momentum")],
                      "low_vol": [c for c in matrix if c.startswith("low_vol")]}.items():
        sub = matrix[cols]
        fs = [per_period_sharpe(sub[c]) for c in cols]
        best = cols[int(np.argmax(fs))]
        out[f"dsr_best_{fam}"] = {"trial": best, "cross_trial_variance": deflated_sharpe_ratio(sub[best], fs),
                                  "sampling_variance": deflated_sharpe_ratio(sub[best], fs, sr_variance=var0)}
        out[f"pbo_{fam}"] = pbo_cscv(sub, n_blocks=16)
        out[f"sensitivity_{fam}"] = {"sharpe_median": float(np.median(fs) * math.sqrt(252)),
                                     "sharpe_q25": float(np.quantile(fs, 0.25) * math.sqrt(252)),
                                     "sharpe_q75": float(np.quantile(fs, 0.75) * math.sqrt(252)),
                                     "share_positive": float(np.mean(np.array(fs) > 0))}
    return out


def print_multiple_testing(summary):
    for fam in ("all", "momentum", "low_vol"):
        d = summary[f"dsr_best_{fam}"]
        a, b = d["cross_trial_variance"], d["sampling_variance"]
        print(f"DSR {fam}: 最优 {d['trial']} 年化夏普 {a['sharpe_annualized']:.2f}（{a['n_trials']} 次试验）| "
              f"跨试验方差口径：运气上限 {a['expected_max_sharpe_annualized']:.2f}，DSR {a['dsr']:.3f} | "
              f"抽样方差口径：运气上限 {b['expected_max_sharpe_annualized']:.2f}，DSR {b['dsr']:.3f} | "
              f"不折扣 PSR {a['psr_vs_zero']:.3f}")
        p = summary[f"pbo_{fam}"]
        print(f"PBO {fam}: {p['pbo']:.3f}（{p['n_configs']} 组，{p['combinations']} 种划分；样本内最优者样本外夏普中位数 "
              f"{p['oos_sharpe_of_is_best_median_annualized']:.2f}）")


def stats_only():
    matrix = pd.read_csv(OUT / "trial_daily_returns.csv", index_col=0, parse_dates=True)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    for k in [k for k in summary if k.startswith(("dsr_", "pbo_", "sensitivity_"))]:
        del summary[k]
    summary.update(multiple_testing(matrix))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print_multiple_testing(summary)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"disclaimer": "只描述这些历史区间与这个有偏股票池上的表现，不构成对未来收益的保证。"}
    wf_rows, study = [], None
    for name, grid in FAMILIES.items():
        study, result, rows = walk_forward(name, grid)
        wf_rows += rows
        o = result["oos_performance"]
        summary[f"wf_{name}"] = {"grid_size": len(expand(grid)), "oos": o, "registry": result["registry"],
                                 "excluded_records": len(result["excluded"])}
        print(f"[{time.time() - t0:5.0f}s] WF {name}: 串联收益 {o['total_return']:+.2%} 基准 {o['benchmark_return']:+.2%} "
              f"夏普 {o['sharpe']:.2f} 回撤 {o['max_drawdown']:.2%}")
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(OUT / "walk_forward_by_year.csv", index=False)
    summary["universe_limitations"] = study.universe.limitations

    seg, matrix, stats, excluded, pit = trial_matrix(study)
    matrix.to_csv(OUT / "trial_daily_returns.csv")
    stats.to_csv(OUT / "trials.csv", index=False)
    print(f"[{time.time() - t0:5.0f}s] 试验矩阵 {matrix.shape}，区间 {seg.as_dict()}，整段排除 {len(excluded)} 只")

    summary["trial_matrix"] = {"segment": seg.as_dict(), "n_trials": int(matrix.shape[1]), "days": int(matrix.shape[0]),
                               "excluded_whole_segment": excluded, "point_in_time_unusable_windows": pit,
                               "constraint_violations": int(stats["limit_violations"].sum())}
    summary.update(multiple_testing(matrix))
    ew = study.equal_weight_returns(seg)
    summary["equal_weight_same_universe"] = performance(ew, study.benchmark_returns(seg))
    summary["turnover_constraint_pairs"] = turnover_pairs(stats)
    summary["walk_forward_by_year"] = wf.to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(wf[["family", "test_year", "chosen", "return", "benchmark", "vs_equal_weight", "sharpe", "turnover",
              "return_cost_x2", "voluntary_drops", "deferred", "limit_violations", "verify_problems",
              "max_equity_diff"]].to_string(index=False))
    print_multiple_testing(summary)
    print("换手约束配对:", json.dumps(summary["turnover_constraint_pairs"], ensure_ascii=False))
    print(f"同池等权买入持有 {seg.as_dict()}: 收益 {summary['equal_weight_same_universe']['total_return']:+.2%} "
          f"夏普 {summary['equal_weight_same_universe']['sharpe']:.2f}")
    print(f"用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stats_only() if "--stats-only" in sys.argv else main()
