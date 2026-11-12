"""
M3 研究可信度审计：只读本地快照与 scripts/research_cross_section.py 的输出，不引入新数据源、不改研究参数。

    python scripts/m3_audit.py      # 约 6 分钟；输出 reports/m3_audit/audit.json 与若干 CSV

检查项：
A. 股票池：各年可交易股票数、与真实指数 50 只的差距、同池等权与指数的逐年差距；
B. 基准口径：指数（价格指数、无成本）与策略（含分红、含成本）的差别，逐年量化分红、费用、期末未计的平仓成本、首日错位；
C. 数据质量排除：触发排除的事件发生在段开始之前（过去信息）还是之后（前视）；
D. 统计：78 组试验收益的相关性与有效试验次数；PSR/DSR 公式的蒙特卡洛校验；相对同池等权的超额收益检验；PBO 对分段数的敏感性；
E. 换手约束：72 组横截面配置在约束前后的换手、费用、滑点、年化收益、最大回撤、夏普（绝对值）。
"""
import json
import logging
import math
import re
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

from quantrisk.research.cross_section import CrossSectionStudy  # noqa: E402
from quantrisk.research.overfitting import (deflated_sharpe_ratio, expected_max_sharpe, pbo_cscv,  # noqa: E402
                                            per_period_sharpe, probabilistic_sharpe_ratio,
                                            sampling_sharpe_variance)
from quantrisk.research.walk_forward import Segment  # noqa: E402
from scripts.research_cross_section import FAMILIES, FULL, config, expand, label  # noqa: E402

RES = ROOT / "reports/research_cross_section"
OUT = ROOT / "reports/m3_audit"
INDEX_SIZE = 50


def universe_audit(study):
    cal = study.calendar
    rows = []
    for y in range(2017, 2025):
        day = cal[cal.year == y][0]
        n = sum(bool(study.universe.eligibility(s, pd.DatetimeIndex([day])).iloc[0]) for s in study.universe.all_symbols())
        rows.append({"year": y, "first_day": str(day.date()), "eligible": n, "index_size": INDEX_SIZE,
                     "missing_vs_index": INDEX_SIZE - n})
    cons = pd.read_csv(study.snapshot / "universe_current_constituents.csv", dtype={"品种代码": str})
    incl = pd.to_datetime(cons["纳入日期"]).dt.year.value_counts().sort_index()
    return {"pool_by_year": rows, "inclusion_year_counts": {int(k): int(v) for k, v in incl.items()},
            "constituents_total": int(len(cons)),
            "included_after_2024": int((pd.to_datetime(cons["纳入日期"]) > "2024-12-31").sum()),
            "delisted_or_removed_in_universe": [s for s in study.universe.all_symbols()
                                                if study.universe.delist_dates.get(s) is not None],
            "extra_symbols_in_snapshot_not_in_universe": sorted(
                set(json.loads((study.snapshot / "manifest.json").read_text(encoding="utf-8"))["symbols"])
                - set(study.universe.all_symbols()))}


def exclusion_audit():
    rows = []
    for fam in FAMILIES:
        s = json.loads((RES / f"wf_{fam}" / "study.json").read_text(encoding="utf-8"))
        for e in s["excluded"]:
            if e.get("mode") == "point_in_time":
                rows.append({"family": fam, **e["segment"], "symbol": e["symbol"], "event_date": ",".join(e["event_dates"]),
                             "classification": f"按时点：{e['unusable_from']} 至 {e['unusable_to']} 不可用"
                             if e["unusable_from"] else "按时点：段内无不可用日"})
                continue
            m = re.findall(r"(\d{4}-\d{2}-\d{2}): (UNRECORDED_ADJUSTMENT|CONFLICT\w*|ANNOUNCED_LATE)", e["reason"])
            if not m:
                kind = "无行情（未上市/已无数据）" if "段内无行情" in e["reason"] else "其他"
                rows.append({"family": fam, **e["segment"], "symbol": e["symbol"], "event_date": None,
                             "classification": kind})
                continue
            first = min(d for d, _ in m)
            rows.append({"family": fam, **e["segment"], "symbol": e["symbol"], "event_date": first,
                         "classification": "前视：事件在段开始之后" if first >= e["segment"]["start"]
                         else "过去信息：事件在预热期"})
    return pd.DataFrame(rows).drop_duplicates(subset=["start", "end", "symbol", "classification"])


def benchmark_audit(studies):
    rows = []
    for fam, study in studies.items():
        s = json.loads((RES / f"wf_{fam}" / "study.json").read_text(encoding="utf-8"))
        for f, fold in zip(s["folds"], study.folds):
            seg = fold.test
            out = study.run_segment(seg, f["chosen_params"])
            m = out["metrics"]
            led = out["ledger"]
            # 期末持仓若全部卖出需要的成本（本研究未计入）：佣金 + 印花税 + 过户费 + 滑点
            pos = out["positions"].xs(led.index[-1], level="date")
            mv = float(pos["market_value"].sum())
            stamp = 0.0005 if seg.end >= pd.Timestamp("2023-08-28") else 0.001
            transfer = 0.00001 if seg.end >= pd.Timestamp("2022-04-29") else 0.00002
            exit_cost = mv * (study.cfg.market["commission_rate"] + stamp + transfer + study.cfg.market["slippage"])
            bench = study.benchmark_returns(seg)
            ew = study.equal_weight_returns(seg)
            rows.append({"family": fam, "test_year": seg.start.year, "chosen": label(f["chosen_params"]),
                         "strategy_return": m["total_return"], "equal_weight_return": float((1 + ew).prod() - 1),
                         "index_price_return": float((1 + bench).prod() - 1),
                         "index_first_day_return": float(bench.iloc[0]),
                         "strategy_first_day_return": float(out["returns"].iloc[0]),
                         "dividends_pct": m["dividends_total"] / study.cfg.initial_capital,
                         "fees_pct": m["fees_total"] / study.cfg.initial_capital,
                         "slippage_pct": m["slippage_cost_vs_open"] / study.cfg.initial_capital,
                         "unbooked_exit_cost_pct": exit_cost / study.cfg.initial_capital,
                         "gross_exposure_mean": m["gross_exposure_mean"]})
    df = pd.DataFrame(rows)
    summary = {}
    for fam, g in df.groupby("family"):
        chain = lambda c: float((1 + g[c]).prod() - 1)
        summary[fam] = {"strategy": chain("strategy_return"), "equal_weight_same_pool": chain("equal_weight_return"),
                        "index_price": chain("index_price_return"),
                        "dividends_pct_sum": float(g["dividends_pct"].sum()), "fees_pct_sum": float(g["fees_pct"].sum()),
                        "slippage_pct_sum": float(g["slippage_pct"].sum()),
                        "unbooked_exit_cost_pct_sum": float(g["unbooked_exit_cost_pct"].sum())}
    return df, summary


def stats_audit(study, matrix):
    out = {}
    corr = matrix.corr().values
    n = corr.shape[0]
    iu = np.triu_indices(n, 1)
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    rho = float(corr[iu].mean())
    out["correlation"] = {"mean_pairwise": rho, "min_pairwise": float(corr[iu].min()),
                          "n_eff_participation_ratio": float(eig.sum() ** 2 / (eig ** 2).sum()),
                          "n_eff_mean_correlation": float(n / (1 + (n - 1) * rho))}
    for fam in ("momentum", "low_vol", "ma_cross"):
        cols = [i for i, c in enumerate(matrix.columns) if c.startswith(fam)]
        sub = corr[np.ix_(cols, cols)]
        out["correlation"][f"mean_pairwise_{fam}"] = float(sub[np.triu_indices(len(cols), 1)].mean())

    # 原始收益的相关性主要来自共同的市场涨跌（全部是只做多的 A 股组合），据此估计的有效试验次数会严重偏低；
    # 配置之间真正的差别是“选了哪些股票”，因此有效试验次数用相对同池等权的超额收益估计。
    seg = Segment(pd.Timestamp(matrix.index[0]), pd.Timestamp(matrix.index[-1]))
    ew = study.equal_weight_returns(seg).reindex(matrix.index)
    active = matrix.sub(ew, axis=0)
    ac = active.corr().values
    ae = np.clip(np.linalg.eigvalsh(ac), 0, None)
    arho = float(ac[iu].mean())
    out["correlation"].update({"active_mean_pairwise": arho,
                               "active_n_eff_participation_ratio": float(ae.sum() ** 2 / (ae ** 2).sum()),
                               "active_n_eff_mean_correlation": float(n / (1 + (n - 1) * arho))})
    neff = max(2, int(round(out["correlation"]["active_n_eff_participation_ratio"])))
    t = len(matrix)
    var0 = sampling_sharpe_variance(t)

    srs = [per_period_sharpe(matrix[c]) for c in matrix]
    best = matrix.columns[int(np.argmax(srs))]
    out["dsr_absolute"] = {"best": best, "psr_vs_zero": probabilistic_sharpe_ratio(matrix[best]),
                           "cross_trial_var_N78": deflated_sharpe_ratio(matrix[best], srs)["dsr"],
                           "sampling_var_N78": deflated_sharpe_ratio(matrix[best], srs, sr_variance=var0)["dsr"],
                           f"sampling_var_N{neff}": deflated_sharpe_ratio(matrix[best], srs, n_trials=neff,
                                                                          sr_variance=var0)["dsr"]}

    # 相对同池等权买入持有：同引擎、同成本、同股票池、同日期，幸存者偏差部分抵消
    a_srs = [per_period_sharpe(active[c]) for c in active]
    a_best = active.columns[int(np.argmax(a_srs))]
    out["active_vs_equal_weight"] = {
        "equal_weight_sharpe_annualized": per_period_sharpe(ew) * math.sqrt(252),
        "best_active": a_best, "best_information_ratio_annualized": max(a_srs) * math.sqrt(252),
        "share_configs_with_positive_ir": float(np.mean(np.array(a_srs) > 0)),
        "ir_by_family_median": {fam: float(np.median([s for c, s in zip(active.columns, a_srs) if c.startswith(fam)])
                                       * math.sqrt(252)) for fam in ("momentum", "low_vol", "ma_cross")},
        "psr_best_active_vs_zero": probabilistic_sharpe_ratio(active[a_best]),
        "dsr_best_active_sampling_var_N78": deflated_sharpe_ratio(active[a_best], a_srs, sr_variance=var0)["dsr"],
        f"dsr_best_active_sampling_var_N{neff}": deflated_sharpe_ratio(active[a_best], a_srs, n_trials=neff,
                                                                       sr_variance=var0)["dsr"]}

    # Walk-Forward 样本外（2021–2024）相对同池等权的超额收益，以及对同池等权与指数的 beta / alpha（OLS，未做 HAC 调整）
    ews = pd.concat([study.equal_weight_returns(f.test) for f in study.folds])
    oos_out = {}
    for fam in FAMILIES:
        oos = pd.read_csv(RES / f"wf_{fam}" / "oos_daily_returns.csv", index_col=0, parse_dates=True)["return"]
        idx = pd.concat([study.benchmark_returns(f.test) for f in study.folds]).reindex(oos.index).fillna(0.0)
        a = oos - ews.reindex(oos.index)
        row = {"information_ratio_vs_equal_weight": per_period_sharpe(a) * math.sqrt(252),
               "psr_active_vs_zero": probabilistic_sharpe_ratio(a),
               "active_sum_by_year": {int(k): float(v) for k, v in a.groupby(a.index.year).sum().items()}}
        for name, b in (("equal_weight", ews.reindex(oos.index)), ("index_price", idx)):
            x = np.c_[np.ones(len(b)), b.values]
            coef = np.linalg.lstsq(x, oos.values, rcond=None)[0]
            resid = oos.values - x @ coef
            se = np.sqrt(np.diag(resid @ resid / (len(resid) - 2) * np.linalg.inv(x.T @ x)))
            row[f"beta_vs_{name}"] = float(coef[1])
            row[f"alpha_annualized_vs_{name}"] = float(coef[0] * 252)
            row[f"alpha_t_vs_{name}"] = float(coef[0] / se[0])
        oos_out[fam] = row
    out["oos_vs_equal_weight"] = oos_out
    out["pbo_by_blocks"] = {b: pbo_cscv(matrix, n_blocks=b)["pbo"] for b in (8, 10, 12, 16)}
    out["pbo_active_16"] = pbo_cscv(active, n_blocks=16)["pbo"]

    # 公式校验（蒙特卡洛）：真实夏普为 0 时 PSR>0.95 的频率应约为 5%；N 次独立试验最大夏普的均值应接近 expected_max_sharpe
    rng = np.random.default_rng(0)
    sims = rng.normal(0, 0.01, (4000, 500))
    psr = np.array([probabilistic_sharpe_ratio(x) for x in sims])
    sr_hat = sims.mean(axis=1) / sims.std(axis=1, ddof=1)
    var = float(sr_hat.var(ddof=1))
    maxes = [sr_hat[rng.choice(4000, 50, replace=False)].max() for _ in range(2000)]
    out["formula_checks"] = {"psr_false_positive_rate_at_0.95": float((psr > 0.95).mean()),
                             "sharpe_sampling_variance_mc": var, "sharpe_sampling_variance_formula": 1 / 499,
                             "expected_max_of_50_mc": float(np.mean(maxes)),
                             "expected_max_of_50_formula": expected_max_sharpe(50, var)}
    registry = [json.loads(x) for x in (ROOT / "reports/experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()]
    out["test_window_reuse"] = [{"run_id": e["run_id"], "strategy": e["config"]["strategy"],
                                 "prior_runs_on_same_test_windows": len(e["prior_runs_on_same_test_windows"]),
                                 "code_dirty": e["code_version"]["dirty"]} for e in registry]
    return out


def cost_audit(study):
    cal = study.calendar
    seg = Segment(cal[cal >= FULL[0]][0], cal[cal <= FULL[1]][-1])
    years = len(cal[(cal >= seg.start) & (cal <= seg.end)]) / 252
    rows = []
    for fam, grid in FAMILIES.items():
        for p in expand(grid):
            out = study.run_segment(seg, p)
            m, r = out["metrics"], out["returns"]
            rows.append({**{k: p.get(k, 0) for k in ("factor", "lookback", "skip", "top_n", "buffer")},
                         "max_turnover": "none" if p["max_turnover"] is None else p["max_turnover"],
                         "annualized_turnover_two_sided": m["annualized_turnover"],
                         "fees_pct_per_year": m["fees_total"] / study.cfg.initial_capital / years,
                         "slippage_pct_per_year": m["slippage_cost_vs_open"] / study.cfg.initial_capital / years,
                         "annualized_return": (1 + m["total_return"]) ** (1 / years) - 1,
                         "max_drawdown": m["max_drawdown"], "sharpe": per_period_sharpe(r) * math.sqrt(252)})
    df = pd.DataFrame(rows)
    df["cost_pct_per_year"] = df["fees_pct_per_year"] + df["slippage_pct_per_year"]
    key = ["factor", "lookback", "skip", "top_n"]
    variants = {"无约束": (0, "none"), "保留缓冲4": (4, "none"), "换出上限0.25": (0, 0.25), "两者": (4, 0.25)}
    cols = ["annualized_turnover_two_sided", "cost_pct_per_year", "fees_pct_per_year", "slippage_pct_per_year",
            "annualized_return", "max_drawdown", "sharpe"]
    table = {}
    for fam in ("momentum", "low_vol"):
        g = df[df["factor"] == fam]
        table[fam] = {name: g[(g["buffer"] == b) & (g["max_turnover"] == t)][cols].median().to_dict()
                      for name, (b, t) in variants.items()}
    return df, table


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    studies = {fam: CrossSectionStudy(config(grid)) for fam, grid in FAMILIES.items()}
    study = studies["momentum"]
    audit = {"universe": universe_audit(study)}
    ex = exclusion_audit()
    ex.to_csv(OUT / "exclusions.csv", index=False)
    audit["exclusions"] = ex.groupby(["family", "classification"]).size().unstack(fill_value=0).to_dict(orient="index")
    audit["exclusions_lookahead"] = ex[ex["classification"].str.startswith("前视")].to_dict(orient="records")
    print(f"[{time.time() - t0:4.0f}s] 股票池与排除记录完成")
    bdf, bsum = benchmark_audit(studies)
    bdf.to_csv(OUT / "benchmark_by_year.csv", index=False)
    audit["benchmark"] = bsum
    print(f"[{time.time() - t0:4.0f}s] 基准口径完成")
    matrix = pd.read_csv(RES / "trial_daily_returns.csv", index_col=0, parse_dates=True)
    audit["stats"] = stats_audit(study, matrix)
    print(f"[{time.time() - t0:4.0f}s] 统计审计完成")
    cdf, ctable = cost_audit(study)
    cdf.to_csv(OUT / "turnover_cost_by_config.csv", index=False)
    audit["turnover_cost_median"] = ctable
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[{time.time() - t0:4.0f}s] 完成")
    print(json.dumps({k: audit[k] for k in ("universe", "exclusions", "benchmark", "stats", "turnover_cost_median")},
                     ensure_ascii=False, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
