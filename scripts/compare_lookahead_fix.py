"""
前视修复前后对比：读取 reports/before_lookahead_fix_20261109/（修复前的原始输出，未改动）与当前输出，汇总差异。

    python scripts/compare_lookahead_fix.py     # 输出 reports/lookahead_fix/comparison.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "reports/before_lookahead_fix_20261109"
AFTER = ROOT / "reports"
OUT = ROOT / "reports/lookahead_fix"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wf_rows(study, label):
    rows = []
    for f in study["folds"]:
        p = f["test_performance"]
        rows.append({"study": label, "test_year": int(f["test"]["start"][:4]), "chosen": json.dumps(f["chosen_params"], sort_keys=True),
                     "return": p["total_return"], "max_drawdown": p["max_drawdown"], "sharpe": p["sharpe"],
                     "turnover": p["turnover"], "vs_equal_weight": p["excess_vs_equal_weight"],
                     "equal_weight": f["equal_weight_return"], "benchmark": p["benchmark_return"]})
    o = study["oos_performance"]
    rows.append({"study": label, "test_year": "串联", "chosen": "", "return": o["total_return"],
                 "max_drawdown": o["max_drawdown"], "sharpe": o["sharpe"], "turnover": None,
                 "vs_equal_weight": None, "equal_weight": None, "benchmark": o["benchmark_return"]})
    return rows


def side(base, tag):
    out = {}
    out["wf"] = (wf_rows(load(base / "m3_walk_forward/study.json"), "ma_cross")
                 + wf_rows(load(base / "research_cross_section/wf_momentum/study.json"), "momentum")
                 + wf_rows(load(base / "research_cross_section/wf_low_vol/study.json"), "low_vol"))
    s = load(base / "research_cross_section/summary.json")
    out["multiple_testing"] = {
        fam: {"best": s[f"dsr_best_{fam}"]["trial"],
              "sharpe_annualized": s[f"dsr_best_{fam}"]["cross_trial_variance"]["sharpe_annualized"],
              "psr_vs_zero": s[f"dsr_best_{fam}"]["cross_trial_variance"]["psr_vs_zero"],
              "dsr_cross_trial": s[f"dsr_best_{fam}"]["cross_trial_variance"]["dsr"],
              "dsr_sampling": s[f"dsr_best_{fam}"]["sampling_variance"]["dsr"],
              "pbo": s[f"pbo_{fam}"]["pbo"],
              "sharpe_median": s[f"sensitivity_{fam}"]["sharpe_median"]}
        for fam in ("all", "momentum", "low_vol")}
    out["trial_matrix"] = {k: s["trial_matrix"][k] for k in ("n_trials", "days", "excluded_whole_segment")}
    out["trial_matrix"]["point_in_time_windows"] = [
        {k: e.get(k) for k in ("symbol", "event_dates", "unusable_from", "unusable_to")}
        for e in s["trial_matrix"].get("point_in_time_unusable_windows", [])]
    out["equal_weight_full_period"] = s["equal_weight_same_universe"]["total_return"]
    a = load(base / "m3_audit/audit.json")
    st = a["stats"]
    out["audit"] = {"active_best": st["active_vs_equal_weight"]["best_active"],
                    "active_best_ir": st["active_vs_equal_weight"]["best_information_ratio_annualized"],
                    "active_psr": st["active_vs_equal_weight"]["psr_best_active_vs_zero"],
                    "active_dsr_N78": st["active_vs_equal_weight"]["dsr_best_active_sampling_var_N78"],
                    "active_dsr_Neff": next(v for k, v in st["active_vs_equal_weight"].items()
                                            if k.startswith("dsr_best_active_sampling_var_N") and not k.endswith("N78")),
                    "active_n_eff": st["correlation"]["active_n_eff_participation_ratio"],
                    "ir_by_family_median": st["active_vs_equal_weight"]["ir_by_family_median"],
                    "pbo_by_blocks": st["pbo_by_blocks"], "pbo_active_16": st["pbo_active_16"],
                    "oos_vs_equal_weight": {f: {k: v[k] for k in ("information_ratio_vs_equal_weight", "psr_active_vs_zero",
                                                                 "beta_vs_equal_weight", "alpha_annualized_vs_equal_weight",
                                                                 "alpha_t_vs_equal_weight")}
                                            for f, v in st["oos_vs_equal_weight"].items()},
                    "benchmark": a["benchmark"], "turnover_cost_median": a["turnover_cost_median"],
                    "exclusions": a["exclusions"]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before, after = side(BEFORE, "before"), side(AFTER, "after")
    lag = AFTER / "research_cross_section_lag1"
    result = {"before": before, "after": after}
    if (lag / "summary.json").exists():
        s = load(lag / "summary.json")
        result["after_knowledge_lag_1"] = {
            "wf": wf_rows(load(lag / "wf_momentum/study.json"), "momentum") + wf_rows(load(lag / "wf_low_vol/study.json"), "low_vol"),
            "multiple_testing": {fam: {"dsr_sampling": s[f"dsr_best_{fam}"]["sampling_variance"]["dsr"],
                                       "pbo": s[f"pbo_{fam}"]["pbo"], "best": s[f"dsr_best_{fam}"]["trial"]}
                                 for fam in ("all", "momentum", "low_vol")}}
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    b = pd.DataFrame(before["wf"]).set_index(["study", "test_year"])
    a = pd.DataFrame(after["wf"]).set_index(["study", "test_year"])
    j = b.join(a, lsuffix="_before", rsuffix="_after")
    pd.set_option("display.width", 250)
    cols = ["return_before", "return_after", "max_drawdown_before", "max_drawdown_after", "sharpe_before", "sharpe_after",
            "turnover_before", "turnover_after", "vs_equal_weight_before", "vs_equal_weight_after"]
    print(j[cols].round(4).to_string())
    print("chosen params changed:", [(i, r["chosen_before"], r["chosen_after"]) for i, r in j.iterrows()
                                     if r["chosen_before"] != r["chosen_after"]])
    print(json.dumps({"before": before["multiple_testing"], "after": after["multiple_testing"]}, ensure_ascii=False, indent=1))
    print(json.dumps({"before": before["audit"]["oos_vs_equal_weight"], "after": after["audit"]["oos_vs_equal_weight"]},
                     ensure_ascii=False, indent=1))
    if "after_knowledge_lag_1" in result:
        print("lag1:", json.dumps(result["after_knowledge_lag_1"]["multiple_testing"], ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
