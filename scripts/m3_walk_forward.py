"""
M3 样本外研究：上证 50 当前成分股（按纳入日期入池）、2018–2024 年真实行情、均线交叉策略的 Walk-Forward。
每个测试年的组合账本另由 scripts/verify_portfolio.py 独立核算。

    python scripts/m3_walk_forward.py
    输出：reports/m3_walk_forward/（study.json、各测试年明细、verification.json）
          reports/experiments/registry.jsonl（实验登记，追加写入）

结果只描述这些历史区间上的表现，不构成对未来收益的保证；股票池存在幸存者偏差（见 study.json 的 universe_limitations）。
"""
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from quantrisk.research.walk_forward import StudyConfig, WalkForwardStudy  # noqa: E402
from quantrisk.strategies import get_strategy  # noqa: E402
from scripts.verify_portfolio import verify  # noqa: E402

OUT = ROOT / "reports/m3_walk_forward"
CONFIG = StudyConfig(snapshot=str(ROOT / "data/snapshots/research_sse50"), strategy="ma_cross",
                     param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 60]},
                     test_years=[2021, 2022, 2023, 2024], train_years=2, embargo_days=5, warmup_days=120, top_k=3,
                     initial_capital=10_000_000, position_pct=0.1, max_weight=0.1, max_gross=0.95)


def main():
    study = WalkForwardStudy(CONFIG, lambda p: get_strategy("ma_cross", p))
    result = study.run(registry_path=ROOT / "reports/experiments/registry.jsonl", report_dir=OUT)
    verification = {}
    for f in result["folds"]:
        year = f["test"]["start"][:4]
        d = OUT / f"test_{year}"
        (d / "run_manifest.json").write_text(json.dumps({"params": {"initial_capital": CONFIG.initial_capital}}),
                                             encoding="utf-8")
        v = verify(CONFIG.snapshot, d, CONFIG.initial_capital, CONFIG.market["slippage"],
                   CONFIG.market["commission_rate"], CONFIG.market["min_commission"],
                   CONFIG.max_weight, CONFIG.max_gross)
        verification[year] = {k: v[k] for k in ("fills_checked", "max_abs_equity_diff", "max_abs_cash_diff",
                                                "max_abs_position_diff", "problems")}
        p = f["test_performance"]
        print(f"{year} 参数 {f['chosen_params']} | 收益 {p['total_return']:+.2%} 基准 {p['benchmark_return']:+.2%} "
              f"超额 {p['excess_return']:+.2%} 等权超额 {p['excess_vs_equal_weight']:+.2%} 回撤 {p['max_drawdown']:.2%} "
              f"夏普 {p['sharpe']:.2f} 换手 {p['turnover']:.2f} 股票池 {p['universe_size_at_test_start']} "
              f"| 核算差 {v['max_abs_equity_diff']:.1e} 问题 {len(v['problems'])}")
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    o = result["oos_performance"]
    print(f"\n样本外合计（{len(result['folds'])} 个测试年串联）: 收益 {o['total_return']:+.2%} 基准 {o['benchmark_return']:+.2%} "
          f"超额 {o['excess_return']:+.2%} 夏普 {o['sharpe']:.2f} 回撤 {o['max_drawdown']:.2%} 信息比率 {o['information_ratio']:.2f}")
    print("实验登记:", result["registry"], "| 排除记录:", len(result["excluded"]))


if __name__ == "__main__":
    main()
