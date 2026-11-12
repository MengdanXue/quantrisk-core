"""
前视修复的归因诊断（只用于解释结果变化，不改参数、不参与任何选择）：
修复后 600309、603501 不再被误报排除。把这两只从股票池中去掉、其余条件不变，重算 78 组试验矩阵的 DSR 与 PBO，
看修复后 PBO 的下降是否主要来自这两只股票。

    python scripts/lookahead_fix_diagnostic.py    # 约 5 分钟；输出 reports/lookahead_fix/diagnostic_without_600309_603501.json
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

from quantrisk.data.universe import PointInTimeUniverse  # noqa: E402
from quantrisk.research.cross_section import CrossSectionStudy  # noqa: E402
from scripts.research_cross_section import FAMILIES, SNAPSHOT, config, multiple_testing, trial_matrix  # noqa: E402

DROP = ("600309", "603501")


def main():
    uni = PointInTimeUniverse.from_research_snapshot(SNAPSHOT)
    for s in DROP:
        uni.memberships.pop(s)
    study = CrossSectionStudy(config(FAMILIES["momentum"]), universe=uni)
    seg, matrix, stats, excluded, pit = trial_matrix(study)
    mt = multiple_testing(matrix)
    out = {"dropped": DROP, "segment": seg.as_dict(),
           **{k: {"pbo": v["pbo"]} for k, v in mt.items() if k.startswith("pbo_")},
           **{k: {"trial": v["trial"], "dsr_sampling": v["sampling_variance"]["dsr"],
                  "sharpe_annualized": v["sampling_variance"]["sharpe_annualized"]}
              for k, v in mt.items() if k.startswith("dsr_")}}
    dest = ROOT / "reports/lookahead_fix/diagnostic_without_600309_603501.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
