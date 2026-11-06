"""运行指标对照并生成 docs/VALIDATION.md。

    python scripts/validation_report.py
"""
import importlib.metadata
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from quantrisk.indicators.registry import indicator_registry  # noqa: E402
from tests.conftest import random_walk  # noqa: E402
from tests.talib_cases import CASES, CONVENTION_CASES, compare  # noqa: E402

SEEDS = (0, 1, 2)
# 已比对但不一致、原因未逐行核对的项：(指标名, 输出列, TA-Lib 函数, 说明)
UNRESOLVED = [
    ("ParabolicSAR", "PSAR", "SAR", "均为 Wilder 算法；差异疑似来自起始方向与反转当根的处理，未逐行核对"),
    ("MESAAdaptiveMA", "MAMA", "MAMA", "Ehlers 希尔伯特变换实现细节不同，未逐行核对"),
]


def _ours(name, col, data):
    out = indicator_registry.create_indicator(name).calculate(data)
    return out[col] if col else (out if isinstance(out, pd.Series) else out.iloc[:, 0])


def worst_error(name, col, ref):
    return max(compare(_ours(name, col, d), ref(d), tail=500)[0] for d in (random_walk(n=1500, seed=s) for s in SEEDS))


def main():
    import talib

    rows_match = [(n, c, worst_error(n, c, r)) for n, c, r, _ in CASES]
    rows_conv = [(n, c, worst_error(n, c, r), note) for n, c, r, note in CONVENTION_CASES]
    rows_unres = []
    for n, c, fn, note in UNRESOLVED:
        ref = {"SAR": lambda d: pd.Series(talib.SAR(d["high"].to_numpy(), d["low"].to_numpy(), 0.02, 0.2), index=d.index),
               "MAMA": lambda d: pd.Series(talib.MAMA(d["close"].to_numpy(), 0.5, 0.05)[0], index=d.index)}[fn]
        rows_unres.append((n, c, worst_error(n, c, ref), note))

    # 部分输出列一致、部分列口径不同的指标（如 BollingerBands）计入“口径不同”
    conv = sorted({n for n, _, _, _ in rows_conv})
    matched = sorted({n for n, _, _ in rows_match} - set(conv))
    total = indicator_registry.get_total_count()
    lines = [
        "# 指标数值验证",
        "",
        "> 本文件由 `python scripts/validation_report.py` 生成，对应测试为 `tests/test_validation.py`。",
        "",
        f"数据：几何布朗运动模拟日线，1500 根，随机种子 {', '.join(map(str, SEEDS))}；比较最后 500 根（跳过预热期），"
        f"指标为“最大绝对误差 ÷ 参考序列最大绝对值”，取 3 个种子中的最大值。参考实现：TA-Lib {importlib.metadata.version('ta-lib')}。",
        "",
        "## 汇总",
        "",
        "| 状态 | 指标数 | 说明 |",
        "|---|---:|---|",
        f"| 与 TA-Lib 一致 | {len(matched)} | 相对误差 < 1e-9 |",
        f"| 口径不同，按声明公式验证 | {len(conv)} | 与 TA-Lib 的差异来自定义选择，已对照指标自己声明的公式验证 |",
        f"| 不一致，原因未核对 | {len(rows_unres)} | 见下表，使用时请注意 |",
        f"| 未验证 | {total - len(matched) - len(conv) - len(rows_unres)} | 其余指标只保证能运行 |",
        "",
        "## 与 TA-Lib 一致",
        "",
        "| 指标 | 输出列 | 最大相对误差 |",
        "|---|---|---:|",
        *[f"| {n} | {c or '—'} | {e:.1e} |" for n, c, e in rows_match],
        "",
        "## 口径不同（已按声明公式验证）",
        "",
        "| 指标 | 输出列 | 与声明公式的最大相对误差 | 与 TA-Lib 的差异 |",
        "|---|---|---:|---|",
        *[f"| {n} | {c or '—'} | {e:.1e} | {note} |" for n, c, e, note in rows_conv],
        "",
        "## 不一致，原因未核对",
        "",
        "| 指标 | 输出列 | 与 TA-Lib 的最大相对误差 | 说明 |",
        "|---|---|---:|---|",
        *[f"| {n} | {c} | {e:.1e} | {note} |" for n, c, e, note in rows_unres],
        "",
        "## 验证过程中修复的问题",
        "",
        "| 指标 | 问题 | 修复 |",
        "|---|---|---|",
        "| Aroon / AroonOscillator | 实现与自身注释公式相反：最高价出现在 1 天前时 Aroon Up 显示 4，应为 96 | 改为 (n − 距极值周期数) / n × 100 |",
        "| TRIMA | 两层均线都用 n/2+1，合计覆盖 n+1 根K线，与注释公式和标准定义都不符 | 偶数周期 SMA(n/2) 再 SMA(n/2+1)，奇数周期两层均为 (n+1)/2 |",
        "",
    ]
    out = ROOT / "docs" / "VALIDATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out, len(matched), len(conv), len(rows_unres))


if __name__ == "__main__":
    main()
