"""
实验配置：初始化时冻结，之后不可修改

- 配置写入 config.json，其 SHA-256 写入 config.sha256，实验编号 = 名称 + 哈希前 8 位（目录名）；
- 每次加载都重算哈希，与锁文件或目录名不一致即抛 ConfigChangedError：要改参数只能新建实验；
- 本金硬上限 100 元，不允许追加资金、杠杆、借贷；策略参数在初始化时登记，实验期间不可调整。
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .etf_rules import FeeScenario, load_rules

CAPITAL_CAP = 100.0


class ConfigChangedError(RuntimeError):
    """配置文件与初始化时冻结的版本不一致。"""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    etf: str
    start_date: str                         # 第一个决策日（该日收盘后生成第一批建议）
    fee: dict                               # FeeScenario.as_dict()
    selection: dict                         # 选股记录（方法、数据时间、候选表哈希）
    capital: float = CAPITAL_CAP
    friction_budget: float = 0.005          # 单边：显性费用 + 估算价差，占成交额的上限
    half_spread_ticks: int = 1              # 估算的半个买卖价差（报价单位数）
    limit_offset_ticks: int = 1             # 限价 = 信号日收盘价 ± N 个报价单位
    signal_valid_days: int = 5              # 信号发出后，未成交时最多连续挂单的交易日数
    strategy: dict = field(default_factory=lambda: {"name": "ma_cross",
                                                    "params": {"fast_period": 20, "slow_period": 60}})
    close_cutoff_cst: str = "15:30"         # 当日 K 线只在北京时间该时刻之后获取才被接受
    rehearsal: bool = False                 # True：历史回放演练（数据获取晚于决策日），不是前向实验
    benchmark: str = "buy_and_hold_same_etf"
    rules_file: str = ""
    rules_verified_on: str = ""

    def __post_init__(self):
        if not (0 < self.capital <= CAPITAL_CAP):
            raise ValueError(f"实验本金必须在 (0, {CAPITAL_CAP:.0f}] 元内，当前 {self.capital}")
        if not (0 < self.friction_budget < 0.05):
            raise ValueError("摩擦成本预算必须在 (0, 5%) 内")
        if self.strategy.get("name") != "ma_cross":
            raise ValueError("第一阶段只登记了 ma_cross 策略")
        FeeScenario(**self.fee)

    def canonical(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def experiment_id(self) -> str:
        return f"{self.name}-{self.config_hash()[:8]}"

    @property
    def fee_scenario(self) -> FeeScenario:
        return FeeScenario(**self.fee)


def make_config(name: str, etf: str, start_date: str, fee: FeeScenario, selection: dict, **kw) -> ExperimentConfig:
    rules = load_rules(etf)                 # 规则未核实时这里就抛错
    return ExperimentConfig(name=name, etf=etf, start_date=start_date, fee=fee.as_dict(), selection=selection,
                            rules_file=rules.rule_file, rules_verified_on=rules.verified_on, **kw)


def init_experiment(root, cfg: ExperimentConfig) -> Path:
    d = Path(root) / cfg.experiment_id()
    if d.exists():
        raise FileExistsError(f"实验目录已存在，不覆盖: {d}")
    d.mkdir(parents=True)
    (d / "config.json").write_text(cfg.canonical(), encoding="utf-8")
    (d / "config.sha256").write_text(cfg.config_hash() + "\n", encoding="utf-8")
    return d


def load_experiment(exp_dir) -> ExperimentConfig:
    d = Path(exp_dir)
    raw = (d / "config.json").read_text(encoding="utf-8")
    cfg = ExperimentConfig(**json.loads(raw))
    locked = (d / "config.sha256").read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if actual != locked or cfg.config_hash() != locked or not d.name.endswith(locked[:8]):
        raise ConfigChangedError(f"{d.name}：配置文件与冻结版本不一致（锁定 {locked[:12]}…，实际 {actual[:12]}…）。"
                                 "实验期间不允许修改参数；如需调整，请新建实验")
    load_rules(cfg.etf)
    return cfg


def selection_hash(table_csv_text: str) -> str:
    return hashlib.sha256(table_csv_text.encode("utf-8")).hexdigest()


def select_etf(histories, as_of, max_price: float = 0.90, lookback: int = 20) -> dict:
    """
    ETF 选择（初始化时执行一次，之后冻结）：
    在已审计白名单内、交易所规则已核实的 ETF 中，只用 as_of 之前已完成的交易日：
    最近收盘价 ≤ max_price，按最近 lookback 个交易日的平均成交额（元，日线 amount 字段）从高到低取第一名。
    数据不足 lookback 天、或这些天里有成交额缺失的候选直接剔除并记录原因。
    histories：{代码: 含 date、close、amount 的日线 DataFrame}。
    """
    import pandas as pd
    from .etf_rules import RuleNotVerifiedError, UnsupportedETFError, WHITELIST
    as_of = pd.Timestamp(as_of)
    rows = []
    for code, h in sorted(histories.items()):
        row = {"code": code, "name": WHITELIST.get(code, {}).get("name")}
        try:
            load_rules(code)
        except (RuleNotVerifiedError, UnsupportedETFError) as exc:
            row.update(eligible=False, reason=str(exc))
            rows.append(row)
            continue
        h = h.assign(date=pd.to_datetime(h["date"])).sort_values("date")
        h = h[h["date"] < as_of].tail(lookback)
        if len(h) < lookback or h["amount"].isna().any():
            row.update(eligible=False, reason=f"已完成交易日不足 {lookback} 天或成交额缺失")
        else:
            row.update(window=[str(h["date"].iloc[0].date()), str(h["date"].iloc[-1].date())],
                       last_close=float(h["close"].iloc[-1]), avg_amount=float(h["amount"].mean()),
                       eligible=float(h["close"].iloc[-1]) <= max_price)
            row["reason"] = "" if row["eligible"] else f"最近收盘价 {row['last_close']} > {max_price}"
        rows.append(row)
    ok = sorted((r for r in rows if r["eligible"]), key=lambda r: (-r["avg_amount"], r["code"]))
    if not ok:
        raise ValueError("没有符合条件的 ETF")
    return {"selected": ok[0]["code"], "rule": f"白名单 ∧ 规则已核实 ∧ 最近收盘价 ≤ {max_price} 元；"
                                               f"按最近 {lookback} 个已完成交易日平均成交额降序",
            "as_of": str(as_of.date()), "candidates": rows}
