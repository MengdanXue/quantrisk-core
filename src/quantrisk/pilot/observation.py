"""
前向观察（只读：不改写任何账本或配置）

- 基线（observation/baseline.json，只写一次）：配置、交易规则文件、ETF 选择记录、pilot 代码的 SHA-256。
  每日检查：配置、规则或选择记录与基线不一致即停止；代码哈希变化只记录（若影响结果，流水重算比对会拦截）；
- 运行日志（observation/run_log.jsonl，追加）：每次标准操作的时间与各步骤结果（成功 / 行情过期 / 流水不一致 / 其他错误）；
- 每日观察表：从三本模拟账的流水逐日汇总行情获取时间、信号、订单生成或拒绝原因、次日成交结果、费用与价差、资金与净值；
- 复盘：观察满 N 个交易日后才生成，评价数据时效、流程纪律与一致性、执行、成本、账户表现（收益不是唯一标准）。
历史回放演练（rehearsal）一律拒绝进入前向观察。
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .etf_rules import RULES_DIR
from .experiment_config import ExperimentConfig
from .ledgers import MANUAL, PAPER_LEDGERS, Journal
from .market_data import CST

TARGET_DAYS = 20


class ObservationError(RuntimeError):
    """前向观察的前提不满足（演练实验、基线被改动、观察期未满等）。"""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_sha(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((root / "src/quantrisk/pilot").glob("*.py")) + [root / "scripts/etf_pilot.py"]:
        if p.exists():
            h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def _require_forward(cfg: ExperimentConfig):
    if cfg.rehearsal:
        raise ObservationError(f"{cfg.experiment_id()} 是历史回放演练，不能进入前向观察")


def _paths(exp_dir: Path, cfg: ExperimentConfig, selection_dir: Path, root: Path) -> Dict[str, Path]:
    return {"config": exp_dir / "config.json", "rules": RULES_DIR / cfg.rules_file,
            "selection": selection_dir / cfg.selection["file"], "root": root}


def current_fingerprint(exp_dir, cfg: ExperimentConfig, selection_dir, root) -> dict:
    p = _paths(Path(exp_dir), cfg, Path(selection_dir), Path(root))
    return {"config_sha256": _sha(p["config"]), "rules_sha256": _sha(p["rules"]),
            "selection_sha256": _sha(p["selection"]), "pilot_code_sha256": _code_sha(p["root"])}


def ensure_baseline(exp_dir, cfg: ExperimentConfig, selection_dir, root, target_days: int = TARGET_DAYS) -> dict:
    _require_forward(cfg)
    obs = Path(exp_dir) / "observation"
    obs.mkdir(exist_ok=True)
    path = obs / "baseline.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fp = current_fingerprint(exp_dir, cfg, selection_dir, root)
    if fp["selection_sha256"] != cfg.selection.get("sha256"):
        raise ObservationError("选择记录文件与配置中登记的哈希不一致")
    base = {"experiment_id": cfg.experiment_id(), "observation_start": cfg.start_date, "target_trading_days": target_days,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fp}
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    return base


def check_baseline(exp_dir, cfg: ExperimentConfig, selection_dir, root) -> dict:
    """返回 {hard: [...], soft: [...]}：hard 非空时必须停止。"""
    base = ensure_baseline(exp_dir, cfg, selection_dir, root)
    now = current_fingerprint(exp_dir, cfg, selection_dir, root)
    hard = [k for k in ("config_sha256", "rules_sha256", "selection_sha256") if now[k] != base[k]]
    soft = ["pilot_code_sha256"] if now["pilot_code_sha256"] != base["pilot_code_sha256"] else []
    return {"hard": hard, "soft": soft, "baseline": base, "current": now}


def log_run(exp_dir, entry: dict):
    obs = Path(exp_dir) / "observation"
    obs.mkdir(exist_ok=True)
    rec = {"logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with (obs / "run_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def run_log(exp_dir) -> List[dict]:
    p = Path(exp_dir) / "observation" / "run_log.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def _cst(ts: str) -> Optional[pd.Timestamp]:
    if not ts or ts == "None":
        return None
    t = pd.Timestamp(ts)
    return (t if t.tzinfo else t.tz_localize("UTC")).tz_convert(CST)


def daily_log(exp_dir, cfg: ExperimentConfig) -> pd.DataFrame:
    """逐个前向交易日汇总（以严格模拟账为主，附基准与触价敏感性账的净值）。"""
    _require_forward(cfg)
    by = {L: {r["day"]: r for r in Journal(exp_dir, L).records()} for L in PAPER_LEDGERS}
    rows = []
    for day, r in by["PAPER_STRICT"].items():
        ev = {e["type"]: e for e in r["events"]}
        dec, res, close = ev.get("DECISION"), ev.get("ORDER_RESULT"), ev["DAY_CLOSE"]
        prop = (dec or {}).get("proposal") or {}
        order = prop.get("order") or {}
        seen = _cst((dec or {}).get("data", {}).get("bar_first_seen_utc"))
        cutoff = pd.Timestamp(day).tz_localize(CST) + pd.Timedelta(hours=15, minutes=30)
        bench, touch = by["BENCHMARK"].get(day), by["PAPER_TOUCH"].get(day)
        eq = lambda rec: next(e["equity"] for e in rec["events"] if e["type"] == "DAY_CLOSE") if rec else None
        rows.append({
            "date": day, "bar_fetched_at_cst": seen.strftime("%Y-%m-%d %H:%M") if seen is not None else None,
            "fetch_delay_hours_after_1530": round((seen - cutoff).total_seconds() / 3600, 2) if seen is not None else None,
            "close": close["close"], "signal": (dec or {}).get("signal"),
            "intent": ((dec or {}).get("intent") or {}).get("side"),
            "decision": prop.get("action") or ("NO_INTENT" if dec else "NO_BAR"), "side": prop.get("side"),
            "reject_reason": prop.get("reason") or None, "limit_price": order.get("limit_price"),
            "est_explicit_fee": order.get("est_explicit_total"), "est_friction": order.get("est_friction"),
            "est_total_ratio": order.get("est_total_ratio"),
            "prev_order_result": res["status"] if res else None, "prev_order_reason": (res or {}).get("reason"),
            "fill_price": (res or {}).get("price"), "fill_fee": (res or {}).get("fee_total"),
            "spread_cost_vs_open": (res or {}).get("spread_cost_vs_open"),
            "cash": close["cash"], "holding": close["holding"], "equity": close["equity"],
            "cum_return": close["equity"] / cfg.capital - 1,
            "benchmark_equity": eq(bench), "benchmark_cum_return": (eq(bench) / cfg.capital - 1) if bench else None,
            "touch_equity": eq(touch)})
    return pd.DataFrame(rows)


def write_daily_log(exp_dir, cfg: ExperimentConfig, out_dir) -> Path:
    df = daily_log(exp_dir, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "daily_log.csv", index=False)
    lines = [f"# 前向观察记录：{cfg.experiment_id()}", "",
             f"起始决策日 {cfg.start_date}；已观察 {len(df)} / {TARGET_DAYS} 个交易日；全部为模拟成交（PAPER），"
             "真实成交见 MANUAL_LIVE（未开户时为空）。", "",
             "| 日期 | 行情获取（北京时间） | 收盘 | 信号 | 意图 | 决策 | 拒绝原因 | 限价 | 上一订单结果 | 成交价 | 费用 | 价差成本 | 现金 | 持仓 | 净值 | 累计收益 | 基准累计 |",
             "|---|---|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = lambda v, fmt="{}": "—" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)
    for r in df.to_dict("records"):
        lines.append(f"| {r['date']} | {f(r['bar_fetched_at_cst'])} | {f(r['close'], '{:.3f}')} | {f(r['signal'])} | {f(r['intent'])} | "
                     f"{f(r['decision'])} | {f(r['reject_reason'])} | {f(r['limit_price'], '{:.3f}')} | {f(r['prev_order_result'])} | "
                     f"{f(r['fill_price'], '{:.3f}')} | {f(r['fill_fee'], '{:.2f}')} | {f(r['spread_cost_vs_open'], '{:.3f}')} | "
                     f"{r['cash']:.2f} | {r['holding']} | {r['equity']:.2f} | {r['cum_return']:+.2%} | {f(r['benchmark_cum_return'], '{:+.2%}')} |")
    (out / "daily_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def review(exps: List[tuple], market_dir, out_path, target_days: int = TARGET_DAYS) -> Path:
    """
    exps：[(exp_dir, cfg, 独立复算结果, 基线检查结果)]。任一实验前向交易日不足 target_days 即拒绝。
    评价维度：数据时效、流程纪律与一致性、执行、成本、账户表现；收益只是其中一项，且样本极小。
    """
    sections, summary = [], []
    rev = Path(market_dir) / "revisions.csv"
    n_rev = max(0, sum(1 for _ in rev.open(encoding="utf-8")) - 1) if rev.exists() else 0
    for exp_dir, cfg, ver, base in exps:
        df = daily_log(exp_dir, cfg)
        if len(df) < target_days:
            raise ObservationError(f"{cfg.experiment_id()} 只观察了 {len(df)} 个交易日，未满 {target_days} 个，不生成复盘")
        df = df.head(target_days)
        runs = run_log(exp_dir)
        status = pd.Series([r.get("status") for r in runs]).value_counts().to_dict()
        delay = df["fetch_delay_hours_after_1530"].dropna()
        late = int((delay > 24).sum())
        dec = df["decision"].value_counts().to_dict()
        res = df["prev_order_result"].dropna().value_counts().to_dict()
        rej = df["reject_reason"].dropna().map(lambda s: "超预算" if "超过预算" in s else "价格跳变预热" if "跳变" in s
                                                else "完整性" if "完整性" in s else "资金不足" if "不足一手" in s else "其他").value_counts().to_dict()
        eq = df["equity"]
        mdd = float((eq / eq.cummax() - 1).min())
        manual = [r for r in Journal(exp_dir, MANUAL).records() if r["type"] == "BROKER_FILL"]
        summary.append((cfg.experiment_id(), len(df), status, len(ver["problems"]), base["hard"], eq.iloc[-1]))
        sections += [f"## {cfg.experiment_id()}（佣金情景 {cfg.fee['name']}）", "",
                     f"观察区间：{df['date'].iloc[0]} 至 {df['date'].iloc[-1]}，共 {len(df)} 个交易日。", "",
                     "### 1. 数据时效", "",
                     f"- K 线获取时间相对当日 15:30 的延迟（小时）：中位数 {delay.median():.2f}，最大 {delay.max():.2f}；"
                     f"超过 24 小时才获取（即补处理）的交易日 {late} 个",
                     f"- 数据源事后修订记录：{n_rev} 条（只记录、不覆盖）", "",
                     "### 2. 流程纪律与一致性", "",
                     f"- 标准操作运行次数及结果：{status}",
                     f"- 基线检查：配置/规则/选择记录变化 {base['hard'] or '无'}；代码变化 {base['soft'] or '无'}",
                     f"- 独立复算问题：{len(ver['problems'])} 个{('：' + '；'.join(ver['problems'][:5])) if ver['problems'] else ''}",
                     "- 流水与重算不一致：见运行日志中的 MISMATCH 次数（出现即已停止，未覆盖历史）", "",
                     "### 3. 执行", "",
                     f"- 非零信号 {int((df['signal'].fillna(0) != 0).sum())} 次；决策分布 {dec}；拒绝原因 {rej or '无'}",
                     f"- 次日订单结果 {res or '无订单'}（UNCERTAIN_TOUCH 为仅盘中触价，不计入严格账本）", "",
                     "### 4. 成本", "",
                     f"- 显性费用合计 {df['fill_fee'].fillna(0).sum():.2f} 元；价差成本合计（含在成交价中）{df['spread_cost_vs_open'].fillna(0).sum():.3f} 元",
                     f"- 订单估算单边成本占比：{df['est_total_ratio'].dropna().round(4).tolist() or '无订单'}（预算 {cfg.friction_budget:.2%}）", "",
                     "### 5. 账户（模拟）", "",
                     f"- 严格模拟：期末权益 {eq.iloc[-1]:.2f} 元，累计 {df['cum_return'].iloc[-1]:+.2%}，最大回撤 {mdd:.2%}",
                     f"- 基准（同一 ETF 买入持有）：累计 {df['benchmark_cum_return'].iloc[-1]:+.2%}" if df["benchmark_cum_return"].notna().iloc[-1] else "- 基准：无数据",
                     f"- 触价敏感性账：期末权益 {df['touch_equity'].iloc[-1]:.2f} 元",
                     f"- 真实成交（MANUAL_LIVE）：{'尚无真实交易数据' if not manual else f'{len(manual)} 笔'}", ""]
    head = ["# ETF Pilot 前向观察复盘", "",
            f"生成时间：{datetime.now(timezone.utc).astimezone(CST).isoformat(timespec='minutes')}（北京时间）。"
            "全部为前向模拟（PAPER），不含历史回放；不连接券商、不下单。", "",
            "**评价不以收益为主**：20 个交易日、100 元、1 只 ETF 的样本不能说明任何策略是否有效。"
            "这份复盘首先检查数据是否按时、流程是否守纪律、账本是否可复算、执行与成本是否符合设定。", "",
            "| 实验 | 观察交易日 | 运行结果 | 独立复算问题 | 基线变化 | 期末权益 |", "|---|---:|---|---:|---|---:|"]
    head += [f"| {a} | {b} | {c} | {d} | {e or '无'} | {f:.2f} |" for a, b, c, d, e, f in summary]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + [""] + sections), encoding="utf-8")
    return out
