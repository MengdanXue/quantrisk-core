"""
组合回测入口、指标与审计报告

- run_strategy_portfolio：对每只证券用同一类策略独立生成信号，交给 daily.run_portfolio 统一撮合与记账；
- portfolio_metrics：收益、回撤、波动、换手、风险暴露、费用与滑点，全部可由输出明细复核；
- signal_audit：每个信号 → 订单 → 成交的完整链路；
- write_report：把以上内容和运行清单（参数、代码版本、数据快照哈希、复现命令）写入目录。
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .daily import AShareMarket, Instrument, RiskBudget, run_portfolio


def run_strategy_portfolio(strategy_factory: Callable[[str], object], datasets: List, initial_capital: float,
                           budget: RiskBudget = RiskBudget(), market_kwargs: Optional[dict] = None,
                           universe=None, trade_start=None, allow_unusable: bool = False) -> dict:
    """
    datasets：quantrisk.data.ashare.AShareDataset 列表（同一交易日历）。
    universe：可选的按时点股票池（quantrisk.data.universe.PointInTimeUniverse）。
    trade_start：可选；此前的数据只用于计算信号（指标预热），交易与记账从 trade_start 开始。
    数据集质量检查未通过时默认拒绝运行（allow_unusable=True 时运行，并在结果中记录）。
    """
    market_kwargs = market_kwargs or {}
    unusable = {ds.symbol: ds.quality.get("issues") for ds in datasets if ds.quality.get("usable") is False}
    if unusable and not allow_unusable:
        from ...data.ashare import DataQualityError
        raise DataQualityError(f"以下数据集未通过质量检查，拒绝回测: {unusable}")
    calendars = {tuple(ds.calendar) for ds in datasets}
    if len(calendars) != 1:
        raise ValueError("组合内各证券的交易日历不一致")
    instruments, signals = [], {}
    for ds in datasets:
        if ds.bars.empty:
            continue
        strategy = strategy_factory(ds.symbol)
        data = strategy.preprocess_data(ds.signal_data.copy())
        sig = strategy.generate_signals(data)
        bars, actions = ds.bars, tuple(ds.corporate_actions)
        if trade_start is not None:
            t0 = pd.Timestamp(trade_start)
            sig, bars = sig[sig.index >= t0], bars[bars.index >= t0]
            actions = tuple(a for a in actions if a.ex_date >= t0)
            if bars.empty:
                continue
        signals[ds.symbol] = sig
        eligible = None if universe is None else universe.eligibility(ds.symbol, datasets[0].calendar)
        instruments.append(Instrument(ds.symbol, strategy, sig, bars, AShareMarket(ds.symbol, **market_kwargs),
                                      actions, eligible))
    calendar = datasets[0].calendar
    if trade_start is not None:
        calendar = calendar[calendar >= pd.Timestamp(trade_start)]
    out = run_portfolio(instruments, initial_capital, calendar=calendar, budget=budget)
    out["signals"] = signals
    out["data_quality_overrides"] = unusable
    out["initial_capital"] = initial_capital
    out["metrics"] = portfolio_metrics(out)
    out["signal_audit"] = signal_audit(out)
    return out


def portfolio_metrics(out: dict) -> Dict[str, float]:
    led, capital = out["ledger"], out["initial_capital"]
    equity = led["equity"]
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / capital - 1
    peak = equity.cummax().clip(lower=capital)
    drawdown = equity / peak - 1
    fills = pd.DataFrame(out["fills"])
    traded = float((fills["quantity"] * fills["price"]).sum()) if len(fills) else 0.0
    slippage_cost = float((fills["quantity"] * (fills["price"] - fills["open"]).abs()).sum()) if len(fills) else 0.0
    fees = float(fills["fee_total"].sum()) if len(fills) else 0.0
    pos = out["positions"]
    days = len(led)
    return {
        "initial_capital": capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / capital - 1),
        "annualized_volatility": float(returns.std() * np.sqrt(252)) if days > 1 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "turnover": traded / float(equity.mean()),                   # 全期成交额 ÷ 平均权益
        "annualized_turnover": traded / float(equity.mean()) * 252 / days,
        "gross_exposure_mean": float(led["gross_exposure"].mean()),
        "gross_exposure_max": float(led["gross_exposure"].max()),
        "max_single_weight": float(pos["weight"].max()),
        "max_positions": int(led["positions"].max()),
        "fees_total": fees,
        "slippage_cost_vs_open": slippage_cost,
        "dividends_total": float(sum(j["amount"] for j in out["journal"] if j["type"] == "DIVIDEND")),
        "orders": len(out["orders"]),
        "fills": len(out["fills"]),
        "orders_not_fully_filled": sum(o["status"] != "FILLED" for o in out["orders"]),
        "min_cash": float(led["cash"].min()),
    }


def signal_audit(out: dict) -> pd.DataFrame:
    """每个非零信号一行：是否生成订单、订单状态与原因、成交日期/价格、相对信号日收盘价的偏差。"""
    orders = {(o["symbol"], o["signal_date"]): o for o in out["orders"]}
    fills: Dict[str, List[dict]] = {}
    for f in out["fills"]:
        fills.setdefault(f["order_id"], []).append(f)
    rows = []
    for symbol, sig in out["signals"].items():
        pos = out["positions"].xs(symbol, level="symbol")
        for d, v in sig.items():
            v = int(getattr(v, "value", v)) if pd.notna(v) else 0
            if v == 0:
                continue
            day = str(pd.Timestamp(d).date())
            o = orders.get((symbol, day))
            prev_qty = pos["quantity"].get(pd.Timestamp(d), np.nan)
            row = {"symbol": symbol, "signal_date": day, "signal": v, "position_at_signal_close": prev_qty}
            if o is None:
                if v > 0:
                    reason = "已持仓，买入信号不加仓" if prev_qty > 0 else "已有挂单或当日无行情"
                else:
                    reason = "无持仓，卖出信号无动作" if prev_qty == 0 else "已有挂单或当日无行情"
                row.update(action="无订单", reason=reason)
            else:
                fl = fills.get(o["order_id"], [])
                qty = sum(f["quantity"] for f in fl)
                avg = sum(f["quantity"] * f["price"] for f in fl) / qty if qty else np.nan
                row.update(action=o["side"], order_id=o["order_id"], order_qty=o["quantity"], filled_qty=qty,
                           status=o["status"], reason=o["reason"], sizing_note=o.get("sizing_note", ""),
                           fill_date=fl[0]["date"] if fl else None, fill_price=avg,
                           signal_close=o.get("signal_close"),
                           price_vs_signal_close=(avg / o["signal_close"] - 1) if qty else np.nan,
                           fees=sum(f["fee_total"] for f in fl))
            rows.append(row)
    return pd.DataFrame(rows)


def _git_version(root: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    diff = git("diff", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(diff or untracked),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked_files": untracked.splitlines()}


def write_report(out: dict, out_dir, params: dict, snapshot_manifest: dict, command: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out["ledger"].to_csv(out_dir / "equity_daily.csv")
    out["positions"].to_csv(out_dir / "positions_daily.csv")
    pd.DataFrame(out["orders"]).to_csv(out_dir / "orders.csv", index=False)
    fills = pd.DataFrame(out["fills"])
    if len(fills):
        fees = pd.DataFrame(list(fills["fees"]))
        fills = pd.concat([fills.drop(columns=["fees"]), fees], axis=1)
        fills["slippage_vs_open"] = fills["quantity"] * (fills["price"] - fills["open"]).abs()
    fills.to_csv(out_dir / "fills.csv", index=False)
    pd.DataFrame(out["journal"]).to_csv(out_dir / "journal.csv", index=False)
    pd.DataFrame(out["trades"]).to_csv(out_dir / "round_trips.csv", index=False)
    out["signal_audit"].to_csv(out_dir / "signal_audit.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(out["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    root = Path(__file__).resolve().parents[4]
    import importlib.metadata as md
    manifest = {
        "command": command,
        "params": params,
        "budget": vars(out["budget"]),
        "code_version": _git_version(root),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {p: md.version(p) for p in ("pandas", "numpy")},
        "data_snapshot": {"source": snapshot_manifest["source"], "fetched_at": snapshot_manifest["fetched_at"],
                          "akshare_version": snapshot_manifest["akshare_version"],
                          "files": {k: v["sha256"] for k, v in snapshot_manifest["files"].items()}},
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out_dir.glob("*.csv"))},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    return out_dir
