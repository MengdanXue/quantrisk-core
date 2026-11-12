"""
A 股日线数据：下载、快照、校验与离线复现

数据来源（通过 akshare 调用新浪财经公开接口）：
- 不复权日线：ak.stock_zh_a_daily(symbol="sh600519", adjust="")，成交量单位为股
- 后复权因子：ak.stock_zh_a_daily(symbol=..., adjust="hfq-factor")，按除权除息日给出累计因子
- 前复权因子：ak.stock_zh_a_daily(symbol=..., adjust="qfq-factor")，仅用于检查前复权是否包含未来信息
- 分红送转（主）：ak.stock_dividend_cninfo(symbol="600519")，巨潮资讯（证监会指定信息披露平台），每 10 股口径；
  同一除权日的多条记录（如年度分红 + 特别分红）合并
- 分红送转（交叉核对）：ak.stock_history_dividend_detail(symbol="600519", indicator="分红")，新浪；
  已发现其遗漏同日第二笔派息（300750，2024-04-30），只用于核对，不作为入账依据（旧快照无巨潮文件时除外）
- 交易日历：ak.tool_trade_date_hist_sina()

价格口径：
- 撮合与估值用不复权价（真实成交价格），涨跌停按不复权前收盘或除权除息参考价计算；
- 信号用“截至当日”的后复权价：由公司行为记录按除权公式逐个累乘得到复权因子（只用当天及以前已发生的事件），
  并以区间首日的新浪后复权因子为起点；新浪后复权因子本身只用于交叉核对；
- 不使用前复权价：前复权会按回测区间之后发生的分红送转回溯调整历史价格。

本模块从不生成或替换为模拟数据；下载失败、快照缺失或哈希不符时直接抛出异常。
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..strategies.backtest.daily import CorporateAction, valid_bar_mask

SOURCE = "sina finance via akshare (corporate actions: cninfo via akshare)"
CNINFO_COLUMNS = ['实施方案公告日期', '分红类型', '送股比例', '转增比例', '派息比例', '股权登记日', '除权日', '派息日',
                  '股份到账日', '实施方案分红说明', '报告时间']


class DataUnavailableError(RuntimeError):
    """真实数据无法获取或快照不可用。"""


def _exchange_prefix(symbol: str) -> str:
    code = str(symbol)[-6:]
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("0", "2", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    raise ValueError(f"无法判断交易所: {symbol}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------- 下载 ----------------

def _ak():
    try:
        import akshare
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailableError("需要安装 akshare：pip install -e '.[data]'") from exc
    return akshare


def download_snapshot(symbols: List[str], start: str, end: str, out_dir, pause: float = 1.0) -> Path:
    """下载真实行情并写入快照目录（CSV + manifest.json）。任一步失败即抛错，不写入部分快照。"""
    ak = _ak()
    out = Path(out_dir)
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    files: Dict[str, dict] = {}

    def save(df: pd.DataFrame, name: str, call: str):
        time.sleep(pause)          # 控制请求频率，避免触发数据源限流
        path = tmp / name
        df.to_csv(path, index=False, lineterminator="\n")
        files[name] = {"call": call, "rows": int(len(df)), "sha256": _sha256(path)}

    try:
        cal = ak.tool_trade_date_hist_sina()
        save(cal.rename(columns={"trade_date": "date"}), "calendar.csv", "tool_trade_date_hist_sina()")
        s, e = start.replace("-", ""), end.replace("-", "")
        for sym in symbols:
            full = _exchange_prefix(sym) + sym
            raw = ak.stock_zh_a_daily(symbol=full, start_date=s, end_date=e, adjust="")
            if raw is None or raw.empty:
                raise DataUnavailableError(f"{sym} 在 {start}~{end} 没有返回行情")
            save(raw, f"{sym}_daily_raw.csv", f'stock_zh_a_daily("{full}", "{s}", "{e}", adjust="")')
            save(ak.stock_zh_a_daily(symbol=full, adjust="hfq-factor"), f"{sym}_hfq_factor.csv",
                 f'stock_zh_a_daily("{full}", adjust="hfq-factor")')
            save(ak.stock_zh_a_daily(symbol=full, adjust="qfq-factor"), f"{sym}_qfq_factor.csv",
                 f'stock_zh_a_daily("{full}", adjust="qfq-factor")')
            save(ak.stock_history_dividend_detail(symbol=sym, indicator="分红"), f"{sym}_dividends.csv",
                 f'stock_history_dividend_detail("{sym}", indicator="分红")')
            try:
                cninfo = ak.stock_dividend_cninfo(symbol=sym)
                note = ""
            except KeyError as exc:
                # akshare 在巨潮返回空记录时会在取列时抛 KeyError；明确记录为“无记录”，而不是吞掉其他错误
                if "实施方案公告日期" not in str(exc):
                    raise
                cninfo = pd.DataFrame(columns=CNINFO_COLUMNS)
                note = "巨潮返回空记录（akshare 对空结果抛出 KeyError '实施方案公告日期'）"
            save(cninfo, f"{sym}_dividends_cninfo.csv", f'stock_dividend_cninfo("{sym}")')
            if note:
                files[f"{sym}_dividends_cninfo.csv"]["note"] = note
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(f"真实数据下载失败（不会使用模拟数据替代）: {type(exc).__name__}: {exc}") from exc

    import importlib.metadata
    manifest = {
        "source": SOURCE,
        "akshare_version": importlib.metadata.version("akshare"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbols": symbols, "start": start, "end": end,
        "files": files,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if out.exists():
        raise FileExistsError(f"快照目录已存在，不覆盖: {out}")
    tmp.rename(out)
    return out


# ---------------- 读取与校验 ----------------

_VERIFIED: Dict[str, dict] = {}


def load_manifest(snapshot_dir) -> dict:
    """读取快照清单并校验全部文件的 SHA-256；同一进程内每个快照目录只在首次读取时完整校验一次。"""
    key = str(Path(snapshot_dir).resolve())
    if key in _VERIFIED:
        return _VERIFIED[key]
    manifest = _verify_manifest(snapshot_dir)
    _VERIFIED[key] = manifest
    return manifest


def _verify_manifest(snapshot_dir) -> dict:
    d = Path(snapshot_dir)
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise DataUnavailableError(f"快照不存在: {mpath}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        path = d / name
        if not path.exists():
            raise DataUnavailableError(f"快照文件缺失: {path}")
        actual = _sha256(path)
        if actual != meta["sha256"]:
            raise DataUnavailableError(f"快照文件哈希不符: {name} 期望 {meta['sha256'][:12]}… 实际 {actual[:12]}…")
    return manifest


@dataclass
class AShareDataset:
    symbol: str
    bars: pd.DataFrame                 # 不复权 OHLCV（成交量：股），索引为有行情的交易日
    signal_data: pd.DataFrame          # 截至当日后复权的 OHLCV，用于生成信号
    calendar: pd.DatetimeIndex         # 区间内的交易日
    corporate_actions: List[CorporateAction]
    quality: Dict = field(default_factory=dict)
    manifest: Dict = field(default_factory=dict)


def _factor_asof(factor: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """每个交易日取生效日期不晚于当天的最新因子（不会用到当天之后的因子）。"""
    f = factor.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    col = [c for c in f.columns if c != "date"][0]
    s = pd.merge_asof(pd.DataFrame({"date": dates}), f[["date", col]], on="date", direction="backward")
    return pd.Series(s[col].astype(float).to_numpy(), index=dates)


class DataQualityError(DataUnavailableError):
    """数据被判定为不可用于回测。"""


def _delist_dates(d: Path) -> Dict[str, pd.Timestamp]:
    out = {}
    for name, code_col, date_col in (("delisted_sh.csv", "公司代码", "暂停上市日期"),
                                     ("delisted_sz.csv", "证券代码", "终止上市日期")):
        path = d / name
        if path.exists():
            df = pd.read_csv(path, dtype={code_col: str})
            for _, r in df.iterrows():
                dt = pd.to_datetime(r[date_col], errors="coerce")
                if pd.notna(dt):
                    out[str(r[code_col]).zfill(6)] = dt
    return out


def load_dataset(snapshot_dir, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> AShareDataset:
    """
    读取快照并做数据质量检查。quality["usable"] 为 False 时，quality["issues"] 给出原因；
    组合回测入口默认拒绝使用不可用的数据集。本函数从不补造行情。
    """
    d = Path(snapshot_dir)
    manifest = load_manifest(d)
    if (start or manifest["start"]) is None or (end or manifest["end"]) is None:
        raise ValueError("该快照没有默认区间，必须指定 start 与 end")
    start = pd.Timestamp(start or manifest["start"])
    end = pd.Timestamp(end or manifest["end"])
    issues: List[str] = []

    cal = pd.to_datetime(pd.read_csv(d / "calendar.csv")["date"])
    calendar = pd.DatetimeIndex(cal[(cal >= start) & (cal <= end)]).sort_values()
    if len(calendar) == 0 or calendar[-1] < end - pd.Timedelta(days=10):
        raise DataUnavailableError(f"交易日历未覆盖回测区间至 {end.date()}")

    full = pd.read_csv(d / f"{symbol}_daily_raw.csv")
    full["date"] = pd.to_datetime(full["date"])
    full = full.set_index("date").sort_index()
    if full.index.has_duplicates:
        raise DataQualityError(f"{symbol} 行情存在重复日期")
    full_history = "完整历史" in manifest["files"].get(f"{symbol}_daily_raw.csv", {}).get("note", "")
    listing_date = full.index[0] if full_history else None
    bars = full.loc[start:end, ["open", "high", "low", "close", "volume", "amount"]].astype(float)

    off_calendar = sorted(set(bars.index) - set(calendar))
    if off_calendar:
        raise DataQualityError(f"{symbol} 行情日期不在交易日历中: {off_calendar[:5]}")
    delisted = _delist_dates(d).get(symbol)

    if bars.empty:
        quality = {"usable": False, "issues": ["区间内没有任何行情"], "rows": 0, "listing_date": listing_date,
                   "delist_date": delisted}
        return AShareDataset(symbol, bars, bars.copy(), calendar, [], quality, manifest)

    valid = valid_bar_mask(bars) & (bars["volume"] > 0)
    first, last = bars.index[0], bars.index[-1]
    within = calendar[(calendar >= first) & (calendar <= last)]
    missing = [str(x.date()) for x in within if x not in bars.index]
    pre_listing = [str(x.date()) for x in calendar if x < first]
    after_end = [str(x.date()) for x in calendar if x > last]
    events: List[dict] = []           # 带日期的问题：按时点判定可用性时使用（point_in_time_usable）
    if after_end:
        if delisted is not None and delisted <= end:
            data_end = f"已退市（退市名单日期 {delisted.date()}），{last.date()} 后无行情"
        elif full.index[-1] > last:
            data_end = "区间截止"
        else:
            data_end = f"{last.date()} 后无行情且不在退市名单中（数据中断或过期）"
            issues.append(data_end)
            events.append({"date": after_end[0], "type": "DATA_INTERRUPTED", "permanent": True, "detail": data_end})
    else:
        data_end = "覆盖至区间末"

    sina_actions = _actions_from_sina(pd.read_csv(d / f"{symbol}_dividends.csv"), start, end)
    cninfo_path = d / f"{symbol}_dividends_cninfo.csv"
    if cninfo_path.exists():
        cninfo_raw = pd.read_csv(cninfo_path)
        dup = int(cninfo_raw.duplicated().sum())
        actions = _actions_from_cninfo(cninfo_raw.drop_duplicates(), start, end)
        action_source = "cninfo"
    else:
        cninfo_raw, dup = None, 0
        actions = sina_actions
        action_source = "sina（快照中没有巨潮记录）"

    hfq = _factor_asof(pd.read_csv(d / f"{symbol}_hfq_factor.csv"), bars.index)
    validation = validate_corporate_actions(bars, actions, sina_actions if cninfo_raw is not None else None,
                                            hfq, cninfo_raw)
    for v in validation:
        if v["status"].startswith(("CONFLICT", "UNRECORDED", "ANNOUNCED_LATE")):
            issues.append(f"公司行为 {v['ex_date']}: {v['status']}")
            events.append({"date": v["ex_date"], "type": v["status"], "permanent": False,
                           "detail": f"公司行为 {v['ex_date']}: {v['status']}"})

    limit_anomalies = check_limit_band(symbol, bars, actions, listing_date)
    if limit_anomalies:
        issues.append(f"{len(limit_anomalies)} 个交易日价格超出涨跌停范围且无法由公司行为解释")
        events += [{"date": d, "type": "LIMIT_BAND_ANOMALY", "permanent": False,
                    "detail": f"{d} 价格超出涨跌停范围且无法由公司行为解释"} for d in limit_anomalies]

    factor = adjustment_factor(bars, actions, start_value=float(hfq.iloc[0]))
    signal = bars[["open", "high", "low", "close"]].mul(factor, axis=0)
    signal["volume"] = bars["volume"]
    signal.attrs["symbol"] = symbol

    quality = {
        "usable": not issues,
        "issues": issues,
        "rows": int(len(bars)),
        "calendar_days": int(len(calendar)),
        "listing_date": None if listing_date is None else str(listing_date.date()),
        "delist_date": None if delisted is None else str(delisted.date()),
        "data_end_status": data_end,
        "pre_listing_days": len(pre_listing),
        "after_data_end_days": len(after_end),
        "suspended_or_missing_days": missing,
        "invalid_bar_days": [str(x.date()) for x in bars.index[~valid]],
        "limit_band_anomalies": limit_anomalies,
        "corporate_action_source": action_source,
        "cninfo_duplicate_rows_removed": dup,
        "corporate_actions": [
            {"ex_date": str(a.ex_date.date()), "cash_per_share": a.cash_per_share, "bonus_per_share": a.bonus_per_share}
            for a in actions],
        "corporate_action_validation": validation,
        "corporate_action_reconciliation": reconcile_actions(actions, sina_actions),
        "hfq_factor_check": check_hfq_against_actions(bars, hfq, actions),
        "st_status": "未知：数据源没有带日期的风险警示历史，按非 ST 处理",
        "issue_events": sorted(events, key=lambda e: e["date"]),
    }
    return AShareDataset(symbol, bars, signal, calendar, actions, quality, manifest)


def point_in_time_usable(ds: AShareDataset, contamination_days: int, knowledge_lag_days: int = 0) -> pd.DataFrame:
    """
    按时点判定数据集每个交易日能否参与选股（quality["usable"] 是对整个加载区间的判定，含区间内之后才发生的事件）。

    每个带日期的问题事件 e（未记录的公司行为如配股、来源冲突、价格超出涨跌停、数据中断）：
    - e 之前：不可用的依据尚未出现，照常可用——不会用之后才发生的事件提前排除；
    - 从 e 之后第 knowledge_lag_days 个交易日的收盘起不可用（默认 0：事件当日收盘时，价格缺口已出现在日线数据中）；
    - 恢复：信号回看窗口不再覆盖 e，即 e 之后满 contamination_days 个交易日；数据中断类事件不恢复。
    事件日期取数据中价格实际变化的交易日（配股为除权日），不使用公告日期。
    返回 DataFrame(index=交易日, columns=[usable, reason])。
    """
    cal = pd.DatetimeIndex(ds.calendar)
    usable = pd.Series(True, index=cal)
    reason = pd.Series("", index=cal, dtype=object)
    for e in ds.quality.get("issue_events", []):
        pos = int(cal.searchsorted(pd.Timestamp(e["date"])))
        start = pos + knowledge_lag_days
        stop = len(cal) if e.get("permanent") else max(start, pos + contamination_days)
        if start >= len(cal) or start >= stop:
            continue
        usable.iloc[start:stop] = False
        reason.iloc[start:stop] = reason.iloc[start:stop].where(reason.iloc[start:stop] == "",
                                                                 reason.iloc[start:stop] + "; ") + e["detail"]
    return pd.DataFrame({"usable": usable, "reason": reason})


def validate_corporate_actions(bars: pd.DataFrame, primary: List[CorporateAction],
                               secondary: Optional[List[CorporateAction]], hfq: pd.Series,
                               cninfo_raw: Optional[pd.DataFrame] = None, tol: float = 1e-3) -> List[dict]:
    """
    逐个除权日的独立验证（区间内）：
    - VERIFIED_TWO_SOURCES：巨潮与新浪记录一致；
    - VERIFIED_BY_FACTOR：两个记录来源不一致或只有一个，但主来源与后复权因子跳变一致（相对误差 < tol）；
    - CONFLICT：主来源与其他来源都不一致；
    - VERIFIED_BY_FACTOR_NEXT_TRADED_DAY：除权日停牌，复牌首日的因子跳变与主来源按停牌前收盘价推算的结果一致；
    - EX_DATE_NOT_TRADED：除权日停牌，且复牌首日的因子跳变无法与记录对上（该日会另被判为 UNRECORDED_ADJUSTMENT）；
    - UNRECORDED_ADJUSTMENT：后复权因子在某日跳变，但两个来源都没有对应记录（可能是配股等未建模事件）；
    - ANNOUNCED_LATE：巨潮实施公告日期晚于除权日。
    """
    idx = bars.index
    prim = {a.ex_date: a for a in primary}
    sec = {a.ex_date: a for a in (secondary or [])}
    out = []

    def factor_ratio(day):
        if day not in idx:
            return None
        i = idx.get_loc(day)
        return None if i == 0 else float(hfq.iloc[i] / hfq.iloc[i - 1])

    def implied(a, day):
        i = idx.get_loc(day)
        pc = bars["close"].iloc[i - 1]
        return pc * (1 + a.bonus_per_share) / (pc - a.cash_per_share)

    absorbed = set()          # 除权日停牌时，因子跳变出现在复牌首日；核对一致的复牌首日不再判为未记录事件
    for day in sorted(set(prim) | set(sec)):
        p, q = prim.get(day), sec.get(day)
        rec = {"ex_date": str(day.date()), "primary": None if p is None else (p.cash_per_share, p.bonus_per_share),
               "secondary": None if q is None else (q.cash_per_share, q.bonus_per_share)}
        fr = factor_ratio(day)
        rec["factor_ratio"] = fr
        if day not in idx:
            # 除权日该股无行情（如停牌）：后复权因子的跳变出现在之后第一个有行情的交易日，按停牌前收盘价核对
            nxt = idx[idx > day]
            rec["status"] = "EX_DATE_NOT_TRADED"
            if p is not None and len(nxt) and idx.get_loc(nxt[0]) > 0 and nxt[0] not in prim and nxt[0] not in sec:
                first = nxt[0]
                r = factor_ratio(first)
                rec.update(first_traded_day=str(first.date()), factor_ratio=r)
                if r is not None and abs(r / implied(p, first) - 1) < tol:
                    rec["status"] = "VERIFIED_BY_FACTOR_NEXT_TRADED_DAY"
                    rec["factor_rel_diff"] = abs(r / implied(p, first) - 1)
                    absorbed.add(first)
        elif p is not None and q is not None and abs(p.cash_per_share - q.cash_per_share) < 1e-9 \
                and abs(p.bonus_per_share - q.bonus_per_share) < 1e-9:
            rec["status"] = "VERIFIED_TWO_SOURCES"
        elif p is not None and fr is not None and abs(fr / implied(p, day) - 1) < tol:
            rec["status"] = "VERIFIED_BY_FACTOR"
        elif p is None:
            rec["status"] = "CONFLICT_MISSING_IN_PRIMARY"
        else:
            rec["status"] = "CONFLICT"
        if fr is not None and p is not None:
            rec["factor_rel_diff"] = abs(fr / implied(p, day) - 1)
        out.append(rec)

    known = set(prim) | set(sec) | absorbed
    for i in range(1, len(idx)):
        r = float(hfq.iloc[i] / hfq.iloc[i - 1])
        if abs(r - 1) > 1e-6 and idx[i] not in known:
            out.append({"ex_date": str(idx[i].date()), "primary": None, "secondary": None, "factor_ratio": r,
                        "status": "UNRECORDED_ADJUSTMENT"})

    if cninfo_raw is not None and len(cninfo_raw):
        c = cninfo_raw.copy()
        c["除权日"] = pd.to_datetime(c["除权日"], errors="coerce")
        c["实施方案公告日期"] = pd.to_datetime(c["实施方案公告日期"], errors="coerce")
        late = c[(c["除权日"].isin(list(prim))) & (c["实施方案公告日期"] > c["除权日"])]
        for _, r in late.iterrows():
            out.append({"ex_date": str(r["除权日"].date()), "status": "ANNOUNCED_LATE",
                        "announced": str(r["实施方案公告日期"].date())})
    return sorted(out, key=lambda x: x["ex_date"])


def check_limit_band(symbol: str, bars: pd.DataFrame, actions: List[CorporateAction],
                     listing_date=None) -> List[str]:
    """按非 ST 的涨跌幅规则检查每日最高/最低价；新股上市初期与无前收盘的交易日跳过。"""
    from ..strategies.backtest.ashare_rules import (UnsupportedRuleError, classify_board, ex_right_reference_price,
                                                    limit_prices, price_limit_pct)
    board = classify_board(symbol)
    acts = {a.ex_date: a for a in actions}
    idx = bars.index
    close, high, low = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    listed_pos = None if listing_date is None else int((idx < listing_date).sum())
    out = []
    for i in range(1, len(idx)):
        day = idx[i]
        if listed_pos is not None and i - listed_pos < 5 and (day - listing_date).days < 30:
            continue                                     # 新股上市初期，涨跌幅规则特殊
        ref = close[i - 1]
        if day in acts:
            ref = ex_right_reference_price(ref, acts[day].cash_per_share, acts[day].bonus_per_share)
        try:
            lo, up = limit_prices(ref, price_limit_pct(board, day))
        except UnsupportedRuleError:
            continue
        if lo is None:
            continue
        if high[i] > up + 1e-9 or low[i] < lo - 1e-9:
            out.append(str(day.date()))
    return out


def _actions_from_sina(div: pd.DataFrame, start, end) -> List[CorporateAction]:
    div = div.copy()
    div["除权除息日"] = pd.to_datetime(div["除权除息日"], errors="coerce")
    div = div[(div["进度"] == "实施") & div["除权除息日"].between(start, end)]
    return _merge_same_day([
        CorporateAction(ex_date=row["除权除息日"], cash_per_share=float(row["派息"]) / 10,
                        bonus_per_share=(float(row["送股"]) + float(row["转增"])) / 10)
        for _, row in div.iterrows()])


def _actions_from_cninfo(div: pd.DataFrame, start, end) -> List[CorporateAction]:
    if div.empty:
        return []
    div = div.copy()
    div["除权日"] = pd.to_datetime(div["除权日"], errors="coerce")
    div = div[div["除权日"].between(start, end)]
    num = lambda v: 0.0 if pd.isna(v) else float(v)
    return _merge_same_day([
        CorporateAction(ex_date=row["除权日"], cash_per_share=num(row["派息比例"]) / 10,
                        bonus_per_share=(num(row["送股比例"]) + num(row["转增比例"])) / 10)
        for _, row in div.iterrows()])


def _merge_same_day(actions: List[CorporateAction]) -> List[CorporateAction]:
    """同一除权日的多条记录（如年度分红 + 特别分红）合并为一条。"""
    merged: Dict[pd.Timestamp, List[float]] = {}
    for a in actions:
        cash, bonus = merged.get(a.ex_date, [0.0, 0.0])
        merged[a.ex_date] = [cash + a.cash_per_share, bonus + a.bonus_per_share]
    return [CorporateAction(ex_date=k, cash_per_share=round(v[0], 10), bonus_per_share=round(v[1], 10))
            for k, v in sorted(merged.items())]


def reconcile_actions(primary: List[CorporateAction], secondary: List[CorporateAction]) -> List[dict]:
    """逐个除权日比较两个来源的每股现金与送转比例。"""
    a = {x.ex_date: x for x in primary}
    b = {x.ex_date: x for x in secondary}
    out = []
    for day in sorted(set(a) | set(b)):
        p, q = a.get(day), b.get(day)
        out.append({
            "ex_date": str(day.date()),
            "primary_cash": p.cash_per_share if p else None, "secondary_cash": q.cash_per_share if q else None,
            "primary_bonus": p.bonus_per_share if p else None, "secondary_bonus": q.bonus_per_share if q else None,
            "match": bool(p and q and abs(p.cash_per_share - q.cash_per_share) < 1e-9
                          and abs(p.bonus_per_share - q.bonus_per_share) < 1e-9),
        })
    return out


def adjustment_factor(bars: pd.DataFrame, actions: List[CorporateAction], start_value: float = 1.0) -> pd.Series:
    """
    截至当日的复权因子：从区间首日的 start_value 开始，每个除权日乘以
    前收盘 ×（1 + 每股送转）÷（前收盘 − 每股现金），只使用当天及以前已发生的事件。
    """
    factor = pd.Series(1.0, index=bars.index)
    value = start_value
    pending = sorted((a for a in actions if a.ex_date > bars.index[0]), key=lambda a: a.ex_date)
    j = 0
    for i, day in enumerate(bars.index):
        while j < len(pending) and pending[j].ex_date <= day:
            a = pending[j]
            prev_close = bars["close"].iloc[i - 1]
            value *= prev_close * (1 + a.bonus_per_share) / (prev_close - a.cash_per_share)
            j += 1
        factor.iloc[i] = value
    return factor


def check_hfq_against_actions(bars: pd.DataFrame, hfq: pd.Series, actions: List[CorporateAction]) -> List[dict]:
    """用分红送转记录独立推算除权日的因子变化，与数据源的后复权因子比对。"""
    out = []
    idx = bars.index
    for a in actions:
        if a.ex_date not in idx:
            continue
        i = idx.get_loc(a.ex_date)
        if i == 0:
            continue
        prev_close = bars["close"].iloc[i - 1]
        implied = prev_close * (1 + a.bonus_per_share) / (prev_close - a.cash_per_share)
        source = hfq.iloc[i] / hfq.iloc[i - 1]
        out.append({"ex_date": str(a.ex_date.date()), "implied_ratio": implied, "source_ratio": source,
                    "rel_diff": abs(source / implied - 1)})
    return out


def download_research_snapshot(index_code: str, extra_symbols: List[str], out_dir, pause: float = 1.5) -> Path:
    """
    研究用快照（仅本地使用）：指数当前成分股（含纳入日期）的完整历史不复权日线、后复权因子、两源公司行为；
    额外证券（如已退市股票）；指数日线（基准）；交易日历；沪深退市名单。
    任一步失败即抛错，不写入部分快照。
    """
    ak = _ak()
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"快照目录已存在，不覆盖: {out}")
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    files: Dict[str, dict] = {}

    def save(df: pd.DataFrame, name: str, call: str, note: str = ""):
        time.sleep(pause)
        path = tmp / name
        df.to_csv(path, index=False, lineterminator="\n")
        files[name] = {"call": call, "rows": int(len(df)), "sha256": _sha256(path), **({"note": note} if note else {})}

    try:
        save(ak.tool_trade_date_hist_sina().rename(columns={"trade_date": "date"}), "calendar.csv",
             "tool_trade_date_hist_sina()")
        cons = ak.index_stock_cons(symbol=index_code)
        save(cons, "universe_current_constituents.csv", f'index_stock_cons("{index_code}")',
             "只有当前成分股及其最近一次纳入日期，不含已调出股票")
        prefix = "sh" if index_code.startswith("000") else "sz"
        save(ak.stock_zh_index_daily(symbol=prefix + index_code), "benchmark_index_daily.csv",
             f'stock_zh_index_daily("{prefix}{index_code}")')
        save(ak.stock_info_sh_delist(symbol="全部"), "delisted_sh.csv", 'stock_info_sh_delist("全部")')
        save(ak.stock_info_sz_delist(symbol="终止上市公司"), "delisted_sz.csv", 'stock_info_sz_delist("终止上市公司")')
        symbols = sorted(set(cons["品种代码"].astype(str).str.zfill(6)) | set(extra_symbols))
        for sym in symbols:
            full = _exchange_prefix(sym) + sym
            raw = ak.stock_zh_a_daily(symbol=full, adjust="")
            if raw is None or raw.empty:
                raise DataUnavailableError(f"{sym} 没有返回行情")
            save(raw, f"{sym}_daily_raw.csv", f'stock_zh_a_daily("{full}", adjust="")', "完整历史（自上市起）")
            save(ak.stock_zh_a_daily(symbol=full, adjust="hfq-factor"), f"{sym}_hfq_factor.csv",
                 f'stock_zh_a_daily("{full}", adjust="hfq-factor")')
            save(ak.stock_history_dividend_detail(symbol=sym, indicator="分红"), f"{sym}_dividends.csv",
                 f'stock_history_dividend_detail("{sym}", indicator="分红")')
            try:
                cninfo, note = ak.stock_dividend_cninfo(symbol=sym), ""
            except KeyError as exc:
                if "实施方案公告日期" not in str(exc):
                    raise
                cninfo = pd.DataFrame(columns=CNINFO_COLUMNS)
                note = "巨潮返回空记录（akshare 对空结果抛出 KeyError '实施方案公告日期'）"
            save(cninfo, f"{sym}_dividends_cninfo.csv", f'stock_dividend_cninfo("{sym}")', note)
    except (DataUnavailableError, FileExistsError):
        raise
    except Exception as exc:
        raise DataUnavailableError(f"真实数据下载失败（不会使用模拟数据替代）: {type(exc).__name__}: {exc}") from exc

    import importlib.metadata
    manifest = {"source": SOURCE, "akshare_version": importlib.metadata.version("akshare"),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "index": index_code, "symbols": symbols, "start": None, "end": None,
                "usage": "仅本地研究使用，不提交、不公开、不再分发", "files": files}
    (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(out)
    return out
