"""
ETF 行情存储（追加写入，带获取时间戳）

- 来源：akshare 调用新浪免费接口：fund_etf_hist_sina（日线，价格 3 位小数，成交量单位为份）、
  fund_etf_dividend_sina（累计分红，日期为除息日；单次分红由累计值差分得到）、tool_trade_date_hist_sina（交易日历）。
- 每根 K 线、每条分红、每个交易日都记录首次获取时间（first_seen_utc）与获取批次（fetch_id）；
- 当日 K 线只有在北京时间收盘截止时刻（默认 15:30）之后获取才接受，之前获取的视为未完成 K 线并丢弃；
- 已存储的数据不会被覆盖：数据源事后修订的值写入 revisions.csv，账本与决策始终使用首次获取的值；
- 行情过期（缺少最近一个已收盘交易日的 K 线）时抛 StaleDataError，调用方必须停止生成交易建议；
- 从不生成或补造行情。
"""

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .etf_rules import ETFRules

CST = timezone(timedelta(hours=8))
BAR_FIELDS = ["open", "high", "low", "close", "volume", "amount"]


class StaleDataError(RuntimeError):
    """缺少最近一个已收盘交易日的行情。"""


class DataIntegrityError(RuntimeError):
    """行情数据存在无法解释的问题。"""


class AkshareETFFetcher:
    """免费数据源（新浪，经 akshare）。"""

    def _ak(self):
        import akshare
        return akshare

    def source(self) -> str:
        return f"sina via akshare {self._ak().__version__}"

    def bars(self, code: str) -> pd.DataFrame:
        df = self._ak().fund_etf_hist_sina(symbol=code)
        return df[["date"] + BAR_FIELDS]

    def dividends(self, code: str) -> pd.DataFrame:
        df = self._ak().fund_etf_dividend_sina(symbol=code)
        return df.rename(columns={"日期": "date", "累计分红": "cumulative"})[["date", "cumulative"]]

    def calendar(self) -> List[str]:
        return [str(pd.Timestamp(d).date()) for d in self._ak().tool_trade_date_hist_sina()["trade_date"]]


def _atomic_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _read(path: Path, columns) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"date": str, "ex_date": str})
    return pd.DataFrame(columns=columns)


class ETFMarketStore:
    def __init__(self, root, code: str, fetcher=None, close_cutoff_cst: str = "15:30"):
        self.dir = Path(root) / code
        self.dir.mkdir(parents=True, exist_ok=True)
        self.code = code
        self.fetcher = fetcher
        h, m = (int(x) for x in close_cutoff_cst.split(":"))
        self.cutoff = time(h, m)

    # ---------- 获取 ----------
    def fetch(self, now_utc: Optional[datetime] = None) -> dict:
        if self.fetcher is None:
            raise RuntimeError("没有配置数据源")
        now = now_utc or datetime.now(timezone.utc)
        now_cst = now.astimezone(CST)
        fetch_id = now.isoformat(timespec="seconds")
        today = now_cst.date()
        complete_today = now_cst.time() >= self.cutoff
        summary = {"fetch_id": fetch_id, "now_cst": now_cst.isoformat(timespec="seconds"), "code": self.code,
                   "source": self.fetcher.source() if hasattr(self.fetcher, "source") else "unknown",
                   "new_bars": 0, "new_dividends": 0, "new_calendar_days": 0, "revisions": 0,
                   "dropped_incomplete_bars": []}

        cal = _read(self.dir / "calendar.csv", ["date", "first_seen_utc", "fetch_id"])
        new_days = sorted(set(self.fetcher.calendar()) - set(cal["date"]))
        if new_days:
            cal = pd.concat([cal, pd.DataFrame({"date": new_days, "first_seen_utc": fetch_id, "fetch_id": fetch_id})])
            _atomic_write_csv(cal.sort_values("date"), self.dir / "calendar.csv")
        summary["new_calendar_days"] = len(new_days)

        raw = self.fetcher.bars(self.code).copy()
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        keep = []
        for _, r in raw.iterrows():
            d = date.fromisoformat(r["date"])
            if d > today or (d == today and not complete_today):
                summary["dropped_incomplete_bars"].append(r["date"])     # 未收盘或未来日期，不接受
                continue
            keep.append(r)
        raw = pd.DataFrame(keep, columns=["date"] + BAR_FIELDS)
        summary["revisions"] += self._merge(raw, "bars.csv", "date", BAR_FIELDS, fetch_id, summary, "new_bars")

        div = self.fetcher.dividends(self.code).copy()
        div["date"] = pd.to_datetime(div["date"]).dt.strftime("%Y-%m-%d")
        mask = div["date"].map(lambda x: date.fromisoformat(x) <= today).astype(bool).to_numpy()
        div = div[mask].sort_values("date")
        div["cash"] = div["cumulative"].astype(float).diff().fillna(div["cumulative"].astype(float)).round(6)
        div = div.rename(columns={"date": "ex_date"})[["ex_date", "cumulative", "cash"]]
        summary["revisions"] += self._merge(div, "dividends.csv", "ex_date", ["cumulative", "cash"], fetch_id, summary,
                                            "new_dividends")
        with (self.dir / "fetch_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return summary

    def _merge(self, new: pd.DataFrame, name: str, key: str, fields, fetch_id: str, summary: dict, counter: str) -> int:
        path = self.dir / name
        old = _read(path, [key] + list(fields) + ["first_seen_utc", "fetch_id"])
        known = old.set_index(key) if len(old) else None
        rev_rows, add = [], []
        for _, r in new.iterrows():
            k = r[key]
            if known is not None and k in known.index:
                for f in fields:
                    a, b = float(known.at[k, f]), float(r[f])
                    if abs(a - b) > 1e-9:
                        rev_rows.append({"file": name, "key": k, "field": f, "stored": a, "new": b, "fetch_id": fetch_id})
                continue
            add.append({key: k, **{f: r[f] for f in fields}, "first_seen_utc": fetch_id, "fetch_id": fetch_id})
        if add:
            merged = pd.concat([old, pd.DataFrame(add)]).sort_values(key)
            _atomic_write_csv(merged, path)
        summary[counter] += len(add)
        if rev_rows:
            rpath = self.dir / "revisions.csv"
            rev = pd.concat([_read(rpath, list(rev_rows[0])), pd.DataFrame(rev_rows)])
            _atomic_write_csv(rev, rpath)
        return len(rev_rows)

    # ---------- 读取 ----------
    def calendar(self) -> pd.DatetimeIndex:
        cal = _read(self.dir / "calendar.csv", ["date"])
        return pd.DatetimeIndex(pd.to_datetime(cal["date"])).sort_values()

    def bars(self, through=None) -> pd.DataFrame:
        b = _read(self.dir / "bars.csv", ["date"] + BAR_FIELDS + ["first_seen_utc", "fetch_id"])
        b["date"] = pd.to_datetime(b["date"])
        b = b.set_index("date").sort_index()
        for f in BAR_FIELDS:
            b[f] = b[f].astype(float)
        return b if through is None else b.loc[:pd.Timestamp(through)]

    def dividends(self, through=None) -> pd.DataFrame:
        d = _read(self.dir / "dividends.csv", ["ex_date", "cumulative", "cash", "first_seen_utc", "fetch_id"])
        d["ex_date"] = pd.to_datetime(d["ex_date"])
        d["cash"] = d["cash"].astype(float)
        d = d.set_index("ex_date").sort_index()
        return d if through is None else d.loc[:pd.Timestamp(through)]

    # ---------- 新鲜度与完整性 ----------
    def last_closed_trading_day(self, now_utc: datetime) -> pd.Timestamp:
        now_cst = now_utc.astimezone(CST)
        cal = self.calendar()
        today = pd.Timestamp(now_cst.date())
        eligible = cal[(cal < today) | ((cal == today) & (now_cst.time() >= self.cutoff))]
        if len(eligible) == 0:
            raise StaleDataError("交易日历为空或未覆盖当前日期")
        if cal.max() < today:
            raise StaleDataError(f"交易日历只覆盖到 {cal.max().date()}，未覆盖 {today.date()}")
        return eligible[-1]

    def check_fresh(self, now_utc: datetime) -> pd.Timestamp:
        need = self.last_closed_trading_day(now_utc)
        b = self.bars()
        last = b.index.max() if len(b) else None
        if last is None or last < need:
            raise StaleDataError(f"{self.code} 行情最新只到 {None if last is None else last.date()}，"
                                 f"缺少已收盘交易日 {need.date()} 的数据：停止生成交易建议")
        return need

    def integrity_issues(self, rules: ETFRules, through=None) -> List[dict]:
        """逐日检查：价格不在报价单位上、K 线不自洽、成交量为 0、非交易日、交易日缺数据、无法由分红解释的涨跌幅越界。"""
        b = self.bars(through)
        cal = self.calendar()
        div = self.dividends(through)["cash"].to_dict()
        out = []
        if len(b) == 0:
            return [{"date": None, "kind": "NO_DATA", "issue": "没有任何行情"}]
        for d in b.index.difference(cal):
            out.append({"date": str(d.date()), "kind": "OFF_CALENDAR", "issue": "行情日期不在交易日历中"})
        span = cal[(cal >= b.index.min()) & (cal <= b.index.max())]
        for d in span.difference(b.index):
            out.append({"date": str(d.date()), "kind": "GAP", "issue": "交易日缺少行情（停牌或数据缺失，未核实）"})
        prev = None
        for d, r in b.iterrows():
            if not all(rules.on_tick(r[f]) for f in ("open", "high", "low", "close")):
                out.append({"date": str(d.date()), "kind": "OFF_TICK", "issue": "价格不在 0.001 元报价单位上"})
            if not (r["low"] <= min(r["open"], r["close"]) and r["high"] >= max(r["open"], r["close"])):
                out.append({"date": str(d.date()), "kind": "BAD_BAR", "issue": "K 线不自洽（高低价与开收盘价矛盾）"})
            if r["volume"] <= 0:
                out.append({"date": str(d.date()), "kind": "ZERO_VOLUME", "issue": "成交量为 0"})
            if prev is not None:
                lo, up = rules.limit_prices(rules.ex_dividend_reference(prev, div[d]) if d in div else prev)
                if r["high"] > up + 1e-9 or r["low"] < lo - 1e-9:
                    out.append({"date": str(d.date()), "kind": "UNEXPLAINED_JUMP",
                                "issue": "价格超出涨跌幅限制且无法由分红解释（可能是份额拆分/折算等未建模事件）"})
            prev = r["close"]
        return sorted(out, key=lambda x: x["date"] or "")
