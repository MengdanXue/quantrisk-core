"""
按时点（point-in-time）的股票池

某只股票在交易日 d 可交易，当且仅当：
1. d 落在它的某个成员区间 [纳入日, 调出日) 内；
2. 已上市满 seasoning_days 个交易日（避开新股上市初期的特殊涨跌幅与定价）；
3. d 早于退市日期（如有）。

数据限制（如实记录在 limitations 中）：akshare 的 index_stock_cons 只提供当前成分股及其最近一次纳入日期，
没有已调出股票的历史区间。用它构造的股票池对“纳入”是按时点的，但缺少历史上被调出的股票，
因此存在幸存者偏差；只有在提供完整的历史成员区间时，本模块才能消除这一偏差。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PointInTimeUniverse:
    memberships: Dict[str, List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]]
    listing_dates: Dict[str, Optional[pd.Timestamp]] = field(default_factory=dict)
    delist_dates: Dict[str, Optional[pd.Timestamp]] = field(default_factory=dict)
    seasoning_days: int = 60
    limitations: List[str] = field(default_factory=list)
    trading_calendar: Optional[pd.DatetimeIndex] = None    # 完整交易日历，用于计算上市以来的交易日数

    def is_member(self, symbol: str, day) -> bool:
        day = pd.Timestamp(day)
        return any(start <= day and (end is None or day < end) for start, end in self.memberships.get(symbol, []))

    def eligibility(self, symbol: str, calendar: pd.DatetimeIndex) -> pd.Series:
        cal = pd.DatetimeIndex(calendar)
        member = pd.Series([self.is_member(symbol, d) for d in cal], index=cal)
        listing = self.listing_dates.get(symbol)
        if listing is None:
            seasoned = pd.Series(True, index=cal)
        else:
            # 上市以来的交易日数：在完整交易日历上计数，只依赖上市日期（公开信息）
            full = self.trading_calendar if self.trading_calendar is not None else cal
            listed_pos = full.searchsorted(listing)
            days_listed = full.searchsorted(cal) - listed_pos
            seasoned = pd.Series((cal >= listing) & (days_listed >= self.seasoning_days), index=cal)
        delist = self.delist_dates.get(symbol)
        alive = pd.Series(True, index=cal) if delist is None else pd.Series(cal < delist, index=cal)
        return member & seasoned & alive

    def members(self, day) -> List[str]:
        return sorted(s for s in self.memberships if self.is_member(s, day))

    def all_symbols(self) -> List[str]:
        return sorted(self.memberships)

    @classmethod
    def from_research_snapshot(cls, snapshot_dir, seasoning_days: int = 60) -> "PointInTimeUniverse":
        from .ashare import _delist_dates, load_manifest
        d = Path(snapshot_dir)
        manifest = load_manifest(d)
        cons = pd.read_csv(d / "universe_current_constituents.csv", dtype={"品种代码": str})
        memberships, listing = {}, {}
        for _, r in cons.iterrows():
            sym = str(r["品种代码"]).zfill(6)
            memberships[sym] = [(pd.Timestamp(r["纳入日期"]), None)]
            raw = pd.read_csv(d / f"{sym}_daily_raw.csv", usecols=["date"], nrows=1)
            listing[sym] = pd.Timestamp(raw["date"].iloc[0])
        delist = {s: v for s, v in _delist_dates(d).items() if s in memberships}
        limitations = [
            "成员区间只来自当前成分股的最近一次纳入日期：历史上被调出的股票不在股票池中（幸存者偏差未消除）",
            "曾多次进出指数的股票只保留最近一次纳入，此前的成员区间缺失",
            f"快照下载于 {manifest['fetched_at']}，成分股为该时点的名单",
        ]
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(d / "calendar.csv")["date"])).sort_values()
        return cls(memberships, listing, delist, seasoning_days, limitations, calendar)
