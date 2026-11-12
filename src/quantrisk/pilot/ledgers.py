"""
账本流水（追加写入、哈希链、严格隔离）

四本账各自一个 JSONL 文件，互不相通：
- PAPER_STRICT：严格模拟（只认开盘价满足限价的成交）；
- PAPER_TOUCH：敏感性分析（盘中触价也按限价成交）；
- BENCHMARK：同一只 ETF 买入持有，同样的成交规则与费用；
- MANUAL_LIVE：只接受人工录入、经券商确认的真实成交。

每条记录带 seq、prev_hash、hash（SHA-256），读取时逐条校验；账本类型与来源必须匹配：
模拟记录（source=SIMULATED）写不进 MANUAL_LIVE，人工记录（source=BROKER_CONFIRMED_MANUAL_ENTRY）写不进模拟账本。
写入为“整行追加 + fsync”；崩溃造成的最后一行不完整时，读取会拒绝，repair=True 时把残行移到 .partial 并记录。
账本状态（现金、持仓、批次）始终由流水重放得到，因此中断后可以恢复，资金变化可以复算。
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PAPER_LEDGERS = ("PAPER_STRICT", "PAPER_TOUCH", "BENCHMARK")
MANUAL = "MANUAL_LIVE"
SOURCES = {**{k: "SIMULATED" for k in PAPER_LEDGERS}, MANUAL: "BROKER_CONFIRMED_MANUAL_ENTRY"}


class LedgerIsolationError(RuntimeError):
    """试图把一种来源的记录写进另一种账本。"""


class CorruptJournalError(RuntimeError):
    """流水被改动、残缺或哈希链断裂。"""


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class Journal:
    def __init__(self, exp_dir, ledger: str):
        if ledger not in SOURCES:
            raise ValueError(f"未知账本 {ledger}")
        self.ledger = ledger
        self.path = Path(exp_dir) / "journals" / f"{ledger}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self, repair: bool = False) -> List[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n") + 1
            if not repair:
                raise CorruptJournalError(f"{self.path.name} 最后一行不完整（可能是写入时中断），"
                                          "运行 repair 会把残行移到 .partial 文件后继续")
            with open(self.path.with_suffix(".partial"), "ab") as fh:
                fh.write(raw[cut:] + b"\n")
            with open(self.path, "r+b") as fh:
                fh.truncate(cut)
            raw = raw[:cut]
        out, prev = [], None
        for i, line in enumerate(raw.decode("utf-8").splitlines()):
            rec = json.loads(line)
            if rec.get("ledger") != self.ledger or rec.get("source") != SOURCES[self.ledger]:
                raise LedgerIsolationError(f"{self.path.name} 第 {i + 1} 行的账本/来源为 "
                                           f"{rec.get('ledger')}/{rec.get('source')}，与本账本不符")
            if rec.get("seq") != i or rec.get("prev_hash") != prev or rec.get("hash") != _digest(rec):
                raise CorruptJournalError(f"{self.path.name} 第 {i + 1} 行校验失败（序号、前序哈希或内容被改动）")
            prev = rec["hash"]
            out.append(rec)
        return out

    def append(self, payload: Dict, source: str) -> dict:
        if source != SOURCES[self.ledger]:
            raise LedgerIsolationError(f"来源 {source} 的记录不能写入 {self.ledger} 账本")
        existing = self.records()
        rec = {"seq": len(existing), "prev_hash": existing[-1]["hash"] if existing else None,
               "ledger": self.ledger, "source": source,
               "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
        rec["hash"] = _digest(rec)
        line = (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with open(self.path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        return rec
