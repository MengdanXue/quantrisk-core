"""
ETF Pilot 命令行（本工具不连接券商、不下单）

    python scripts/etf_pilot.py select                                  # 按冻结规则选 ETF（真实 20 日成交额）
    python scripts/etf_pilot.py init --scenario A [--start 2026-11-10] [--rehearsal-start 2025-11-03]
    python scripts/etf_pilot.py fetch --exp <实验编号>                   # 拉取行情（北京时间 15:30 之后才接受当日 K 线）
    python scripts/etf_pilot.py run --exp <实验编号> [--repair]          # 推进模拟账本；生成真实账户的交易建议（已登记本金时）
    真实账户（只接受 actual_broker=true 的实验；每条都要 --evidence-kind APP_SCREENSHOT|SETTLEMENT_STATEMENT|BROKER_MESSAGE --evidence-ref <文件/编号>）：
    python scripts/etf_pilot.py record-capital    --exp <编号> --amount 100 --date 2026-11-20 ...
    python scripts/etf_pilot.py record-order      --exp <编号> --date 2026-11-20 --time 09:16:03 --side BUY --qty 100 --limit 0.578 \\
        --order-id 2026112000012 [--suggestion S-20261119] ...
    python scripts/etf_pilot.py record-fill       --exp <编号> --date 2026-11-20 --time 09:30:05 --order-id 2026112000012 --qty 100 \\
        --price 0.577 --trade-id T0000987 [--fee-hint commission=0.01] ...
    python scripts/etf_pilot.py record-cancel     --exp <编号> --date ... --time ... --order-id ... --qty <撤单数量> [--cancel-id ...] ...
    python scripts/etf_pilot.py record-reject     --exp <编号> --date ... --time ... --order-id ... --reason "<券商原文>" ...
    python scripts/etf_pilot.py record-expired    --exp <编号> --date ... --order-id ... --qty <剩余数量> ...      # 15:00 后、核实 App 后
    python scripts/etf_pilot.py record-settlement --exp <编号> --date ... --order-id ... --qty 100 --amount 57.70 \\
        --fee commission=0.01 [--fee transfer_fee=0] --statement-ref <交割单编号> ...
    python scripts/etf_pilot.py record-dividend   --exp <编号> --date 2027-01-20 --amount 1.2 [--broker-ref ...] ...
    python scripts/etf_pilot.py void              --exp <编号> --hash <记录hash> --reason "..." ...
    python scripts/etf_pilot.py orders            --exp <编号>                 # 委托状态机视图
    python scripts/etf_pilot.py live              --exp <编号>                 # 只重新生成真实账户日报
    python scripts/etf_pilot.py report --exp <编号>                      # 输出到 reports/etf_pilot/<编号>/
    python scripts/etf_pilot.py verify --exp <编号>                      # 重算比对 + 独立复算
    python scripts/etf_pilot.py status
    python scripts/etf_pilot.py daily [--exp <编号> ...]                # 标准每日操作（15:30 后）：基线检查→行情→运行→报告→观察表→复算
    python scripts/etf_pilot.py observe --exp <编号>                     # 只重新生成每日观察表
    python scripts/etf_pilot.py review [--exp <编号> ...]               # 观察满 20 个交易日后生成复盘

数据与实验记录：data/pilot/（不提交、不公开）；报告：reports/etf_pilot/。
"""
import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from quantrisk.pilot.etf_rules import SCENARIOS, WHITELIST  # noqa: E402
from quantrisk.pilot.experiment_config import (init_experiment, load_experiment, make_config,  # noqa: E402
                                               select_etf)
from quantrisk.pilot.experiment_report import build_report  # noqa: E402
from quantrisk.pilot.live_report import write_live_daily  # noqa: E402
from quantrisk.pilot.manual_execution import (ManualEntryError, order_book, record_cancel, record_capital,  # noqa: E402
                                              record_dividend, record_expired, record_fill, record_order,
                                              record_reject, record_settlement, void_record)
from quantrisk.pilot.experiment_config import ConfigChangedError  # noqa: E402
from quantrisk.pilot.ledgers import CorruptJournalError  # noqa: E402
from quantrisk.pilot.market_data import CST, AkshareETFFetcher, DataIntegrityError, ETFMarketStore, StaleDataError  # noqa: E402
from quantrisk.pilot.observation import (ObservationError, check_baseline, log_run, review,  # noqa: E402
                                         write_daily_log)
from quantrisk.pilot.runner import ReplayMismatchError, manual_suggestion, sync  # noqa: E402
from scripts.verify_etf_ledger import verify as independent_verify  # noqa: E402

PILOT = ROOT / "data/pilot"
MARKET, EXPS, SELECTION = PILOT / "market", PILOT / "experiments", PILOT / "selection"
REPORTS = ROOT / "reports/etf_pilot"


def _store(cfg):
    return ETFMarketStore(MARKET, cfg.etf, AkshareETFFetcher(), cfg.close_cutoff_cst)


def cmd_select(a):
    ak = AkshareETFFetcher()._ak()
    now = datetime.now(timezone.utc)
    hist = {}
    for code in WHITELIST:
        hist[code] = ak.fund_etf_hist_sina(symbol=code)
        time.sleep(1.5)
    res = select_etf(hist, now.astimezone(CST).date(), max_price=a.max_price)
    res.update({"fetched_at_utc": now.isoformat(timespec="seconds"), "source": AkshareETFFetcher().source(),
                "whitelist_audit": "ETF_PILOT_AUDIT.md 第 3 节（2026-11-09）"})
    SELECTION.mkdir(parents=True, exist_ok=True)
    path = SELECTION / f"selection_{now.astimezone(CST):%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("selected", "rule", "as_of")}, ensure_ascii=False))
    for c in res["candidates"]:
        print(f"  {c['code']} {c.get('name')} 收盘 {c.get('last_close')} 20日均成交额 {c.get('avg_amount', 0):,.0f} "
              f"{'✓' if c['eligible'] else '✗ ' + c['reason']}")
    print("已保存:", path.relative_to(ROOT))


def cmd_init(a):
    files = sorted(SELECTION.glob("selection_*.json"))
    if not files:
        sys.exit("还没有选择记录，请先运行 select")
    sel = json.loads(files[-1].read_text(encoding="utf-8"))
    sel_ref = {"file": files[-1].name, "selected": sel["selected"], "rule": sel["rule"], "as_of": sel["as_of"],
               "fetched_at_utc": sel["fetched_at_utc"],
               "sha256": hashlib.sha256(files[-1].read_bytes()).hexdigest()}
    rehearsal = a.rehearsal_start is not None
    start = a.rehearsal_start or a.start or str(datetime.now(timezone.utc).astimezone(CST).date())
    name = f"etf{sel['selected'][2:]}_{SCENARIOS[a.scenario].name}" + ("_rehearsal" if rehearsal else "")
    cfg = make_config(name, sel["selected"], start, SCENARIOS[a.scenario], sel_ref, rehearsal=rehearsal)
    d = init_experiment(EXPS, cfg)
    print("实验已初始化并冻结:", d.name, "| 起始决策日", start, "| 历史回放演练" if rehearsal else "| 前向实验")


def _exp(a):
    d = EXPS / a.exp
    return d, load_experiment(d)


def cmd_fetch(a):
    d, cfg = _exp(a)
    print(json.dumps(_store(cfg).fetch(), ensure_ascii=False))


def cmd_run(a):
    d, cfg = _exp(a)
    store = _store(cfg)
    print(json.dumps(sync(d, cfg, store, repair=a.repair), ensure_ascii=False))
    s = manual_suggestion(d, cfg, store)
    print("真实账户建议:", json.dumps({k: s.get(k) for k in ("status", "message", "suggestion_id", "intent", "proposal")},
                                   ensure_ascii=False, default=str))


def _evidence(a):
    return {"kind": a.evidence_kind, "ref": a.evidence_ref}


def _kv(items):
    out = {}
    for kv in items or []:
        k, v = kv.split("=")
        out[k] = float(v)
    return out


def _print(rec):
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str))


def cmd_record_capital(a):
    d, cfg = _exp(a)
    _print(record_capital(d, cfg, a.amount, a.date, _evidence(a), a.note))


def cmd_record_order(a):
    d, cfg = _exp(a)
    _print(record_order(d, cfg, _store(cfg), a.date, a.time, a.side, a.qty, a.limit, a.order_id, _evidence(a),
                        suggestion_id=a.suggestion, code=a.code, note=a.note))


def cmd_record_fill(a):
    d, cfg = _exp(a)
    _print(record_fill(d, cfg, _store(cfg), a.date, a.time, a.order_id, a.qty, a.price, a.trade_id, _evidence(a),
                       fee_hint=_kv(a.fee_hint) or None, note=a.note))


def cmd_record_cancel(a):
    d, cfg = _exp(a)
    _print(record_cancel(d, cfg, _store(cfg), a.date, a.time, a.order_id, a.qty, _evidence(a), a.cancel_id, a.note))


def cmd_record_reject(a):
    d, cfg = _exp(a)
    _print(record_reject(d, cfg, _store(cfg), a.date, a.time, a.order_id, a.reason, _evidence(a), a.note))


def cmd_record_expired(a):
    d, cfg = _exp(a)
    _print(record_expired(d, cfg, _store(cfg), a.date, a.order_id, a.qty, _evidence(a), a.note))


def cmd_record_settlement(a):
    d, cfg = _exp(a)
    _print(record_settlement(d, cfg, _store(cfg), a.date, a.order_id, a.qty, a.amount, _kv(a.fee), a.statement_ref,
                             _evidence(a), a.note))


def cmd_record_dividend(a):
    d, cfg = _exp(a)
    _print(record_dividend(d, cfg, a.date, a.amount, _evidence(a), a.broker_ref, a.note))


def cmd_void(a):
    d, cfg = _exp(a)
    _print(void_record(d, cfg, a.hash, a.reason, _evidence(a)))


def cmd_orders(a):
    d, cfg = _exp(a)
    from quantrisk.pilot.ledgers import Journal
    book = order_book(Journal(d, "MANUAL_LIVE").records())
    if not book:
        print("没有真实委托记录")
    for oid, o in book.items():
        od = o["order"]
        print(f"{oid} {od['trade_date']} {od['side']} Q={o['Q']} F={o['F']} C={o['C']} R={o['R']} → {o['status']}"
              f"{' 交割待录' if o['fees_pending'] else ''} flags={od.get('flags') or []}")


def cmd_live(a):
    d, cfg = _exp(a)
    print("真实账户日报:", write_live_daily(d, cfg, _store(cfg), SELECTION, ROOT, REPORTS / d.name / "observation").relative_to(ROOT))


def cmd_report(a):
    d, cfg = _exp(a)
    out = build_report(d, cfg, _store(cfg), REPORTS / d.name)
    print("报告:", out.relative_to(ROOT))


def cmd_verify(a):
    d, cfg = _exp(a)
    store = _store(cfg)
    rep = sync(d, cfg, store, append=False)
    ind = independent_verify(d, MARKET / cfg.etf)
    print(json.dumps({"replay": rep, "independent": ind}, ensure_ascii=False, indent=1))
    sys.exit(1 if ind["problems"] else 0)


def cmd_status(a):
    for d in sorted(EXPS.glob("*")):
        try:
            cfg = load_experiment(d)
            ok = "配置未改动"
        except Exception as exc:  # noqa: BLE001
            cfg, ok = None, f"配置异常: {exc}"
        n = sum(1 for _ in (d / "journals" / "PAPER_STRICT.jsonl").open()) if (d / "journals" / "PAPER_STRICT.jsonl").exists() else 0
        print(d.name, "|", ok, "| 已处理交易日", n, "| 演练" if cfg and cfg.rehearsal else "")


def _forward_ids(a):
    if a.exp:
        return a.exp
    return [d.name for d in sorted(EXPS.glob("*")) if (d / "config.json").exists()
            and not json.loads((d / "config.json").read_text(encoding="utf-8")).get("rehearsal")]


def cmd_daily(a):
    """标准每日操作。任何一步异常都写入运行日志并停止该实验，不覆盖历史、不跳过。"""
    fetched, bad = {}, 0
    for eid in _forward_ids(a):
        d = EXPS / eid
        entry = {"command": "daily", "experiment": eid}
        try:
            cfg = load_experiment(d)
            base = check_baseline(d, cfg, SELECTION, ROOT)
            entry["baseline_soft_changes"] = base["soft"]
            if base["hard"]:
                raise ObservationError(f"基线被改动：{base['hard']}")
            store = _store(cfg)
            if cfg.etf not in fetched:
                fetched[cfg.etf] = store.fetch()
            entry["fetch"] = {k: fetched[cfg.etf][k] for k in ("fetch_id", "now_cst", "new_bars", "revisions",
                                                                "dropped_incomplete_bars")}
            entry["sync"] = sync(d, cfg, store)
            entry["suggestion"] = manual_suggestion(d, cfg, store).get("status")
            build_report(d, cfg, store, REPORTS / eid)
            write_daily_log(d, cfg, REPORTS / eid / "observation")
            write_live_daily(d, cfg, store, SELECTION, ROOT, REPORTS / eid / "observation")
            ver = independent_verify(d, MARKET / cfg.etf)
            entry["verify_problems"] = ver["problems"]
            entry["status"] = "OK" if not ver["problems"] else "VERIFY_FAILED"
        except StaleDataError as exc:
            entry.update(status="STALE", error=str(exc))
        except ReplayMismatchError as exc:
            entry.update(status="MISMATCH", error=str(exc))
        except CorruptJournalError as exc:
            entry.update(status="CORRUPT_JOURNAL", error=str(exc))
        except (ConfigChangedError, ObservationError) as exc:
            entry.update(status="CONFIG_OR_BASELINE_CHANGED", error=str(exc))
        except DataIntegrityError as exc:
            entry.update(status="DATA_INTEGRITY", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            entry.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        log_run(d, entry)
        bad += entry["status"] != "OK"
        s = entry.get("sync", {})
        print(f"[{entry['status']}] {eid} 处理至 {s.get('through')} 新增 {s.get('appended', {}).get('PAPER_STRICT')} 天"
              + (f" | {entry['error']}" if "error" in entry else ""))
    sys.exit(1 if bad else 0)


def cmd_observe(a):
    d, cfg = _exp(a)
    print("观察表:", write_daily_log(d, cfg, REPORTS / d.name / "observation").relative_to(ROOT))


def cmd_review(a):
    exps = []
    for eid in _forward_ids(a):
        d = EXPS / eid
        cfg = load_experiment(d)
        exps.append((d, cfg, independent_verify(d, MARKET / cfg.etf), check_baseline(d, cfg, SELECTION, ROOT)))
    try:
        out = review(exps, MARKET / exps[0][1].etf, REPORTS / f"forward_review_{datetime.now(timezone.utc).astimezone(CST):%Y%m%d}.md")
    except ObservationError as exc:
        sys.exit(f"不生成复盘：{exc}")
    print("复盘:", out.relative_to(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--max-price", type=float, default=0.90); s.set_defaults(f=cmd_select)
    s = sub.add_parser("init"); s.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    s.add_argument("--start"); s.add_argument("--rehearsal-start"); s.set_defaults(f=cmd_init)
    for name, f in (("fetch", cmd_fetch), ("report", cmd_report), ("verify", cmd_verify)):
        s = sub.add_parser(name); s.add_argument("--exp", required=True); s.set_defaults(f=f)
    s = sub.add_parser("run"); s.add_argument("--exp", required=True); s.add_argument("--repair", action="store_true")
    s.set_defaults(f=cmd_run)
    def real(name, f, *args):
        sp = sub.add_parser(name)
        sp.add_argument("--exp", required=True)
        sp.add_argument("--evidence-kind", required=True, choices=("APP_SCREENSHOT", "SETTLEMENT_STATEMENT", "BROKER_MESSAGE"))
        sp.add_argument("--evidence-ref", required=True)
        sp.add_argument("--note", default="")
        for spec in args:
            sp.add_argument(*spec[0], **spec[1])
        sp.set_defaults(f=f)
    A = lambda *names, **kw: (names, kw)
    real("record-capital", cmd_record_capital, A("--amount", type=float, required=True), A("--date", required=True))
    real("record-order", cmd_record_order, A("--date", required=True), A("--time", required=True), A("--side", required=True),
         A("--qty", type=int, required=True), A("--limit", type=float, required=True), A("--order-id", required=True),
         A("--suggestion"), A("--code"))
    real("record-fill", cmd_record_fill, A("--date", required=True), A("--time", required=True), A("--order-id", required=True),
         A("--qty", type=int, required=True), A("--price", type=float, required=True), A("--trade-id", required=True),
         A("--fee-hint", action="append"))
    real("record-cancel", cmd_record_cancel, A("--date", required=True), A("--time", required=True), A("--order-id", required=True),
         A("--qty", type=int, required=True), A("--cancel-id"))
    real("record-reject", cmd_record_reject, A("--date", required=True), A("--time", required=True), A("--order-id", required=True),
         A("--reason", required=True))
    real("record-expired", cmd_record_expired, A("--date", required=True), A("--order-id", required=True), A("--qty", type=int, required=True))
    real("record-settlement", cmd_record_settlement, A("--date", required=True), A("--order-id", required=True),
         A("--qty", type=int, required=True), A("--amount", type=float, required=True), A("--fee", action="append", required=True),
         A("--statement-ref", required=True))
    real("record-dividend", cmd_record_dividend, A("--date", required=True), A("--amount", type=float, required=True), A("--broker-ref"))
    real("void", cmd_void, A("--hash", required=True), A("--reason", required=True))
    s = sub.add_parser("orders"); s.add_argument("--exp", required=True); s.set_defaults(f=cmd_orders)
    s = sub.add_parser("live"); s.add_argument("--exp", required=True); s.set_defaults(f=cmd_live)
    s = sub.add_parser("status"); s.set_defaults(f=cmd_status)
    s = sub.add_parser("daily"); s.add_argument("--exp", action="append"); s.set_defaults(f=cmd_daily)
    s = sub.add_parser("observe"); s.add_argument("--exp", required=True); s.set_defaults(f=cmd_observe)
    s = sub.add_parser("review"); s.add_argument("--exp", action="append"); s.set_defaults(f=cmd_review)
    a = p.parse_args()
    try:
        a.f(a)
    except ManualEntryError as exc:
        sys.exit(f"拒绝录入：{exc}")


if __name__ == "__main__":
    main()
