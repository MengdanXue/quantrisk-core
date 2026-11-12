"""
ETF Pilot 账本独立复算（不导入 quantrisk，只读实验目录的 config.json、journals/*.jsonl 与行情存储的 bars.csv）。

逐项复核：
1. 每本账的哈希链、序号、账本与来源是否匹配（模拟账只能是 SIMULATED，MANUAL_LIVE 只能是人工录入）；
2. 从本金开始逐笔重放：现金分红 = 持仓 × 每份分红；买入现金减少 = 数量 × 成交价 + 费用，卖出相反；
   每日收盘的现金、持仓、权益 = 现金 + 持仓 × 收盘价，必须与流水记录一致；现金任何时候不得为负；
3. 每笔模拟成交：价格在 0.001 元报价单位上；买入价 ≤ 限价，卖出价 ≥ 限价；价格在当日最低价与最高价之间；
   严格成交还要求开盘价满足限价；严格账本中不得出现盘中触价成交；
4. 费用按配置的佣金情景独立重算（各项四舍五入到分）；申报数量（买入 100 份整数倍）；T+1（卖出日晚于买入日）；
5. 前向实验的成交时间：决策日及以前的 K 线与分红，首次获取时间都不得晚于成交日 09:15（北京时间）；
   记为 LATE_DATA_NOT_EXECUTABLE 的订单，行情也必须确实晚于这个时刻（不能把本可成交的单作废）。

    python scripts/verify_etf_ledger.py <实验目录> <行情目录（含 bars.csv）>
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SOURCES = {"PAPER_STRICT": "SIMULATED", "PAPER_TOUCH": "SIMULATED", "BENCHMARK": "SIMULATED",
           "MANUAL_LIVE": "BROKER_CONFIRMED_MANUAL_ENTRY"}
TOL = 1e-6
CST = timezone(timedelta(hours=8))
SUBMIT_DEADLINE = (9, 15)


def utc(ts):
    t = datetime.fromisoformat(str(ts))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def first_seen_by_date(market_dir):
    """[(日期, 首次获取时间)]：K 线按交易日，分红按除息日。"""
    rows = [(r["date"], utc(r["first_seen_utc"])) for r in csv.DictReader(open(market_dir / "bars.csv", encoding="utf-8"))]
    div = market_dir / "dividends.csv"
    if div.exists():
        rows += [(r["ex_date"], utc(r["first_seen_utc"])) for r in csv.DictReader(open(div, encoding="utf-8"))]
    return rows


def deadline_of(day):
    y, m, d = (int(x) for x in day.split("-"))
    return datetime(y, m, d, *SUBMIT_DEADLINE, tzinfo=CST)


def cents(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def on_tick(p):
    d = Decimal(str(p))
    return d == (d / Decimal("0.001")).quantize(Decimal(1), rounding=ROUND_HALF_UP) * Decimal("0.001")


def fees_of(fee, price, qty):
    amount = Decimal(str(price)) * qty
    out = {"commission": cents(max(amount * Decimal(str(fee["commission_rate"])), Decimal(str(fee["min_commission"]))))}
    if fee.get("transfer_fee_rate"):
        out["transfer_fee"] = cents(amount * Decimal(str(fee["transfer_fee_rate"])))
    if fee.get("other_fee_rate"):
        out["other_fee"] = cents(amount * Decimal(str(fee["other_fee_rate"])))
    return out


def read_journal(path, ledger, problems):
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        problems.append(f"{ledger}: 最后一行不完整")
    recs, prev = [], None
    for i, line in enumerate(raw.splitlines()):
        r = json.loads(line)
        body = {k: v for k, v in r.items() if k != "hash"}
        h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if r["seq"] != i or r["prev_hash"] != prev or r["hash"] != h:
            problems.append(f"{ledger} 第 {i + 1} 行：哈希链或序号不符")
        if r["ledger"] != ledger or r["source"] != SOURCES[ledger]:
            problems.append(f"{ledger} 第 {i + 1} 行：账本/来源为 {r['ledger']}/{r['source']}")
        prev = r["hash"]
        recs.append(r)
    return recs


MANUAL_CHILD = ("BROKER_FILL", "BROKER_CANCEL", "BROKER_REJECT", "BROKER_EXPIRED", "BROKER_SETTLEMENT")
FEE_KEYS = ("commission", "transfer_fee", "exchange_fee", "other")


def apportion(fees, qtys):
    """与 manual_execution.apportion_fees 独立实现：按数量比例、按分四舍五入、尾差归最后一笔。"""
    out = [{} for _ in qtys]
    tot = sum(qtys)
    for k, v in fees.items():
        v = cents(v)
        parts = [cents(v * q / tot) for q in qtys[:-1]]
        parts.append(round(v - sum(parts), 2))
        for i, x in enumerate(parts):
            out[i][k] = x
    return out


def nat_key(r):
    t = r["type"]
    return {"CAPITAL": ("CAPITAL",), "BROKER_ORDER": (t, r.get("broker_order_id")), "BROKER_FILL": (t, r.get("broker_trade_id")),
            "BROKER_CANCEL": (t, r.get("broker_cancel_id")), "BROKER_REJECT": (t, r.get("broker_order_id")),
            "BROKER_EXPIRED": (t, r.get("broker_order_id")), "BROKER_SETTLEMENT": (t, r.get("broker_order_id"), r.get("trade_date")),
            "BROKER_DIVIDEND": (t, r.get("broker_ref")), "VOID": (t, r.get("target_hash"))}.get(t, (t, r.get("hash")))


def verify_manual(manual, cfg, bars, exp_dir, problems, checked):
    """V2–V11：真实账本的键唯一、引用、状态机、价格、费用、现金持仓、标记、凭据、无虚构。"""
    P = lambda msg: problems.append("MANUAL_LIVE: " + msg)
    real = bool(cfg.get("fee", {}).get("actual_broker"))
    checked["manual_records"] = len(manual)
    if manual and not real:                                                              # V11
        P(f"实验佣金情景 actual_broker=false，却有 {len(manual)} 条真实记录（不允许在模拟实验下记真实账）")
    by_hash = {r["hash"]: r for r in manual}
    voids = [r for r in manual if r["type"] == "VOID"]
    voided = {}
    for v in voids:
        if v["target_hash"] not in by_hash:
            P(f"VOID 指向不存在的记录 {v['target_hash'][:12]}")
        elif by_hash[v["target_hash"]]["type"] in ("CAPITAL", "VOID"):
            P(f"VOID 作废了 {by_hash[v['target_hash']]['type']}")
        elif by_hash[v["target_hash"]]["seq"] >= v["seq"]:
            P("VOID 早于被作废记录")
        voided[v["target_hash"]] = voided.get(v["target_hash"], 0) + 1
    for h, n in voided.items():
        if n > 1:
            P(f"记录 {h[:12]} 被作废 {n} 次")
    active = [r for r in manual if r["type"] != "VOID" and r["hash"] not in voided]
    for r in manual:                                                                     # V11 凭据
        ev = r.get("evidence") or {}
        if not ev.get("kind") or not ev.get("ref"):
            P(f"seq {r['seq']} {r['type']} 缺少凭据 evidence")
    keys = {}
    for r in active:                                                                     # V2 键唯一
        k = nat_key(r)
        if k in keys:
            P(f"自然键重复：{k}（seq {keys[k]} 与 {r['seq']}）")
        keys[k] = r["seq"]
    reentries = {}
    for r in manual:
        if r.get("reentry_of"):
            reentries[nat_key(r)] = reentries.get(nat_key(r), 0) + 1
            if r["reentry_of"] not in voided:
                P(f"seq {r['seq']} 标为重录，但其原记录未被作废")
    for k, n in reentries.items():
        if n > 1:
            P(f"自然键 {k} 重录超过一次")
    caps = [r for r in active if r["type"] == "CAPITAL"]
    if len(caps) > 1:
        P("本金登记超过一次")
    if not caps:
        if active:
            P("没有本金登记却有其他真实记录")
        return
    cap = caps[0]
    if cap["amount"] > cfg["capital"] + TOL or cap["amount"] <= 0:
        P("登记本金超过配置上限或非正")
    orders = {r["broker_order_id"]: r for r in active if r["type"] == "BROKER_ORDER"}
    kids = {oid: {"fills": [], "cancels": [], "reject": [], "expired": [], "settle": []} for oid in orders}
    for r in active:
        if r["type"] in MANUAL_CHILD:
            oid = r.get("broker_order_id")
            if oid not in orders:                                                        # V4 引用
                P(f"seq {r['seq']} {r['type']} 引用不存在的委托 {oid}")
                continue
            if r["seq"] < orders[oid]["seq"]:
                P(f"seq {r['seq']} {r['type']} 早于其委托记录")
            if r["trade_date"] != orders[oid]["trade_date"]:
                P(f"{r['type']} seq {r['seq']} 日期 {r['trade_date']} ≠ 委托日期 {orders[oid]['trade_date']}")
            slot = {"BROKER_FILL": "fills", "BROKER_CANCEL": "cancels", "BROKER_REJECT": "reject",
                    "BROKER_EXPIRED": "expired", "BROKER_SETTLEMENT": "settle"}[r["type"]]
            kids[oid][slot].append(r)
    events = []                                                                          # 入账事件
    sug_path = exp_dir / "suggestions.jsonl"
    sugs = {}
    if sug_path.exists():
        for line in sug_path.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            sugs[x["suggestion_id"]] = x
    dates = sorted(bars)
    for oid, od in orders.items():                                                       # V3 状态机 / V5 价格 / V6 费用
        k = kids[oid]
        Q = od["qty"]
        F = sum(x["qty"] for x in k["fills"])
        C = sum(x["cancelled_qty"] for x in k["cancels"])
        R = Q - F - C
        if F < 0 or C < 0 or R < 0:
            P(f"委托 {oid}：F={F} C={C} 超出 Q={Q}")
        if len(k["reject"]) > 1 or len(k["expired"]) > 1 or len(k["settle"]) > 1:
            P(f"委托 {oid}：REJECT/EXPIRED/SETTLEMENT 超过一条")
        if k["reject"] and (F or C):
            P(f"委托 {oid}：拒单却有成交或撤单")
        if k["expired"] and k["expired"][0]["expired_qty"] != R:                        # R 不含作废数量
            P(f"委托 {oid}：作废数量 {k['expired'][0]['expired_qty']} ≠ 剩余 {R}")
        if Q % 100 and od["side"] == "BUY":
            P(f"委托 {oid}：买入数量 {Q} 不是 100 份整数倍")
        if not on_tick(od["limit_price"]):
            P(f"委托 {oid}：限价不在报价单位上")
        d = od["trade_date"]
        prior = [x for x in dates if x < d]
        if prior:
            prev = bars[prior[-1]]["close"]
            lo, up = round(prev * 0.9 + 1e-12, 3), round(prev * 1.1 + 1e-12, 3)
            if not (lo - TOL <= od["limit_price"] <= up + TOL):
                P(f"委托 {oid}：限价 {od['limit_price']} 超出独立计算的涨跌停带 [{lo}, {up}]")
        # 标记复算（V8，建议相关）
        exp_flags = set()
        sid = od.get("suggestion_id")
        sg = sugs.get(sid) if sid else None
        if sg is None:
            exp_flags.add("NO_SUGGESTION")
        else:
            if prior and sg.get("decision_day") != prior[-1]:
                exp_flags.add("SUGGESTION_STALE")
            prop = sg.get("proposal") or {}
            if prop.get("action") == "REJECT":
                exp_flags.add("SUGGESTION_REJECTED")
            o = prop.get("order") or {}
            if o:
                if o.get("side") != od["side"]:
                    exp_flags.add("SIDE_MISMATCH")
                if int(o.get("qty", -1)) != Q:
                    exp_flags.add("QTY_MISMATCH")
                if abs(float(o.get("limit_price", -1)) - od["limit_price"]) > 1e-9:
                    exp_flags.add("LIMIT_MISMATCH")
            elif sg.get("intent") is None:
                exp_flags.add("NO_SUGGESTION")
        got = set(od.get("flags") or []) - {"LATE_ENTRY", "PRICE_UNVERIFIED"}
        if got != exp_flags:
            P(f"委托 {oid}：标记 {sorted(got)} ≠ 独立复算 {sorted(exp_flags)}")
        fills = sorted(k["fills"], key=lambda x: (x["fill_time"], x["seq"]))
        for x in fills:
            if not on_tick(x["price"]):
                P(f"成交 {x['broker_trade_id']}：价格不在报价单位上")
            if (od["side"] == "BUY" and x["price"] > od["limit_price"] + TOL) or (od["side"] == "SELL" and x["price"] < od["limit_price"] - TOL):
                P(f"成交 {x['broker_trade_id']}：价格 {x['price']} 违反限价 {od['limit_price']}")
            bar = bars.get(x["trade_date"])
            if bar is not None and not (bar["low"] - TOL <= x["price"] <= bar["high"] + TOL):
                P(f"成交 {x['broker_trade_id']}：价格 {x['price']} 不在当日区间")
            if x["fill_time"] < od["order_time"]:
                P(f"成交 {x['broker_trade_id']}：时间早于委托")
            if x.get("fees"):
                P(f"成交 {x['broker_trade_id']}：成交记录携带费用字段（费用只能来自交割单）")
        fee_parts = [{} for _ in fills]
        if k["settle"]:
            st = k["settle"][0]
            if st["settled_qty"] != F:
                P(f"交割 {oid}：数量 {st['settled_qty']} ≠ 成交 {F}")
            amt = sum(x["qty"] * x["price"] for x in fills)
            if abs(st["settled_amount"] - amt) > 0.005:
                P(f"交割 {oid}：金额 {st['settled_amount']} ≠ {amt:.4f}")
            if float(st["fees"].get("stamp_duty", 0)) > 0:
                P(f"交割 {oid}：出现印花税")
            if any(v < 0 for v in st["fees"].values()):
                P(f"交割 {oid}：费用为负")
            if abs(round(sum(st["fees"].values()), 2) - st["fee_total"]) > TOL:
                P(f"交割 {oid}：fee_total 与各项之和不符")
            fee_parts = apportion(st["fees"], [x["qty"] for x in fills]) if fills else []
            # OVER_BUDGET_ACTUAL 复算
            tot = round(sum(st["fees"].values()), 2)
            slip = 0.0
            if prior:
                signed = sum(x["qty"] * (x["price"] - bars[prior[-1]]["close"]) for x in fills)
                slip = max(0.0, signed if od["side"] == "BUY" else -signed)
            ratio = (tot + slip) / amt if amt else 0
            exp = {"OVER_BUDGET_ACTUAL"} if amt and ratio > cfg["friction_budget"] + 1e-12 else set()
            got = set(st.get("flags") or []) - {"LATE_ENTRY", "FEE_HINT_MISMATCH"}
            if got != exp:
                P(f"交割 {oid}：标记 {sorted(got)} ≠ 独立复算 {sorted(exp)}")
        for x, fp in zip(fills, fee_parts):
            events.append((x["trade_date"], x["fill_time"], x["seq"], "FILL", od["side"], x["qty"], x["price"], round(sum(fp.values()), 2)))
    for r in active:
        if r["type"] == "BROKER_DIVIDEND":
            if r["amount"] <= 0:
                P("分红金额非正")
            events.append((r["trade_date"], "23:59:59", r["seq"], "DIV", None, 0, 0.0, r["amount"]))
    events.sort()
    cash, qty, buy_days = cap["amount"], 0, []
    for d, t, seq, kind, side, q, px, amt in events:                                    # V7 现金与持仓 / T+1 / 整手
        if d < cap["trade_date"]:
            P(f"{d} 的入账事件早于本金登记日 {cap['trade_date']}")
        if kind == "DIV":
            cash += amt
            continue
        if side == "BUY":
            cash -= q * px + amt
            qty += q
            buy_days.append(d)
        else:
            if buy_days and max(buy_days) >= d:
                P(f"{d}: 卖出违反 T+1（最近买入日 {max(buy_days)}）")
            if q > qty:
                P(f"{d}: 卖出 {q} 超过持仓 {qty}")
            cash += q * px - amt
            qty -= q
            if qty == 0:
                buy_days = []
        if cash < -TOL:
            P(f"{d}: 现金为负 {cash:.4f}")
    checked["manual_cash"], checked["manual_holding"] = round(cash, 6), qty
    checked["manual_orders"] = len(orders)
    checked["manual_fee_total"] = round(sum(sum(k["settle"][0]["fees"].values()) for k in kids.values() if k["settle"]), 2)


def verify_live_daily(exp_dir, checked, problems):
    """日报（live_daily.csv）：末行现金/持仓与独立复算一致；任何一天双轨信号 DIFF 都算问题。没有本金时也检查。"""
    live = exp_dir.parent.parent.parent.parent / "reports" / "etf_pilot" / exp_dir.name / "observation" / "live_daily.csv"
    if not live.exists():
        return
    rows = list(csv.DictReader(open(live, encoding="utf-8")))
    checked["live_daily_rows"] = len(rows)
    tail = [r for r in rows if r.get("live_cash") not in (None, "", "nan")]
    if tail and "manual_cash" in checked:
        last = tail[-1]
        if abs(float(last["live_cash"]) - checked["manual_cash"]) > TOL or int(float(last["live_holding"])) != checked["manual_holding"]:
            problems.append(f"MANUAL_LIVE: 日报末行现金/持仓 {last['live_cash']}/{last['live_holding']} ≠ 复算 "
                            f"{checked['manual_cash']:.4f}/{checked['manual_holding']}")
    elif tail:
        problems.append("MANUAL_LIVE: 日报有真实现金列，但流水中没有有效的本金登记")
    diff = [r["date"] for r in rows if r.get("signal_consistency") == "DIFF"]
    if diff:
        problems.append(f"MANUAL_LIVE: 日报 {len(diff)} 天 PAPER 原始意图与建议不一致（signal_consistency=DIFF）：{diff}")


def verify_hashes(exp_dir, root, problems, checked):
    """V10：配置 / 规则 / 选择记录 / pilot 代码的 SHA-256 与基线比对（独立计算）。"""
    base_p = exp_dir / "observation" / "baseline.json"
    if not base_p.exists():
        return
    base = json.loads(base_p.read_text(encoding="utf-8"))
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cur = {"config_sha256": sha(exp_dir / "config.json"),
           "rules_sha256": sha(root / "src/quantrisk/pilot" / cfg["rules_file"]),
           "selection_sha256": sha(exp_dir.parent.parent / "selection" / cfg["selection"]["file"])}
    h = hashlib.sha256()
    for p in sorted((root / "src/quantrisk/pilot").glob("*.py")) + [root / "scripts/etf_pilot.py"]:
        if p.exists():
            h.update(p.name.encode() + p.read_bytes())
    cur["pilot_code_sha256"] = h.hexdigest()
    hard = [k for k in ("config_sha256", "rules_sha256", "selection_sha256") if cur[k] != base.get(k)]
    if hard:
        problems.append(f"基线硬变化：{hard}")
    checked["hashes"] = {**cur, "pilot_code_changed_since_baseline": cur["pilot_code_sha256"] != base.get("pilot_code_sha256")}


def verify(exp_dir, market_dir):
    exp_dir, market_dir = Path(exp_dir), Path(market_dir)
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    bars = {r["date"]: {k: float(r[k]) for k in ("open", "high", "low", "close")}
            for r in csv.DictReader(open(market_dir / "bars.csv", encoding="utf-8"))}
    seen = first_seen_by_date(market_dir)
    forward = not cfg.get("rehearsal")
    problems, checked = [], {"fills": 0, "days": 0, "dividends": 0, "timing": 0}
    for ledger in ("PAPER_STRICT", "PAPER_TOUCH", "BENCHMARK"):
        recs = read_journal(exp_dir / "journals" / f"{ledger}.jsonl", ledger, problems)
        cash, qty, buy_days = float(cfg["capital"]), 0, []
        decision_day = None
        for r in recs:
            day = r["day"]
            for e in r["events"]:
                t = e["type"]
                if forward and t == "ORDER_RESULT":
                    checked["timing"] += 1
                    if decision_day is None:
                        problems.append(f"{ledger} {day}: 有撮合结果但找不到前一交易日的决策")
                    else:
                        avail = max(ts for dt, ts in seen if dt <= decision_day)
                        late = avail > deadline_of(day)
                        if late and e["status"] != "LATE_DATA_NOT_EXECUTABLE":
                            problems.append(f"{ledger} {day}: 决策日 {decision_day} 的行情获取于 {avail.astimezone(CST).isoformat()}，"
                                            f"晚于成交日 09:15，却记为 {e['status']}")
                        if not late and e["status"] == "LATE_DATA_NOT_EXECUTABLE":
                            problems.append(f"{ledger} {day}: 行情在成交日 09:15 前已获取，不应记为不可成交")
                if t == "DIVIDEND":
                    checked["dividends"] += 1
                    amt = qty * e["per_share"]
                    if abs(amt - e["amount"]) > TOL:
                        problems.append(f"{ledger} {day}: 分红 {e['amount']} ≠ {qty} × {e['per_share']}")
                    cash += amt
                elif t == "ORDER_RESULT" and e["status"] in ("STRICT_FILL", "TOUCH_FILL"):
                    checked["fills"] += 1
                    p, q, side, lim = e["price"], e["qty"], e["side"], e["limit_price"]
                    bar = bars.get(day)
                    if ledger != "PAPER_TOUCH" and e["status"] == "TOUCH_FILL":
                        problems.append(f"{ledger} {day}: 严格账本中出现盘中触价成交")
                    if not on_tick(p):
                        problems.append(f"{ledger} {day}: 成交价 {p} 不在报价单位上")
                    if (side == "BUY" and p > lim + TOL) or (side == "SELL" and p < lim - TOL):
                        problems.append(f"{ledger} {day}: 成交价 {p} 违反限价 {lim}")
                    if bar is None or not (bar["low"] - TOL <= p <= bar["high"] + TOL):
                        problems.append(f"{ledger} {day}: 成交价 {p} 不在当日价格区间内")
                    if e["status"] == "STRICT_FILL" and bar is not None and \
                            ((side == "BUY" and bar["open"] > lim + TOL) or (side == "SELL" and bar["open"] < lim - TOL)):
                        problems.append(f"{ledger} {day}: 严格成交但开盘价 {bar['open']} 不满足限价 {lim}")
                    fees = fees_of(cfg["fee"], p, q)
                    if fees != {k: float(v) for k, v in e["fees"].items()}:
                        problems.append(f"{ledger} {day}: 费用 {e['fees']} ≠ 独立重算 {fees}")
                    total = round(sum(fees.values()), 2)
                    if side == "BUY":
                        if q % 100:
                            problems.append(f"{ledger} {day}: 买入数量 {q} 不是 100 份整数倍")
                        cash -= q * p + total
                        qty += q
                        buy_days.append(day)
                    else:
                        if buy_days and max(buy_days) >= day:
                            problems.append(f"{ledger} {day}: 违反 T+1（最近买入日 {max(buy_days)}）")
                        if q > qty:
                            problems.append(f"{ledger} {day}: 卖出 {q} 超过持仓 {qty}")
                        cash += q * p - total
                        qty -= q
                        if qty == 0:
                            buy_days = []
                    if abs(cash - e["cash_after"]) > TOL or qty != e["holding_after"]:
                        problems.append(f"{ledger} {day}: 成交后现金/持仓 {e['cash_after']}/{e['holding_after']} ≠ 复算 {cash}/{qty}")
                elif t == "DECISION":
                    decision_day = e["data"]["last_bar"]
                elif t == "DAY_CLOSE":
                    checked["days"] += 1
                    if cash < -TOL:
                        problems.append(f"{ledger} {day}: 现金为负 {cash}")
                    if abs(cash - e["cash"]) > TOL or qty != e["holding"]:
                        problems.append(f"{ledger} {day}: 收盘现金/持仓 {e['cash']}/{e['holding']} ≠ 复算 {cash}/{qty}")
                    if abs(e["equity"] - (cash + qty * (e["close"] or 0))) > TOL:
                        problems.append(f"{ledger} {day}: 权益 {e['equity']} ≠ 现金 + 持仓 × 收盘价")
    manual = read_journal(exp_dir / "journals" / "MANUAL_LIVE.jsonl", "MANUAL_LIVE", problems)
    verify_manual(manual, cfg, bars, exp_dir, problems, checked)
    verify_live_daily(exp_dir, checked, problems)
    try:
        verify_hashes(exp_dir, Path(__file__).resolve().parents[1], problems, checked)
    except FileNotFoundError as exc:
        problems.append(f"哈希复算缺少文件：{exc}")
    return {"problems": problems, "checked": checked}


if __name__ == "__main__":
    out = verify(sys.argv[1], sys.argv[2])
    print(json.dumps(out, ensure_ascii=False, indent=2))
    sys.exit(1 if out["problems"] else 0)
