# ETF Pilot：人工实盘账本（MANUAL_LIVE）设计

- 日期：2026-11-10 设计，2026-11-11 实施并测试（`src/quantrisk/pilot/manual_execution.py`、`live_report.py`、`scripts/verify_etf_ledger.py`，`tests/test_etf_pilot.py` T1–T25）
- 来源：双轨实验实施计划的第 7–11 节（设计部分）原文；计划中与个人账户核实、开发排期有关的章节不公开。
- 标注约定：**[代码]** 附 `文件:行号`（以设计时代码为准，实施后行号可能变化）；**[规则]** 为交易规则，需以交易所/券商正式文件核实；**[推断]** 为未直接验证的判断。

---

## 7. 真实轨道的数据结构

所有记录都写入 `data/pilot/experiments/<编号>/journals/MANUAL_LIVE.jsonl`，复用现有封装（`seq / prev_hash / hash / ledger / source / written_at_utc`，[代码] `ledgers.py:79-82`），`source` 固定为 `BROKER_CONFIRMED_MANUAL_ENTRY`。**系统永远不会主动往这本账里写任何记录**；每一条都来自人工录入，且必须能对应到券商 App 或交割单上的一条凭据。

公共字段（每条都有）：

| 字段 | 含义 |
|---|---|
| `type` | 记录类型，见下 |
| `trade_date` | 交易日（YYYY-MM-DD），必须在交易日历内 |
| `evidence` | 凭据：`{"kind": "APP_SCREENSHOT" / "SETTLEMENT_STATEMENT" / "BROKER_MESSAGE", "ref": "<文件名或编号>", "captured_at": "..."}`，必填 |
| `note` | 自由文本 |
| `entered_at_utc` | 录入时刻 |
| `flags` | 录入时计算并固化的异常标记列表（第 10 节），可为空 |

### 7.1 `CAPITAL`（已有，不变）

`amount ≤ cfg.capital`，只能一条。[代码] `manual_execution.py:59-65`。

### 7.2 `BROKER_ORDER`（新）委托

```json
{"type": "BROKER_ORDER", "trade_date": "2026-11-20", "order_time": "09:16:03",
 "broker_order_id": "202611200001234", "code": "sz159596", "side": "BUY",
 "qty": 100, "limit_price": 0.578, "order_kind": "LIMIT",
 "suggestion_id": "S-20261119", "flags": [], "evidence": {...}}
```

校验（不通过即拒绝）：`code == cfg.etf`；`side ∈ {BUY, SELL}`；`qty` 为 100 的整数倍（买）/ 符合卖出申报单位且 ≤ 可卖（卖）；`limit_price` 在 0.001 报价单位上且在当日涨跌停价之内（当日有 K 线时核对，否则记"未核对"）；`order_kind` 第一阶段只接受 `LIMIT`（SOP 只用限价单）；`broker_order_id` 在本账本内唯一；`order_time` 不早于对应建议的 `generated_at`（建议生成之前不可能照建议下单）。

### 7.3 `BROKER_FILL`（修改）成交

```json
{"type": "BROKER_FILL", "trade_date": "2026-11-20", "fill_time": "09:30:05",
 "broker_trade_id": "T0000987", "broker_order_id": "202611200001234",
 "code": "sz159596", "side": "BUY", "qty": 100, "price": 0.577,
 "fee_hint": null, "evidence": {...}}
```

与现有版本的差别：**必须引用已存在的 `BROKER_ORDER`**；**不再携带入账费用**（现有 `fees` 字段废弃，改为可选 `fee_hint`，只做展示、不进账本，见第 9 节）。校验：`broker_trade_id` 唯一；`side/code` 与委托一致；`price` 在报价单位上、买 ≤ 限价 / 卖 ≥ 限价、在当日高低价之间（[代码] 现有 `manual_execution.py:104-108` 保留）；累计成交数量 ≤ 委托数量 − 已撤数量；委托未进入终态；`fill_time ≥ order_time`；同一 `trade_date`。

### 7.4 `BROKER_CANCEL`（新）撤单（含部分撤单）

```json
{"type": "BROKER_CANCEL", "trade_date": "2026-11-20", "cancel_time": "10:02:11",
 "broker_order_id": "202611200001234", "broker_cancel_id": "C0000321",
 "cancelled_qty": 100, "evidence": {...}}
```

`cancelled_qty ≤ 剩余未成交数量`；`broker_cancel_id` 唯一（券商不提供时用 `"<order_id>#<cancel_time>"`）；委托未进入终态。`cancelled_qty < 剩余` 即部分撤单，委托继续开放。

### 7.5 `BROKER_REJECT`（新）券商/交易所拒单

```json
{"type": "BROKER_REJECT", "trade_date": "...", "reject_time": "...", "broker_order_id": "...",
 "reason_text": "<券商原文>", "evidence": {...}}
```

只能在该委托**没有任何成交、没有任何撤单**时录入；每个委托最多一条。

### 7.6 `BROKER_EXPIRED`（新）收盘未成交作废

```json
{"type": "BROKER_EXPIRED", "trade_date": "...", "broker_order_id": "...", "expired_qty": 100, "evidence": {...}}
```

`expired_qty == 剩余未成交数量`（必须相等，防止漏录成交）；每个委托最多一条；只能在 `trade_date` 收盘后录入（录入时刻 ≥ 当日 15:00 北京时间）。**系统不会自动生成这条记录**：一个委托过了交易日仍无终态，只会在日报里标 `OPEN_PAST_DAY`，等人对照 App 录入。

### 7.7 `BROKER_SETTLEMENT`（新）交割单费用

```json
{"type": "BROKER_SETTLEMENT", "trade_date": "2026-11-20", "broker_order_id": "202611200001234",
 "settled_qty": 100, "settled_amount": 57.70,
 "fees": {"commission": 0.01, "transfer_fee": 0.0, "exchange_fee": 0.0, "other": 0.0},
 "statement_ref": "<交割单编号或截图文件名>", "evidence": {...}}
```

每个 `(broker_order_id, trade_date)` 最多一条；`settled_qty` 必须等于该委托当日累计成交数量，`settled_amount` 必须等于 Σ(成交价 × 数量)（误差 ≤ 0.005）；`fees` 各项 ≥ 0、按分记账；`stamp_duty` 若出现且 > 0 → 硬拒绝（ETF 无印花税，[规则] VERIFIED），提示核对是否录错品种。

### 7.8 `BROKER_DIVIDEND`（已有，加唯一键）

增加 `broker_ref`（券商资金流水号或 `"<pay_date>#<amount>"`）唯一；`trade_date` 为**实际到账日**（真实账本不按除息日自动计分红，[代码] `manual_execution.py:124-129`）。

### 7.9 `VOID`（新）作废一条错录

```json
{"type": "VOID", "trade_date": "...", "target_hash": "<被作废记录的 hash>", "reason": "...", "evidence": {...}}
```

唯一的修正手段。被作废的记录保留在流水里（哈希链不动），重放时跳过；作废后同一 `broker_*_id` 允许重新录入一次（新记录带 `reentry_of: <target_hash>`）。`VOID` 不能作废 `CAPITAL`，不能作废另一条 `VOID`，一个 hash 最多被作废一次。日报展示"修正次数"，复盘里作为流程纪律指标。

### 7.10 不入账本的辅助文件

- `suggestions.jsonl`（已有，[代码] `runner.py:288-293`）：系统生成的建议，按 `suggestion_id = S-<决策日>` 幂等；**不属于账本**，不参与资金计算。
- `observation/live_daily.csv`（新）：第 11 节日报表，每次 `daily` 由流水与行情**重新生成**，不是记录来源。

## 8. 委托状态机

状态**不存储**，每次由该委托的事件序列推导（和账本一样，"状态 = 流水重放"）。

定义：`Q` = 委托数量，`F` = Σ 有效成交数量，`C` = Σ 有效撤单数量，`R = Q − F − C` = 剩余开放数量。

```
                    BROKER_ORDER
                         │
                         ▼
                  ┌─────────────┐
      ┌───────────│  SUBMITTED  │───────────┐
      │ REJECT    │ F=0, C=0    │  CANCEL   │
      │ (F=0,C=0) │ R=Q         │  c=R      │
      ▼           └──────┬──────┘           ▼
 ┌──────────┐            │ FILL/CANCEL  ┌───────────┐
 │ REJECTED │            │ (R>0 之后)   │ CANCELLED │  F=0
 └──────────┘            ▼              └───────────┘
                  ┌─────────────┐
                  │    OPEN     │◄──┐ FILL f<R 或 CANCEL c<R
                  │ 0<R<Q 或    │───┘ （部分成交 / 部分撤单，仍开放）
                  │ F>0 且 R>0  │
                  └──┬───┬───┬──┘
          FILL f=R   │   │   │  EXPIRED e=R（收盘后）
          ┌──────────┘   │   └───────────┐
          ▼              │ CANCEL c=R    ▼
   ┌─────────────┐       ▼        ┌──────────────────┐
   │   FILLED    │  ┌─────────────────┐ │ EXPIRED (F=0)    │
   │ F=Q         │  │ PARTIAL_CLOSED  │ │ 或 PARTIAL_CLOSED│
   └─────────────┘  │ 0<F<Q, R=0      │ │ (F>0)            │
                    └─────────────────┘ └──────────────────┘
```

状态判定（按优先级）：

| 条件 | 状态 | 终态 |
|---|---|---|
| 存在 `BROKER_REJECT` | `REJECTED` | 是 |
| `R == 0` 且 `F == Q` | `FILLED` | 是 |
| `R == 0` 且 `0 < F < Q` | `PARTIAL_CLOSED`（子类型：`by_cancel` / `by_expiry` / `mixed`） | 是 |
| `R == 0` 且 `F == 0`，有 `BROKER_EXPIRED` | `EXPIRED` | 是 |
| `R == 0` 且 `F == 0`，只有撤单 | `CANCELLED` | 是 |
| `R > 0` 且 `F == 0` 且 `C == 0` | `SUBMITTED` | 否 |
| `R > 0`（其余） | `OPEN`（日报写 `PARTIALLY_FILLED` 或 `PARTIALLY_CANCELLED`） | 否 |

不变量（录入时强制，独立复算再查一遍）：

1. `0 ≤ F ≤ Q`，`0 ≤ C ≤ Q − F`，`R ≥ 0`；
2. 进入终态后不得再有该委托的任何事件（`VOID` 除外）；
3. `REJECT` 要求 `F = C = 0`；`EXPIRED` 要求 `e == R > 0` 且录入时刻 ≥ 当日 15:00 CST；
4. 所有事件 `trade_date` 相同（A 股限价委托当日有效；不支持隔夜委托）；时间单调：`order_time ≤ 每个 fill/cancel 时间`；
5. 每个委托最多一条 `REJECT`、一条 `EXPIRED`、一条 `SETTLEMENT`；
6. 账本只被 `BROKER_FILL`（数量、价格）、`BROKER_SETTLEMENT`（费用）、`BROKER_DIVIDEND`（现金）改变；`ORDER / CANCEL / REJECT / EXPIRED / VOID` 本身不动现金和持仓。

### 8.1 与 T+1、现金的关系

- 卖出委托：录入 `BROKER_ORDER(SELL)` 时检查 `qty ≤ sellable`（当日及以前买入、且非当日买入，[代码] 现有 `manual_execution.py:100-102`）；多个开放卖单的数量合计也 ≤ sellable。
- 买入委托：`Σ 开放买单的 qty × limit_price` ≤ 现金（真实账户的冻结资金逻辑，不复用 `Account.freeze`——真实账户现金以券商为准，这里只做一致性检查）。

## 9. 费用记录与核算口径

**原则**：费用只有一个入账来源——`BROKER_SETTLEMENT`（交割单 / 券商费用明细）；成交记录不带费用；情景估算只用于对照。这样费用既不会漏，也不可能算两次。

| 规则 | 内容 |
|---|---|
| 入账单位 | 券商计费单位是"每个委托每日"（最低佣金按委托收，不按笔），所以 `SETTLEMENT` 以 `(broker_order_id, trade_date)` 为键，唯一 |
| 分摊 | 一个委托多笔成交时，费用按成交数量**按比例分摊到每笔成交**，各笔按分四舍五入，尾差归最后一笔。分摊只用于逐笔成本（`Account.buy/sell` 的 `fees` 参数，[代码] `ledger.py:86-98`），总额不变 |
| 时序 | 成交当晚可能只有成交回报没有交割单。`FILL` 先录、`SETTLEMENT` 后录；在 `SETTLEMENT` 录入前，重放时该委托费用按 0 计，日报净值标 `FEES_PENDING`（临时值），**不用情景估算代替** |
| 重放确定性 | 费用并入账本的时点是 `SETTLEMENT` 的 `trade_date`（成交日），不是录入日。因此补录交割单后，从成交日起的现金序列会改变——这是正确的（真实现金本来就被扣了），日报里对应天数标 `RESTATED` |
| `fee_hint` | 成交回报里 App 显示的费用，只存只看，不入账；与 `SETTLEMENT` 不一致时打 `FEE_HINT_MISMATCH` 标记 |
| 情景对照 | `cfg.fee_scenario.breakdown(VWAP, F)`（[代码] `etf_rules.py:153-161`）与 `SETTLEMENT.fees` 的差额写日报第 5 项；差额不改任何账 |
| 费用构成 | `commission / transfer_fee / exchange_fee / other` 四项；`stamp_duty > 0` 硬拒绝；`transfer_fee > 0` 首次出现时，日报提示"过户费 UNVERIFIED 规则有了实证，待更新规则文件"（规则文件本身不在观察期内改） |
| 不允许 | 在 `FILL` 和 `SETTLEMENT` 两处都录费用；一个委托两条 `SETTLEMENT`；用情景估算补费用；用 `PAPER` 的费用代替 |

独立复算（第 11.2 节）用不导入本库的脚本重算：Σ 入账费用 == Σ `SETTLEMENT.fees` 总额（排除被 `VOID` 的）；每条 `SETTLEMENT` 的 `settled_qty / settled_amount` 与成交一致；无 `SETTLEMENT` 的已成交委托数 == `FEES_PENDING` 数。

## 10. 异常标记与审计规则

标记分两级：**ERROR = 拒绝写入**（第 7、8 节的校验），**FLAG = 如实写入并固化在记录的 `flags` 字段**。FLAG 永远不阻止录入——真实发生的交易必须进账本，否则账本与现实不符。

| 标记 | 挂在 | 触发条件 | 审计含义 |
|---|---|---|---|
| `NO_SUGGESTION` | ORDER | `suggestion_id` 为空，或 `suggestions.jsonl` 里找不到 | 不按系统建议下的单 |
| `SUGGESTION_STALE` | ORDER | 引用的建议 `decision_day` 不是 `trade_date` 的前一交易日 | 用了过期建议 |
| `SUGGESTION_REJECTED` | ORDER | 建议存在但 `proposal.action == "REJECT"`（如超预算） | 系统明确拒绝仍下单 |
| `SIDE_MISMATCH` / `QTY_MISMATCH` / `LIMIT_MISMATCH` | ORDER | 与建议 `order.side / qty / limit_price` 不同 | 偏离建议 |
| `ORDER_BEFORE_SUGGESTION` | ORDER | `order_time` 早于建议 `generated_at` | **ERROR**（时间上不可能照建议下单；若确实是不看建议下的单，去掉 `suggestion_id` 改为 `NO_SUGGESTION`） |
| `OVER_BUDGET_ACTUAL` | SETTLEMENT | （实收费用 + 实际价差成本）/ 成交额 > `friction_budget`。实际价差成本 = Σ 成交数量 × (成交价 − 决策日收盘价)（买为正、卖取负），负值按 0 计 | 事后超预算；复盘统计 |
| `FEES_PENDING` | （日报）| 已成交委托尚无 `SETTLEMENT` | 临时净值 |
| `FEE_HINT_MISMATCH` | SETTLEMENT | `fee_hint` 合计与 `SETTLEMENT` 合计差 > 0.005 | 核对 App 与交割单 |
| `OPEN_PAST_DAY` | （日报）| 委托 `trade_date` 已过，仍非终态 | 等人录 `EXPIRED` / `CANCEL`；期间"下一笔条件"为不满足 |
| `OUTSIDE_LIMIT_BAND` | ORDER | 限价超出当日涨跌停价 | **ERROR**（交易所不会接受） |
| `PRICE_UNVERIFIED` | FILL / ORDER | 录入时行情存储还没有该日 K 线 | 日报在该日 K 线到达后补核：若成交价不在高低价之间 → `VERIFY_FAILED` |
| `CORRECTION` | VOID | 任一 `VOID` | 流程纪律：修正次数 |
| `LATE_ENTRY` | 任意 | `entered_at` 晚于 `trade_date` 次日 15:30 | 录入纪律 |

审计规则：

1. 标记在录入时计算并写入记录，之后不改；独立复算用同样的输入重算一遍，不一致即 `VERIFY_FAILED`。
2. 标记统计进日报（当日新增）和复盘（累计）：建议遵循率 = 无 `NO_SUGGESTION / *_MISMATCH / SUGGESTION_REJECTED` 的委托数 ÷ 委托总数。
3. 带 `SUGGESTION_REJECTED` 或 `OVER_BUDGET_ACTUAL` 的交易**照常入账、照常计入真实净值**，不剔除——真实账户的收益是什么就是什么；但复盘分别给出"全部交易"和"仅遵循建议的交易"两组统计。
4. 不允许用任何方式把模拟成交标成真实（已有机制，[代码] `ledgers.py:76-77`）；也不允许反向：真实成交不会写进 PAPER。

## 11. 真实账户日报（七项）与独立复算

### 11.1 七项指标定义

每日 `daily` 后生成 `reports/etf_pilot/<编号>/observation/live_daily.md|csv`，每交易日一行。没有 `CAPITAL` 时所有数值列为 `NO_ACCOUNT`，不出现数字。

| # | 指标 | 定义 / 计算 | 数据来源 |
|---|---|---|---|
| 1 | PAPER 资金、持仓、净值、收益 | 现有 `DAY_CLOSE`（PAPER_STRICT；PAPER_TOUCH 括号内） | PAPER 流水 |
| 2 | MANUAL_LIVE 资金、持仓、净值、收益 | 现金 = 本金 − Σ买入(价×量) + Σ卖出(价×量) − Σ入账费用 + Σ分红；持仓 = Σ买 − Σ卖；净值 = 现金 + 持仓 × 当日收盘；累计收益 = 净值/本金 − 1；最大回撤按净值序列。标 `FEES_PENDING` / `RESTATED` | MANUAL_LIVE 流水 + bars.csv |
| 3 | 双轨信号是否一致 | 比较 D 日 PAPER_STRICT `DECISION.intent.side`（持仓过滤前的原始意图需要额外输出 `raw_intent`）与 `S-D` 建议 `intent.side`；三态 `SAME / DIFF / NA`。两者同源（[代码] `runner.py:136`、`:273`），`DIFF` 即为缺陷，当日状态置 `VERIFY_FAILED` | PAPER 流水、suggestions.jsonl |
| 4 | 成交价与滑点差异 | 真实 VWAP（Σ价×量/Σ量）对比：(a) PAPER_STRICT 同日模拟成交价；(b) 决策日收盘；(c) 限价。各以元、报价单位数、bp 表示；PAPER 当日无成交时 (a) 为 `NA`，并注明 PAPER 结果（`NO_FILL / LATE / EXPIRED`） | 两本流水 |
| 5 | 实际手续费与模拟费用差异 | `SETTLEMENT.fees` 合计 − `fee_scenario.breakdown(VWAP, F)` 合计；以及 − PAPER 同日费用；按项列出。无 `SETTLEMENT` 写 `FEES_PENDING` | SETTLEMENT、配置 |
| 6 | 数据质量与 SHA256 | 行情获取时刻、相对 15:30 延迟、新增 K 线数、修订数、最近 60 日完整性问题数；配置 / 规则 / 选择记录 / pilot 代码四个哈希 + 与基线是否一致；流水文件（4 本）当日末 hash 链末端 | run_log、baseline.json、journals |
| 7 | 是否满足下一笔真实交易条件 | 逐项布尔并给原因：① 本金已登记；② 行情新鲜（最近已收盘交易日 K 线已到）；③ 无 `OPEN_PAST_DAY` 委托；④ 无 `PRICE_UNVERIFIED` 待核；⑤ 有意图且与持仓相容；⑥ `propose()` 为 `PROPOSE`；⑦ 预估摩擦 ≤ 0.5%；⑧ 现金 ≥ 限价成交额 + 预估费用；⑨ 无阻断项（跳变预热、完整性）；⑩ 当前时刻早于次日 09:15。全部为真 → `ELIGIBLE`，否则 `NOT_ELIGIBLE` + 首个不满足项。`FEES_PENDING` 不阻断，只提示 | propose() 结果、流水、时钟 |

### 11.2 独立复算（扩展 `scripts/verify_etf_ledger.py`，继续不导入本库）

现有脚本已对模拟账做逐日重算（[代码] `verify_etf_ledger.py:89-168`），对 MANUAL_LIVE 只查本金和现金/持仓非负（`:169-190`）。扩展为：

| 检查 | 方法 |
|---|---|
| V1 哈希链、来源 | 现有 `read_journal` |
| V2 键唯一 | `broker_order_id`、`broker_trade_id`、`broker_cancel_id`、`broker_ref`、`(order, date)` 的 `SETTLEMENT`、`target_hash` 各自唯一（排除被 VOID 的） |
| V3 状态机 | 对每个委托按第 8 节重算 `F / C / R` 与状态；检查六条不变量 |
| V4 引用完整 | 每条 FILL / CANCEL / REJECT / EXPIRED / SETTLEMENT 的 `broker_order_id` 存在且早于自己；`VOID.target_hash` 存在 |
| V5 价格 | 成交价在报价单位上、满足限价、在当日高低价之间（bars.csv）；限价在涨跌停带内（用前收 ± 10% 四舍五入到 0.001 独立计算） |
| V6 费用 | Σ 入账费用 == Σ SETTLEMENT；`settled_qty / settled_amount` 与成交一致；分摊重算（比例、四舍五入、尾差归最后一笔）与账本逐笔一致；`stamp_duty == 0` |
| V7 现金与持仓 | 逐日重放得到现金、持仓、净值，与 `live_daily.csv` 每行比对（TOL 0.005）；T+1、整手、现金非负 |
| V8 标记复算 | 用 `suggestions.jsonl` 和配置重算每条 ORDER / SETTLEMENT 的 `flags`，与记录一致 |
| V9 信号一致 | `DECISION.raw_intent`（PAPER_STRICT）与 `S-D.intent` 比对 == 日报第 3 项 |
| V10 哈希 | 重算配置 / 规则 / 选择记录 / pilot 代码文件 sha256，与日报第 6 项、baseline.json 比对 |
| V11 无虚构 | `MANUAL_LIVE` 每条记录 `evidence` 非空；A/B 实验（`actual_broker == false`）的 MANUAL_LIVE 流水必须为空 |

任一失败 → `daily` 对该实验返回 `VERIFY_FAILED`，按 SOP 停止并报告。
