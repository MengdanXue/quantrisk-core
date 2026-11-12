# ETF Pilot 使用说明

单只境内普通股票型宽基 ETF 的小资金实验，分两种模式：PAPER（模拟）和 MANUAL_LIVE（人工录入真实成交）。

- **本工具不连接任何券商，不包含任何下单代码。** 真实交易由本人在券商 App 中确认执行，成交后手工录入。
- 设计与审计见 [etf_pilot/ENGINE_AUDIT_20261109.md](etf_pilot/ENGINE_AUDIT_20261109.md)（开发前的引擎兼容性审计；代码注释里的 `ETF_PILOT_AUDIT.md` 即此文件）与 [etf_pilot/MANUAL_LIVE_DESIGN.md](etf_pilot/MANUAL_LIVE_DESIGN.md)（人工实盘账本设计）。

## 1. 已冻结的实验设定（2026-11-09）

| 项目 | 设定 |
|---|---|
| 标的 | **sz159596 华宝 A50ETF**（跟踪中证 A50 指数） |
| 选择规则 | 已审计白名单 ∧ 交易所规则已核实 ∧ 最近收盘价 ≤ 0.90 元，按最近 20 个已完成交易日的平均成交额降序取第一 |
| 选择依据 | 2026-10-12 至 11-06 共 20 个交易日，日均成交额 1440 万元；11-06 收盘价 0.578 元；数据于 2026-11-09 14:34（北京时间）从新浪获取。记录文件：`data/pilot/selection/selection_20261109T143418.json` |
| 策略 | 现有 `ma_cross`，fast 20 / slow 60，其余参数为默认值（EMA、价格过滤）；不做参数优化 |
| 本金 | 100 元硬上限；不追加、不借贷、无杠杆 |
| 佣金情景 | A：最低 5 元；B：万分之 0.85、无最低（仅理论敏感性，不代表任何券商报价）。开户后用实际费率新建实验 |
| 摩擦成本预算 | 单边 0.5%，等于显性费用加上估算价差（半个买卖价差，1 个报价单位）；两项分别记录，超出就拒绝生成订单 |
| 限价 | 买入：信号日收盘价 + 1 个报价单位；卖出：收盘价 − 1 个报价单位。信号有效期 5 个交易日 |
| 基准 | 同一只 ETF 买入持有：同样的成交规则、费用和预算 |

**交易规则来源**：`src/quantrisk/pilot/rules_szse_etf_20261109.json`，核实日期 2026-11-09。

- 《深圳证券交易所交易规则（2026 年修订）》，2026-07-06 起施行：
  - 第 3.3.11 条：报价单位 0.001 元；
  - 第 3.3.8 条：买入 100 份整数倍，余额不足 100 份须一次性卖出；
  - 第 3.3.13、3.3.14、3.3.19 条：涨跌幅 10%，按四舍五入计算涨跌停价；
  - 第 4.4 节：除息参考价；
  - 第 3.1.4、3.1.5 条：股票型 ETF 实行 T+1。
- 深交所投教指南：ETF 无印花税。
- 深交所收费表：经手费双边各 0.004%，通常包含在佣金中。
- **过户费未核实**，作为费用配置项处理。

## 2. 每日操作（前向实验）

> 前向观察阶段请使用标准流程：`scripts/etf_pilot.py daily`，异常处理与 20 个交易日复盘见 [ETF_PILOT_FORWARD_SOP.md](ETF_PILOT_FORWARD_SOP.md)。下面是单步命令。

**每个交易日北京时间 15:30 之后**执行：

```bash
P=python   # 已安装本库及 akshare 的解释器（pip install -e ".[dev,data]"）
$P scripts/etf_pilot.py status                                      # 列出实验与已处理交易日
$P scripts/etf_pilot.py fetch  --exp etf159596_B_0.85bp_nomin-6c96b650
$P scripts/etf_pilot.py run    --exp etf159596_B_0.85bp_nomin-6c96b650
$P scripts/etf_pilot.py run    --exp etf159596_A_min5-e4a402b4
$P scripts/etf_pilot.py report --exp etf159596_B_0.85bp_nomin-6c96b650   # 输出到 reports/etf_pilot/<编号>/
$P scripts/etf_pilot.py verify --exp etf159596_B_0.85bp_nomin-6c96b650   # 重算比对 + 独立复算
```

- 两个实验共用同一份行情存储（`data/pilot/market/sz159596/`），`fetch` 执行一次就够。
- 漏了几天没关系：下次运行会按顺序补齐。补处理的交易日，只用当天及以前的数据做决策。
- `run` 会被拒绝的情况：
  - 缺少最近一个已收盘交易日的行情（行情过期）：报 `StaleDataError`，不生成任何建议；
  - 已写入的记录与重算结果不一致（数据源事后修订了过去的数据，或代码逻辑变了）：报 `ReplayMismatchError`，不覆盖历史；
  - 流水最后一行不完整（写入时中断）：报 `CorruptJournalError`。确认之后，用 `run --repair` 把残行移到 `.partial` 文件，再继续；
  - 配置文件被改动：报 `ConfigChangedError`。实验期间不能改参数，要改就新建实验。

## 3. 开户之后（MANUAL_LIVE）

真实轨道只能挂在**真实费率实验**下（`actual_broker=True`）；A/B 两个模拟费率实验上的一切 `record-*` 都会被拒绝。设计依据见 [etf_pilot/MANUAL_LIVE_DESIGN.md](etf_pilot/MANUAL_LIVE_DESIGN.md)。

1. 用实际费率新建实验：在 `src/quantrisk/pilot/etf_rules.py` 的 `SCENARIOS` 中加入真实费率情景，设 `actual_broker=True` 并注明券商和核实日期，然后 `init --scenario <新情景>`。已经冻结的实验不要改。
2. 登记实验本金（只能登记一次，不超过 100 元）。每条真实记录都必须带凭据：`--evidence-kind` 取 `APP_SCREENSHOT / SETTLEMENT_STATEMENT / BROKER_MESSAGE`，`--evidence-ref` 写截图文件名或交割单编号。
   ```bash
   $P scripts/etf_pilot.py record-capital --exp <编号> --amount 100 --date 2026-12-01 \
       --evidence-kind APP_SCREENSHOT --evidence-ref capital_20261201.png
   ```
3. 之后每天 `daily` 会写入 `suggestions.jsonl`。这只是**交易建议**（方向、数量、限价、预估费用）；是否下单由你在券商 App 中决定。
4. 按券商回报逐条录入事件（`E="--evidence-kind APP_SCREENSHOT --evidence-ref <文件名>"`）：
   ```bash
   $P scripts/etf_pilot.py record-order  --exp <编号> --date 2026-12-02 --time 09:25:30 --side BUY --qty 100 --limit 0.579 \
       --order-id <券商委托编号> --suggestion S-20261201 $E
   $P scripts/etf_pilot.py record-fill   --exp <编号> --date 2026-12-02 --time 09:30:12 --order-id <委托编号> --qty 100 \
       --price 0.578 --trade-id <券商成交编号> [--fee-hint commission=0.10] $E
   $P scripts/etf_pilot.py record-cancel --exp <编号> --date ... --time ... --order-id ... --qty <撤单数量> [--cancel-id ...] $E
   $P scripts/etf_pilot.py record-reject --exp <编号> --date ... --time ... --order-id ... --reason "资金不足" $E
   $P scripts/etf_pilot.py record-expired --exp <编号> --date ... --order-id ... --qty <剩余数量> $E      # 当日 15:00 之后
   $P scripts/etf_pilot.py record-settlement --exp <编号> --date ... --order-id ... --qty <成交总量> --amount <成交金额> \
       --fee commission=0.10 --fee transfer_fee=0.00 --statement-ref <交割单编号> --evidence-kind SETTLEMENT_STATEMENT --evidence-ref ...
   $P scripts/etf_pilot.py record-dividend --exp <编号> --date <到账日> --amount ... [--broker-ref ...] $E
   ```
   - **委托状态不存储**，每次由事件推导：Q 委托数量、F 成交、C 撤单、E 收盘作废，R = Q − F − C − E。终态为 `FILLED / PARTIAL_CLOSED / CANCELLED / EXPIRED / REJECTED`；非终态为 `SUBMITTED / OPEN`。
   - **费用只认交割单**：成交记录不带费用（`--fee-hint` 只作核对，不入账）；交割单按成交数量比例分摊到各笔成交，按分四舍五入、尾差归最后一笔。交割单录入前该委托标 `FEES_PENDING`，费用按 0 计；补录后日报对受影响日期标 `RESTATED`。
   - **收盘未成交作废（EXPIRED）系统不会自动生成**：在券商 App 核实委托状态后人工录入，且只能在当日 15:00 之后。系统只在日报第 7 项提示 `OPEN_PAST_DAY`。
   - **ERROR 直接拒绝**：报价单位、整手、T+1、现金、涨跌停带（`OUTSIDE_LIMIT_BAND`）、成交价在当日区间内、违反限价、超过剩余数量、跨日事件、终态后再录事件、委托早于建议生成时刻（`ORDER_BEFORE_SUGGESTION`）、印花税 > 0、交割数量或金额与成交不符。
   - **FLAG 照常入账**：`NO_SUGGESTION`、`SUGGESTION_STALE`、`SUGGESTION_REJECTED`、`SIDE/QTY/LIMIT_MISMATCH`、`PRICE_UNVERIFIED`（当日 K 线未到，到达后由独立复算补核）、`OVER_BUDGET_ACTUAL`、`FEE_HINT_MISMATCH`、`LATE_ENTRY`（晚于次日 15:30 录入）。真实发生的交易必须进账本。
5. **幂等与修正**：同一券商编号原样重录返回已有记录，不新增行；内容不同则拒绝。录错只能 `void --hash <记录哈希> --reason ...` 后重录，同一编号只允许重录一次；不能作废本金登记或作废记录本身；委托下还有有效事件时要先逐条作废子事件。
6. 查看：`orders --exp <编号>`（委托与状态）、`live --exp <编号>`（真实账户日报）。

## 4. 文件与账本

| 位置 | 内容 |
|---|---|
| `data/pilot/market/<代码>/` | `bars.csv`、`dividends.csv`、`calendar.csv`（每行带首次获取时间）、`revisions.csv`（数据源事后修订的记录，不覆盖原值）、`fetch_log.jsonl` |
| `data/pilot/experiments/<编号>/config.json`、`config.sha256` | 冻结的配置与哈希 |
| `.../journals/PAPER_STRICT.jsonl` | 严格模拟：只认开盘价满足限价的成交 |
| `.../journals/PAPER_TOUCH.jsonl` | 敏感性：盘中触价也按限价计为成交 |
| `.../journals/BENCHMARK.jsonl` | 同一只 ETF 买入持有 |
| `.../journals/MANUAL_LIVE.jsonl` | 只有人工录入的券商事件（本金、委托、成交、撤单、拒单、作废、交割、分红、VOID） |
| `.../suggestions.jsonl` | 给真实账户的交易建议（不属于任何账本） |
| `reports/etf_pilot/<编号>/` | `report.md`、`nav.png`、`nav_daily.csv`、`orders_audit.csv`、`fills.csv`、`metrics.json` |
| `reports/etf_pilot/<编号>/observation/` | `daily_log.*`（PAPER 观察表）、`live_daily.*` 与 `live_summary.json`（真实账户日报七项） |

- 每本账是独立的 JSONL 文件，带序号和哈希链；账本类型和来源必须匹配，模拟记录写不进 MANUAL_LIVE，反过来也写不进去。
- 账本状态全部由流水重放得到。`scripts/verify_etf_ledger.py` 不导入本库，可以独立复算现金、持仓、费用，并核对限价、价格区间和 T+1；对真实账本另行复核自然键唯一、引用完整、状态机、交割单分摊、标记、凭据、日报末行与双轨信号一致性，以及配置/规则/选择记录的哈希基线。

## 5. 模拟成交规则

- **严格成交**：次日开盘价满足限价（买：开盘价 ≤ 限价；卖：开盘价 ≥ 限价），且开盘不在涨停（买）或跌停（卖）。
  - 成交价 = 开盘价 ± 半个价差，再限制在限价以内、当日最高价和最低价之间。
- **不确定**：开盘价不满足限价，只是盘中触及。这类订单不计入严格账本，敏感性账本按限价计为成交。
- **作废**：当日无行情、开盘就在涨跌停，或者价格没有触及限价。
- **价差不重复计费**：价差体现在成交价相对开盘价的偏离里（单独统计），显性费用按佣金情景另算。

## 6. 已知限制

1. **份额拆分/折算没有数据源**。sz159596 在 2025-10-20 价格从 1.249 变为 0.628。2026-11-12 已从基金年报核实：《华宝中证A50交易型开放式指数证券投资基金2025年年度报告》第 23 页记载，本基金于 2025-10-17 进行份额拆分，**拆分比例为 2**；同一年报的份额变动表中拆分变动份额为 +488,841,520 份。系统没有拆分数据源，冻结实验仍按以下方式处理（偏保守，不编造份额）：
   - 把它识别为"无法解释的价格跳变"，信号从跳变日起重新计算；
   - 跳变后 120 个交易日内不生成建议；
   - 跳变当天如果模拟账本仍有持仓，直接停止，不编造份额变动。
2. **日线数据无法证明盘中能否成交**。严格模拟只认开盘价，所以结果偏保守；价差是假设值（1 个报价单位），不是历史实际价差。
3. **佣金情景都不是真实券商报价**：
   - 情景 B 在约 58 元的成交额上，佣金四舍五入后为 0.00 元，比任何真实券商都低；
   - 情景 A 下所有买入都会因为超出预算被拒绝。情景 A 的参数（万 2.5、每笔起点 5 元）与 2026-11 核对的一家大型券商网页公示 ETF 费率相同，所以“100 元本金下 A 必被拒”可以视为按真实常见费率会发生的情况；具体账户的实际费率仍以交割单为准。
4. 过户费没有核实；只核实了深交所规则，上交所的 ETF 暂时一律拒绝。
5. ETF 分红只有累计值，单次分红由差分得到。现金分红按除息日计入模拟账本，实际到账要晚几天。
6. **历史回放演练**（目录名带 `_rehearsal` 的实验）的行情获取时间都晚于决策日，只能检验流程和账本，不能当作前向或实盘结果。
7. 100 元只能持有 100 份、1 只 ETF。这个实验验证的是执行链路，**不能验证任何策略的收益**。
