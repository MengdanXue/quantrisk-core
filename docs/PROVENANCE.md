# 代码来源与各阶段贡献

这份清单说明 quantrisk-core 里哪些能力来自原项目，哪些是在本仓库中修复、扩展或新写的。依据是 git 历史和工作区文件，不靠回忆。

## 0. 原项目是什么

- **TraderX**（产品名 QuantRiskPro V2.0）是作者 2025–2026 年独立编写的交易管理系统原型，覆盖行情、交易管理、前后端等。那个仓库保持私有，其中的文档和数据文件**没有**进入本仓库。仓库内的 LICENSE 为 MIT，版权行为 "Copyright (c) 2025 TraderX Team"（作者当时为该项目起的名字）。
- 本仓库第一个提交 `53b3c03` 从 TraderX 的 `backend-python/src`（修订 `156d265e`）复制了 85 个 Python 文件，只保留与具体业务、数据源、部署无关的计算部分。没有导入的：绑定原项目配置中心和数据源的旧指标函数，以及两个不是作者本人编写的估值计算器。
- 从 TraderX 带来的文件头（`作者: TraderX Team`、`Copyright (c) 2025 TraderX. All rights reserved.` 等）**原样保留**，用来标明来源。署名与许可的说明见仓库根目录 `NOTICE`。
- **未核实**：TraderX 内部的代码是否还含有其他来源的片段，本仓库没有逐文件追溯。指标公式是公开的技术分析定义，与 TA-Lib 的对照只比较数值，不涉及代码。

## 1. 原有 TraderX 能力（`53b3c03` 导入，之后只做了下面列出的修复）

| 模块 | 内容 |
|---|---|
| `quantrisk.indicators` | 385 个注册指标、17 类；注册表、批量计算器、可选 Numba 加速 |
| `quantrisk.strategies` | 20 个单标的策略；双标的配对交易框架；旧回测引擎（`backtest/engine.py`、`portfolio.py`、`events.py`、`execution.py`、`performance.py`、`report.py`）、蒙特卡洛、Walk-Forward 优化 |
| `quantrisk.factor_research` | 因子库、因子分析（IC、分组收益）、因子合成 |
| `quantrisk.valuation` | DCF 估值 |

这些能力的**数值正确性大多没有验证**：38 个常用指标与 TA-Lib 做过对照，其余 347 个没有；蒙特卡洛、Walk-Forward 优化、`report.py` 没有验证（见 README「已知问题」）。

## 2. 开源整理与第一批修复（已提交，2026-11-06）

| 提交 | 内容 |
|---|---|
| `102b6b4` | 修复旧引擎逐 bar 收益的前视偏差 |
| `a262a04`、`c12c393` | 打包、首批测试、本地浏览器演示 |
| `7a1a7db` | 截图与前视偏差案例 |
| `34f2985`、`5843702` | 指标修复：Aroon 上下线颠倒；TRIMA 窗口多算 1 根 |
| `a63123c` | 配对交易框架移出单标的策略注册表 |
| `1f855b1`、`f83d412` | TA-Lib 对照、截断不变性与费用测试；TA-Lib 改为可选依赖 |

## 3. M0：可信回测（未提交，2026-11-06）

目标：现金、持仓、权益、交易费用全部可核对。

- 新写：`backtest/ledger.py`（资金账本：可用现金 + 冻结现金 + 持仓市值）；
- 修改：`backtest/execution.py`（撮合）、`strategies/base.py`（T 日收盘出信号、T+1 开盘撮合）、`events.py`、`engine.py`、`portfolio.py`（旧引擎停用）；
- 验证：`tests/test_backtest_ledger.py`、`scripts/verify_buy_and_hold.py`（独立核算）。

## 4. M1：真实 A 股回测（未提交，2026-11-06）

目标：用真实历史数据复现回测，不用随机行情冒充。

- 新写：`backtest/ashare_rules.py`（带版本日期的涨跌停、申报单位、T+1、印花税、过户费）、`src/quantrisk/data/ashare.py`（akshare 下载、快照 manifest 与 SHA-256 校验、除权除息、停牌）；
- 验证：`tests/test_ashare_rules.py`、`tests/test_ashare_real.py`、`scripts/ashare_acceptance.py`：600519、000001 的 2024 年行情与独立核算逐日对账。

## 5. M2：组合回测与风险控制（未提交，2026-11-06）

目标：多股票共享现金、风险预算、能解释收益与回撤来源。

- 新写：`backtest/daily.py`（日频主循环、`RiskBudget`：下单与成交两道硬性检查、被动超限事件、可选减仓）、`backtest/portfolio_run.py`（组合入口与审计报告）；
- 验证：`tests/test_portfolio.py`、`tests/test_risk.py`、`scripts/verify_portfolio.py`、`scripts/portfolio_acceptance.py`：6 只股票 2024 年行情逐日独立核算。

## 6. M3：样本外研究与前视偏差修复（未提交，2026-11-06 – 11-09）

- 新写：`data/universe.py`（按时点股票池）、`research/walk_forward.py`（组合级 Walk-Forward，训练/验证/测试隔离、实验登记）、`research/cross_section.py`（动量、低波动横截面因子与两层换手约束）、`research/overfitting.py`（PSR、Deflated Sharpe Ratio、CSCV/PBO）；
- 前视修复：数据质量原先按整段判定，段内之后才发生的配股会让股票从段首起被排除；改为按时点判定（`point_in_time_usable`，Walk-Forward 默认），见 [research/M3_LOOKAHEAD_FIX.md](research/M3_LOOKAHEAD_FIX.md)；
- 研究可信度审计：[research/M3_RESEARCH_AUDIT.md](research/M3_RESEARCH_AUDIT.md)。结论是**没有证据表明策略有超额收益**：股票池有幸存者偏差（同池等权跑赢上证 50 指数 62 个百分点），DSR 与 PBO 都不支持；
- 验证：`tests/test_research.py`、`tests/test_cross_section.py`、`tests/test_point_in_time_quality.py`、`scripts/m3_walk_forward.py`、`scripts/research_cross_section.py`、`scripts/m3_audit.py`、`scripts/compare_lookahead_fix.py`。

（M2 与 M3 在同一天前后衔接，`backtest/walk_forward.py` 和 `data/universe.py` 的归属按会话记录划在 M3，边界是近似的。）

## 7. ETF Pilot 独立实验模块（未提交，2026-11-09 – 11-11）

单只 ETF、100 元以内的模拟与人工实盘记录。不连接券商，没有下单代码。

- 新写：`src/quantrisk/pilot/` 全部 12 个模块与深交所规则文件 `rules_szse_etf_20261109.json`；`scripts/etf_pilot.py`（命令行）、`scripts/verify_etf_ledger.py`（**不导入本库**的独立复算器）；
- 复用：策略信号直接调用原有 `ma_cross`（20/60），不改参数；
- 设计文档：[etf_pilot/ENGINE_AUDIT_20261109.md](etf_pilot/ENGINE_AUDIT_20261109.md)、[etf_pilot/MANUAL_LIVE_DESIGN.md](etf_pilot/MANUAL_LIVE_DESIGN.md)；
- 验证：`tests/test_etf_pilot.py` 66 个测试，全部用合成数据。

## 8. 真实行情前向观察与独立账本复算（运行记录，不在仓库中）

这部分是**运行**而不是代码，产生的行情、流水、报告都在 `data/` 与 `reports/`，不随仓库分发：

- **历史回放**（HISTORICAL_REPLAY）：sz159596 2025-11-10 至 2026-11-06 共 241 个交易日，两种佣金情景；2026-11-12 用当前代码逐条复现 11-09 写入的流水，独立复算 0 问题。行情获取时间晚于全部决策日，只检验流程和账本；
- **前向模拟**（FORWARD PAPER）：2026-11-09 起按 [ETF_PILOT_FORWARD_SOP.md](ETF_PILOT_FORWARD_SOP.md) 每个交易日运行，计划 20 个交易日；
- **真实资金**（MANUAL_LIVE）：没有进行。按核对过的常见券商费率（每笔佣金起点 5 元），100 元本金下任何买入都会被 0.5% 成本预算拒绝，所以没有初始化真实费率实验。
