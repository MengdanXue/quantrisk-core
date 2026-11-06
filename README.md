# quantrisk-core

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个纯 Python 的量化工具库：技术指标、策略回测、因子研究、DCF 估值，附带一个本地浏览器演示页面。

代码从个人项目 QuantRiskPro 中抽出，只保留和具体业务、数据源、部署无关的计算部分。抽取后做了两件事：

- **回测**：发现并修复了收益计算里的一个**前视偏差**（[详见下文](#回测里的前视偏差修复前后对比)），并给全部 20 个策略加上了防前视的“截断不变性”测试；
- **指标**：把 38 个常用指标和 TA-Lib 逐个对照，其中 29 个结果一致，7 个是口径不同（已按各自声明的公式验证），2 个不一致、原因尚未查明，过程中修复了 Aroon 和 TRIMA 两个实现错误（[验证报告](docs/VALIDATION.md)）。

> 仅用于学习与研究，不构成任何投资建议。

![演示页面](docs/images/demo-overview.png)

## 目录

- [功能概览](#功能概览)
- [指标验证](#指标验证)
- [快速开始](#快速开始)
- [本地演示](#本地演示)
- [回测规则](#回测规则)
- [回测里的前视偏差：修复前后对比](#回测里的前视偏差修复前后对比)
- [项目结构](#项目结构)
- [测试](#测试)
- [已知问题](#已知问题)

## 功能概览

| 模块 | 内容 |
|---|---|
| `quantrisk.indicators` | 385 个注册指标，分 17 类：趋势、动量、波动率、成交量、统计、形态、周期、多因子、情绪、微观结构、衍生品、跨市场、宏观联动、执行质量、高级统计、机器学习特征、自定义；统一注册表、批量计算器，可选 Numba 加速 |
| `quantrisk.strategies` | 20 个单标的策略（均线交叉、海龟、MACD、布林回归、RSI 回归、唐奇安通道、波动率突破、网格、时序动量等）；双标的配对交易框架 `AdvancedPairsTradingStrategy`（协整检验、半衰期、卡尔曼对冲比率）；以及回测引擎、组合回测、蒙特卡洛模拟、Walk-Forward 优化、报告生成 |
| `quantrisk.factor_research` | 因子库、因子分析（IC、分组收益）、因子合成 |
| `quantrisk.valuation` | DCF 估值 |

只给 OHLCV 时，有 310 个指标能直接算出结果；其余 75 个需要额外的输入列（买卖盘、财报字段、参照资产等），缺列时会明确抛出 `MissingColumnError` 或 `KeyError`。

## 指标验证

用 TA-Lib 作为参考实现，在 3 组模拟行情上逐点比较（完整结果见 [docs/VALIDATION.md](docs/VALIDATION.md)）：

| 状态 | 数量 | 指标 |
|---|---:|---|
| ✅ 与 TA-Lib 一致（相对误差 < 1e-9） | 29 | SMA、EMA、WMA、DEMA、TEMA、TRIMA、KAMA、T3、ADX、+DI、−DI、Aroon、Aroon 振荡器、TRIX、线性回归线、线性回归斜率、CCI、MACD（三条线）、动量、ROC、RSI、随机指标（K/D/快 K）、终极振荡器、威廉 %R、PPO、真实波幅、A/D 线、Chaikin 振荡器、MFI |
| 🟦 口径不同，已按声明公式验证 | 7 | 标准差、方差、布林带（样本标准差 ddof=1）；APO（EMA）；CMO（Chande 原始定义）；ATR（EMA 平滑，非 Wilder）；OBV（从 0 起算） |
| ⚠️ 不一致，原因未查明 | 2 | 抛物线 SAR、MESA 自适应均线 |
| ⬜ 未验证 | 347 | 只保证能运行，使用前请自行核对 |

验证过程中修复了两个实现错误：**Aroon** 的计算方向与自身公式相反（最高价在 1 天前时显示 4，应为 96）；**TRIMA** 的两层均线窗口多算了 1 根 K 线。

## 快速开始

```bash
git clone https://github.com/MengdanXue/quantrisk-core.git
cd quantrisk-core
pip install -e ".[dev]"     # 最小安装：pip install -e .
```

`statsmodels`（协整类指标）和 `scikit-learn`（ML 特征类指标）是可选依赖，分别对应 `[stats]` 和 `[ml]`。

### 计算指标

```python
import importlib, pkgutil
import quantrisk.indicators
from quantrisk.indicators.registry import indicator_registry

# 指标通过装饰器注册，先导入全部子模块
for m in pkgutil.walk_packages(quantrisk.indicators.__path__, "quantrisk.indicators."):
    importlib.import_module(m.name)

# df：以日期为索引，含 open / high / low / close / volume 列的 DataFrame
rsi = indicator_registry.create_indicator("RSI").calculate(df)
macd = indicator_registry.create_indicator("MACD").calculate(df)   # 返回 MACD / Signal / Histogram 三列

print(indicator_registry.get_category_stats())     # 各类别指标数量
print(indicator_registry.search("bollinger"))       # 按关键字查找
```

### 运行回测

```python
from quantrisk.strategies import get_strategy, list_strategies

result = get_strategy("ma_cross", {"position_pct": 0.5}).run(
    df, initial_capital=1_000_000, commission=0.001, slippage=0.0005
)
print(result.metrics)          # total_return / sharpe_ratio / max_drawdown / win_rate / ...
result.equity_curve.plot()     # 权益曲线（pandas Series）
```

成交规则见下文[回测规则](#回测规则)。

## 本地演示

```bash
pip install -e ".[demo]"
uvicorn demo.app:app --port 8765
```

打开 <http://127.0.0.1:8765>。演示使用几何布朗运动生成的模拟行情，不读取任何外部数据。

**策略回测**：选择策略，调整天数、随机种子、漂移、波动、手续费和仓位比例，查看权益曲线（蓝）与买入持有（灰）的对比，以及收益、夏普、回撤、胜率等指标。

![策略回测](docs/images/demo-backtest.png)

**指标计算**：下拉框列出 308 个只需 OHLCV 就能计算的指标。量级差别大的输出（例如通道宽度）会自动画在右侧坐标轴上。

![指标计算](docs/images/demo-indicator.png)

## 回测规则

`BaseStrategy.run()` 的成交与计费规则如下，均有测试覆盖：

| 项目 | 规则 |
|---|---|
| 信号 | `generate_signals()` 在每根 K 线收盘后给出信号，只能使用截至当根的数据 |
| 成交价 | 当根收盘价；买入价 × (1 + 滑点)，卖出价 × (1 − 滑点) |
| 手续费 | 按成交金额比例收取，买卖双向 |
| 仓位 | 开仓金额 = 可用资金 × `position_pct`，强信号再乘 1.5；所需资金超过可用资金时不成交 |
| 方向 | 只做多：买入信号开仓，卖出信号平仓，不加仓、不做空 |
| 收益 | 每根 K 线收益 = 本根期末权益 ÷ 上一根期末权益 − 1；成交当根只体现交易成本，价格变动从下一根开始计入 |

防作弊测试（`tests/test_strategies.py`）：

- **截断不变性**：对每个策略，只用前 k 根数据回测的信号和权益，必须与全量数据回测的前 k 根完全一致；只要用到未来数据，结果就会不同。20 个策略全部通过。为证明这个检查确实有效，另有一个使用居中滚动均线（常见的隐蔽前视写法）的反例策略，测试要求它必须被抓出来。
- **收盘成交不占便宜**：一个“偷看当根涨跌”的策略在收盘成交规则下不应赚钱。
- **成本核算**：价格不变时完成一次买卖，权益损失必须恰好等于手续费加滑点。

## 回测里的前视偏差：修复前后对比

原实现在成交那根 K 线上，用**成交后的持仓**乘以“上一根到这一根”的价格变化来计算当根收益：

- 买入当根：明明是收盘才买入，却拿到了当根的涨幅；
- 卖出当根：明明持有到收盘才卖出，却漏掉了当根的跌幅。

突破类策略恰好在“当根上涨”时买入、“当根下跌”时卖出，所以这个偏差会被系统性放大。下图是同一个策略在同一份**随机游走**数据上的结果。随机游走本来不可预测，任何策略的期望收益都应接近零：

![修复前后对比](docs/images/lookahead-before-after.png)

修复方式是把当根收益改为“本根期末权益 ÷ 上一根期末权益 − 1”（[`strategies/base.py`](src/quantrisk/strategies/base.py)）。`tests/test_strategies.py::test_no_lookahead_when_trading_at_close` 构造了一个“偷看当根涨跌”的策略：在收盘成交的规则下，它不应该赚到钱。这个测试在原代码上 5 个随机种子全部失败，修复后全部通过。

复现上图：

```bash
python scripts/lookahead_comparison.py /path/to/a/CJK/font.ttf
```

## 项目结构

```
quantrisk-core/
├── src/quantrisk/
│   ├── indicators/          # 指标：base / registry / engine + 17 个类别子包
│   ├── strategies/          # 策略 + backtest/（引擎、组合、蒙特卡洛、WFO、报告）
│   ├── factor_research/     # 因子库、分析、合成
│   └── valuation/           # DCF
├── demo/                    # FastAPI 演示服务 + 单页前端（无外部依赖）
├── scripts/                 # 前视偏差对比图、指标验证报告的生成脚本
├── tests/                   # pytest
└── docs/images/             # README 截图
```

## 测试

```bash
pytest
```

当前共 189 个测试，覆盖：

- 指标与 TA-Lib 的逐点对照（需要另外安装：`pip install -e ".[validate]"`，未安装时这部分自动跳过），以及口径不同的指标对照其声明公式；
- 全部指标在只有 OHLCV 输入时“要么能算、要么明确报缺列”；
- 20 个策略都能跑通，并通过截断不变性检查；前视偏差回归测试；成本核算；
- 演示服务的 API。

## 已知问题

- 抛物线 SAR 和 MESA 自适应均线与 TA-Lib 的结果不一致，原因尚未查明。
- 347 个指标的数值还没有验证。
- 双标的配对交易框架 `AdvancedPairsTradingStrategy` 的接口和单标的策略不同，没有放进 `get_strategy()` 的注册表，也还没有测试。
- 部分代码注释和指标显示名是中文。

## 许可证

MIT，见 [LICENSE](LICENSE)。
