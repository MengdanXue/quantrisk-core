# quantrisk-core

一个纯 Python 的量化工具库：技术指标、策略回测、因子研究、DCF 估值。代码从个人项目 QuantRiskPro（TraderX）中抽出，只保留和具体业务、数据源、部署无关的计算部分，其余代码不在本仓库。

> 仅用于学习与研究，不构成任何投资建议。

## 包含什么

| 模块 | 内容 |
|---|---|
| `quantrisk.indicators` | 385 个注册指标，分 17 类：趋势、动量、波动率、成交量、统计、形态、周期、多因子、情绪、微观结构、衍生品、跨市场、宏观联动、执行质量、高级统计、机器学习特征、自定义；另有统一注册表、批量计算器和可选的 Numba 加速 |
| `quantrisk.strategies` | 20 个可直接运行的策略（均线交叉、海龟、MACD、布林回归、波动率突破、网格、时序动量等），以及事件驱动回测引擎、组合回测、蒙特卡洛、Walk-Forward 优化、报告生成 |
| `quantrisk.factor_research` | 因子库、因子分析（IC/分层）、因子合成 |
| `quantrisk.valuation` | DCF 估值 |

只给 OHLCV 时，有 310 个指标可以直接算；其余 75 个需要额外的输入列（买卖盘、财报字段、参照资产等），缺列时会明确抛出 `MissingColumnError` 或 `KeyError`。

## 安装

```bash
pip install -e ".[dev]"      # 含测试和演示依赖
# 或最小安装：pip install -e .
```

`statsmodels`（协整类指标）和 `scikit-learn`（ML 特征类指标）是可选依赖，分别对应 `[stats]` 和 `[ml]`。

## 用法

```python
import importlib, pkgutil
import quantrisk.indicators
from quantrisk.indicators.registry import indicator_registry
from quantrisk.strategies import get_strategy

# 指标通过装饰器注册，先导入全部子模块
for m in pkgutil.walk_packages(quantrisk.indicators.__path__, "quantrisk.indicators."):
    importlib.import_module(m.name)

rsi = indicator_registry.create_indicator("RSI").calculate(df)   # df 需有 open/high/low/close/volume 列
result = get_strategy("ma_cross").run(df, initial_capital=1_000_000, commission=0.001)
print(result.metrics["sharpe_ratio"], result.metrics["max_drawdown"])
```

## 本地演示

```bash
pip install -e ".[demo]"
uvicorn demo.app:app --port 8765
```

打开 <http://127.0.0.1:8765>。页面用几何布朗运动生成的模拟行情：选策略跑回测，看权益曲线和绩效指标，也可以选指标画图。演示不读取任何外部数据。

## 测试

```bash
pytest
```

## 与原项目相比的改动

- **修复回测收益的前视偏差**（`strategies/base.py`）。原实现在成交当根 K 线上，用成交后的持仓回溯计算上一根到这一根的收益：买入当天白拿了当天涨幅，卖出当天漏掉了当天跌幅。在随机游走数据上，波动率突破策略原来算出约 +29%、夏普 7.4，修复后约 −5.7%。`tests/test_strategies.py::test_no_lookahead_when_trading_at_close` 用于防止回归。
- 去掉了和原项目配置中心、数据源绑定的旧版指标函数（`calculator_v2`、`sigma`、`bands` 等），以及 API、数据库、前端和部署代码。

## 已知问题

- `advanced_pairs` 在策略注册表里，但它的构造函数不接受 `params` 参数，所以 `get_strategy("advanced_pairs")` 会报错。演示和测试都暂时跳过了它。
- 部分代码注释和指标显示名是中文。

## 许可证

MIT，见 [LICENSE](LICENSE)。
