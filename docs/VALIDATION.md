# 指标数值验证

> 本文件由 `python scripts/validation_report.py` 生成，对应测试为 `tests/test_validation.py`。

数据：几何布朗运动模拟日线，1500 根，随机种子 0, 1, 2；比较最后 500 根（跳过预热期），指标为“最大绝对误差 ÷ 参考序列最大绝对值”，取 3 个种子中的最大值。参考实现：TA-Lib 0.8.3。

## 汇总

| 状态 | 指标数 | 说明 |
|---|---:|---|
| 与 TA-Lib 一致 | 29 | 相对误差 < 1e-9 |
| 口径不同，按声明公式验证 | 7 | 与 TA-Lib 的差异来自定义选择，已对照指标自己声明的公式验证 |
| 不一致，原因未核对 | 2 | 见下表，使用时请注意 |
| 未验证 | 347 | 其余指标只保证能运行 |

## 与 TA-Lib 一致

| 指标 | 输出列 | 最大相对误差 |
|---|---|---:|
| SMA | — | 1.8e-15 |
| EMA | — | 5.5e-16 |
| WMA | — | 1.1e-14 |
| DEMA | — | 1.1e-15 |
| TEMA | — | 2.4e-15 |
| TRIMA | — | 6.1e-14 |
| KAMA | — | 1.8e-16 |
| T3 | — | 4.5e-15 |
| ADX | ADX | 8.5e-16 |
| DI_Plus | — | 9.5e-16 |
| DI_Minus | — | 7.1e-16 |
| Aroon | Aroon_Up | 7.1e-17 |
| Aroon | Aroon_Down | 7.1e-17 |
| AroonOscillator | — | 7.1e-17 |
| TRIX | TRIX | 7.0e-14 |
| LinearRegressionLine | LR_Value | 1.1e-13 |
| LinearRegressionLine | LR_Slope | 2.1e-12 |
| CCI | — | 1.3e-14 |
| MACD | MACD | 4.0e-14 |
| MACD | MACD_Signal | 3.1e-14 |
| MACD | MACD_Histogram | 4.5e-14 |
| Momentum | — | 0.0e+00 |
| ROC | — | 1.1e-15 |
| RSI | — | 6.3e-16 |
| StochasticOscillator | Stoch_K | 5.9e-16 |
| StochasticOscillator | Stoch_D | 2.7e-15 |
| StochasticOscillator | Fast_K | 1.4e-16 |
| UltimateOscillator | — | 3.5e-16 |
| WilliamsR | — | 1.5e-16 |
| LinearRegressionSlope | — | 1.2e-12 |
| PPOMACD | PPO | 3.0e-14 |
| TrueRange | — | 0.0e+00 |
| BollingerBands | BB_Middle | 1.8e-15 |
| AccumulationDistribution | — | 0.0e+00 |
| ChaikinOscillator | — | 1.6e-14 |
| MoneyFlowIndex | — | 1.7e-15 |

## 口径不同（已按声明公式验证）

| 指标 | 输出列 | 与声明公式的最大相对误差 | 与 TA-Lib 的差异 |
|---|---|---:|---|
| StandardDeviation | StdDev | 0.0e+00 | 样本标准差（ddof=1），TA-Lib 为总体标准差 |
| Variance | — | 0.0e+00 | 样本方差（ddof=1），TA-Lib 为总体方差 |
| BollingerBands | BB_Upper | 0.0e+00 | 标准差用 ddof=1，TA-Lib 用总体标准差 |
| BollingerBands | BB_Lower | 0.0e+00 | 同上 |
| APO | — | 0.0e+00 | 用 EMA，TA-Lib 默认用 SMA（matype=0） |
| CMO | — | 1.9e-16 | Chande 原始定义（简单求和），TA-Lib 用 Wilder 平滑 |
| ATR | — | 0.0e+00 | EMA(span=n) 平滑，TA-Lib 用 Wilder 平滑（alpha=1/n） |
| OBV | — | 0.0e+00 | 从 0 起算，TA-Lib 从第一根成交量起算（差一个常数） |

## 不一致，原因未核对

| 指标 | 输出列 | 与 TA-Lib 的最大相对误差 | 说明 |
|---|---|---:|---|
| ParabolicSAR | PSAR | 2.7e-02 | 均为 Wilder 算法；差异疑似来自起始方向与反转当根的处理，未逐行核对 |
| MESAAdaptiveMA | MAMA | 7.1e-02 | Ehlers 希尔伯特变换实现细节不同，未逐行核对 |

## 验证过程中修复的问题

| 指标 | 问题 | 修复 |
|---|---|---|
| Aroon / AroonOscillator | 实现与自身注释公式相反：最高价出现在 1 天前时 Aroon Up 显示 4，应为 96 | 改为 (n − 距极值周期数) / n × 100 |
| TRIMA | 两层均线都用 n/2+1，合计覆盖 n+1 根K线，与注释公式和标准定义都不符 | 偶数周期 SMA(n/2) 再 SMA(n/2+1)，奇数周期两层均为 (n+1)/2 |
