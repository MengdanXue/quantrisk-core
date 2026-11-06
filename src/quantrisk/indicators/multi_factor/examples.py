"""
多因子指标使用示例

演示如何使用各种多因子指标进行量化分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from ..registry.indicator_registry import indicator_registry
from .factor_indicators import (
    PBFactor, PEFactor, ROEFactor, RevenueGrowthFactor,
    PriceMomentumFactor, DownsideVolatilityFactor, MarketCapFactor,
    MultiFactorScore, FactorRotationSignal
)


def example_1_basic_factor_usage():
    """
    示例1: 基本因子使用

    演示如何计算单个因子
    """
    print("=" * 80)
    print("示例1: 基本因子使用")
    print("=" * 80)

    # 创建模拟数据
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        'shares_outstanding': np.full(n, 1000000),
        'total_assets': np.random.uniform(50e6, 100e6, n),
        'total_liabilities': np.random.uniform(20e6, 40e6, n),
    })

    # 计算市净率因子
    pb_factor = PBFactor(normalization='zscore')
    pb_scores = pb_factor.calculate(data)

    print(f"PB因子统计:")
    print(f"  均值: {pb_scores.mean():.4f}")
    print(f"  标准差: {pb_scores.std():.4f}")
    print(f"  最小值: {pb_scores.min():.4f}")
    print(f"  最大值: {pb_scores.max():.4f}")
    print(f"\n最新PB因子得分: {pb_scores.iloc[-1]:.4f}")


def example_2_compare_normalizations():
    """
    示例2: 比较不同归一化方法

    演示zscore、minmax、rank三种归一化的差异
    """
    print("\n" + "=" * 80)
    print("示例2: 比较归一化方法")
    print("=" * 80)

    # 创建数据
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)),
    })

    normalizations = ['zscore', 'minmax', 'rank']

    print("\n价格动量因子 (20日) 在不同归一化下的结果:\n")
    print(f"{'方法':<10} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print("-" * 50)

    for norm in normalizations:
        factor = PriceMomentumFactor(normalization=norm, period=20)
        scores = factor.calculate(data)

        print(f"{norm:<10} {scores.mean():>10.4f} {scores.std():>10.4f} "
              f"{scores.min():>10.4f} {scores.max():>10.4f}")


def example_3_multi_factor_selection():
    """
    示例3: 多因子选股

    使用多个因子组合进行股票筛选
    """
    print("\n" + "=" * 80)
    print("示例3: 多因子选股")
    print("=" * 80)

    # 创建10只股票的模拟数据
    np.random.seed(42)
    n_stocks = 10
    n_periods = 100

    # 为每只股票生成数据
    stock_data = {}
    for i in range(n_stocks):
        stock_data[f'Stock_{i+1}'] = pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.02)),
            'shares_outstanding': np.full(n_periods, np.random.randint(500000, 2000000)),
            'total_assets': np.random.uniform(30e6, 150e6, n_periods),
            'total_liabilities': np.random.uniform(10e6, 60e6, n_periods),
            'net_income': np.random.uniform(1e6, 10e6, n_periods),
        })

    # 计算多个因子
    factor_scores = {}

    for stock_name, data in stock_data.items():
        scores = {}

        # 价值因子
        pb = PBFactor(normalization='rank')
        scores['value'] = pb.calculate(data).iloc[-1]

        # 质量因子
        roe = ROEFactor(normalization='rank')
        scores['quality'] = roe.calculate(data).iloc[-1]

        # 动量因子
        momentum = PriceMomentumFactor(normalization='rank', period=20)
        scores['momentum'] = momentum.calculate(data).iloc[-1]

        # 综合得分 (等权)
        scores['composite'] = (scores['value'] + scores['quality'] + scores['momentum']) / 3

        factor_scores[stock_name] = scores

    # 转换为DataFrame并排序
    scores_df = pd.DataFrame(factor_scores).T
    scores_df = scores_df.sort_values('composite', ascending=False)

    print("\n股票因子得分排名 (分数越高越好):\n")
    print(scores_df.to_string())

    print("\n\n推荐买入Top 3股票:")
    for i, (stock, row) in enumerate(scores_df.head(3).iterrows(), 1):
        print(f"  {i}. {stock}: 综合得分 {row['composite']:.4f}")


def example_4_factor_rotation():
    """
    示例4: 因子轮动策略

    动态选择表现最好的因子
    """
    print("\n" + "=" * 80)
    print("示例4: 因子轮动策略")
    print("=" * 80)

    # 创建数据
    np.random.seed(42)
    n = 120  # 需要足够长的历史数据
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        'market_index': 1000 * np.exp(np.cumsum(np.random.randn(n) * 0.015)),
    })

    # 创建因子轮动模型
    rotation = FactorRotationSignal(
        lookback_period=60,
        rebalance_period=20,
        top_n_factors=2
    )

    # 计算轮动信号
    signals = rotation.calculate(data)

    # 显示最近的轮动信号
    print("\n最近10个交易日的因子权重和综合信号:\n")
    print(signals.tail(10).to_string())

    # 统计因子被选中的频率
    print("\n\n因子选中频率统计:")
    for col in signals.columns:
        if col != 'composite_signal':
            selected_count = (signals[col] > 0).sum()
            frequency = selected_count / len(signals) * 100
            print(f"  {col}: {frequency:.1f}%")


def example_5_custom_multi_factor_model():
    """
    示例5: 自定义多因子模型

    创建自定义权重的多因子模型
    """
    print("\n" + "=" * 80)
    print("示例5: 自定义多因子模型")
    print("=" * 80)

    # 创建完整的模拟数据
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        'shares_outstanding': np.full(n, 1000000),
        'total_assets': np.random.uniform(50e6, 100e6, n),
        'total_liabilities': np.random.uniform(20e6, 40e6, n),
        'net_income': np.random.uniform(1e6, 5e6, n),
        'revenue': np.random.uniform(10e6, 20e6, n),
    })

    # 定义三种不同的投资风格
    strategies = {
        '价值投资': {
            'value': 0.60,
            'quality': 0.30,
            'growth': 0.05,
            'momentum': 0.05,
            'low_volatility': 0.00
        },
        '成长投资': {
            'value': 0.10,
            'quality': 0.20,
            'growth': 0.50,
            'momentum': 0.20,
            'low_volatility': 0.00
        },
        '平衡投资': {
            'value': 0.20,
            'quality': 0.20,
            'growth': 0.20,
            'momentum': 0.20,
            'low_volatility': 0.20
        },
    }

    print("\n不同投资风格的多因子得分对比:\n")
    print(f"{'投资风格':<12} {'最新得分':>12} {'平均得分':>12} {'得分波动':>12}")
    print("-" * 50)

    for strategy_name, weights in strategies.items():
        # 创建多因子模型
        model = MultiFactorScore(normalization='zscore', weights=weights)
        scores = model.calculate(data)

        print(f"{strategy_name:<12} {scores.iloc[-1]:>12.4f} "
              f"{scores.mean():>12.4f} {scores.std():>12.4f}")


def example_6_factor_exposure_analysis():
    """
    示例6: 因子暴露分析

    分析组合的因子暴露度
    """
    print("\n" + "=" * 80)
    print("示例6: 因子暴露分析")
    print("=" * 80)

    # 创建投资组合 (5只股票)
    np.random.seed(42)
    portfolio_weights = {
        'Stock_A': 0.25,
        'Stock_B': 0.20,
        'Stock_C': 0.20,
        'Stock_D': 0.20,
        'Stock_E': 0.15,
    }

    # 为每只股票计算因子值
    factor_exposures = {}
    n = 100

    for stock, weight in portfolio_weights.items():
        data = pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            'shares_outstanding': np.full(n, np.random.randint(500000, 2000000)),
            'total_assets': np.random.uniform(30e6, 150e6, n),
            'total_liabilities': np.random.uniform(10e6, 60e6, n),
            'net_income': np.random.uniform(1e6, 10e6, n),
        })

        # 计算各因子
        exposures = {
            'PB': PBFactor(normalization='zscore').calculate(data).iloc[-1],
            'ROE': ROEFactor(normalization='zscore').calculate(data).iloc[-1],
            'Momentum': PriceMomentumFactor(normalization='zscore').calculate(data).iloc[-1],
            'Size': MarketCapFactor(normalization='zscore').calculate(data).iloc[-1],
        }

        factor_exposures[stock] = exposures

    # 计算加权因子暴露
    exposure_df = pd.DataFrame(factor_exposures).T
    weighted_exposure = (exposure_df.T * pd.Series(portfolio_weights)).T.sum()

    print("\n各股票的因子暴露:\n")
    print(exposure_df.to_string())

    print("\n\n组合的加权因子暴露:\n")
    for factor, value in weighted_exposure.items():
        print(f"  {factor}: {value:>8.4f}")


def run_all_examples():
    """运行所有示例"""
    examples = [
        example_1_basic_factor_usage,
        example_2_compare_normalizations,
        example_3_multi_factor_selection,
        example_4_factor_rotation,
        example_5_custom_multi_factor_model,
        example_6_factor_exposure_analysis,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n错误: {example_func.__name__} 执行失败")
            print(f"原因: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("所有示例运行完成!")
    print("=" * 80)


if __name__ == '__main__':
    run_all_examples()
