"""
另类数据指标 (20个)

包含社交媒体、新闻情绪、链上数据、资金流向、另类信号等非传统数据源指标
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


# ============================================================================
# 社交媒体指标 (4个)
# ============================================================================

@indicator_registry.register('sentiment')
class SocialSentiment(BaseIndicator):
    """社交媒体情绪分析"""

    _metadata = IndicatorMetadata(
        name='SocialSentiment',
        display_name='社交媒体情绪',
        category='sentiment',
        description='基于价格波动模拟社交媒体情绪指数',
        formula='Social Sentiment = f(price_momentum, volatility, volume)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 14, 'sentiment_window': 7},
        complexity_level=2
    )

    def __init__(self, period: int = 14, sentiment_window: int = 7):
        self.period = period
        self.sentiment_window = sentiment_window
        super().__init__(period=period, sentiment_window=sentiment_window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.sentiment_window, 'sentiment_window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 价格动量（代理社交讨论热度）
        returns = close.pct_change()
        momentum = returns.rolling(window=self.period).mean() * 100

        # 波动率（代理情绪波动）
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252) * 100

        # 成交量异常（代理病毒传播）
        vol_ma = volume.rolling(window=self.period).mean()
        vol_spike = (volume / vol_ma - 1) * 100

        # 综合情绪得分 (-100到100)
        sentiment_raw = momentum * 0.5 + vol_spike.clip(-50, 50) * 0.3 - (volatility - 20).clip(0, 50) * 0.2
        sentiment = sentiment_raw.rolling(window=self.sentiment_window).mean()

        # 情绪强度
        sentiment_strength = abs(sentiment).rolling(window=self.sentiment_window).mean()

        # 情绪类别
        sentiment_label = pd.Series('Neutral', index=data.index)
        sentiment_label[sentiment > 30] = 'Very Positive'
        sentiment_label[(sentiment > 10) & (sentiment <= 30)] = 'Positive'
        sentiment_label[(sentiment < -10) & (sentiment >= -30)] = 'Negative'
        sentiment_label[sentiment < -30] = 'Very Negative'

        return pd.DataFrame({
            'Social_Sentiment': sentiment,
            'Sentiment_Strength': sentiment_strength,
            'Momentum_Component': momentum,
            'Volume_Component': vol_spike
        })


@indicator_registry.register('sentiment')
class MentionVolume(BaseIndicator):
    """提及量指标"""

    _metadata = IndicatorMetadata(
        name='MentionVolume',
        display_name='提及量',
        category='sentiment',
        description='基于成交量和波动率模拟社交媒体提及量',
        formula='Mention Volume = f(volume_spike, volatility, price_change)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=1
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 成交量异常度
        vol_zscore = (volume - volume.rolling(window=self.period).mean()) / \
                     volume.rolling(window=self.period).std()

        # 价格异常度
        price_zscore = (returns - returns.rolling(window=self.period).mean()) / \
                       returns.rolling(window=self.period).std()

        # 提及量指数（成交量和价格异常的组合）
        mention_index = (abs(vol_zscore) * 0.6 + abs(price_zscore) * 0.4) * 100

        # 平滑提及量
        mention_smooth = mention_index.rolling(window=5).mean()

        # 提及量趋势
        mention_trend = mention_smooth - mention_smooth.shift(self.period)

        # 提及量爆发检测
        mention_burst = (mention_index > mention_index.rolling(window=self.period).quantile(0.9)).astype(int)

        return pd.DataFrame({
            'Mention_Volume': mention_index,
            'Mention_Smooth': mention_smooth,
            'Mention_Trend': mention_trend,
            'Mention_Burst': mention_burst
        })


@indicator_registry.register('sentiment')
class InfluencerActivity(BaseIndicator):
    """意见领袖活动指标"""

    _metadata = IndicatorMetadata(
        name='InfluencerActivity',
        display_name='意见领袖活动',
        category='sentiment',
        description='基于大额交易检测模拟意见领袖活动',
        formula='Influencer Activity = Large Trade Detection',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20, 'threshold': 2.0},
        complexity_level=2
    )

    def __init__(self, period: int = 20, threshold: float = 2.0):
        self.period = period
        self.threshold = threshold
        super().__init__(period=period, threshold=threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.threshold, 'threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 大额交易检测（日内波动 × 成交量）
        intraday_range = (high - low) / close
        vol_zscore = (volume - volume.rolling(window=self.period).mean()) / \
                     volume.rolling(window=self.period).std()

        # 影响力指数
        impact_score = intraday_range * 100 * (1 + vol_zscore.clip(0, None))

        # 意见领袖活动检测
        influencer_activity = (vol_zscore > self.threshold).astype(int)

        # 活动累计
        activity_count = influencer_activity.rolling(window=self.period).sum()

        # 影响力方向
        returns = close.pct_change()
        influence_direction = np.sign(returns) * influencer_activity

        # 累积影响力
        cumulative_influence = influence_direction.cumsum()

        return pd.DataFrame({
            'Impact_Score': impact_score,
            'Influencer_Activity': influencer_activity,
            'Activity_Count': activity_count,
            'Cumulative_Influence': cumulative_influence
        })


@indicator_registry.register('sentiment')
class ViralityScore(BaseIndicator):
    """病毒传播度指标"""

    _metadata = IndicatorMetadata(
        name='ViralityScore',
        display_name='病毒传播度',
        category='sentiment',
        description='测量市场信息传播速度和广度',
        formula='Virality = f(volume_acceleration, price_momentum, volatility_spike)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 10},
        complexity_level=2
    )

    def __init__(self, period: int = 10):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 成交量加速度
        vol_change = volume.pct_change()
        vol_acceleration = vol_change.diff()

        # 价格动量加速
        momentum = returns.rolling(window=5).sum()
        momentum_accel = momentum.diff()

        # 波动率突增
        volatility = returns.rolling(window=self.period).std()
        vol_spike = volatility / volatility.rolling(window=self.period * 2).mean() - 1

        # 病毒传播度得分
        virality = (
            abs(vol_acceleration).clip(0, 1) * 40 +
            abs(momentum_accel) * 100 * 30 +
            vol_spike.clip(0, 2) * 30
        )

        # 平滑
        virality_smooth = virality.rolling(window=3).mean()

        # 传播阶段
        stage = pd.Series('Dormant', index=data.index)
        stage[virality_smooth > 20] = 'Emerging'
        stage[virality_smooth > 40] = 'Spreading'
        stage[virality_smooth > 60] = 'Viral'
        stage[virality_smooth > 80] = 'Explosive'

        return pd.DataFrame({
            'Virality_Score': virality_smooth,
            'Volume_Acceleration': vol_acceleration,
            'Momentum_Acceleration': momentum_accel,
            'Volatility_Spike': vol_spike
        })


# ============================================================================
# 新闻情绪 (4个)
# ============================================================================

@indicator_registry.register('sentiment')
class NewsSentiment(BaseIndicator):
    """新闻情绪分析"""

    _metadata = IndicatorMetadata(
        name='NewsSentiment',
        display_name='新闻情绪',
        category='sentiment',
        description='基于价格缺口和开盘行为模拟新闻情绪',
        formula='News Sentiment = f(gaps, opening_momentum, intraday_trend)',
        required_columns=['open', 'high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']

        # 跳空（隔夜新闻）
        gap = (open_price - close.shift(1)) / close.shift(1) * 100

        # 开盘动量
        opening_momentum = (close - open_price) / open_price * 100

        # 新闻情绪得分
        news_sentiment = gap * 0.6 + opening_momentum * 0.4

        # 平滑
        sentiment_smooth = news_sentiment.rolling(window=5).mean()

        # 情绪强度
        sentiment_intensity = abs(news_sentiment).rolling(window=self.period).mean()

        # 正负新闻统计
        positive_news = (news_sentiment > 0).astype(int).rolling(window=self.period).sum()
        negative_news = (news_sentiment < 0).astype(int).rolling(window=self.period).sum()
        news_ratio = positive_news / (negative_news + 1)  # 避免除零

        return pd.DataFrame({
            'News_Sentiment': sentiment_smooth,
            'Sentiment_Intensity': sentiment_intensity,
            'Gap_Component': gap,
            'News_Ratio': news_ratio
        })


@indicator_registry.register('sentiment')
class HeadlineImpact(BaseIndicator):
    """标题冲击力"""

    _metadata = IndicatorMetadata(
        name='HeadlineImpact',
        display_name='标题冲击力',
        category='sentiment',
        description='测量重大新闻对市场的冲击程度',
        formula='Headline Impact = Gap Size × Volume × Follow-through',
        required_columns=['open', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        close = data['close']
        volume = data['volume']

        # 跳空幅度
        gap = abs((open_price - close.shift(1)) / close.shift(1)) * 100

        # 成交量放大
        vol_ma = volume.rolling(window=self.period).mean()
        vol_ratio = volume / vol_ma

        # 跟进强度（当天延续）
        intraday_move = abs((close - open_price) / open_price) * 100

        # 冲击力指数
        impact = gap * vol_ratio * (1 + intraday_move / 10)

        # 平滑
        impact_smooth = impact.rolling(window=3).mean()

        # 重大事件检测
        major_event = (impact > impact.rolling(window=self.period).quantile(0.95)).astype(int)

        # 事件衰减
        impact_decay = impact.rolling(window=5, center=False).apply(
            lambda x: x.iloc[-1] / (x.max() + 0.01) if len(x) > 0 else 0
        )

        return pd.DataFrame({
            'Headline_Impact': impact_smooth,
            'Impact_Raw': impact,
            'Major_Event': major_event,
            'Impact_Decay': impact_decay
        })


@indicator_registry.register('sentiment')
class MediaCoverage(BaseIndicator):
    """媒体报道量"""

    _metadata = IndicatorMetadata(
        name='MediaCoverage',
        display_name='媒体报道量',
        category='sentiment',
        description='基于价格和成交量异常估算媒体关注度',
        formula='Media Coverage = f(price_volatility, volume_spike, momentum)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20, 'coverage_window': 10},
        complexity_level=2
    )

    def __init__(self, period: int = 20, coverage_window: int = 10):
        self.period = period
        self.coverage_window = coverage_window
        super().__init__(period=period, coverage_window=coverage_window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.coverage_window, 'coverage_window')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 价格波动度
        volatility = returns.rolling(window=self.coverage_window).std() * np.sqrt(252)
        vol_percentile = volatility.rolling(window=self.period).apply(
            lambda x: (x.rank() / len(x) * 100).iloc[-1] if len(x) > 0 else 50
        )

        # 成交量异常
        vol_ma = volume.rolling(window=self.period).mean()
        vol_spike = (volume / vol_ma - 1).clip(-0.5, 5) * 100

        # 价格异常
        price_move = abs(returns.rolling(window=self.coverage_window).sum()) * 100

        # 媒体覆盖度估算
        coverage = vol_percentile * 0.4 + vol_spike * 0.3 + price_move * 0.3

        # 覆盖度等级
        coverage_level = pd.Series(1, index=data.index)
        coverage_level[coverage > 40] = 2
        coverage_level[coverage > 60] = 3
        coverage_level[coverage > 80] = 4
        coverage_level[coverage > 95] = 5

        # 覆盖度趋势
        coverage_trend = coverage - coverage.shift(self.coverage_window)

        return pd.DataFrame({
            'Media_Coverage': coverage,
            'Coverage_Level': coverage_level,
            'Coverage_Trend': coverage_trend,
            'Volatility_Component': vol_percentile
        })


@indicator_registry.register('sentiment')
class EventSurprise(BaseIndicator):
    """事件意外度"""

    _metadata = IndicatorMetadata(
        name='EventSurprise',
        display_name='事件意外度',
        category='sentiment',
        description='测量市场事件的意外程度',
        formula='Event Surprise = Price Deviation × Volume Anomaly',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30, 'std_threshold': 2.0},
        complexity_level=2
    )

    def __init__(self, period: int = 30, std_threshold: float = 2.0):
        self.period = period
        self.std_threshold = std_threshold
        super().__init__(period=period, std_threshold=std_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.std_threshold, 'std_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 价格Z-score
        ret_mean = returns.rolling(window=self.period).mean()
        ret_std = returns.rolling(window=self.period).std()
        price_zscore = (returns - ret_mean) / ret_std

        # 成交量Z-score
        vol_mean = volume.rolling(window=self.period).mean()
        vol_std = volume.rolling(window=self.period).std()
        volume_zscore = (volume - vol_mean) / vol_std

        # 事件意外度（价格和成交量的联合异常）
        surprise = abs(price_zscore) * abs(volume_zscore)

        # 意外事件检测
        surprise_event = (surprise > self.std_threshold).astype(int)

        # 正向/负向意外
        positive_surprise = ((price_zscore > self.std_threshold) &
                           (volume_zscore > 1)).astype(int)
        negative_surprise = ((price_zscore < -self.std_threshold) &
                           (volume_zscore > 1)).astype(int)

        # 意外度等级
        surprise_level = pd.Series(0, index=data.index)
        surprise_level[surprise > 2] = 1
        surprise_level[surprise > 4] = 2
        surprise_level[surprise > 6] = 3

        return pd.DataFrame({
            'Event_Surprise': surprise,
            'Surprise_Event': surprise_event,
            'Surprise_Level': surprise_level,
            'Positive_Surprise': positive_surprise,
            'Negative_Surprise': negative_surprise
        })


# ============================================================================
# 链上数据 (4个) - 使用价格/成交量数据模拟
# ============================================================================

@indicator_registry.register('sentiment')
class WhaleActivity(BaseIndicator):
    """巨鲸活动指标"""

    _metadata = IndicatorMetadata(
        name='WhaleActivity',
        display_name='巨鲸活动',
        category='sentiment',
        description='基于大额交易检测巨鲸活动',
        formula='Whale Activity = Large Volume Transactions',
        required_columns=['high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20, 'volume_threshold': 2.5},
        complexity_level=2
    )

    def __init__(self, period: int = 20, volume_threshold: float = 2.5):
        self.period = period
        self.volume_threshold = volume_threshold
        super().__init__(period=period, volume_threshold=volume_threshold)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_float(self.volume_threshold, 'volume_threshold')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 大额交易检测
        vol_ma = volume.rolling(window=self.period).mean()
        vol_std = volume.rolling(window=self.period).std()
        vol_zscore = (volume - vol_ma) / vol_std

        # 巨鲸活动标记
        whale_activity = (vol_zscore > self.volume_threshold).astype(int)

        # 活动方向（买入/卖出）
        price_impact = (close - (high + low) / 2) / ((high - low) + 1e-10)
        whale_direction = np.sign(price_impact) * whale_activity

        # 累积巨鲸流向
        whale_flow = (whale_direction * volume).cumsum()

        # 巨鲸活跃度
        whale_count = whale_activity.rolling(window=self.period).sum()

        # 巨鲸主导度
        whale_volume = (volume * whale_activity).rolling(window=self.period).sum()
        total_volume = volume.rolling(window=self.period).sum()
        whale_dominance = whale_volume / total_volume * 100

        return pd.DataFrame({
            'Whale_Activity': whale_activity,
            'Whale_Direction': whale_direction,
            'Whale_Flow': whale_flow,
            'Whale_Count': whale_count,
            'Whale_Dominance': whale_dominance
        })


@indicator_registry.register('sentiment')
class ExchangeFlow(BaseIndicator):
    """交易所流入流出"""

    _metadata = IndicatorMetadata(
        name='ExchangeFlow',
        display_name='交易所流入流出',
        category='sentiment',
        description='模拟交易所资金流入流出',
        formula='Exchange Flow = f(volume_change, price_momentum)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 成交量变化
        vol_change = volume - volume.shift(1)

        # 流向估算（成交量变化 × 价格方向）
        flow_direction = np.sign(returns)
        exchange_flow = vol_change * flow_direction

        # 流入/流出统计
        inflow = exchange_flow.where(exchange_flow > 0, 0)
        outflow = abs(exchange_flow.where(exchange_flow < 0, 0))

        # 净流入
        net_flow = inflow - outflow
        net_flow_cumulative = net_flow.cumsum()

        # 流入流出比
        inflow_sum = inflow.rolling(window=self.period).sum()
        outflow_sum = outflow.rolling(window=self.period).sum()
        flow_ratio = inflow_sum / (outflow_sum + 1)  # 避免除零

        # 流动性压力
        flow_pressure = (outflow - inflow) / volume

        return pd.DataFrame({
            'Exchange_Flow': exchange_flow,
            'Net_Flow': net_flow_cumulative,
            'Flow_Ratio': flow_ratio,
            'Inflow': inflow_sum,
            'Outflow': outflow_sum
        })


@indicator_registry.register('sentiment')
class ActiveAddresses(BaseIndicator):
    """活跃地址数"""

    _metadata = IndicatorMetadata(
        name='ActiveAddresses',
        display_name='活跃地址数',
        category='sentiment',
        description='基于成交量模拟活跃地址数',
        formula='Active Addresses Proxy = f(volume, volatility)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30},
        complexity_level=1
    )

    def __init__(self, period: int = 30):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 活跃度代理（归一化成交量）
        vol_ma = volume.rolling(window=self.period).mean()
        vol_std = volume.rolling(window=self.period).std()
        active_proxy = (volume - vol_ma) / vol_std * 10 + 100
        active_proxy = active_proxy.clip(0, None)

        # 活跃度趋势
        active_ma = active_proxy.rolling(window=14).mean()
        active_trend = active_proxy - active_ma

        # 新用户代理（成交量突增）
        new_users = (volume > volume.rolling(window=self.period).quantile(0.9)).astype(int)

        # 活跃率
        returns_active = abs(returns) > returns.rolling(window=self.period).std()
        activity_rate = returns_active.astype(int).rolling(window=self.period).mean() * 100

        return pd.DataFrame({
            'Active_Addresses': active_proxy,
            'Active_Trend': active_trend,
            'New_Users_Signal': new_users,
            'Activity_Rate': activity_rate
        })


@indicator_registry.register('sentiment')
class NetworkHashRate(BaseIndicator):
    """网络算力"""

    _metadata = IndicatorMetadata(
        name='NetworkHashRate',
        display_name='网络算力',
        category='sentiment',
        description='基于价格稳定性模拟网络算力',
        formula='Hash Rate Proxy = f(price_stability, volume_consistency)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30},
        complexity_level=2
    )

    def __init__(self, period: int = 30):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 价格稳定性（低波动率=高算力）
        returns = close.pct_change()
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        price_stability = 1 / (volatility + 0.01)  # 反向指标

        # 成交量一致性
        vol_cv = volume.rolling(window=self.period).std() / \
                 volume.rolling(window=self.period).mean()
        volume_consistency = 1 / (vol_cv + 0.01)

        # 算力代理（归一化到0-100）
        hashrate_raw = price_stability * 0.6 + volume_consistency * 0.4
        hashrate = (hashrate_raw - hashrate_raw.rolling(window=self.period * 2).min()) / \
                  (hashrate_raw.rolling(window=self.period * 2).max() -
                   hashrate_raw.rolling(window=self.period * 2).min() + 1e-10) * 100

        # 算力趋势
        hashrate_ma = hashrate.rolling(window=14).mean()
        hashrate_trend = hashrate - hashrate_ma

        # 难度调整代理
        difficulty_proxy = hashrate.rolling(window=14).mean() / \
                          hashrate.rolling(window=self.period).mean()

        return pd.DataFrame({
            'HashRate_Proxy': hashrate,
            'HashRate_Trend': hashrate_trend,
            'Difficulty_Proxy': difficulty_proxy,
            'Price_Stability': price_stability
        })


# ============================================================================
# 资金流向 (4个)
# ============================================================================

@indicator_registry.register('sentiment')
class InstitutionalFlow(BaseIndicator):
    """机构资金流"""

    _metadata = IndicatorMetadata(
        name='InstitutionalFlow',
        display_name='机构资金流',
        category='sentiment',
        description='基于收盘行为估算机构资金流向',
        formula='Institutional Flow = f(closing_strength, large_trades)',
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 收盘强度（机构倾向尾盘操作）
        closing_strength = (close - low) / (high - low + 1e-10)
        closing_strength = closing_strength.fillna(0.5)

        # 大单检测
        vol_ma = volume.rolling(window=self.period).mean()
        large_trade = (volume > vol_ma * 1.5).astype(int)

        # 机构流向 = 收盘强度 × 大单标记 × 成交量
        inst_flow = (closing_strength - 0.5) * 2 * large_trade * volume

        # 累积流向
        inst_flow_cumulative = inst_flow.cumsum()

        # 机构活跃度
        inst_activity = large_trade.rolling(window=self.period).sum()

        # 机构情绪
        inst_sentiment = inst_flow.rolling(window=self.period).sum() / \
                        (volume.rolling(window=self.period).sum() + 1)

        return pd.DataFrame({
            'Inst_Flow': inst_flow,
            'Inst_Flow_Cumulative': inst_flow_cumulative,
            'Inst_Activity': inst_activity,
            'Inst_Sentiment': inst_sentiment
        })


@indicator_registry.register('sentiment')
class RetailFlow(BaseIndicator):
    """散户资金流"""

    _metadata = IndicatorMetadata(
        name='RetailFlow',
        display_name='散户资金流',
        category='sentiment',
        description='基于开盘行为估算散户资金流向',
        formula='Retail Flow = f(opening_action, chasing_behavior)',
        required_columns=['open', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 14},
        complexity_level=2
    )

    def __init__(self, period: int = 14):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        close = data['close']
        volume = data['volume']

        # 开盘跳空（散户情绪驱动）
        gap = (open_price - close.shift(1)) / close.shift(1)

        # 追涨杀跌行为
        prev_return = close.pct_change().shift(1)
        chasing = gap * np.sign(prev_return)  # 同向追涨杀跌

        # 散户流向
        retail_flow = chasing * volume

        # 累积流向
        retail_flow_cumulative = retail_flow.cumsum()

        # 追高比例
        chase_ratio = (chasing > 0).astype(int).rolling(window=self.period).mean()

        # 散户错误率（逆向指标）
        future_return = close.pct_change().shift(-5)  # 5日后收益
        wrong_direction = ((retail_flow > 0) & (future_return < 0)) | \
                         ((retail_flow < 0) & (future_return > 0))
        retail_error_rate = wrong_direction.astype(int).rolling(window=self.period).mean()

        return pd.DataFrame({
            'Retail_Flow': retail_flow,
            'Retail_Flow_Cumulative': retail_flow_cumulative,
            'Chase_Ratio': chase_ratio,
            'Retail_Error_Rate': retail_error_rate
        })


@indicator_registry.register('sentiment')
class ETFFlow(BaseIndicator):
    """ETF资金流"""

    _metadata = IndicatorMetadata(
        name='ETFFlow',
        display_name='ETF资金流',
        category='sentiment',
        description='基于稳定交易模式估算ETF资金流',
        formula='ETF Flow = f(steady_volume, price_tracking)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 稳定交易量（ETF特征）
        vol_ma = volume.rolling(window=self.period).mean()
        vol_std = volume.rolling(window=self.period).std()
        vol_stability = 1 / (vol_std / vol_ma + 0.01)

        # 价格跟踪度（ETF跟踪指数）
        returns = close.pct_change()
        ret_ma = returns.rolling(window=self.period).mean()
        tracking_error = returns.rolling(window=self.period).std()
        tracking_quality = 1 / (tracking_error + 0.01)

        # ETF流向代理
        etf_flow = vol_stability * tracking_quality * volume * np.sign(ret_ma)

        # 累积流向
        etf_flow_cumulative = etf_flow.cumsum()

        # ETF占比估算
        total_volume = volume.rolling(window=self.period).sum()
        etf_volume_proxy = (vol_stability > vol_stability.rolling(window=self.period).median()).astype(float) * volume
        etf_ratio = etf_volume_proxy.rolling(window=self.period).sum() / total_volume * 100

        return pd.DataFrame({
            'ETF_Flow': etf_flow,
            'ETF_Flow_Cumulative': etf_flow_cumulative,
            'ETF_Ratio': etf_ratio,
            'Tracking_Quality': tracking_quality
        })


@indicator_registry.register('sentiment')
class CrossBorderFlow(BaseIndicator):
    """跨境资金流"""

    _metadata = IndicatorMetadata(
        name='CrossBorderFlow',
        display_name='跨境资金流',
        category='sentiment',
        description='基于时区交易模式估算跨境资金',
        formula='Cross Border Flow = f(overnight_moves, gap_analysis)',
        required_columns=['open', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20},
        complexity_level=2
    )

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        close = data['close']
        volume = data['volume']

        # 隔夜变动（跨时区交易）
        overnight_return = (open_price - close.shift(1)) / close.shift(1)

        # 日内变动
        intraday_return = (close - open_price) / open_price

        # 跨境流向（隔夜变动为主）
        cross_border_flow = overnight_return * volume

        # 累积流向
        cb_flow_cumulative = cross_border_flow.cumsum()

        # 跨境流入强度
        inflow_strength = (overnight_return > 0).astype(int) * abs(overnight_return) * 100
        outflow_strength = (overnight_return < 0).astype(int) * abs(overnight_return) * 100

        # 净流入
        net_cb_flow = (inflow_strength - outflow_strength).rolling(window=self.period).sum()

        # 跨境活跃度
        cb_activity = abs(overnight_return).rolling(window=self.period).sum()

        return pd.DataFrame({
            'CrossBorder_Flow': cross_border_flow,
            'CB_Flow_Cumulative': cb_flow_cumulative,
            'Net_CB_Flow': net_cb_flow,
            'CB_Activity': cb_activity
        })


# ============================================================================
# 另类信号 (4个)
# ============================================================================

@indicator_registry.register('sentiment')
class GoogleTrends(BaseIndicator):
    """谷歌搜索趋势"""

    _metadata = IndicatorMetadata(
        name='GoogleTrends',
        display_name='谷歌搜索趋势',
        category='sentiment',
        description='基于价格关注度模拟搜索趋势',
        formula='Google Trends Proxy = f(price_extremes, volatility, volume)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30},
        complexity_level=2
    )

    def __init__(self, period: int = 30):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 价格极端度（引发搜索）
        price_zscore = abs((close - close.rolling(window=self.period).mean()) /
                          close.rolling(window=self.period).std())

        # 波动率（关注度）
        volatility = returns.rolling(window=14).std() * np.sqrt(252)
        vol_zscore = (volatility - volatility.rolling(window=self.period).mean()) / \
                     volatility.rolling(window=self.period).std()

        # 成交量异常
        vol_ma = volume.rolling(window=self.period).mean()
        vol_spike = volume / vol_ma - 1

        # 搜索趋势指数（0-100）
        search_trend = (price_zscore.clip(0, 3) / 3 * 40 +
                       abs(vol_zscore).clip(0, 3) / 3 * 30 +
                       vol_spike.clip(0, 2) / 2 * 30)

        # 平滑
        search_smooth = search_trend.rolling(window=7).mean()

        # 搜索趋势变化
        trend_change = search_smooth.diff(7)

        # 搜索热度等级
        heat_level = pd.Series(1, index=data.index)
        heat_level[search_smooth > 30] = 2
        heat_level[search_smooth > 50] = 3
        heat_level[search_smooth > 70] = 4
        heat_level[search_smooth > 90] = 5

        return pd.DataFrame({
            'Search_Trend': search_smooth,
            'Trend_Change': trend_change,
            'Heat_Level': heat_level,
            'Price_Component': price_zscore
        })


@indicator_registry.register('sentiment')
class SatelliteData(BaseIndicator):
    """卫星数据指标"""

    _metadata = IndicatorMetadata(
        name='SatelliteData',
        display_name='卫星数据',
        category='sentiment',
        description='基于价格模式模拟卫星监测数据',
        formula='Satellite Data = f(trend_consistency, cycle_detection)',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30, 'cycle_length': 20},
        complexity_level=3
    )

    def __init__(self, period: int = 30, cycle_length: int = 20):
        self.period = period
        self.cycle_length = cycle_length
        super().__init__(period=period, cycle_length=cycle_length)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.cycle_length, 'cycle_length')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']

        # 趋势一致性（模拟供应链/库存监测）
        ma_short = close.rolling(window=10).mean()
        ma_long = close.rolling(window=self.period).mean()
        trend_alignment = ((close > ma_short) & (ma_short > ma_long)).astype(int) - \
                         ((close < ma_short) & (ma_short < ma_long)).astype(int)
        trend_consistency = trend_alignment.rolling(window=self.cycle_length).mean()

        # 周期检测（模拟生产周期）
        detrended = close - ma_long

        # 简单周期强度
        cycle_strength = detrended.rolling(window=self.cycle_length).std() / \
                        close.rolling(window=self.cycle_length).std()

        # 活动指数（模拟物流/运输活动）
        vol_momentum = volume.pct_change().rolling(window=self.cycle_length).mean()
        activity_index = (vol_momentum + 1) * 100

        # 供需平衡代理
        supply_demand = trend_consistency * 50 + cycle_strength * 50

        return pd.DataFrame({
            'Satellite_Index': supply_demand,
            'Trend_Consistency': trend_consistency,
            'Cycle_Strength': cycle_strength,
            'Activity_Index': activity_index
        })


@indicator_registry.register('sentiment')
class WeatherImpact(BaseIndicator):
    """天气影响指标"""

    _metadata = IndicatorMetadata(
        name='WeatherImpact',
        display_name='天气影响',
        category='sentiment',
        description='基于季节性模式模拟天气影响',
        formula='Weather Impact = f(seasonality, cyclical_patterns)',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 30, 'seasonal_window': 90},
        complexity_level=2
    )

    def __init__(self, period: int = 30, seasonal_window: int = 90):
        self.period = period
        self.seasonal_window = seasonal_window
        super().__init__(period=period, seasonal_window=seasonal_window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.seasonal_window, 'seasonal_window')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']

        # 季节性成分
        seasonal_ma = close.rolling(window=self.seasonal_window).mean()
        seasonal_dev = (close - seasonal_ma) / seasonal_ma * 100

        # 波动周期
        returns = close.pct_change()
        volatility = returns.rolling(window=self.period).std()
        vol_seasonal = volatility - volatility.rolling(window=self.seasonal_window).mean()

        # 天气影响代理（季节性异常）
        weather_impact = seasonal_dev * 0.6 + vol_seasonal * 100 * 0.4

        # 影响强度
        impact_strength = abs(weather_impact).rolling(window=14).mean()

        # 异常天气检测
        anomaly = (abs(weather_impact) > abs(weather_impact).rolling(window=self.period).quantile(0.9)).astype(int)

        # 季节趋势
        seasonal_trend = seasonal_dev - seasonal_dev.shift(self.period)

        return pd.DataFrame({
            'Weather_Impact': weather_impact,
            'Impact_Strength': impact_strength,
            'Weather_Anomaly': anomaly,
            'Seasonal_Trend': seasonal_trend
        })


@indicator_registry.register('sentiment')
class GeopoliticalRisk(BaseIndicator):
    """地缘政治风险"""

    _metadata = IndicatorMetadata(
        name='GeopoliticalRisk',
        display_name='地缘政治风险',
        category='sentiment',
        description='基于市场不确定性测量地缘政治风险',
        formula='Geopolitical Risk = f(volatility_spike, gap_frequency, uncertainty)',
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        output_type='dataframe',
        default_params={'period': 30, 'risk_window': 60},
        complexity_level=3
    )

    def __init__(self, period: int = 30, risk_window: int = 60):
        self.period = period
        self.risk_window = risk_window
        super().__init__(period=period, risk_window=risk_window)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.risk_window, 'risk_window')

    @property
    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 波动率突增（风险事件）
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        vol_baseline = volatility.rolling(window=self.risk_window).median()
        vol_spike = (volatility / vol_baseline - 1) * 100

        # 跳空频率（政治事件）
        gap = abs((open_price - close.shift(1)) / close.shift(1))
        gap_frequency = (gap > gap.rolling(window=self.risk_window).quantile(0.9)).astype(int)
        gap_count = gap_frequency.rolling(window=self.period).sum()

        # 不确定性（高低差异）
        range_pct = (high - low) / close * 100
        uncertainty = range_pct.rolling(window=self.period).mean()

        # 避险情绪（成交量 + 波动率）
        safe_haven = (volume > volume.rolling(window=self.period).mean()).astype(int) * \
                     (volatility > vol_baseline).astype(int)
        safe_haven_index = safe_haven.rolling(window=self.period).sum()

        # 地缘风险指数
        geo_risk = (
            vol_spike.clip(0, 100) * 0.4 +
            gap_count * 5 * 0.3 +
            uncertainty * 3 * 0.3
        )

        # 风险等级
        risk_level = pd.Series('Low', index=data.index)
        risk_level[geo_risk > 30] = 'Medium'
        risk_level[geo_risk > 60] = 'High'
        risk_level[geo_risk > 90] = 'Critical'

        return pd.DataFrame({
            'Geopolitical_Risk': geo_risk,
            'Volatility_Spike': vol_spike,
            'Gap_Frequency': gap_count,
            'Uncertainty': uncertainty,
            'Safe_Haven_Index': safe_haven_index
        })
