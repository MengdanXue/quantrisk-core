"""
情绪指标 (15个)

基于市场情绪分析的技术指标
"""

from typing import List
import pandas as pd
import numpy as np

from ..base.indicator_base import BaseIndicator, IndicatorMetadata
from ..base.indicator_validator import IndicatorValidator
from ..registry.indicator_registry import indicator_registry


@indicator_registry.register('sentiment')
class FearGreedIndex(BaseIndicator):
    """恐惧贪婪指数"""

    _metadata = IndicatorMetadata(
        name='FearGreedIndex',
        display_name='恐惧贪婪指数',
        category='sentiment',
        description='综合多个指标的市场情绪指数',
        formula='FGI = weighted average of momentum, volatility, volume',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # 动量成分
        momentum = (close - close.shift(self.period)) / close.shift(self.period)
        momentum_score = momentum.rolling(window=self.period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )

        # 波动率成分
        returns = close.pct_change()
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        vol_score = 100 - volatility.rolling(window=self.period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )

        # 成交量成分
        vol_ma = volume.rolling(window=self.period).mean()
        vol_ratio = volume / vol_ma
        volume_score = vol_ratio.rolling(window=self.period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )

        # 价格强度
        price_strength = (close - low.rolling(window=self.period).min()) / \
                        (high.rolling(window=self.period).max() - low.rolling(window=self.period).min())
        strength_score = price_strength * 100

        # 综合指数
        fgi = (momentum_score * 0.3 + vol_score * 0.2 + volume_score * 0.2 + strength_score * 0.3)

        # 情绪分类
        sentiment = pd.Series('Neutral', index=data.index)
        sentiment[fgi < 25] = 'Extreme Fear'
        sentiment[(fgi >= 25) & (fgi < 45)] = 'Fear'
        sentiment[(fgi >= 55) & (fgi < 75)] = 'Greed'
        sentiment[fgi >= 75] = 'Extreme Greed'

        return pd.DataFrame({
            'FGI': fgi,
            'Momentum_Score': momentum_score,
            'Volatility_Score': vol_score,
            'Volume_Score': volume_score,
            'Strength_Score': strength_score
        })


@indicator_registry.register('sentiment')
class PutCallRatioProxy(BaseIndicator):
    """看跌看涨比率代理"""

    _metadata = IndicatorMetadata(
        name='PutCallRatioProxy',
        display_name='PCR代理指标',
        category='sentiment',
        description='使用价格和波动率模拟PCR情绪',
        formula='PCR Proxy based on price momentum and volatility',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        returns = close.pct_change()

        # 下跌天数比例（代理看空情绪）
        down_days = (returns < 0).rolling(window=self.period).sum()
        up_days = (returns > 0).rolling(window=self.period).sum()
        pcr_proxy = down_days / up_days.replace(0, np.nan)

        # 下跌幅度占比
        neg_returns = returns.where(returns < 0, 0).abs()
        pos_returns = returns.where(returns > 0, 0)
        magnitude_ratio = neg_returns.rolling(window=self.period).sum() / \
                         pos_returns.rolling(window=self.period).sum().replace(0, np.nan)

        # 组合指标
        combined = (pcr_proxy + magnitude_ratio) / 2

        # 信号
        signal = pd.Series(0, index=data.index)
        signal[combined > 1.5] = 1  # 极度悲观 -> 潜在底部
        signal[combined < 0.5] = -1  # 极度乐观 -> 潜在顶部

        return pd.DataFrame({
            'PCR_Proxy': pcr_proxy,
            'Magnitude_Ratio': magnitude_ratio,
            'Combined_Sentiment': combined,
            'Contrarian_Signal': signal
        })


@indicator_registry.register('sentiment')
class MarketBreadth(BaseIndicator):
    """市场广度指标"""

    _metadata = IndicatorMetadata(
        name='MarketBreadth',
        display_name='市场广度',
        category='sentiment',
        description='衡量上涨下跌的市场广度',
        formula='Breadth = Advance-Decline Analysis',
        required_columns=['close'],
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
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        returns = close.pct_change()

        # 上涨下跌统计
        advances = (returns > 0).astype(int)
        declines = (returns < 0).astype(int)

        # 累积线
        ad_line = (advances - declines).cumsum()

        # 广度比率
        adv_sum = advances.rolling(window=self.period).sum()
        dec_sum = declines.rolling(window=self.period).sum()
        breadth_ratio = adv_sum / (adv_sum + dec_sum)

        # McClellan振荡器代理
        ema_19 = ad_line.ewm(span=19, adjust=False).mean()
        ema_39 = ad_line.ewm(span=39, adjust=False).mean()
        mcclellan = ema_19 - ema_39

        return pd.DataFrame({
            'AD_Line': ad_line,
            'Breadth_Ratio': breadth_ratio,
            'McClellan_Proxy': mcclellan
        })


@indicator_registry.register('sentiment')
class SmartMoneyIndex(BaseIndicator):
    """聪明钱指数"""

    _metadata = IndicatorMetadata(
        name='SmartMoneyIndex',
        display_name='聪明钱指数',
        category='sentiment',
        description='基于开盘和收盘行为判断机构资金流向',
        formula='SMI = Yesterday_Close - Today_Open + Today_Close',
        required_columns=['open', 'close'],
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
        return ['open', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        close = data['close']

        # 聪明钱流向
        # 开盘跳空（散户情绪）
        opening_gap = open_price - close.shift(1)

        # 日内变化（机构行为）
        intraday = close - open_price

        # SMI = 累积（收盘变化 - 开盘跳空）
        smi = (intraday - opening_gap).cumsum()

        # 归一化
        smi_ma = smi.rolling(window=self.period).mean()
        smi_std = smi.rolling(window=self.period).std()
        smi_z = (smi - smi_ma) / smi_std

        # 背离检测
        close_ma = close.rolling(window=self.period).mean()
        price_trend = (close > close_ma).astype(int)
        smi_trend = (smi > smi_ma).astype(int)
        divergence = (price_trend != smi_trend).astype(int)

        return pd.DataFrame({
            'SMI': smi,
            'SMI_ZScore': smi_z,
            'SMI_Divergence': divergence
        })


@indicator_registry.register('sentiment')
class ArmsIndex(BaseIndicator):
    """Arms指数(TRIN)"""

    _metadata = IndicatorMetadata(
        name='ArmsIndex',
        display_name='Arms指数',
        category='sentiment',
        description='Richard Arms开发的交易指数',
        formula='TRIN = (Adv/Dec) / (Adv Vol/Dec Vol)',
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

        # 上涨/下跌成交量
        up_volume = volume.where(returns > 0, 0)
        down_volume = volume.where(returns < 0, 0)

        # 上涨/下跌天数
        up_days = (returns > 0).astype(int)
        down_days = (returns < 0).astype(int)

        # 滚动统计
        adv = up_days.rolling(window=self.period).sum()
        dec = down_days.rolling(window=self.period).sum()
        adv_vol = up_volume.rolling(window=self.period).sum()
        dec_vol = down_volume.rolling(window=self.period).sum()

        # TRIN
        ad_ratio = adv / dec.replace(0, np.nan)
        vol_ratio = adv_vol / dec_vol.replace(0, np.nan)
        trin = ad_ratio / vol_ratio.replace(0, np.nan)

        # 解读
        signal = pd.Series(0, index=data.index)
        signal[trin > 2] = 1  # 超卖
        signal[trin < 0.5] = -1  # 超买

        return pd.DataFrame({
            'TRIN': trin,
            'AD_Ratio': ad_ratio,
            'Vol_Ratio': vol_ratio,
            'TRIN_Signal': signal
        })


@indicator_registry.register('sentiment')
class VIXProxy(BaseIndicator):
    """VIX代理指标"""

    _metadata = IndicatorMetadata(
        name='VIXProxy',
        display_name='VIX代理',
        category='sentiment',
        description='使用历史波动率模拟恐慌指数',
        formula='VIX Proxy = Normalized Historical Volatility',
        required_columns=['close'],
        output_type='dataframe',
        default_params={'period': 20, 'lookback': 252},
        complexity_level=2
    )

    def __init__(self, period: int = 20, lookback: int = 252):
        self.period = period
        self.lookback = lookback
        super().__init__(period=period, lookback=lookback)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.lookback, 'lookback')

    @property
    def required_columns(self) -> List[str]:
        return ['close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        returns = close.pct_change()

        # 历史波动率
        hv = returns.rolling(window=self.period).std() * np.sqrt(252) * 100

        # 波动率的百分位
        def calc_percentile(x):
            return (x.rank() / len(x) * 100).iloc[-1]

        hv_percentile = hv.rolling(window=self.lookback).apply(calc_percentile, raw=False)

        # 波动率期限结构
        hv_short = returns.rolling(window=10).std() * np.sqrt(252) * 100
        hv_long = returns.rolling(window=30).std() * np.sqrt(252) * 100
        term_structure = hv_short / hv_long

        # 恐慌程度
        fear_level = pd.Series('Normal', index=data.index)
        fear_level[hv_percentile > 80] = 'High Fear'
        fear_level[hv_percentile > 95] = 'Extreme Fear'
        fear_level[hv_percentile < 20] = 'Complacency'

        return pd.DataFrame({
            'VIX_Proxy': hv,
            'VIX_Percentile': hv_percentile,
            'Term_Structure': term_structure
        })


@indicator_registry.register('sentiment')
class HighLowIndex(BaseIndicator):
    """新高新低指数"""

    _metadata = IndicatorMetadata(
        name='HighLowIndex',
        display_name='新高新低指数',
        category='sentiment',
        description='统计创新高和新低的比例',
        formula='HLI = New Highs / (New Highs + New Lows)',
        required_columns=['high', 'low', 'close'],
        output_type='dataframe',
        default_params={'period': 52},
        complexity_level=1
    )

    def __init__(self, period: int = 52):
        self.period = period
        super().__init__(period=period)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')

    @property
    def required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']

        # 滚动最高和最低
        rolling_high = high.rolling(window=self.period).max()
        rolling_low = low.rolling(window=self.period).min()

        # 新高新低标记
        new_high = (high >= rolling_high).astype(int)
        new_low = (low <= rolling_low).astype(int)

        # 累积统计
        new_high_count = new_high.rolling(window=20).sum()
        new_low_count = new_low.rolling(window=20).sum()

        # 指数
        total = new_high_count + new_low_count
        hli = new_high_count / total.replace(0, np.nan)

        # 10日平滑
        hli_smooth = hli.rolling(window=10).mean()

        return pd.DataFrame({
            'New_High': new_high,
            'New_Low': new_low,
            'HLI': hli,
            'HLI_Smooth': hli_smooth
        })


@indicator_registry.register('sentiment')
class SentimentZone(BaseIndicator):
    """情绪区间"""

    _metadata = IndicatorMetadata(
        name='SentimentZone',
        display_name='情绪区间',
        category='sentiment',
        description='基于多指标的情绪区间判断',
        formula='Sentiment Zone Classification',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # 布林带位置
        ma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        bb_position = (close - ma) / (2 * std)

        # 成交量情绪
        vol_ma = volume.rolling(window=self.period).mean()
        vol_sentiment = volume / vol_ma - 1

        # 综合情绪得分
        sentiment_score = (rsi - 50) / 50 * 0.4 + bb_position * 0.3 + vol_sentiment.clip(-1, 1) * 0.3

        # 情绪区间
        zone = pd.Series(0, index=data.index)
        zone[sentiment_score > 0.5] = 2  # 极度乐观
        zone[(sentiment_score > 0.2) & (sentiment_score <= 0.5)] = 1  # 乐观
        zone[(sentiment_score < -0.2) & (sentiment_score >= -0.5)] = -1  # 悲观
        zone[sentiment_score < -0.5] = -2  # 极度悲观

        return pd.DataFrame({
            'Sentiment_Score': sentiment_score,
            'Sentiment_Zone': zone,
            'RSI_Component': rsi,
            'BB_Component': bb_position
        })


@indicator_registry.register('sentiment')
class CrowdSentiment(BaseIndicator):
    """群体情绪"""

    _metadata = IndicatorMetadata(
        name='CrowdSentiment',
        display_name='群体情绪',
        category='sentiment',
        description='基于价格动作推断群体情绪',
        formula='Crowd Sentiment Analysis',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 买入压力
        buying_pressure = (close - low) / (high - low)
        buying_pressure = buying_pressure.fillna(0.5)

        # 卖出压力
        selling_pressure = (high - close) / (high - low)
        selling_pressure = selling_pressure.fillna(0.5)

        # 成交量加权情绪
        vol_weighted_bp = (buying_pressure * volume).rolling(window=self.period).sum() / \
                         volume.rolling(window=self.period).sum()

        vol_weighted_sp = (selling_pressure * volume).rolling(window=self.period).sum() / \
                         volume.rolling(window=self.period).sum()

        # 情绪比率
        sentiment_ratio = vol_weighted_bp / vol_weighted_sp.replace(0, np.nan)

        # 群体行为指标
        crowd_sentiment = (sentiment_ratio - 1) / (sentiment_ratio + 1) * 100

        # 极端情绪检测
        extreme = pd.Series(0, index=data.index)
        extreme[crowd_sentiment > 40] = 1  # 极度乐观
        extreme[crowd_sentiment < -40] = -1  # 极度悲观

        return pd.DataFrame({
            'Buying_Pressure': vol_weighted_bp,
            'Selling_Pressure': vol_weighted_sp,
            'Crowd_Sentiment': crowd_sentiment,
            'Extreme_Sentiment': extreme
        })


@indicator_registry.register('sentiment')
class ContrarianIndicator(BaseIndicator):
    """逆向情绪指标"""

    _metadata = IndicatorMetadata(
        name='ContrarianIndicator',
        display_name='逆向情绪',
        category='sentiment',
        description='用于逆向操作的极端情绪检测',
        formula='Contrarian Signal at Extremes',
        required_columns=['close', 'volume'],
        output_type='dataframe',
        default_params={'period': 20, 'lookback': 100},
        complexity_level=2
    )

    def __init__(self, period: int = 20, lookback: int = 100):
        self.period = period
        self.lookback = lookback
        super().__init__(period=period, lookback=lookback)

    def _validate_params(self):
        IndicatorValidator.validate_positive_int(self.period, 'period')
        IndicatorValidator.validate_positive_int(self.lookback, 'lookback')

    @property
    def required_columns(self) -> List[str]:
        return ['close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()

        # 动量极端
        momentum = close / close.shift(self.period) - 1
        mom_percentile = momentum.rolling(window=self.lookback).apply(
            lambda x: (x.rank() / len(x) * 100).iloc[-1]
        )

        # 成交量极端
        vol_ma = volume.rolling(window=self.period).mean()
        vol_ratio = volume / vol_ma
        vol_percentile = vol_ratio.rolling(window=self.lookback).apply(
            lambda x: (x.rank() / len(x) * 100).iloc[-1]
        )

        # 波动率极端
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        vol_extreme = volatility.rolling(window=self.lookback).apply(
            lambda x: (x.rank() / len(x) * 100).iloc[-1]
        )

        # 综合逆向得分
        contrarian_score = (
            (100 - mom_percentile) * 0.4 +  # 高动量 -> 卖出信号
            (100 - vol_percentile) * 0.3 +  # 高成交量 -> 反转信号
            vol_extreme * 0.3  # 高波动 -> 潜在转折
        )

        # 逆向信号
        signal = pd.Series(0, index=data.index)
        signal[(contrarian_score > 70) & (mom_percentile < 20)] = 1  # 买入
        signal[(contrarian_score > 70) & (mom_percentile > 80)] = -1  # 卖出

        return pd.DataFrame({
            'Momentum_Percentile': mom_percentile,
            'Volume_Percentile': vol_percentile,
            'Volatility_Percentile': vol_extreme,
            'Contrarian_Score': contrarian_score,
            'Contrarian_Signal': signal
        })


@indicator_registry.register('sentiment')
class InstitutionalFlow(BaseIndicator):
    """机构资金流向"""

    _metadata = IndicatorMetadata(
        name='InstitutionalFlow',
        display_name='机构资金流',
        category='sentiment',
        description='推断机构资金流入流出',
        formula='Institutional Flow Estimation',
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

        # 大单推测（基于日内波动和成交量）
        range_pct = (high - low) / open_price
        vol_impact = range_pct / (volume.rolling(window=self.period).mean() / volume)

        # 方向性
        direction = np.sign(close - open_price)

        # 机构流量
        inst_flow = direction * vol_impact * volume
        inst_flow_cum = inst_flow.cumsum()

        # 平滑
        inst_flow_ma = inst_flow.rolling(window=self.period).mean()

        # 与价格的背离
        close_ma = close.rolling(window=self.period).mean()
        price_trend = (close > close_ma).astype(int)
        flow_trend = (inst_flow_ma > 0).astype(int)
        divergence = (price_trend != flow_trend).astype(int)

        return pd.DataFrame({
            'Inst_Flow': inst_flow,
            'Inst_Flow_Cum': inst_flow_cum,
            'Inst_Flow_MA': inst_flow_ma,
            'Flow_Divergence': divergence
        })


@indicator_registry.register('sentiment')
class MomentumExhaustion(BaseIndicator):
    """动量耗尽"""

    _metadata = IndicatorMetadata(
        name='MomentumExhaustion',
        display_name='动量耗尽',
        category='sentiment',
        description='检测趋势动量耗尽的信号',
        formula='Momentum Exhaustion Detection',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 价格动量
        price_mom = close - close.shift(self.period)
        price_mom_norm = price_mom / close.shift(self.period) * 100

        # 动量加速度
        mom_accel = price_mom_norm.diff()

        # 成交量确认
        vol_ma = volume.rolling(window=self.period).mean()
        vol_trend = (volume > vol_ma).astype(int)

        # 动量与成交量背离
        price_up = (price_mom > 0).astype(int)
        vol_up = (volume > volume.shift(1)).astype(int)

        # 耗尽信号
        # 价格创新高但动量减弱
        exhaustion_bull = ((high == high.rolling(window=self.period).max()) &
                          (mom_accel < 0) &
                          (volume < vol_ma)).astype(int)

        # 价格创新低但动量减弱
        exhaustion_bear = ((low == low.rolling(window=self.period).min()) &
                          (mom_accel > 0) &
                          (volume < vol_ma)).astype(int)

        exhaustion = exhaustion_bull - exhaustion_bear

        return pd.DataFrame({
            'Momentum': price_mom_norm,
            'Momentum_Accel': mom_accel,
            'Volume_Trend': vol_trend,
            'Exhaustion_Signal': exhaustion
        })


@indicator_registry.register('sentiment')
class RetailSentiment(BaseIndicator):
    """散户情绪"""

    _metadata = IndicatorMetadata(
        name='RetailSentiment',
        display_name='散户情绪',
        category='sentiment',
        description='推断散户交易情绪',
        formula='Retail Sentiment Estimation',
        required_columns=['open', 'close', 'volume'],
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
        return ['open', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        open_price = data['open']
        close = data['close']
        volume = data['volume']

        # 开盘情绪（散户倾向在开盘交易）
        opening_action = open_price - close.shift(1)
        opening_sentiment = opening_action / close.shift(1) * 100

        # 散户跟风指标
        # 假设散户在大涨后买入，大跌后卖出
        prev_return = close.pct_change().shift(1)
        chasing = prev_return.rolling(window=3).mean() * 100

        # 追涨杀跌程度
        chase_intensity = abs(chasing).rolling(window=self.period).mean()

        # 散户情绪综合
        retail_sentiment = opening_sentiment.rolling(window=self.period).mean() + chasing

        # 逆向信号（散户极端时反向操作）
        signal = pd.Series(0, index=data.index)
        signal[retail_sentiment > retail_sentiment.rolling(window=60).quantile(0.9)] = -1  # 散户极度乐观
        signal[retail_sentiment < retail_sentiment.rolling(window=60).quantile(0.1)] = 1  # 散户极度悲观

        return pd.DataFrame({
            'Opening_Sentiment': opening_sentiment,
            'Chasing_Behavior': chasing,
            'Retail_Sentiment': retail_sentiment,
            'Contrarian_Signal': signal
        })


@indicator_registry.register('sentiment')
class SentimentDivergence(BaseIndicator):
    """情绪背离"""

    _metadata = IndicatorMetadata(
        name='SentimentDivergence',
        display_name='情绪背离',
        category='sentiment',
        description='检测价格与情绪的背离',
        formula='Sentiment vs Price Divergence',
        required_columns=['high', 'low', 'close', 'volume'],
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
        return ['high', 'low', 'close', 'volume']

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 价格趋势
        price_trend = (close - close.shift(self.period)) / close.shift(self.period) * 100

        # 成交量趋势
        vol_trend = (volume - volume.rolling(window=self.period).mean()) / \
                   volume.rolling(window=self.period).mean() * 100

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_trend = rsi.diff(self.period)

        # 背离检测
        # 看涨背离：价格新低但RSI未新低
        price_lower = close < close.shift(self.period)
        rsi_higher = rsi > rsi.shift(self.period)
        bullish_div = (price_lower & rsi_higher).astype(int)

        # 看跌背离：价格新高但RSI未新高
        price_higher = close > close.shift(self.period)
        rsi_lower = rsi < rsi.shift(self.period)
        bearish_div = (price_higher & rsi_lower).astype(int)

        divergence = bullish_div - bearish_div

        return pd.DataFrame({
            'Price_Trend': price_trend,
            'RSI': rsi,
            'RSI_Trend': rsi_trend,
            'Divergence': divergence,
            'Bullish_Div': bullish_div,
            'Bearish_Div': bearish_div
        })
