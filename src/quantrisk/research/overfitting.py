"""
多重检验与回测过拟合的度量

- probabilistic_sharpe_ratio：观测到的夏普比率超过某基准值的概率，考虑样本长度、偏度与峰度
  （Bailey & López de Prado, 2012, "The Sharpe Ratio Efficient Frontier"）；
- deflated_sharpe_ratio：以 N 次试验下“纯运气能得到的最大夏普”的期望为基准的 PSR
  （Bailey & López de Prado, 2014, "The Deflated Sharpe Ratio"）。试验次数必须如实计入所有评估过的配置；
- pbo_cscv：组合对称交叉验证（CSCV）估计回测过拟合概率 PBO——在样本内选出的最优配置，在样本外排名落到
  中位数以下的概率（Bailey, Borwein, López de Prado & Zhu, 2017, "The Probability of Backtest Overfitting"）。

夏普比率在本模块内一律按“每期”（日频，不年化）计算；年化只用于展示。
"""

import itertools
import math
from statistics import NormalDist
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def per_period_sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def _moments(returns):
    r = np.asarray(returns, dtype=float)
    m, sd = r.mean(), r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    z = (r - m) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())          # 偏度、峰度（非超额，正态为 3）


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    t = len(r)
    if t < 3:
        raise ValueError("样本太短")
    sr = per_period_sharpe(r)
    skew, kurt = _moments(r)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        raise ValueError("偏度/峰度导致方差估计非正")
    return _N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """N 次独立试验、真实夏普均为 0 时，最大样本夏普的近似期望（每期口径）。"""
    if n_trials < 1:
        raise ValueError("试验次数至少为 1")
    if n_trials == 1:
        return 0.0
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                     + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def sampling_sharpe_variance(observations: int) -> float:
    """真实夏普为 0 时，T 个观测的每期样本夏普的近似方差 1/(T-1)。"""
    return 1.0 / (observations - 1)


def deflated_sharpe_ratio(returns, trial_sharpes: Sequence[float], n_trials: Optional[int] = None,
                          sr_variance: Optional[float] = None) -> Dict[str, float]:
    """
    returns：被选中配置的日收益；trial_sharpes：全部试验的每期夏普；n_trials：有效试验次数，默认 len(trial_sharpes)。
    sr_variance：零假设下夏普估计的方差。默认用跨试验方差（原文做法）；试验之间高度相关（同一股票池、同一段行情）
    时它会低估运气成分，可传入 sampling_sharpe_variance(T) 作为保守口径。
    """
    n = n_trials or len(trial_sharpes)
    if sr_variance is not None:
        var = float(sr_variance)
    else:
        var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr0 = expected_max_sharpe(n, var)
    sr = per_period_sharpe(returns)
    skew, kurt = _moments(returns)
    return {"dsr": probabilistic_sharpe_ratio(returns, sr0), "psr_vs_zero": probabilistic_sharpe_ratio(returns, 0.0),
            "sharpe_per_period": sr, "sharpe_annualized": sr * math.sqrt(252),
            "expected_max_sharpe_per_period": sr0, "expected_max_sharpe_annualized": sr0 * math.sqrt(252),
            "n_trials": n, "trial_sharpe_variance": var, "skew": skew, "kurtosis": kurt,
            "observations": int(len(np.asarray(returns)))}


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> Dict[str, object]:
    """
    returns：T × N 的日收益矩阵（每列一个配置，行按时间排序，无缺失）。把行按时间切成 n_blocks 段，
    取全部 C(n_blocks, n_blocks/2) 种样本内/样本外划分；每种划分在样本内按夏普选出最优列，记录它在样本外的相对排名 ω，
    logit λ = ln(ω / (1-ω))。PBO = P(λ ≤ 0)。
    """
    m = np.asarray(returns, dtype=float)
    t, n = m.shape
    if n < 2 or n_blocks % 2 or n_blocks < 2 or t < 2 * n_blocks:
        raise ValueError("需要至少 2 个配置、偶数段数，且每段至少 2 个观测")
    if np.isnan(m).any():
        raise ValueError("收益矩阵含缺失值")
    blocks = np.array_split(np.arange(t), n_blocks)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.array([m[b].sum(axis=0) for b in blocks])
    s2 = np.array([(m[b] ** 2).sum(axis=0) for b in blocks])

    def sharpe_of(sel):
        k, a, b = cnt[sel].sum(), s1[sel].sum(axis=0), s2[sel].sum(axis=0)
        mean = a / k
        var = (b - k * mean ** 2) / (k - 1)
        sd = np.sqrt(np.clip(var, 0, None))
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1), 0.0)

    lambdas, oos_sharpe_of_best, is_best = [], [], []
    all_idx = np.arange(n_blocks)
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.array(combo)
        oos = np.setdiff1d(all_idx, ins)
        sr_in, sr_out = sharpe_of(ins), sharpe_of(oos)
        best = int(np.argmax(sr_in))
        v = sr_out[best]
        rank = (sr_out < v).sum() + 1 + 0.5 * ((sr_out == v).sum() - 1)       # 1..N，并列取平均
        w = rank / (n + 1)
        lambdas.append(math.log(w / (1 - w)))
        oos_sharpe_of_best.append(v)
        is_best.append(sr_in[best])
    lam = np.array(lambdas)
    return {"pbo": float((lam <= 0).mean()), "combinations": int(len(lam)), "n_configs": int(n),
            "n_blocks": n_blocks, "logit_median": float(np.median(lam)),
            "oos_sharpe_of_is_best_median_annualized": float(np.median(oos_sharpe_of_best) * math.sqrt(252)),
            "is_sharpe_of_is_best_median_annualized": float(np.median(is_best) * math.sqrt(252)),
            "prob_oos_loss": float((np.array(oos_sharpe_of_best) < 0).mean())}
