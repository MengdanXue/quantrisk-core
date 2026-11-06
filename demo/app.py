"""Local demo: run indicators and strategy backtests on synthetic prices in the browser.

    uvicorn demo.app:app --port 8765
"""
import importlib
import logging
import math
import pkgutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import quantrisk.indicators
from quantrisk.indicators.registry import indicator_registry
from quantrisk.strategies import STRATEGY_REGISTRY, get_strategy

warnings.filterwarnings("ignore")
logging.getLogger("quantrisk").setLevel(logging.WARNING)

# 指标通过装饰器注册，需先导入全部子模块
for _mod in pkgutil.walk_packages(quantrisk.indicators.__path__, "quantrisk.indicators."):
    importlib.import_module(_mod.name)

# advanced_pairs 的构造函数签名与注册表不一致，见 README「已知问题」
DEMO_STRATEGIES = [name for name in STRATEGY_REGISTRY if name != "advanced_pairs"]
OHLCV = {"open", "high", "low", "close", "volume"}

app = FastAPI(title="quantrisk-core demo")
STATIC = Path(__file__).parent / "static"


def synthetic_prices(days: int, seed: int, drift: float, vol: float) -> pd.DataFrame:
    """几何布朗运动生成的日线 OHLCV，仅供演示。"""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, days)))
    open_ = np.r_[100.0, close[:-1]] * (1 + rng.normal(0, vol / 4, days))
    spread = np.abs(rng.normal(0, vol, days)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100_000, 1_000_000, days).astype(float)
    index = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def _clean(values):
    return [None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), 6) for v in values]


class DataSpec(BaseModel):
    days: int = Field(500, ge=60, le=3000)
    seed: int = 42
    drift: float = Field(0.0003, ge=-0.01, le=0.01)
    vol: float = Field(0.015, gt=0, le=0.1)


class BacktestRequest(DataSpec):
    strategy: str = "ma_cross"
    initial_capital: float = Field(1_000_000, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    position_pct: float = Field(0.5, gt=0, le=1.0)


class IndicatorRequest(DataSpec):
    name: str = "RSI"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/strategies")
def strategies():
    return [
        {"name": n, "description": (STRATEGY_REGISTRY[n].__doc__ or "").strip().split("\n")[0]}
        for n in DEMO_STRATEGIES
    ]


@app.get("/api/indicators")
def indicators():
    """只列出仅需 OHLCV 即可计算的指标。"""
    items = []
    for name in sorted(indicator_registry.list_all()):
        meta = indicator_registry.get_metadata(name)
        needed = set((meta.required_inputs or meta.required_columns or []) if meta else [])
        if needed <= OHLCV:
            items.append({"name": name, "display_name": meta.display_name if meta else name,
                          "category": meta.category if meta else ""})
    return {"total_registered": indicator_registry.get_total_count(), "ohlcv_only": items}


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    if req.strategy not in DEMO_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {req.strategy}")
    data = synthetic_prices(req.days, req.seed, req.drift, req.vol)
    result = get_strategy(req.strategy, {"position_pct": req.position_pct}).run(data, initial_capital=req.initial_capital, commission=req.commission)
    benchmark = data["close"] / data["close"].iloc[0] * req.initial_capital
    metrics = {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) else None)
               for k, v in result.metrics.items()}
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in data.index],
        "close": _clean(data["close"]),
        "equity": _clean(result.equity_curve),
        "benchmark": _clean(benchmark),
        "metrics": metrics,
        "trades": result.trades[-50:],
        "trade_count": len(result.trades),
    }


@app.post("/api/indicator")
def indicator(req: IndicatorRequest):
    data = synthetic_prices(req.days, req.seed, req.drift, req.vol)
    try:
        out = indicator_registry.create_indicator(req.name).calculate(data)
    except Exception as exc:  # 指标缺少输入列等
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")
    if isinstance(out, pd.Series):
        out = out.to_frame(req.name)
    if not isinstance(out, pd.DataFrame):
        raise HTTPException(400, f"indicator returned {type(out).__name__}, not a series")
    numeric = out.select_dtypes("number").iloc[:, :4]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in data.index],
        "close": _clean(data["close"]),
        "series": {str(c): _clean(numeric[c].reindex(data.index)) for c in numeric.columns},
    }
