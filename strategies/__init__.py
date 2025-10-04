from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import pandas as pd

from .base import StrategyDefinition


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def slope_pct(series: pd.Series, lookback: int) -> pd.Series:
    ref = series.shift(lookback)
    return (series.sub(ref)).div(ref).mul(100.0)


def _ema_prepare(df: pd.DataFrame, _params: Dict[str, Any]) -> pd.DataFrame:
    return df


def _ema_generate_signals(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    ema_period = int(params.get("ema_period", 9))
    slope_lookback = int(params.get("slope_lookback", 5))
    min_slope_percent = float(params.get("min_slope_percent", 0.1))

    signals_df = df.copy()
    signals_df["EMA"] = ema(signals_df["Close"], ema_period)
    signals_df["SlopePct"] = slope_pct(signals_df["EMA"], slope_lookback).fillna(0.0)
    signals_df["trend_up"] = signals_df["SlopePct"] > min_slope_percent
    signals_df["trend_dn"] = signals_df["SlopePct"] < -min_slope_percent
    signals_df["prev_close"] = signals_df["Close"].shift(1)
    signals_df["prev_ema"] = signals_df["EMA"].shift(1)
    signals_df["bull_cross"] = (signals_df["prev_close"] <= signals_df["prev_ema"]) & (
        signals_df["Close"] > signals_df["EMA"]
    )
    signals_df["bear_cross"] = (signals_df["prev_close"] >= signals_df["prev_ema"]) & (
        signals_df["Close"] < signals_df["EMA"]
    )
    return signals_df


def _ema_build_orders(
    _df: pd.DataFrame,
    _signals: pd.DataFrame,
    _params: Dict[str, Any],
) -> Dict[str, Any]:
    return {}


EMA_STRATEGY = StrategyDefinition(
    key="ema_trend",
    name="EMA Trend",
    description="Trend-following strategy using EMA crossovers and slope filters.",
    controls=OrderedDict(
        {
            "ema_period": dict(label="EMA period", dtype=int, min=1, max=200, value=9, step=1),
            "slope_lookback": dict(label="Slope lookback (bars)", dtype=int, min=1, max=200, value=5, step=1),
            "min_slope_percent": dict(
                label="Min slope %", dtype=float, min=0.0, max=100.0, value=0.1, step=0.1, format="%.2f"
            ),
            "stop_loss_percent": dict(
                label="Stop loss %", dtype=float, min=0.0, max=50.0, value=2.0, step=0.1, format="%.2f"
            ),
            "trail_percent": dict(
                label="Trailing stop %", dtype=float, min=0.0, max=50.0, value=1.5, step=0.1, format="%.2f"
            ),
            "fee_percent": dict(
                label="Fee % per fill", dtype=float, min=0.0, max=2.0, value=0.26, step=0.01, format="%.4f"
            ),
            "slippage_percent": dict(
                label="Slippage % per fill", dtype=float, min=0.0, max=2.0, value=0.05, step=0.01, format="%.4f"
            ),
            "max_leverage": dict(
                label="Max leverage", dtype=float, min=1.0, max=10.0, value=5.0, step=0.5, format="%.2f"
            ),
            "risk_fraction": dict(
                label="Risk fraction of equity", dtype=float, min=0.01, max=1.0, value=0.25, step=0.01, slider=True
            ),
            "contract_size": dict(
                label="Contract size (BTC per unit)", dtype=float, min=0.000001, max=10.0, value=0.001, step=0.0001, format="%.6f"
            ),
        }
    ),
    ranges={},
    data_requirements={
        "chart_overlays": [
            {"column": "EMA", "label": "EMA ({ema_period})"},
        ],
        "signal_columns": {
            "long": "bull_cross",
            "short": "bear_cross",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "EMA",
            "SlopePct",
            "bull_cross",
            "bear_cross",
            "trend_up",
            "trend_dn",
        ],
    },
    prepare_data=_ema_prepare,
    generate_signals=_ema_generate_signals,
    build_orders=_ema_build_orders,
)


STRATEGY_REGISTRY: Dict[str, StrategyDefinition] = {EMA_STRATEGY.key: EMA_STRATEGY}
DEFAULT_STRATEGY_KEY = EMA_STRATEGY.key

__all__ = ["StrategyDefinition", "STRATEGY_REGISTRY", "DEFAULT_STRATEGY_KEY", "EMA_STRATEGY", "ema", "slope_pct"]
