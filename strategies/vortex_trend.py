from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition
from .indicators import atr, vortex_indicator
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "vortex_period": 14,
    "trend_threshold": 1.05,
    "stop_lookback": 10,
    "stop_loss_percent": 1.5,
    "take_profit_percent": 3.0,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 5.0,
    "risk_fraction": 0.25,
    "contract_size": 0.001,
    "allow_shorts": True,
    "initial_cash": 10_000.0,
}


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    period = int(merged.get("vortex_period", DEFAULT_PARAMS["vortex_period"]))
    stop_lb = int(merged.get("stop_lookback", DEFAULT_PARAMS["stop_lookback"]))

    prepared = df.copy()
    vi = vortex_indicator(prepared, period)
    prepared["VI_Plus"] = vi["vi_plus"]
    prepared["VI_Minus"] = vi["vi_minus"]
    prepared["VI_Strength"] = vi["trend_strength"]
    prepared["ATR"] = atr(prepared, period)
    prepared["StopLong"] = prepared["Low"].rolling(stop_lb, min_periods=1).min()
    prepared["StopShort"] = prepared["High"].rolling(stop_lb, min_periods=1).max()
    return prepared


def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    threshold = float(merged.get("trend_threshold", DEFAULT_PARAMS["trend_threshold"]))

    signals = df.copy()
    prev_plus = signals["VI_Plus"].shift(1)
    prev_minus = signals["VI_Minus"].shift(1)
    bull_cross = (signals["VI_Plus"] > signals["VI_Minus"]) & (prev_plus <= prev_minus)
    bear_cross = (signals["VI_Minus"] > signals["VI_Plus"]) & (prev_minus <= prev_plus)
    signals["long_signal"] = bull_cross & (signals["VI_Plus"] > threshold)
    signals["short_signal"] = bear_cross & (signals["VI_Minus"] > threshold)
    signals["exit_long"] = signals["VI_Plus"] < signals["VI_Minus"]
    signals["exit_short"] = signals["VI_Minus"] < signals["VI_Plus"]
    signals["trend_up"] = signals["VI_Plus"] > signals["VI_Minus"]
    signals["trend_dn"] = signals["VI_Minus"] > signals["VI_Plus"]
    signals["long_stop"] = signals["StopLong"]
    signals["short_stop"] = signals["StopShort"]
    return signals


VORTEX_TREND_STRATEGY = StrategyDefinition(
    key="vortex_trend",
    name="Vortex Trend",
    description="Trend following based on Vortex indicator crossovers and strength thresholds.",
    controls=OrderedDict(
        {
            "vortex_period": dict(label="Vortex period", dtype=int, min=2, max=200, value=14, step=1),
            "trend_threshold": dict(
                label="Trend strength threshold",
                dtype=float,
                min=0.8,
                max=3.0,
                value=1.05,
                step=0.05,
                format="%.2f",
            ),
            "stop_lookback": dict(label="Stop lookback", dtype=int, min=2, max=100, value=10, step=1),
            "stop_loss_percent": dict(
                label="Stop loss %",
                dtype=float,
                min=0.0,
                max=20.0,
                value=1.5,
                step=0.1,
                format="%.2f",
            ),
            "take_profit_percent": dict(
                label="Take profit %",
                dtype=float,
                min=0.0,
                max=30.0,
                value=3.0,
                step=0.1,
                format="%.2f",
            ),
            "fee_percent": dict(
                label="Fee % per fill",
                dtype=float,
                min=0.0,
                max=2.0,
                value=0.26,
                step=0.01,
                format="%.4f",
            ),
            "slippage_percent": dict(
                label="Slippage % per fill",
                dtype=float,
                min=0.0,
                max=2.0,
                value=0.05,
                step=0.01,
                format="%.4f",
            ),
            "max_leverage": dict(
                label="Max leverage",
                dtype=float,
                min=1.0,
                max=10.0,
                value=5.0,
                step=0.5,
                format="%.2f",
            ),
            "risk_fraction": dict(
                label="Risk fraction of equity",
                dtype=float,
                min=0.01,
                max=1.0,
                value=0.25,
                step=0.01,
                slider=True,
            ),
            "contract_size": dict(
                label="Contract size",
                dtype=float,
                min=0.000001,
                max=10.0,
                value=0.001,
                step=0.0001,
                format="%.6f",
            ),
        }
    ),
    range_controls=OrderedDict(
        {
            "vortex_period": dict(label="Vortex period", key="vortex_period", min=7, max=28, step=1, dtype=int),
            "trend_threshold": dict(label="Strength", key="trend_threshold", min=1.0, max=1.6, step=0.05, dtype=float),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "VI_Plus", "label": "VI+"},
            {"column": "VI_Minus", "label": "VI-"},
        ],
        "signal_columns": {
            "long": "long_signal",
            "short": "short_signal",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "VI_Plus",
            "VI_Minus",
            "VI_Strength",
            "long_signal",
            "short_signal",
        ],
        "metadata": {"preferred_timeframes": ["1h", "4h"]},
    },
    prepare_data=_prepare,
    generate_signals=_generate,
    build_orders=lambda _df, _sig, _params: {},
    build_simple_backtest_strategy=make_signal_backtest_builder(
        DEFAULT_PARAMS,
        {
            "long": "long_signal",
            "short": "short_signal",
            "exit_long": "exit_long",
            "exit_short": "exit_short",
            "long_stop": "long_stop",
            "short_stop": "short_stop",
        },
    ),
)


__all__ = ["DEFAULT_PARAMS", "VORTEX_TREND_STRATEGY"]
