from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition
from .indicators import donchian_channels
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "lookback": 55,
    "confirm_lookback": 10,
    "stop_loss_percent": 1.5,
    "take_profit_percent": 3.5,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 5.0,
    "risk_fraction": 0.2,
    "contract_size": 0.001,
    "allow_shorts": True,
    "initial_cash": 10_000.0,
}


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    lookback = int(merged.get("lookback", DEFAULT_PARAMS["lookback"]))
    confirm_lb = int(merged.get("confirm_lookback", DEFAULT_PARAMS["confirm_lookback"]))

    prepared = df.copy()
    channels = donchian_channels(prepared["High"], prepared["Low"], lookback)
    prepared["DonchianUpper"] = channels["upper"]
    prepared["DonchianLower"] = channels["lower"]
    prepared["DonchianMid"] = channels["mid"]
    prepared["UpperConfirm"] = prepared["DonchianUpper"].rolling(confirm_lb, min_periods=1).max()
    prepared["LowerConfirm"] = prepared["DonchianLower"].rolling(confirm_lb, min_periods=1).min()
    return prepared


def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    ignore_trend = bool(merged.get("ignore_trend", False))

    signals = df.copy()
    upper_trigger = signals["UpperConfirm"].shift(1)
    lower_trigger = signals["LowerConfirm"].shift(1)
    signals["long_signal"] = signals["Close"] > upper_trigger
    signals["short_signal"] = signals["Close"] < lower_trigger
    signals["trend_up"] = signals["Close"] > signals["DonchianMid"]
    signals["trend_dn"] = signals["Close"] < signals["DonchianMid"]
    signals["exit_long"] = (~signals["trend_up"]) & (~ignore_trend)
    signals["exit_short"] = (~signals["trend_dn"]) & (~ignore_trend)
    signals["long_stop"] = signals["DonchianLower"]
    signals["short_stop"] = signals["DonchianUpper"]
    return signals


donchian_signal_map = {
    "long": "long_signal",
    "short": "short_signal",
    "exit_long": "exit_long",
    "exit_short": "exit_short",
    "long_stop": "long_stop",
    "short_stop": "short_stop",
}


DONCHIAN_TREND_STRATEGY = StrategyDefinition(
    key="donchian_trend",
    name="Donchian Trend",
    description="Classic Donchian channel breakout with optional trend confirmation.",
    controls=OrderedDict(
        {
            "lookback": dict(label="Channel lookback", dtype=int, min=5, max=200, value=55, step=1),
            "confirm_lookback": dict(label="Confirmation lookback", dtype=int, min=1, max=100, value=10, step=1),
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
                value=3.5,
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
                value=0.2,
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
            "lookback": dict(label="Channel lookback", key="lookback", min=20, max=100, step=5, dtype=int),
            "confirm_lookback": dict(label="Confirmation", key="confirm_lookback", min=5, max=40, step=5, dtype=int),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "DonchianUpper", "label": "Upper"},
            {"column": "DonchianLower", "label": "Lower"},
            {"column": "DonchianMid", "label": "Mid"},
        ],
        "signal_columns": {
            "long": "long_signal",
            "short": "short_signal",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "DonchianUpper",
            "DonchianLower",
            "DonchianMid",
            "UpperConfirm",
            "LowerConfirm",
            "long_signal",
            "short_signal",
        ],
        "metadata": {"preferred_timeframes": ["4h", "1d"]},
    },
    prepare_data=_prepare,
    generate_signals=_generate,
    build_orders=lambda _df, _sig, _params: {},
    build_simple_backtest_strategy=make_signal_backtest_builder(DEFAULT_PARAMS, donchian_signal_map),
)


__all__ = ["DEFAULT_PARAMS", "DONCHIAN_TREND_STRATEGY"]
