from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition
from .indicators import atr, donchian_channels
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "lookback": 20,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 4.0,
    "trail_percent": 0.0,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 5.0,
    "risk_fraction": 0.25,
    "contract_size": 0.001,
    "allow_shorts": True,
    "ignore_trend": False,
    "initial_cash": 10_000.0,
}


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    atr_period = int(merged.get("atr_period", DEFAULT_PARAMS["atr_period"]))
    lookback = int(merged.get("lookback", DEFAULT_PARAMS["lookback"]))
    atr_mult = float(merged.get("atr_multiplier", DEFAULT_PARAMS["atr_multiplier"]))

    prepared = df.copy()
    prepared["ATR"] = atr(prepared, atr_period)
    channels = donchian_channels(prepared["High"], prepared["Low"], lookback)
    prepared["DonchianUpper"] = channels["upper"]
    prepared["DonchianLower"] = channels["lower"]
    prepared["DonchianMid"] = channels["mid"]
    prepared["EntryUpper"] = channels["upper"].shift(1) + prepared["ATR"] * atr_mult
    prepared["EntryLower"] = channels["lower"].shift(1) - prepared["ATR"] * atr_mult
    prepared["LongStop"] = channels["lower"] - prepared["ATR"] * atr_mult
    prepared["ShortStop"] = channels["upper"] + prepared["ATR"] * atr_mult
    return prepared


def _generate(df: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    signals = df.copy()
    signals["long_signal"] = signals["Close"] > signals["EntryUpper"]
    signals["short_signal"] = signals["Close"] < signals["EntryLower"]
    signals["exit_long"] = signals["Close"] < signals["DonchianMid"]
    signals["exit_short"] = signals["Close"] > signals["DonchianMid"]
    signals["long_stop"] = signals["LongStop"]
    signals["short_stop"] = signals["ShortStop"]
    signals["trend_up"] = signals["Close"] > signals["DonchianMid"]
    signals["trend_dn"] = signals["Close"] < signals["DonchianMid"]
    return signals


ATR_BREAKOUT_STRATEGY = StrategyDefinition(
    key="atr_breakout",
    name="ATR Breakout",
    description="Volatility breakout using Donchian channels buffered by ATR for confirmation.",
    controls=OrderedDict(
        {
            "atr_period": dict(label="ATR period", dtype=int, min=2, max=200, value=14, step=1),
            "atr_multiplier": dict(
                label="ATR multiplier",
                dtype=float,
                min=0.1,
                max=5.0,
                value=1.5,
                step=0.1,
                format="%.2f",
            ),
            "lookback": dict(label="Donchian lookback", dtype=int, min=5, max=200, value=20, step=1),
            "stop_loss_percent": dict(
                label="Stop loss %",
                dtype=float,
                min=0.0,
                max=20.0,
                value=2.0,
                step=0.1,
                format="%.2f",
            ),
            "take_profit_percent": dict(
                label="Take profit %",
                dtype=float,
                min=0.0,
                max=30.0,
                value=4.0,
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
            "atr_multiplier": dict(label="ATR multiplier", key="atr_multiplier", min=0.5, max=3.0, step=0.25, dtype=float),
            "lookback": dict(label="Lookback", key="lookback", min=10, max=60, step=5, dtype=int),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "DonchianUpper", "label": "Upper Donchian"},
            {"column": "DonchianLower", "label": "Lower Donchian"},
            {"column": "EntryUpper", "label": "Long trigger"},
            {"column": "EntryLower", "label": "Short trigger"},
        ],
        "signal_columns": {"long": "long_signal", "short": "short_signal", "trend_up": "trend_up", "trend_down": "trend_dn"},
        "preview_columns": [
            "Close",
            "ATR",
            "DonchianUpper",
            "DonchianLower",
            "EntryUpper",
            "EntryLower",
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


__all__ = ["DEFAULT_PARAMS", "ATR_BREAKOUT_STRATEGY"]
