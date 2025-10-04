from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition
from .indicators import rsi
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "rsi_period": 14,
    "entry_long_level": 30.0,
    "exit_long_level": 55.0,
    "entry_short_level": 70.0,
    "exit_short_level": 45.0,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 2.5,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 3.0,
    "risk_fraction": 0.15,
    "contract_size": 0.001,
    "allow_shorts": True,
    "initial_cash": 10_000.0,
}


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    period = int(merged.get("rsi_period", DEFAULT_PARAMS["rsi_period"]))

    prepared = df.copy()
    prepared["RSI"] = rsi(prepared["Close"], period)
    prepared["RSI_MA"] = prepared["RSI"].rolling(window=3, min_periods=1).mean()
    return prepared


def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    entry_long = float(merged.get("entry_long_level", DEFAULT_PARAMS["entry_long_level"]))
    exit_long = float(merged.get("exit_long_level", DEFAULT_PARAMS["exit_long_level"]))
    entry_short = float(merged.get("entry_short_level", DEFAULT_PARAMS["entry_short_level"]))
    exit_short = float(merged.get("exit_short_level", DEFAULT_PARAMS["exit_short_level"]))

    signals = df.copy()
    signals["long_signal"] = signals["RSI"] < entry_long
    signals["short_signal"] = signals["RSI"] > entry_short
    signals["exit_long"] = signals["RSI"] > exit_long
    signals["exit_short"] = signals["RSI"] < exit_short
    signals["trend_up"] = signals["RSI"] > 50
    signals["trend_dn"] = signals["RSI"] < 50
    signals["long_take_profit"] = signals["Close"]
    signals["short_take_profit"] = signals["Close"]
    return signals


RSI_MEAN_REVERSION_STRATEGY = StrategyDefinition(
    key="rsi_mean_reversion",
    name="RSI Mean Reversion",
    description="Mean reversion entries from RSI extremes with flexible exit thresholds.",
    controls=OrderedDict(
        {
            "rsi_period": dict(label="RSI period", dtype=int, min=2, max=200, value=14, step=1),
            "entry_long_level": dict(
                label="Long entry RSI",
                dtype=float,
                min=1.0,
                max=50.0,
                value=30.0,
                step=1.0,
                format="%.1f",
            ),
            "exit_long_level": dict(
                label="Long exit RSI",
                dtype=float,
                min=10.0,
                max=80.0,
                value=55.0,
                step=1.0,
                format="%.1f",
            ),
            "entry_short_level": dict(
                label="Short entry RSI",
                dtype=float,
                min=50.0,
                max=99.0,
                value=70.0,
                step=1.0,
                format="%.1f",
            ),
            "exit_short_level": dict(
                label="Short exit RSI",
                dtype=float,
                min=20.0,
                max=90.0,
                value=45.0,
                step=1.0,
                format="%.1f",
            ),
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
                max=20.0,
                value=2.5,
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
                value=3.0,
                step=0.5,
                format="%.2f",
            ),
            "risk_fraction": dict(
                label="Risk fraction of equity",
                dtype=float,
                min=0.01,
                max=1.0,
                value=0.15,
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
            "entry_long_level": dict(label="Long entry", key="entry_long_level", min=20.0, max=40.0, step=2.0, dtype=float),
            "entry_short_level": dict(label="Short entry", key="entry_short_level", min=60.0, max=80.0, step=2.0, dtype=float),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "RSI", "label": "RSI"},
            {"column": "RSI_MA", "label": "RSI MA"},
        ],
        "signal_columns": {
            "long": "long_signal",
            "short": "short_signal",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "RSI",
            "RSI_MA",
            "long_signal",
            "short_signal",
            "exit_long",
            "exit_short",
        ],
        "metadata": {"preferred_timeframes": ["30m", "1h"]},
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
            "long_take_profit": "long_take_profit",
            "short_take_profit": "short_take_profit",
        },
    ),
)


__all__ = ["DEFAULT_PARAMS", "RSI_MEAN_REVERSION_STRATEGY"]
