from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition
from .indicators import bollinger_bands
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "entry_zscore": 2.0,
    "exit_zscore": 0.5,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 2.5,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 3.0,
    "risk_fraction": 0.2,
    "contract_size": 0.001,
    "allow_shorts": True,
    "initial_cash": 10_000.0,
}


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    period = int(merged.get("bollinger_period", DEFAULT_PARAMS["bollinger_period"]))
    num_std = float(merged.get("bollinger_std", DEFAULT_PARAMS["bollinger_std"]))

    prepared = df.copy()
    bands = bollinger_bands(prepared["Close"], period=period, num_std=num_std)
    prepared["BB_Mid"] = bands["mid"]
    prepared["BB_Upper"] = bands["upper"]
    prepared["BB_Lower"] = bands["lower"]
    prepared["BB_Z"] = bands["zscore"]
    prepared["BB_Bandwidth"] = bands["bandwidth"]
    return prepared


def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    entry_z = float(merged.get("entry_zscore", DEFAULT_PARAMS["entry_zscore"]))
    exit_z = float(merged.get("exit_zscore", DEFAULT_PARAMS["exit_zscore"]))

    signals = df.copy()
    signals["long_signal"] = signals["BB_Z"] < -abs(entry_z)
    signals["short_signal"] = signals["BB_Z"] > abs(entry_z)
    signals["exit_long"] = signals["BB_Z"].abs() < exit_z
    signals["exit_short"] = signals["BB_Z"].abs() < exit_z
    signals["trend_up"] = signals["Close"] > signals["BB_Mid"]
    signals["trend_dn"] = signals["Close"] < signals["BB_Mid"]
    signals["long_take_profit"] = signals["BB_Mid"]
    signals["short_take_profit"] = signals["BB_Mid"]
    signals["long_stop"] = signals["BB_Lower"]
    signals["short_stop"] = signals["BB_Upper"]
    return signals


BOLLINGER_MEAN_REVERSION_STRATEGY = StrategyDefinition(
    key="bollinger_mean_reversion",
    name="Bollinger Mean Reversion",
    description="Contrarian strategy using Bollinger Band z-score extremes.",
    controls=OrderedDict(
        {
            "bollinger_period": dict(label="Bollinger period", dtype=int, min=5, max=200, value=20, step=1),
            "bollinger_std": dict(
                label="Band std dev",
                dtype=float,
                min=0.5,
                max=5.0,
                value=2.0,
                step=0.1,
                format="%.2f",
            ),
            "entry_zscore": dict(
                label="Entry z-score",
                dtype=float,
                min=0.5,
                max=4.0,
                value=2.0,
                step=0.1,
                format="%.2f",
            ),
            "exit_zscore": dict(
                label="Exit z-score",
                dtype=float,
                min=0.1,
                max=2.0,
                value=0.5,
                step=0.1,
                format="%.2f",
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
            "entry_zscore": dict(label="Entry z", key="entry_zscore", min=1.5, max=2.5, step=0.25, dtype=float),
            "exit_zscore": dict(label="Exit z", key="exit_zscore", min=0.25, max=1.0, step=0.25, dtype=float),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "BB_Mid", "label": "BB Mid"},
            {"column": "BB_Upper", "label": "BB Upper"},
            {"column": "BB_Lower", "label": "BB Lower"},
        ],
        "signal_columns": {
            "long": "long_signal",
            "short": "short_signal",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "BB_Mid",
            "BB_Upper",
            "BB_Lower",
            "BB_Z",
            "long_signal",
            "short_signal",
        ],
        "metadata": {"preferred_timeframes": ["30m", "1h", "4h"]},
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
            "long_stop": "long_stop",
            "short_stop": "short_stop",
        },
    ),
)


__all__ = ["DEFAULT_PARAMS", "BOLLINGER_MEAN_REVERSION_STRATEGY"]
