from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .base import StrategyDefinition
from .indicators import funding_metrics
from ._shared import make_signal_backtest_builder


DEFAULT_PARAMS: Dict[str, Any] = {
    "funding_window": 24,
    "entry_zscore": 1.0,
    "exit_zscore": 0.25,
    "basis_threshold_pct": 0.1,
    "basis_exit_pct": 0.02,
    "stop_buffer_percent": 0.75,
    "stop_loss_percent": 1.5,
    "take_profit_percent": 3.0,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 3.0,
    "risk_fraction": 0.15,
    "contract_size": 0.001,
    "allow_shorts": True,
    "initial_cash": 10_000.0,
}

_FUNDING_COLUMNS = {
    "funding_rate",
    "FundingRate",
    "fundingRate",
    "predicted_funding_rate",
    "predictedFundingRate",
}

_ALLOWED_TIMEFRAMES = {"1h", "4h", "1d"}


def _timeframe_to_hours(timeframe: str | None) -> float:
    if not timeframe:
        return 1.0
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        return max(1.0 / 60.0, float(tf[:-1]) / 60.0)
    if tf.endswith("h"):
        return max(1.0, float(tf[:-1]))
    if tf.endswith("d"):
        return max(1.0, float(tf[:-1]) * 24.0)
    return 1.0


def _prepare(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    window = int(merged.get("funding_window", DEFAULT_PARAMS["funding_window"]))
    timeframe = str(merged.get("timeframe", ""))
    hours = _timeframe_to_hours(timeframe)
    frequency_per_day = max(1.0, 24.0 / max(hours, 1e-9))

    prepared = df.copy()
    metrics = funding_metrics(prepared, window=window, frequency_per_day=frequency_per_day)
    for column in metrics.columns:
        prepared[column] = metrics[column]

    basis_series = pd.Series(np.nan, index=prepared.index)
    spot_close = pd.Series(np.nan, index=prepared.index)

    symbol_groups = params.get("symbol_groups", {}) or {}
    reference_symbols = symbol_groups.get("reference") or []
    if isinstance(reference_symbols, str):
        reference_symbols = [reference_symbols] if reference_symbols else []
    reference_symbol = reference_symbols[0] if reference_symbols else None

    if reference_symbol:
        symbol_datasets = params.get("symbol_datasets", {}) or {}
        ref_group = symbol_datasets.get("reference") or {}
        spot_df = None
        if isinstance(ref_group, dict):
            spot_df = ref_group.get(reference_symbol)
            if spot_df is None and ref_group:
                spot_df = next(iter(ref_group.values()))
        if isinstance(spot_df, pd.DataFrame) and not spot_df.empty and "Close" in spot_df:
            spot_close = pd.to_numeric(spot_df["Close"], errors="coerce").reindex(prepared.index).ffill()
            basis_series = (prepared["Close"] - spot_close) / spot_close.replace(0, np.nan) * 100.0

    prepared["SpotClose"] = spot_close
    prepared["BasisPct"] = basis_series
    return prepared


def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged = {**DEFAULT_PARAMS, **dict(params)}
    entry_z = float(merged.get("entry_zscore", DEFAULT_PARAMS["entry_zscore"]))
    exit_z = float(merged.get("exit_zscore", DEFAULT_PARAMS["exit_zscore"]))
    basis_entry = float(merged.get("basis_threshold_pct", DEFAULT_PARAMS["basis_threshold_pct"]))
    basis_exit = float(merged.get("basis_exit_pct", DEFAULT_PARAMS["basis_exit_pct"]))
    stop_buffer = float(merged.get("stop_buffer_percent", DEFAULT_PARAMS["stop_buffer_percent"]))

    signals = df.copy()
    funding_z = signals.get("funding_rate_zscore", pd.Series(np.nan, index=signals.index))
    basis = signals.get("BasisPct", pd.Series(np.nan, index=signals.index))

    long_cond = (funding_z < -abs(entry_z)) & (basis < -abs(basis_entry))
    short_cond = (funding_z > abs(entry_z)) & (basis > abs(basis_entry))
    exit_long = (funding_z > -abs(exit_z)) | (basis > -abs(basis_exit))
    exit_short = (funding_z < abs(exit_z)) | (basis < abs(basis_exit))

    signals["long_signal"] = long_cond.fillna(False)
    signals["short_signal"] = short_cond.fillna(False)
    signals["exit_long"] = exit_long.fillna(False)
    signals["exit_short"] = exit_short.fillna(False)
    signals["trend_up"] = (funding_z < 0).fillna(False)
    signals["trend_dn"] = (funding_z > 0).fillna(False)
    signals["long_stop"] = signals["Close"] * (1 - stop_buffer / 100.0)
    signals["short_stop"] = signals["Close"] * (1 + stop_buffer / 100.0)
    return signals


def _validate(params: Mapping[str, Any]) -> tuple[bool, str | None]:
    timeframe = str(params.get("timeframe") or "")
    if timeframe and timeframe not in _ALLOWED_TIMEFRAMES:
        return False, "Funding premium strategy supports 1h, 4h or 1d timeframes due to funding cadence."

    symbol_groups = params.get("symbol_groups", {}) or {}
    perp_entries = symbol_groups.get("perp") or []
    if isinstance(perp_entries, str):
        perp_entries = [perp_entries] if perp_entries else []
    if not perp_entries:
        return False, "Select a perpetual instrument that includes funding data."

    symbol_datasets = params.get("symbol_datasets", {}) or {}
    perp_group = symbol_datasets.get("perp") or {}
    perp_symbol = perp_entries[0]
    perp_df = None
    if isinstance(perp_group, dict):
        perp_df = perp_group.get(perp_symbol)
        if perp_df is None and perp_group:
            perp_df = next(iter(perp_group.values()))
    if perp_df is None or perp_df.empty:
        return False, "Perpetual data is unavailable for the selected symbol."

    if not any(col in perp_df.columns for col in _FUNDING_COLUMNS):
        return False, "Loaded perpetual data must contain a funding rate column."

    return True, None


FUNDING_PREMIUM_STRATEGY = StrategyDefinition(
    key="funding_premium",
    name="Funding Premium",
    description="Pairs funding rate extremes with spot/perp basis to fade crowded positioning.",
    controls=OrderedDict(
        {
            "funding_window": dict(label="Funding lookback (bars)", dtype=int, min=4, max=200, value=24, step=1),
            "entry_zscore": dict(
                label="Entry funding z", dtype=float, min=0.5, max=5.0, value=1.0, step=0.1, format="%.2f"
            ),
            "exit_zscore": dict(
                label="Exit funding z", dtype=float, min=0.1, max=2.0, value=0.25, step=0.05, format="%.2f"
            ),
            "basis_threshold_pct": dict(
                label="Basis entry threshold %", dtype=float, min=0.0, max=5.0, value=0.1, step=0.05, format="%.2f"
            ),
            "basis_exit_pct": dict(
                label="Basis exit threshold %", dtype=float, min=0.0, max=2.0, value=0.02, step=0.01, format="%.2f"
            ),
            "stop_buffer_percent": dict(
                label="Stop buffer %", dtype=float, min=0.1, max=5.0, value=0.75, step=0.05, format="%.2f"
            ),
            "stop_loss_percent": dict(
                label="Stop loss %", dtype=float, min=0.0, max=20.0, value=1.5, step=0.1, format="%.2f"
            ),
            "take_profit_percent": dict(
                label="Take profit %", dtype=float, min=0.0, max=20.0, value=3.0, step=0.1, format="%.2f"
            ),
            "fee_percent": dict(
                label="Fee % per fill", dtype=float, min=0.0, max=2.0, value=0.26, step=0.01, format="%.4f"
            ),
            "slippage_percent": dict(
                label="Slippage % per fill", dtype=float, min=0.0, max=2.0, value=0.05, step=0.01, format="%.4f"
            ),
            "max_leverage": dict(
                label="Max leverage", dtype=float, min=1.0, max=10.0, value=3.0, step=0.5, format="%.2f"
            ),
            "risk_fraction": dict(
                label="Risk fraction of equity", dtype=float, min=0.01, max=1.0, value=0.15, step=0.01, slider=True
            ),
            "contract_size": dict(
                label="Contract size", dtype=float, min=0.000001, max=10.0, value=0.001, step=0.0001, format="%.6f"
            ),
        }
    ),
    range_controls=OrderedDict(
        {
            "entry_zscore": dict(label="Entry funding z", key="entry_zscore", min=0.5, max=2.0, step=0.25, dtype=float),
            "basis_threshold_pct": dict(
                label="Basis entry %", key="basis_threshold_pct", min=0.05, max=0.5, step=0.05, dtype=float
            ),
        }
    ),
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "symbols": [
            {
                "id": "perp",
                "label": "Perpetual (with funding)",
                "required": True,
                "multiple": False,
                "metadata": {"requires_perpetual": True, "requires_funding": True},
            },
            {
                "id": "reference",
                "label": "Reference spot (optional)",
                "required": False,
                "multiple": False,
                "metadata": {"role": "spot_reference"},
            },
        ],
        "chart_overlays": [
            {"column": "BasisPct", "label": "Basis %"},
        ],
        "signal_columns": {
            "long": "long_signal",
            "short": "short_signal",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "funding_rate",
            "funding_rate_zscore",
            "funding_rate_annualized_pct",
            "BasisPct",
            "long_signal",
            "short_signal",
        ],
        "metadata": {
            "requires_pairs": True,
            "requires_perpetual": True,
            "requires_funding": True,
            "preferred_timeframes": sorted(_ALLOWED_TIMEFRAMES),
        },
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
    validate_context=_validate,
)


__all__ = ["DEFAULT_PARAMS", "FUNDING_PREMIUM_STRATEGY"]
