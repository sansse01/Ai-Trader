from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pandas as pd

PrepareDataFn = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]
GenerateSignalsFn = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]
BuildOrdersFn = Callable[[pd.DataFrame, pd.DataFrame, Mapping[str, Any]], Dict[str, Any]]
SimpleBacktestBuilder = Callable[[pd.DataFrame, pd.DataFrame, Mapping[str, Any]], Any]
TrueStopBacktestBuilder = Callable[[pd.DataFrame, pd.DataFrame, Mapping[str, Any]], Any]
ValidateContextFn = Callable[[Mapping[str, Any]], Tuple[bool, Optional[str]]]


def _identity_prepare(df: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return df


def _identity_signals(df: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return df


def _identity_build_orders(
    _df: pd.DataFrame,
    _signals: pd.DataFrame,
    _params: Mapping[str, Any],
) -> Dict[str, Any]:
    return {}


def _identity_validate(_params: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    return True, None


@dataclass(slots=True)
class StrategyDefinition:
    """Container describing how a trading strategy integrates with the app."""

    key: str
    name: str
    description: str = ""
    controls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    range_controls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    prepare_data: PrepareDataFn = field(default_factory=lambda: _identity_prepare)
    generate_signals: GenerateSignalsFn = field(default_factory=lambda: _identity_signals)
    build_orders: BuildOrdersFn = field(default_factory=lambda: _identity_build_orders)
    build_simple_backtest_strategy: Optional[SimpleBacktestBuilder] = None
    build_true_stop_strategy: Optional[TrueStopBacktestBuilder] = None
    validate_context: ValidateContextFn = field(default_factory=lambda: _identity_validate)


__all__ = ["StrategyDefinition"]
