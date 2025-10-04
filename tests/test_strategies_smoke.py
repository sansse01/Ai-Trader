from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from strategies import STRATEGY_REGISTRY
from tests.expected_signals import (
    EXPECTED_SIGNAL_COUNTS,
    build_strategy_params,
    load_sample_ohlcv,
)


@pytest.fixture(scope="module")
def sample_market_data() -> pd.DataFrame:
    """Return the deterministic OHLCV sample used for smoke testing."""

    return load_sample_ohlcv()


def _normalize_boolean_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(0).astype(float) > 0


@pytest.mark.parametrize("strategy_key", sorted(STRATEGY_REGISTRY.keys()))
def test_strategy_signal_counts(strategy_key: str, sample_market_data: pd.DataFrame) -> None:
    strategy = STRATEGY_REGISTRY[strategy_key]
    params = build_strategy_params(strategy_key, strategy.default_params, sample_market_data)

    prepared = strategy.prepare_data(sample_market_data, params)
    assert isinstance(prepared, pd.DataFrame)
    assert len(prepared) == len(sample_market_data)
    assert_index_equal(prepared.index, sample_market_data.index)

    signals = strategy.generate_signals(prepared, params)
    assert isinstance(signals, pd.DataFrame)
    assert len(signals) == len(prepared)
    assert_index_equal(signals.index, prepared.index)

    signal_columns = strategy.data_requirements.get("signal_columns", {})
    expected_counts = EXPECTED_SIGNAL_COUNTS[strategy_key]

    for label, column in signal_columns.items():
        assert column in signals, f"{strategy_key} missing expected signal column '{column}'"
        normalized = _normalize_boolean_series(signals[column])
        actual = int(normalized.sum())
        assert (
            actual == expected_counts[label]
        ), f"{strategy_key} expected {expected_counts[label]} {label} signals but found {actual}"


@pytest.mark.parametrize("strategy_key", sorted(STRATEGY_REGISTRY.keys()))
def test_strategy_signal_columns_are_boolean(
    strategy_key: str, sample_market_data: pd.DataFrame
) -> None:
    """Ensure the strategies expose signal columns that can be interpreted as booleans."""

    strategy = STRATEGY_REGISTRY[strategy_key]
    params = build_strategy_params(strategy_key, strategy.default_params, sample_market_data)

    prepared = strategy.prepare_data(sample_market_data, params)
    signals = strategy.generate_signals(prepared, params)

    for column in strategy.data_requirements.get("signal_columns", {}).values():
        assert column in signals, f"{strategy_key} missing expected signal column '{column}'"
        normalized = _normalize_boolean_series(signals[column])
        assert normalized.dtype == bool
        assert normalized.isin([True, False]).all()
