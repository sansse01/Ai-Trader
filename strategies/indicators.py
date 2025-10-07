from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14, *, use_ema: bool = True) -> pd.Series:
    """Average True Range using high, low and close columns."""

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high.sub(low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    if use_ema:
        return true_range.ewm(span=period, adjust=False).mean()
    return true_range.rolling(window=period, min_periods=1).mean()


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    lookback: int,
) -> pd.DataFrame:
    """Return upper, lower and mid Donchian channels."""

    upper = high.rolling(window=lookback, min_periods=1).max()
    lower = low.rolling(window=lookback, min_periods=1).min()
    mid = (upper + lower) / 2.0
    width = upper - lower
    return pd.DataFrame({
        "upper": upper,
        "lower": lower,
        "mid": mid,
        "width": width,
    })


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index implementation using Wilder smoothing."""

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    output = 100 - (100 / (1 + rs))
    return output.fillna(50.0)


def vortex_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute the Vortex Indicator (VI+ / VI-)."""

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high.sub(low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    tr_sum = tr.rolling(window=period, min_periods=1).sum()
    vi_plus = vm_plus.rolling(window=period, min_periods=1).sum() / tr_sum.replace(0, np.nan)
    vi_minus = vm_minus.rolling(window=period, min_periods=1).sum() / tr_sum.replace(0, np.nan)

    return pd.DataFrame({
        "vi_plus": vi_plus,
        "vi_minus": vi_minus,
        "trend_strength": (vi_plus - vi_minus).abs(),
    })


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Standard Bollinger Bands along with bandwidth and z-score."""

    ma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    zscore = (series - ma) / std.replace(0, np.nan)
    return pd.DataFrame({
        "mid": ma,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
        "zscore": zscore,
    })


def funding_metrics(
    df: pd.DataFrame,
    rate_column: str | None = None,
    *,
    window: int = 24,
    frequency_per_day: float = 3.0,
) -> pd.DataFrame:
    """Return helper metrics derived from perpetual swap funding data."""

    candidates: Iterable[str]
    if rate_column is None:
        candidates = (
            "funding_rate",
            "FundingRate",
            "fundingRate",
            "predicted_funding_rate",
            "predictedFundingRate",
        )
        rate_column = next((col for col in candidates if col in df.columns), None)
    if rate_column is None or rate_column not in df.columns:
        raise KeyError("No funding rate column found in DataFrame.")

    rate = pd.to_numeric(df[rate_column], errors="coerce")
    metrics = pd.DataFrame(index=df.index)
    metrics["funding_rate"] = rate

    annualization_factor = float(frequency_per_day) * 365.0
    metrics["funding_rate_annualized_pct"] = rate * annualization_factor * 100.0

    rolling_window = max(1, int(window))
    window_sum = rate.rolling(window=rolling_window, min_periods=1).sum()
    window_mean = rate.rolling(window=rolling_window, min_periods=1).mean()
    window_std = rate.rolling(window=rolling_window, min_periods=1).std(ddof=0)

    metrics["funding_rate_window_sum"] = window_sum
    metrics["funding_rate_window_mean"] = window_mean
    metrics["funding_rate_zscore"] = (rate - window_mean) / window_std.replace(0, np.nan)
    metrics["funding_rate_abs_rank"] = rate.abs().rank(pct=True)

    return metrics


__all__ = [
    "atr",
    "donchian_channels",
    "rsi",
    "vortex_indicator",
    "bollinger_bands",
    "funding_metrics",
]
