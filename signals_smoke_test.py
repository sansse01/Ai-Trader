from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency for live smoke runs
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - ccxt is optional for local validation
    ccxt = None

from strategies import STRATEGY_REGISTRY
from tests.expected_signals import (
    EXPECTED_SIGNAL_COUNTS,
    SAMPLE_DATA_PATH,
    build_strategy_params,
    load_sample_ohlcv,
)


def _normalize_boolean_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(0).astype(float) > 0


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{label}={value}" for label, value in counts.items())


def fetch_live_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
    since: str | None = None,
    exchange_id: str = "kraken",
) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt is not installed; install it to enable live smoke runs.")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    since_ms = exchange.parse8601(since) if since else None
    raw = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
    frame = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame.set_index("timestamp", inplace=True)
    return frame


def run_smoke(
    data: pd.DataFrame,
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    results: Dict[str, Dict[str, int]] = {}
    all_ok = True

    for key, strategy in STRATEGY_REGISTRY.items():
        params = build_strategy_params(key, strategy.default_params, data)
        try:
            prepared = strategy.prepare_data(data, params)
            signals = strategy.generate_signals(prepared, params)
        except Exception as exc:  # pragma: no cover - surface strategy errors
            print(f"[{key}] ERROR: {exc}")
            all_ok = False
            continue

        signal_columns = strategy.data_requirements.get("signal_columns", {})
        counts: Dict[str, int] = {}
        for label, column in signal_columns.items():
            if column not in signals:
                print(f"[{key}] missing expected signal column '{column}'")
                all_ok = False
                continue
            normalized = _normalize_boolean_series(signals[column])
            counts[label] = int(normalized.sum())
        results[key] = counts

        if expected_counts and key in expected_counts:
            expected = expected_counts[key]
            mismatches = {
                label: (counts.get(label), expected[label])
                for label in expected
                if counts.get(label) != expected[label]
            }
            if mismatches:
                mismatch_str = ", ".join(
                    f"{label}: actual={actual} expected={expected}" for label, (actual, expected) in mismatches.items()
                )
                print(f"[{key}] signal mismatches -> {mismatch_str}")
                all_ok = False
            else:
                print(f"[{key}] OK -> {_format_counts(counts)}")
        else:
            print(f"[{key}] signals -> {_format_counts(counts)}")

    return all_ok, results


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strategy signal generation.")
    parser.add_argument(
        "--use-live",
        action="store_true",
        help="Fetch fresh OHLCV data via ccxt instead of using the bundled sample dataset.",
    )
    parser.add_argument("--symbol", default="BTC/EUR", help="Trading pair to fetch when using live data.")
    parser.add_argument("--timeframe", default="1h", help="Timeframe to request for live data.")
    parser.add_argument("--limit", type=int, default=360, help="Number of candles to request in live mode.")
    parser.add_argument(
        "--since",
        default="2024-01-01T00:00:00Z",
        help="ISO timestamp to begin the live query from (ignored when --use-live is false).",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=SAMPLE_DATA_PATH,
        help="Path to a CSV file containing sample OHLCV data (ignored when --use-live is true).",
    )
    return parser.parse_args(list(argv))


def load_sample_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if args.use_live:
        try:
            data = fetch_live_ohlcv(args.symbol, args.timeframe, args.limit, args.since)
        except Exception as exc:  # pragma: no cover - live fetch is optional
            print(f"Failed to load live data: {exc}")
            return 1
        expected = None
    else:
        if args.sample_path == SAMPLE_DATA_PATH:
            data = load_sample_ohlcv()
        else:
            data = load_sample_from_path(args.sample_path)
        expected = EXPECTED_SIGNAL_COUNTS

    ok, _ = run_smoke(data, expected)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
