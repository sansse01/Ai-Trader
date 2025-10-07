from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = REPO_ROOT / "tests" / "data" / "sample_ohlcv.csv"

EXPECTED_SIGNAL_COUNTS: Dict[str, Dict[str, int]] = {
    "ema_trend": {"long": 7, "short": 6, "trend_up": 179, "trend_down": 145},
    "atr_breakout": {"long": 0, "short": 0, "trend_up": 198, "trend_down": 162},
    "donchian_trend": {"long": 0, "short": 0, "trend_up": 256, "trend_down": 104},
    "rsi_mean_reversion": {"long": 85, "short": 144, "trend_up": 198, "trend_down": 138},
    "vortex_trend": {"long": 0, "short": 0, "trend_up": 197, "trend_down": 162},
    "bollinger_mean_reversion": {"long": 38, "short": 42, "trend_up": 195, "trend_down": 164},
    "funding_premium": {"long": 0, "short": 0, "trend_up": 170, "trend_down": 189},
}


def load_sample_ohlcv() -> pd.DataFrame:
    """Load the deterministic OHLCV sample used by the smoke tests."""

    df = pd.read_csv(SAMPLE_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def build_strategy_params(
    key: str, base_params: Mapping[str, Any], data: pd.DataFrame
) -> Dict[str, Any]:
    """Return parameters suitable for testing the given strategy."""

    params = dict(base_params)
    if key == "funding_premium":
        params.setdefault("timeframe", "1h")
        params.setdefault("symbol_groups", {"perp": ["perp"], "reference": []})
        params.setdefault("symbol_datasets", {"perp": {"perp": data}})
    return params
