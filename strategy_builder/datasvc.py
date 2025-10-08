"""Market data utilities used by the strategy builder."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_ALLOWED_TFS = {"1m": 1, "15m": 15, "1h": 60, "3h": 180, "1d": 1440}


@dataclass(slots=True)
class DataService:
    """Convenience wrapper for loading and summarising OHLCV data."""

    cache_dir: Path | None = None

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        req_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        missing = set(req_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp").reset_index(drop=True)
        return df

    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return OHLCV history for the requested window.

        The implementation is file-based: it looks for a CSV in ``cache_dir`` named
        ``{symbol}_{timeframe}.csv``. Tests and the CLI provide this file. In a
        production deployment this method can be swapped with an exchange client.
        """

        if timeframe not in _ALLOWED_TFS:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Allowed: {sorted(_ALLOWED_TFS)}")

        if self.cache_dir is None:
            raise RuntimeError("DataService requires cache_dir to read historical data.")

        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)
        path = Path(self.cache_dir) / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {path}")

        df = pd.read_csv(path)
        df = self._normalise(df)
        mask = (df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)
        window = df.loc[mask].copy()
        if window.empty:
            raise ValueError("No data available in requested window")
        LOGGER.info("Loaded %s rows for %s %s", len(window), symbol, timeframe)
        return window

    @staticmethod
    def _returns(df: pd.DataFrame) -> pd.Series:
        close = df["Close"].astype(float)
        return close.pct_change().dropna()

    @staticmethod
    def _annualisation_factor(timeframe: str) -> float:
        minutes = _ALLOWED_TFS[timeframe]
        per_day = 1440 / minutes
        return np.sqrt(per_day * 252)

    @staticmethod
    def _drawdown(equity: Iterable[float]) -> Tuple[float, float]:
        equity_arr = np.asarray(list(equity), dtype=float)
        cum_max = np.maximum.accumulate(equity_arr)
        dd = equity_arr / cum_max - 1.0
        max_dd = dd.min() if len(dd) else 0.0
        end_idx = int(np.argmin(dd)) if len(dd) else 0
        start_idx = int(np.argmax(equity_arr[: end_idx + 1])) if len(dd) else 0
        return float(max_dd), float(start_idx)

    def summarize(self, df: pd.DataFrame) -> Dict[str, object]:
        """Return a privacy preserving statistical summary of the OHLCV dataframe."""

        df = self._normalise(df)
        returns = self._returns(df)
        if returns.empty:
            raise ValueError("Not enough data to build summary")

        timeframe = self._infer_timeframe(df)
        ann_factor = self._annualisation_factor(timeframe)
        ret_mean = float(returns.mean())
        ret_std = float(returns.std(ddof=0))
        skew = float(returns.skew())
        kurt = float(returns.kurt())
        vol_ann = float(ret_std * ann_factor)

        # Equity curve for drawdown
        equity = (1 + returns).cumprod()
        maxdd, _ = self._drawdown(equity)

        # buy & hold metrics
        total_return = float(equity.iloc[-1] - 1.0)
        periods = len(returns)
        years = max(periods / (ann_factor ** 2), 1e-9)
        bh_cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        bh_maxdd = float(maxdd)

        regime_counts = self._regime_counts(df)

        summary = {
            "n": int(len(df)),
            "start": df["Timestamp"].iloc[0].isoformat(),
            "end": df["Timestamp"].iloc[-1].isoformat(),
            "ret_mean": ret_mean,
            "ret_std": ret_std,
            "skew": skew,
            "kurt": kurt,
            "vol_annual": vol_ann,
            "maxdd": maxdd,
            "bh_cagr": bh_cagr,
            "bh_maxdd": bh_maxdd,
            "regime_counts": regime_counts,
            "timeframe": timeframe,
        }
        return summary

    @staticmethod
    def _infer_timeframe(df: pd.DataFrame) -> str:
        diffs = df["Timestamp"].diff().dropna().dt.total_seconds() / 60
        if diffs.empty:
            return "1d"
        avg = diffs.median()
        closest = min(_ALLOWED_TFS.items(), key=lambda kv: abs(kv[1] - avg))
        return closest[0]

    def _regime_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        close = df["Close"].astype(float)
        sma200 = close.rolling(200, min_periods=1).mean()
        donchian_high = close.rolling(55, min_periods=1).max()
        donchian_low = close.rolling(55, min_periods=1).min()
        regime = np.where(close > sma200, "bull", np.where(close < donchian_low, "bear", "range"))
        counts = {state: int((regime == state).sum()) for state in ("bull", "bear", "range")}
        return counts

    @staticmethod
    def summary_hash(summary: Dict[str, object]) -> str:
        payload = json.dumps(summary, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
