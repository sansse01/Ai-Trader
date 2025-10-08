from __future__ import annotations

import types

import pandas as pd
import pytest

import ohlcv_fetcher as fetcher


class _FakeExchangeNamespace:
    def __init__(self, timeframe_seconds: int) -> None:
        self.timeframe_seconds = timeframe_seconds

    def parse_timeframe(self, timeframe: str) -> int:
        return self.timeframe_seconds


class FakeKrakenExchange:
    def __init__(self, start_ms: int, rows: int, frame_ms: int) -> None:
        self.start_ms = start_ms
        self.rows = rows
        self.frame_ms = frame_ms
        self.market = {"BTC/EUR": {}}

    def load_markets(self):
        return self.market

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        limit = limit or 720
        # Simulate ccxt's built-in paginator returning only the first chunk.
        if params and params.get("paginate"):
            count = min(limit, self.rows)
            return [self._build_row(i) for i in range(count)]

        start_idx = 0
        if since is not None:
            start_idx = max(int((since - self.start_ms) // self.frame_ms), 0)

        rows: list[list[float]] = []
        for offset in range(limit):
            idx = start_idx + offset
            if idx >= self.rows:
                break
            rows.append(self._build_row(idx))
        return rows

    def _build_row(self, index: int) -> list[float]:
        ts = self.start_ms + index * self.frame_ms
        return [ts, 1.0, 2.0, 0.5, 1.5, 100.0]


class LateChunkKrakenExchange(FakeKrakenExchange):
    """Exchange that returns the most recent window when pagination is requested."""

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        limit = limit or 720
        if params and params.get("paginate"):
            start_idx = max(self.rows - limit, 0)
            count = min(limit, self.rows - start_idx)
            return [self._build_row(start_idx + i) for i in range(count)]

        return super().fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params=params)


def test_fetch_ohlcv_falls_back_to_manual_pagination(monkeypatch):
    hours = 60 * 24
    frame_ms = 60 * 60 * 1000
    start_ms = 0
    end_ms = start_ms + hours * frame_ms

    fake_exchange = FakeKrakenExchange(start_ms=start_ms, rows=hours, frame_ms=frame_ms)
    fake_module = types.SimpleNamespace(
        kraken=lambda: fake_exchange,
        Exchange=_FakeExchangeNamespace(timeframe_seconds=frame_ms // 1000),
        RateLimitExceeded=Exception,
    )
    monkeypatch.setattr(fetcher, "ccxt", fake_module)

    result = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=end_ms,
        limit_per_page=720,
    )

    assert result.rows == hours
    assert result.reached_target is True
    assert isinstance(result.coverage, pd.Timedelta)
    expected = pd.Timedelta(days=60) - pd.Timedelta(milliseconds=frame_ms)
    assert result.coverage >= expected


def test_fetch_ohlcv_backfills_when_initial_window_is_recent(monkeypatch):
    hours = 60 * 24
    frame_ms = 60 * 60 * 1000
    start_ms = 0
    end_ms = start_ms + hours * frame_ms

    fake_exchange = LateChunkKrakenExchange(start_ms=start_ms, rows=hours, frame_ms=frame_ms)
    fake_module = types.SimpleNamespace(
        kraken=lambda: fake_exchange,
        Exchange=_FakeExchangeNamespace(timeframe_seconds=frame_ms // 1000),
        RateLimitExceeded=Exception,
    )
    monkeypatch.setattr(fetcher, "ccxt", fake_module)

    result = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=end_ms,
        limit_per_page=720,
    )

    assert result.rows == hours
    assert result.reached_target is True
    assert result.start == pd.to_datetime(start_ms, unit="ms", utc=True)
    expected_end = pd.to_datetime(end_ms - frame_ms, unit="ms", utc=True)
    assert result.end == expected_end
