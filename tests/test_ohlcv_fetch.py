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
    timeframes = {"1h": 60}

    def __init__(self, start_ms: int, rows: int, frame_ms: int) -> None:
        self.start_ms = start_ms
        self.rows = rows
        self.frame_ms = frame_ms
        self.market = {"BTC/EUR": {"id": "BTC/EUR"}}
        self.calls: list[dict[str, object]] = []
        self.public_calls: list[dict[str, object]] = []
        self.manual_limit = 720

    def load_markets(self):
        return self.market

    def market(self, symbol: str) -> dict[str, object]:
        info = self.market.get(symbol)
        if info is None:
            raise KeyError(symbol)
        return info

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        limit = limit or 720
        self.calls.append({
            "params": params or {},
            "since": since,
            "limit": limit,
        })
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

    def publicGetOHLC(self, request):
        self.public_calls.append(dict(request))
        since = float(request.get("since", 0))
        since_ms = int(since * 1000)
        start_idx = 0
        if since_ms > self.start_ms:
            start_idx = max(int((since_ms - self.start_ms) // self.frame_ms), 0)
        rows = []
        for offset in range(self.manual_limit):
            idx = start_idx + offset
            if idx >= self.rows:
                break
            rows.append(self._build_raw_row(idx))
        last_idx = start_idx + len(rows)
        last_ms = self.start_ms + last_idx * self.frame_ms
        return {"result": {"BTC/EUR": rows, "last": int(last_ms / 1000)}}

    def _build_row(self, index: int) -> list[float]:
        ts = self.start_ms + index * self.frame_ms
        return [ts, 1.0, 2.0, 0.5, 1.5, 100.0]

    def _build_raw_row(self, index: int) -> list[object]:
        ts = (self.start_ms + index * self.frame_ms) // 1000
        return [
            ts,
            "1.0",
            "2.0",
            "0.5",
            "1.5",
            "100.0",
            "100.0",
            0,
        ]


class LateChunkKrakenExchange(FakeKrakenExchange):
    """Exchange that returns the most recent window when pagination is requested."""

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        limit = limit or 720
        if params and params.get("paginate"):
            start_idx = max(self.rows - limit, 0)
            count = min(limit, self.rows - start_idx)
            return [self._build_row(start_idx + i) for i in range(count)]

        return super().fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params=params)


class AutoPagingKrakenExchange(FakeKrakenExchange):
    """Exchange that fulfils the request entirely via ccxt pagination."""

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        limit = limit or 720
        self.calls.append({
            "params": params or {},
            "since": since,
            "limit": limit,
        })
        if params and params.get("paginate"):
            return [self._build_row(i) for i in range(self.rows)]
        pytest.fail("Manual pagination should not be triggered when auto pagination succeeds")

    def publicGetOHLC(self, request):  # pragma: no cover - autopagination path only
        pytest.fail("publicGetOHLC should not be called when auto pagination succeeds")


class TradesKrakenExchange(LateChunkKrakenExchange):
    """Exchange that exposes deep history via trades instead of OHLCV."""

    def __init__(self, start_ms: int, rows: int, frame_ms: int, trade_frame_ms: int) -> None:
        super().__init__(start_ms=start_ms, rows=rows, frame_ms=frame_ms)
        self.trade_frame_ms = trade_frame_ms
        self.trade_calls: list[dict[str, object]] = []

    def _build_trade(self, index: int) -> dict[str, object]:
        ts = self.start_ms + index * self.trade_frame_ms
        price = 1.0 + (index % 10) * 0.01
        amount = 0.5 + (index % 5) * 0.1
        return {"timestamp": ts, "price": price, "amount": amount}

    def fetch_trades(self, symbol, since=None, limit=None, params=None):
        limit = limit or 1000
        self.trade_calls.append({
            "since": since,
            "limit": limit,
            "params": params or {},
        })
        start_idx = 0
        if since is not None:
            start_idx = max(int((since - self.start_ms) // self.trade_frame_ms), 0)

        trades: list[dict[str, object]] = []
        for offset in range(limit):
            idx = start_idx + offset
            # Provide trades for 120 days worth of data.
            if idx >= 120 * 24 * (self.frame_ms // self.trade_frame_ms):
                break
            trades.append(self._build_trade(idx))
        return trades


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
    assert fake_exchange.public_calls, "publicGetOHLC should be used for manual pagination"


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
    assert fake_exchange.public_calls, "publicGetOHLC should drive the manual backfill"


def test_fetch_ohlcv_uses_ccxt_pagination_when_available(monkeypatch):
    hours = 90 * 24
    frame_ms = 60 * 60 * 1000
    start_ms = 0
    end_ms = start_ms + hours * frame_ms

    fake_exchange = AutoPagingKrakenExchange(start_ms=start_ms, rows=hours, frame_ms=frame_ms)
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
    assert len(fake_exchange.calls) == 1
    params = fake_exchange.calls[0]["params"]
    assert isinstance(params, dict)
    assert params.get("paginate") is True
    expected_calls = max(1, min(result.max_pages, 200))
    assert params.get("paginationCalls") == expected_calls
    assert params.get("until") == end_ms
    assert fake_exchange.public_calls == []


def test_fetch_ohlcv_downloads_dataset_when_api_history_is_short(monkeypatch):
    frame_ms = 60 * 60 * 1000
    # The exchange only exposes the latest 30 days.
    exchange_rows = 30 * 24
    start_ms = 0
    end_ms = start_ms + 120 * 24 * frame_ms

    fake_exchange = LateChunkKrakenExchange(start_ms=start_ms, rows=exchange_rows, frame_ms=frame_ms)
    fake_module = types.SimpleNamespace(
        kraken=lambda: fake_exchange,
        Exchange=_FakeExchangeNamespace(timeframe_seconds=frame_ms // 1000),
        RateLimitExceeded=Exception,
    )
    monkeypatch.setattr(fetcher, "ccxt", fake_module)

    dataset_rows: list[list[float]] = []
    dataset_hours = 120 * 24
    for idx in range(dataset_hours):
        ts = start_ms + idx * frame_ms
        dataset_rows.append([ts, 1.0, 2.0, 0.5, 1.5, 25.0])

    calls: list[tuple] = []

    def fake_dataset(symbol, timeframe, since_ms, target_end_ms, tolerance_ms):
        calls.append((symbol, timeframe, since_ms, target_end_ms, tolerance_ms))
        return dataset_rows

    monkeypatch.setattr(fetcher, "_kraken_dataset_rows", fake_dataset)

    result = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=end_ms,
        limit_per_page=720,
    )

    assert calls, "dataset fallback should be triggered"
    assert result.rows == dataset_hours
    assert result.reached_target is True
    assert result.coverage is not None
    assert result.coverage >= pd.Timedelta(days=120) - pd.Timedelta(milliseconds=frame_ms)


def test_fetch_ohlcv_backfills_with_trades(monkeypatch):
    frame_ms = 60 * 60 * 1000
    # Exchange exposes only 30 days of OHLCV data but trades cover 120 days.
    exchange_rows = 30 * 24
    start_ms = 0
    end_ms = start_ms + 120 * 24 * frame_ms

    fake_exchange = TradesKrakenExchange(
        start_ms=start_ms, rows=exchange_rows, frame_ms=frame_ms, trade_frame_ms=15 * 60 * 1000
    )
    fake_module = types.SimpleNamespace(
        kraken=lambda: fake_exchange,
        Exchange=_FakeExchangeNamespace(timeframe_seconds=frame_ms // 1000),
        RateLimitExceeded=Exception,
    )
    monkeypatch.setattr(fetcher, "ccxt", fake_module)

    dataset_calls: list[tuple] = []

    def fake_dataset(symbol, timeframe, since_ms, target_end_ms, tolerance_ms):
        dataset_calls.append((symbol, timeframe, since_ms, target_end_ms, tolerance_ms))
        return []

    monkeypatch.setattr(fetcher, "_kraken_dataset_rows", fake_dataset)

    result = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=end_ms,
        limit_per_page=720,
    )

    assert result.rows == 120 * 24
    assert result.reached_target is True
    assert fake_exchange.trade_calls, "trade-based backfill should be used"
    assert dataset_calls == []
