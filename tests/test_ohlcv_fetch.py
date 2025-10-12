from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import ohlcv_fetcher as fetcher


def _build_dataset_zip(
    path: Path,
    pair: str,
    timeframe: str,
    start_ms: int,
    frame_ms: int,
    rows: int,
) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["time", "open", "high", "low", "close", "volume", "count"])
    for index in range(rows):
        ts_ms = start_ms + index * frame_ms
        ts_sec = ts_ms // 1000
        price = 1.0 + index * 0.1
        writer.writerow([ts_sec, price, price + 1, price - 0.5, price + 0.2, 100.0 + index, index + 1])

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"ohlc/{pair}/{timeframe}.csv", buffer.getvalue())


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


def test_fetch_ohlcv_falls_back_to_manual_pagination(monkeypatch):
    hours = 60 * 24
    frame_ms = 60 * 60 * 1000
    start_ms = 0
    end_ms = start_ms + hours * frame_ms

    fake_exchange = FakeKrakenExchange(start_ms=start_ms, rows=hours, frame_ms=frame_ms)
    monkeypatch.setattr(fetcher.ccxt, "kraken", lambda: fake_exchange)

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


def test_fetch_ohlcv_uses_cache_roundtrip(tmp_path: Path, monkeypatch):
    frame_ms = 60 * 60 * 1000
    start_ms = 0
    dataset_rows = 3
    total_first_call_rows = 5
    cache_root = tmp_path / "cache"
    zip_path = tmp_path / "ohlc.zip"
    _build_dataset_zip(zip_path, "BTCEUR", "1h", start_ms, frame_ms, dataset_rows)

    download_calls = 0

    def fake_download(root: Path) -> Path:
        nonlocal download_calls
        download_calls += 1
        root.mkdir(parents=True, exist_ok=True)
        dest = root / fetcher._KRAKEN_ZIP_FILENAME
        dest.write_bytes(zip_path.read_bytes())
        return dest

    fake_exchange = FakeKrakenExchange(start_ms=start_ms, rows=total_first_call_rows, frame_ms=frame_ms)

    monkeypatch.setattr(fetcher, "_download_kraken_zip", fake_download)
    monkeypatch.setattr(fetcher.ccxt, "kraken", lambda: fake_exchange)

    result_first = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=start_ms + frame_ms * 6,
        limit_per_page=720,
        cache_root=cache_root,
    )

    assert result_first.rows == total_first_call_rows
    assert download_calls == 1
    cached_after_first = fetcher._read_cached_dataset("BTC/EUR", "1h", cache_root)
    assert len(cached_after_first) == total_first_call_rows

    fake_exchange.rows = total_first_call_rows + 1

    result_second = fetcher.fetch_ohlcv(
        symbol="BTC/EUR",
        timeframe="1h",
        since_ms=start_ms,
        target_end_ms=start_ms + frame_ms * 7,
        limit_per_page=720,
        cache_root=cache_root,
    )

    assert result_second.rows == total_first_call_rows + 1
    assert download_calls == 1
    cached_after_second = fetcher._read_cached_dataset("BTC/EUR", "1h", cache_root)
    assert len(cached_after_second) == total_first_call_rows + 1
    expected_last_ts = pd.to_datetime(start_ms + frame_ms * (total_first_call_rows), unit="ms", utc=True)
    assert cached_after_second.index.max() == expected_last_ts
