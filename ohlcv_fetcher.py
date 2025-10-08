from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional

try:  # pragma: no cover - optional dependency
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxt = None  # type: ignore
import pandas as pd


@dataclass(slots=True)
class OHLCVFetchResult:
    data: pd.DataFrame
    pages: int
    limit_per_page: int
    max_pages: int
    requested_start: Optional[pd.Timestamp] = None
    requested_end: Optional[pd.Timestamp] = None
    target_coverage: Optional[pd.Timedelta] = None
    reached_target: bool = False

    @property
    def rows(self) -> int:
        return int(len(self.data))

    @property
    def start(self) -> Optional[pd.Timestamp]:
        if self.data.empty:
            return None
        return self.data.index.min()

    @property
    def end(self) -> Optional[pd.Timestamp]:
        if self.data.empty:
            return None
        return self.data.index.max()

    @property
    def coverage(self) -> Optional[pd.Timedelta]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    @property
    def truncated(self) -> bool:
        return not self.reached_target


def window_to_timedelta(window_value: int, window_unit: str) -> timedelta:
    unit = window_unit.lower()
    if unit.startswith("day"):
        return timedelta(days=window_value)
    if unit.startswith("month"):
        return timedelta(days=30 * window_value)
    if unit.startswith("year"):
        return timedelta(days=365 * window_value)
    return timedelta(days=window_value)


class MissingCcxtError(RuntimeError):
    """Raised when a ccxt-dependent feature is used without the library installed."""


def _ensure_ccxt() -> "ccxt":
    if ccxt is None:
        raise MissingCcxtError("ccxt is required for live exchange operations. Install it with `pip install ccxt`.")
    return ccxt


def _parse8601(value: str) -> Optional[int]:
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.timestamp() * 1000)


def resolve_since(window_value: int, window_unit: str, override: str = "") -> tuple[int, datetime, timedelta]:
    override_clean = (override or "").strip()
    if override_clean:
        parsed = _parse8601(override_clean)
        if parsed is None:
            raise ValueError("Invalid 'Since' value. Use ISO8601 like 2022-01-01T00:00:00Z")
        dt = datetime.fromtimestamp(parsed / 1000.0, tz=UTC)
        delta = window_to_timedelta(window_value, window_unit)
        return parsed, dt, delta

    delta = window_to_timedelta(window_value, window_unit)

    now = datetime.now(UTC)
    start_dt = now - delta
    since_ms = int(start_dt.timestamp() * 1000)
    return since_ms, start_dt, delta


def normalize_symbol(exchange: "ccxt.Exchange", symbol: str) -> str:
    try:
        markets = exchange.load_markets()
    except Exception:
        return symbol
    if symbol in markets:
        return symbol
    if symbol.upper().startswith("BTC/"):
        alt = "XBT/" + symbol.split("/", 1)[1]
        if alt in markets:
            return alt
    up = symbol.upper()
    if up in markets:
        return up
    return symbol


def _fallback_parse_timeframe(timeframe: str) -> int:
    unit = timeframe[-1:].lower()
    try:
        value = float(timeframe[:-1] or 1)
    except ValueError:
        return 60
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
    }
    seconds = multipliers.get(unit)
    if seconds is None:
        return 60
    return int(max(value * seconds, 1))


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    target_end_ms: Optional[int] = None,
    limit_per_page: int = 720,
) -> OHLCVFetchResult:
    module = _ensure_ccxt()
    if not hasattr(module, "kraken"):
        raise MissingCcxtError("ccxt.kraken() is unavailable; ensure the exchange is supported by your ccxt build.")

    ex = module.kraken()
    sym = normalize_symbol(ex, symbol)
    timeframe_seconds = None
    exchange_cls = getattr(module, "Exchange", None)
    if exchange_cls is not None and hasattr(exchange_cls, "parse_timeframe"):
        try:
            timeframe_seconds = exchange_cls.parse_timeframe(timeframe)
        except Exception:
            timeframe_seconds = None
    if timeframe_seconds is None:
        timeframe_seconds = _fallback_parse_timeframe(timeframe)
    timeframe_ms = max(int(timeframe_seconds * 1000), 1)

    if target_end_ms is None:
        target_end_ms = int(datetime.now(UTC).timestamp() * 1000)
    if target_end_ms <= since_ms:
        target_end_ms = since_ms + timeframe_ms

    desired_span_ms = max(0, target_end_ms - since_ms)
    chunk_span_ms = limit_per_page * timeframe_ms
    approx_pages = math.ceil(desired_span_ms / chunk_span_ms) if chunk_span_ms else 1
    page_budget = max(min(approx_pages + 5, 2400), 4)

    tolerance_ms = max(timeframe_ms, 60_000)
    rate_limit_exc = getattr(module, "RateLimitExceeded", Exception)

    all_rows: list[list[float]] = []
    seen_ts: set[int] = set()
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None
    pages_used = 0
    pages_hint = 0

    def _estimate_pages(row_count: int) -> int:
        if row_count <= 0:
            return 0
        return math.ceil(row_count / max(limit_per_page, 1))

    def _update_bounds(ts: int) -> None:
        nonlocal min_ts, max_ts
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts

    def _has_target_coverage() -> bool:
        if min_ts is None or max_ts is None:
            return False
        coverage_ms = max_ts - min_ts
        return desired_span_ms <= 0 or coverage_ms + tolerance_ms >= desired_span_ms

    market_id = sym
    market_info = None
    try:
        market_info = ex.market(sym)
        market_id = market_info.get("id", market_id)
    except Exception:
        market_info = None

    timeframes_map = getattr(ex, "timeframes", None)
    interval_minutes: Optional[int] = None
    if isinstance(timeframes_map, dict):
        value = timeframes_map.get(timeframe)
        if value is not None:
            try:
                interval_minutes = int(value)
            except Exception:
                interval_minutes = None
    if interval_minutes is None and timeframe_seconds:
        try:
            interval_minutes = max(int(timeframe_seconds // 60), 1)
        except Exception:
            interval_minutes = None

    manual_uses_public = hasattr(ex, "publicGetOHLC")

    def _parse_public_rows(raw_rows: list) -> list[list[float]]:
        if not raw_rows:
            return []
        if market_info is not None and hasattr(ex, "parse_ohlcvs"):
            try:
                parsed = ex.parse_ohlcvs(raw_rows, market_info, timeframe, None, limit_per_page)
                if isinstance(parsed, list):
                    return [list(item) for item in parsed]
            except Exception:
                pass

        parsed_rows: list[list[float]] = []
        for row in raw_rows:
            if not row:
                continue
            try:
                ts_raw = row[0]
                ts = int(float(ts_raw) * 1000)
            except Exception:
                continue
            try:
                open_ = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
            except Exception:
                continue
            volume_index = 6 if len(row) > 6 else 5
            try:
                volume = float(row[volume_index])
            except Exception:
                volume = 0.0
            parsed_rows.append([ts, open_, high, low, close, volume])
        return parsed_rows

    def _fetch_chunk(cursor: int) -> tuple[Optional[list[list[float]]], Optional[int], bool]:
        if manual_uses_public and interval_minutes:
            request = {"pair": market_id, "interval": interval_minutes}
            frame_seconds = max(int(timeframe_seconds or 60), 1)
            try:
                since_seconds = max(int(cursor // 1000), 0)
            except Exception:
                since_seconds = 0
            request["since"] = str(max(since_seconds - frame_seconds, 0))
            try:
                response = ex.publicGetOHLC(request)
            except rate_limit_exc:
                time.sleep(1.25)
                return None, None, False
            except Exception:
                return None, None, False

            result = {}
            if isinstance(response, dict):
                result = response.get("result", {}) or {}
            raw_rows = []
            if isinstance(result, dict):
                raw_rows = result.get(market_id, []) or []
            rows = _parse_public_rows(list(raw_rows))

            last_ms: Optional[int] = None
            if isinstance(result, dict) and "last" in result:
                try:
                    last_ms = int(float(result["last"]) * 1000)
                except Exception:
                    last_ms = None

            return rows, last_ms, True

        try:
            rows = ex.fetch_ohlcv(
                sym,
                timeframe,
                since=int(cursor),
                limit=limit_per_page,
            )
        except rate_limit_exc:
            time.sleep(1.25)
            return None, None, False
        except Exception:
            return None, None, False
        return rows or [], None, True

    def _ingest_chunk(rows: Optional[list[list[float]]]) -> tuple[bool, Optional[int]]:
        if not rows:
            return False, None
        added = False
        newest = None
        for row in rows:
            if not row:
                continue
            ts = int(row[0])
            if ts < since_ms or ts > target_end_ms + tolerance_ms:
                continue
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            all_rows.append(row)
            _update_bounds(ts)
            added = True
            if newest is None or ts > newest:
                newest = ts
        return added, newest

    if limit_per_page > 0:
        auto_calls = max(1, min(page_budget, 200))
        auto_params = {"paginate": True, "paginationCalls": auto_calls, "until": target_end_ms}
        try:
            auto_rows = ex.fetch_ohlcv(
                sym,
                timeframe,
                since=int(since_ms),
                limit=limit_per_page,
                params=auto_params,
            )
        except Exception:
            auto_rows = []

        if auto_rows:
            estimated = _estimate_pages(len(auto_rows))
            if estimated:
                pages_hint = max(pages_hint, estimated)
                pages_used = max(pages_used, min(page_budget, estimated))
            _ingest_chunk(auto_rows)

    if not _has_target_coverage():
        first_chunk, _, ok = _fetch_chunk(since_ms)
        if ok:
            pages_used += 1
            added, newest = _ingest_chunk(first_chunk)
            if added and newest is not None:
                pages_hint = max(pages_hint, 1)
            elif first_chunk:
                pages_hint = max(pages_hint, 1)

    def _forward_paginate(start_cursor: int, pages_so_far: int) -> int:
        cursor = start_cursor
        idle = 0
        max_idle = 3
        while pages_so_far < page_budget and cursor <= target_end_ms + tolerance_ms:
            chunk, hint_cursor, ok = _fetch_chunk(cursor)
            if not ok:
                idle += 1
                if idle >= max_idle:
                    cursor += max(chunk_span_ms, timeframe_ms)
                    idle = 0
                else:
                    cursor += timeframe_ms
                continue

            pages_so_far += 1
            added, newest = _ingest_chunk(chunk)
            if added and newest is not None:
                next_cursor = newest + timeframe_ms
                if hint_cursor is not None:
                    next_cursor = max(next_cursor, hint_cursor)
                if next_cursor <= cursor:
                    cursor += max(chunk_span_ms, timeframe_ms)
                else:
                    cursor = next_cursor
                idle = 0
            else:
                idle += 1
                if idle >= max_idle:
                    cursor += max(chunk_span_ms, timeframe_ms)
                    idle = 0
                else:
                    cursor += timeframe_ms

            if _has_target_coverage():
                break

            time.sleep(0.2)

        return pages_so_far

    def _backfill_paginate(pages_so_far: int) -> int:
        if min_ts is None:
            return pages_so_far

        cursor = max(min_ts - chunk_span_ms, since_ms)
        idle = 0
        max_idle = 3
        while pages_so_far < page_budget and cursor + tolerance_ms >= since_ms:
            chunk, _, ok = _fetch_chunk(cursor)
            if not ok:
                idle += 1
                if idle >= max_idle:
                    if cursor <= since_ms:
                        break
                    cursor = max(cursor - max(chunk_span_ms, timeframe_ms), since_ms)
                    idle = 0
                else:
                    cursor = max(cursor - timeframe_ms, since_ms)
                continue

            pages_so_far += 1
            added, _ = _ingest_chunk(chunk)
            if added:
                cursor = max((min_ts or since_ms) - chunk_span_ms, since_ms)
                idle = 0
            else:
                idle += 1
                if idle >= max_idle:
                    if cursor <= since_ms:
                        break
                    cursor = max(cursor - max(chunk_span_ms, timeframe_ms), since_ms)
                    idle = 0
                else:
                    cursor = max(cursor - timeframe_ms, since_ms)

            if _has_target_coverage():
                break

            time.sleep(0.2)

        return pages_so_far

    if max_ts is None or max_ts + tolerance_ms < target_end_ms:
        start_cursor = since_ms if max_ts is None else max(max_ts + timeframe_ms, since_ms)
        pages_used = _forward_paginate(start_cursor, pages_used)

    if not _has_target_coverage() and min_ts is not None and min_ts > since_ms + tolerance_ms:
        pages_used = _backfill_paginate(pages_used)

    if max_ts is not None and max_ts + tolerance_ms < target_end_ms and not _has_target_coverage():
        start_cursor = since_ms if max_ts is None else max(max_ts + timeframe_ms, since_ms)
        pages_used = _forward_paginate(start_cursor, pages_used)

    unique_rows = len(all_rows)
    if pages_hint <= 0 and unique_rows:
        pages_hint = math.ceil(unique_rows / max(limit_per_page, 1))

    pages = max(pages_hint, pages_used)

    if not all_rows:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty.index = pd.DatetimeIndex([], tz="UTC")
        return OHLCVFetchResult(
            empty,
            pages=pages,
            limit_per_page=limit_per_page,
            max_pages=page_budget,
            requested_start=pd.to_datetime(since_ms, unit="ms", utc=True),
            requested_end=pd.to_datetime(target_end_ms, unit="ms", utc=True),
            target_coverage=pd.to_timedelta(desired_span_ms, unit="ms"),
            reached_target=False,
        )

    all_rows.sort(key=lambda item: item[0])

    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).set_index("Timestamp").sort_index()
    requested_start = pd.to_datetime(since_ms, unit="ms", utc=True)
    requested_end = pd.to_datetime(target_end_ms, unit="ms", utc=True)
    target_coverage = pd.to_timedelta(desired_span_ms, unit="ms")
    coverage = df.index.max() - df.index.min() if not df.empty else pd.Timedelta(0)
    tolerance = pd.to_timedelta(tolerance_ms, unit="ms")
    reached_target = desired_span_ms <= 0 or coverage + tolerance >= target_coverage
    return OHLCVFetchResult(
        df,
        pages=pages,
        limit_per_page=limit_per_page,
        max_pages=page_budget,
        requested_start=requested_start,
        requested_end=requested_end,
        target_coverage=target_coverage,
        reached_target=reached_target,
    )

