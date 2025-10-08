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

    def _update_bounds(ts: int) -> None:
        nonlocal min_ts, max_ts
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts

    def _ingest(rows: list[list[float]]) -> bool:
        added = False
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
        return added

    def _has_target_coverage() -> bool:
        if min_ts is None or max_ts is None:
            return False
        coverage_ms = max_ts - min_ts
        return desired_span_ms <= 0 or coverage_ms + tolerance_ms >= desired_span_ms

    params = {"paginate": True, "paginationCalls": page_budget}
    try:
        raw_rows = ex.fetch_ohlcv(
            sym,
            timeframe,
            since=since_ms,
            limit=limit_per_page,
            params=params,
        )
    except TypeError:
        raw_rows = None
    except Exception:
        raw_rows = None
    else:
        pages_used += 1
        rows_list = list(raw_rows or [])
        if rows_list:
            _ingest(rows_list)
            pages_hint = min(page_budget, max(1, math.ceil(len(rows_list) / max(limit_per_page, 1))))
        else:
            pages_hint = 1

    def _forward_fill(start_cursor: int, pages_so_far: int) -> int:
        cursor = start_cursor
        idle = 0
        max_idle = 3
        while pages_so_far < page_budget and cursor <= target_end_ms + tolerance_ms:
            try:
                chunk = ex.fetch_ohlcv(sym, timeframe, since=int(cursor), limit=limit_per_page)
            except rate_limit_exc:
                time.sleep(1.25)
                continue
            except Exception:
                break
            pages_so_far += 1
            if not chunk:
                idle += 1
                if idle >= max_idle:
                    cursor += max(chunk_span_ms, timeframe_ms)
                    idle = 0
                else:
                    cursor += timeframe_ms
                time.sleep(0.2)
                continue

            idle = 0
            last_ts = cursor
            added = False
            for row in chunk:
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
                if ts > last_ts:
                    last_ts = ts

            time.sleep(0.2)

            if added:
                cursor = last_ts + timeframe_ms
            else:
                cursor += timeframe_ms

            if _has_target_coverage():
                break
        return pages_so_far

    def _backfill(pages_so_far: int) -> int:
        if min_ts is None:
            return pages_so_far

        cursor = min_ts - chunk_span_ms
        idle = 0
        max_idle = 3
        while pages_so_far < page_budget and cursor + tolerance_ms >= since_ms:
            raw_cursor = cursor
            cursor = max(cursor, 0)
            try:
                chunk = ex.fetch_ohlcv(sym, timeframe, since=int(cursor), limit=limit_per_page)
            except rate_limit_exc:
                time.sleep(1.25)
                continue
            except Exception:
                break

            pages_so_far += 1
            if not chunk:
                idle += 1
                if idle >= max_idle:
                    if raw_cursor <= 0:
                        break
                    cursor = raw_cursor - max(chunk_span_ms, timeframe_ms)
                    idle = 0
                else:
                    cursor = raw_cursor - timeframe_ms
                time.sleep(0.2)
                continue

            idle = 0
            added = False
            oldest = min_ts
            for row in chunk:
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
                if oldest is None or ts < oldest:
                    oldest = ts

            time.sleep(0.2)

            if added:
                current_min = min_ts if min_ts is not None else oldest
                if current_min is None:
                    cursor -= chunk_span_ms
                else:
                    cursor = current_min - chunk_span_ms
            else:
                if raw_cursor <= 0:
                    break
                cursor = raw_cursor - max(chunk_span_ms, timeframe_ms)

            if _has_target_coverage():
                break
        return pages_so_far

    if not _has_target_coverage():
        start_cursor = since_ms if max_ts is None else max(max_ts + timeframe_ms, since_ms)
        pages_used = _forward_fill(start_cursor, pages_used)

    if not _has_target_coverage() and min_ts is not None and min_ts > since_ms + tolerance_ms:
        pages_used = _backfill(pages_used)

    if not _has_target_coverage():
        start_cursor = since_ms if max_ts is None else max(max_ts + timeframe_ms, since_ms)
        pages_used = _forward_fill(start_cursor, pages_used)

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

