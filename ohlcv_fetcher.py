from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from urllib.request import Request, urlopen, urlretrieve
import zipfile
from io import BytesIO

import ccxt
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


_KRAKEN_DATASET_URL = "https://data.kraken.com/ohlc.zip"
_KRAKEN_ZIP_FILENAME = "kraken_ohlc.zip"
_KRAKEN_CACHE_FILENAME = "ohlcv.csv"
_KRAKEN_ZIP_MAX_AGE = timedelta(hours=6)
_KRAKEN_DATASET_MAX_AGE = timedelta(hours=6)
_CACHE_DIRNAME = Path("data/kraken_cache")
_OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _kraken_cache_root(cache_root: Optional[Path] = None) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser().resolve()
    return (Path(__file__).resolve().parent / _CACHE_DIRNAME).resolve()


def _symbol_to_cache_pair(symbol: str) -> str:
    return symbol.replace("/", "").replace(" ", "").upper()


def _cache_dataset_path(symbol: str, timeframe: str, cache_root: Path) -> Path:
    pair = _symbol_to_cache_pair(symbol)
    return cache_root / pair / timeframe / _KRAKEN_CACHE_FILENAME


def _is_stale(path: Path, max_age: timedelta) -> bool:
    if not path.exists():
        return True
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except (OSError, ValueError):
        return True
    return datetime.now(UTC) - modified > max_age


def _format_size(size: Optional[int]) -> str:
    if size is None or size <= 0:
        return "an unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = "B"
    for unit_name in units:
        unit = unit_name
        if value < 1024.0 or unit_name == units[-1]:
            break
        value /= 1024.0
    if value >= 10 or unit == "B":
        formatted = f"{value:.0f}"
    else:
        formatted = f"{value:.1f}"
    return f"{formatted} {unit}"


def _kraken_zip_remote_size() -> Optional[int]:
    try:
        request = Request(_KRAKEN_DATASET_URL, method="HEAD")
        with urlopen(request) as response:  # type: ignore[arg-type]
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:
        pass

    try:
        with urlopen(_KRAKEN_DATASET_URL) as response:  # type: ignore[arg-type]
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:
        return None
    return None


def _auto_confirm_env() -> Optional[bool]:
    value = os.environ.get("KRAKEN_CACHE_AUTO_CONFIRM")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


ConfirmDownloadCallback = Callable[[Optional[int]], bool]


def _confirm_dataset_download(
    size: Optional[int],
    override: Optional[ConfirmDownloadCallback] = None,
) -> bool:
    size_label = _format_size(size)
    env_choice = _auto_confirm_env()
    if env_choice is not None:
        if env_choice:
            print(
                f"Automatically approving Kraken OHLC dataset download (~{size_label}) "
                "due to KRAKEN_CACHE_AUTO_CONFIRM."
            )
        else:
            print(
                f"Skipping Kraken OHLC dataset download (~{size_label}) "
                "due to KRAKEN_CACHE_AUTO_CONFIRM."
            )
        return env_choice

    if override is not None:
        return override(size)

    if not sys.stdin or not sys.stdin.isatty() or not sys.stdout or not sys.stdout.isatty():
        print(
            "Kraken OHLC dataset download requires confirmation, but no interactive "
            "terminal is available. Set KRAKEN_CACHE_AUTO_CONFIRM=1 to permit the download."
        )
        return False

    prompt = (
        f"The Kraken OHLC dataset is approximately {size_label}. "
        "Proceed with download? [y/N]: "
    )
    try:
        response = input(prompt)
    except EOFError:
        return False
    return response.strip().lower() in {"y", "yes"}


def _download_kraken_zip(
    cache_root: Path,
    confirm_download: Optional[ConfirmDownloadCallback] = None,
) -> Optional[Path]:
    zip_path = cache_root / _KRAKEN_ZIP_FILENAME
    if zip_path.exists() and not _is_stale(zip_path, _KRAKEN_ZIP_MAX_AGE):
        return zip_path

    cache_root.mkdir(parents=True, exist_ok=True)
    size_bytes = _kraken_zip_remote_size()
    if not _confirm_dataset_download(size_bytes, override=confirm_download):
        if zip_path.exists():
            print(
                "Download cancelled; using existing Kraken OHLC dataset despite staleness."
            )
            return zip_path
        print("Download cancelled; Kraken OHLC dataset not available in cache.")
        return None

    tmp_path = zip_path.with_suffix(".tmp")
    try:
        urlretrieve(_KRAKEN_DATASET_URL, tmp_path)
        tmp_path.replace(zip_path)
        print(
            f"Kraken OHLC dataset downloaded to {zip_path} (~{_format_size(size_bytes)})."
        )
        return zip_path
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if zip_path.exists():
            return zip_path
    return None


def _read_dataset_from_stream(stream: BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(stream)
    except Exception:
        return pd.DataFrame(columns=_OHLC_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)

    lower_cols = {col.lower(): col for col in df.columns}
    time_col = lower_cols.get("time")
    if time_col is None:
        return pd.DataFrame(columns=_OHLC_COLUMNS)

    ts = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
    frame = pd.DataFrame(index=ts)
    frame.index.name = "Timestamp"

    for col in _OHLC_COLUMNS:
        source_col = lower_cols.get(col.lower())
        if source_col is None:
            frame[col] = 0.0
        else:
            frame[col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)

    frame = frame[~frame.index.isna()]
    if frame.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _write_cached_dataset(symbol: str, timeframe: str, frame: pd.DataFrame, cache_root: Path) -> Path:
    dest_path = _cache_dataset_path(symbol, timeframe, cache_root)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return dest_path

    ordered = frame.sort_index()
    for col in _OHLC_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = 0.0
    ordered = ordered[_OHLC_COLUMNS]
    to_write = ordered.reset_index()
    to_write["Timestamp"] = to_write["Timestamp"].dt.tz_convert("UTC")
    to_write.to_csv(dest_path, index=False)
    return dest_path


def _read_cached_dataset(symbol: str, timeframe: str, cache_root: Path) -> pd.DataFrame:
    path = _cache_dataset_path(symbol, timeframe, cache_root)
    if not path.exists():
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    try:
        df = pd.read_csv(path, parse_dates=["Timestamp"])
    except Exception:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    if df.empty or "Timestamp" not in df:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    frame = df.assign(Timestamp=ts).dropna(subset=["Timestamp"]).set_index("Timestamp")
    for col in _OHLC_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame[_OHLC_COLUMNS]
    frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _extract_kraken_dataset(zip_path: Path, symbol: str, timeframe: str, cache_root: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            pair = _symbol_to_cache_pair(symbol)
            pair_lower = pair.lower()
            timeframe_name = f"{timeframe.lower()}.csv"
            target_name: Optional[str] = None
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if not parts:
                    continue
                filename = parts[-1].lower()
                if filename != timeframe_name:
                    continue
                if any(part.lower() == pair_lower for part in parts):
                    target_name = info.filename
                    break
            if target_name is None:
                return pd.DataFrame(columns=_OHLC_COLUMNS)
            with zf.open(target_name) as handle:
                data = handle.read()
    except Exception:
        return pd.DataFrame(columns=_OHLC_COLUMNS)

    frame = _read_dataset_from_stream(BytesIO(data))
    _write_cached_dataset(symbol, timeframe, frame, cache_root)
    return frame


def _kraken_dataset_rows(
    symbol: str,
    timeframe: str,
    cache_root: Optional[Path] = None,
    confirm_download: Optional[ConfirmDownloadCallback] = None,
    required_start_ms: Optional[int] = None,
    required_end_ms: Optional[int] = None,
    tolerance_ms: int = 0,
) -> pd.DataFrame:
    root = _kraken_cache_root(cache_root)
    cached = _read_cached_dataset(symbol, timeframe, root)

    def _covers_required(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False

        start_ok = True
        end_ok = True
        if required_start_ms is not None:
            frame_start_ms = int(frame.index.min().timestamp() * 1000)
            start_ok = frame_start_ms - tolerance_ms <= required_start_ms
        if required_end_ms is not None:
            frame_end_ms = int(frame.index.max().timestamp() * 1000)
            end_ok = frame_end_ms + tolerance_ms >= required_end_ms
        return start_ok and end_ok

    cache_path = _cache_dataset_path(symbol, timeframe, root)
    cache_fresh = not cached.empty and not _is_stale(cache_path, _KRAKEN_DATASET_MAX_AGE)
    if cache_fresh and _covers_required(cached):
        return cached

    needs_download = not cache_fresh or not _covers_required(cached)

    zip_path: Optional[Path] = None
    if needs_download:
        zip_path = _download_kraken_zip(root, confirm_download=confirm_download)
        if zip_path is not None:
            fresh = _extract_kraken_dataset(zip_path, symbol, timeframe, root)
            if not fresh.empty:
                if _covers_required(fresh):
                    return fresh
                cached = fresh
                cache_fresh = True

    if cache_fresh and _covers_required(cached):
        return cached

    if not cached.empty:
        return cached
    return pd.DataFrame(columns=_OHLC_COLUMNS)


def window_to_timedelta(window_value: int, window_unit: str) -> timedelta:
    unit = window_unit.lower()
    if unit.startswith("day"):
        return timedelta(days=window_value)
    if unit.startswith("month"):
        return timedelta(days=30 * window_value)
    if unit.startswith("year"):
        return timedelta(days=365 * window_value)
    return timedelta(days=window_value)


def resolve_since(window_value: int, window_unit: str, override: str = "") -> tuple[int, datetime, timedelta]:
    override_clean = (override or "").strip()
    if override_clean:
        parsed = ccxt.kraken().parse8601(override_clean)
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


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    target_end_ms: Optional[int] = None,
    limit_per_page: int = 720,
    cache_root: Optional[Path | str] = None,
    confirm_download: Optional[ConfirmDownloadCallback] = None,
) -> OHLCVFetchResult:
    ex = ccxt.kraken()
    sym = normalize_symbol(ex, symbol)
    try:
        timeframe_seconds = ccxt.Exchange.parse_timeframe(timeframe)
    except Exception:
        timeframe_seconds = 60
    timeframe_ms = max(int(timeframe_seconds * 1000), 1)

    if target_end_ms is None:
        target_end_ms = int(datetime.now(UTC).timestamp() * 1000)
    if target_end_ms <= since_ms:
        target_end_ms = since_ms + timeframe_ms

    tolerance_ms = max(timeframe_ms, 60_000)
    cache_root_path = _kraken_cache_root(cache_root)
    dataset_symbol = sym
    dataset_rows = _kraken_dataset_rows(
        sym,
        timeframe,
        cache_root_path,
        confirm_download=confirm_download,
        required_start_ms=since_ms,
        required_end_ms=target_end_ms,
        tolerance_ms=tolerance_ms,
    )
    if dataset_rows.empty and symbol != sym:
        alt_rows = _kraken_dataset_rows(
            symbol,
            timeframe,
            cache_root_path,
            confirm_download=confirm_download,
            required_start_ms=since_ms,
            required_end_ms=target_end_ms,
            tolerance_ms=tolerance_ms,
        )
        if not alt_rows.empty:
            dataset_rows = alt_rows
            dataset_symbol = symbol

    desired_span_ms = max(0, target_end_ms - since_ms)
    if not dataset_rows.empty:
        dataset_start_ms = int(dataset_rows.index.min().timestamp() * 1000)
        dataset_end_ms = int(dataset_rows.index.max().timestamp() * 1000)
        missing_before_ms = max(0, since_ms - dataset_start_ms)
        missing_after_ms = max(0, target_end_ms - dataset_end_ms)
        if missing_before_ms <= 0:
            desired_span_ms = missing_after_ms
        else:
            desired_span_ms = missing_before_ms + missing_after_ms

    chunk_span_ms = limit_per_page * timeframe_ms
    approx_pages = math.ceil(desired_span_ms / chunk_span_ms) if chunk_span_ms else 1
    page_budget = max(min(approx_pages + 2, 120), 2)

    params = {"paginate": True, "paginationCalls": page_budget}
    all_rows: list[list[float]] = []
    pages = 0
    manual_attempted = False

    def _manual_fetch() -> tuple[list[list[float]], int]:
        rows: list[list[float]] = []
        seen_ts: set[int] = set()
        cursor = since_ms
        calls = 0
        idle = 0
        max_idle = 3
        while calls < page_budget and cursor <= target_end_ms + tolerance_ms:
            try:
                ohlcv_chunk = ex.fetch_ohlcv(sym, timeframe, since=int(cursor), limit=limit_per_page)
            except ccxt.RateLimitExceeded:
                time.sleep(1.25)
                continue
            except Exception:
                break
            if not ohlcv_chunk:
                idle += 1
                if idle >= max_idle:
                    cursor += max(chunk_span_ms, timeframe_ms)
                    idle = 0
                else:
                    cursor += timeframe_ms
                time.sleep(0.2)
                continue

            idle = 0
            calls += 1
            filtered: list[list[float]] = []
            last_ts = cursor
            for row in ohlcv_chunk:
                ts = int(row[0])
                if ts < since_ms or ts > target_end_ms + tolerance_ms:
                    continue
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                filtered.append(row)
                if ts > last_ts:
                    last_ts = ts

            if filtered:
                rows.extend(filtered)
                advance = last_ts + timeframe_ms
                if advance <= cursor:
                    advance = cursor + timeframe_ms
                cursor = advance
            else:
                cursor += timeframe_ms

            time.sleep(0.2)

            if rows:
                coverage_ms = rows[-1][0] - rows[0][0]
                if desired_span_ms <= 0 or coverage_ms + tolerance_ms >= desired_span_ms:
                    break
        return rows, calls

    manual_calls = 0

    try:
        raw_rows = ex.fetch_ohlcv(
            sym,
            timeframe,
            since=since_ms,
            limit=limit_per_page,
            params=params,
        )
        all_rows = list(raw_rows or [])
        if all_rows:
            pages = min(page_budget, max(1, math.ceil(len(all_rows) / max(limit_per_page, 1))))
    except TypeError:
        all_rows, manual_calls = _manual_fetch()
        pages = manual_calls or pages
        manual_attempted = True
    except Exception:
        all_rows, manual_calls = _manual_fetch()
        pages = manual_calls or pages
        manual_attempted = True

    if all_rows and not manual_attempted:
        ts_values = [int(row[0]) for row in all_rows if row]
        coverage_ms = (max(ts_values) - min(ts_values)) if ts_values else 0
        coverage_met = desired_span_ms <= 0 or coverage_ms + tolerance_ms >= desired_span_ms
        if not coverage_met:
            manual_rows, manual_calls = _manual_fetch()
            manual_attempted = True
            if manual_rows:
                all_rows.extend(manual_rows)
            if manual_calls:
                pages = max(pages, manual_calls)

    if (not all_rows) and not manual_attempted:
        manual_rows, manual_calls = _manual_fetch()
        manual_attempted = True
        if manual_rows:
            all_rows = manual_rows
        if manual_calls:
            pages = manual_calls

    filtered_rows: list[list[float]] = []
    seen_ts: set[int] = set()
    for row in all_rows:
        ts = int(row[0])
        if ts < since_ms or ts > target_end_ms + tolerance_ms:
            continue
        if ts in seen_ts:
            continue
        seen_ts.add(ts)
        filtered_rows.append(row)

    filtered_rows.sort(key=lambda item: item[0])
    df = pd.DataFrame(filtered_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).set_index("Timestamp").sort_index()
    requested_start = pd.to_datetime(since_ms, unit="ms", utc=True)
    requested_end = pd.to_datetime(target_end_ms, unit="ms", utc=True)
    target_coverage = pd.to_timedelta(desired_span_ms, unit="ms")
    tolerance = pd.to_timedelta(tolerance_ms, unit="ms")

    combined_frames = []
    if not dataset_rows.empty:
        combined_frames.append(dataset_rows)
    if not df.empty:
        combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames).sort_index()
    else:
        combined_df = pd.DataFrame(columns=_OHLC_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))

    if not combined_df.empty:
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        for col in _OHLC_COLUMNS:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
        combined_df = combined_df[_OHLC_COLUMNS]
        _write_cached_dataset(dataset_symbol, timeframe, combined_df, cache_root_path)
        if dataset_symbol != sym:
            _write_cached_dataset(sym, timeframe, combined_df, cache_root_path)
    else:
        _write_cached_dataset(dataset_symbol, timeframe, combined_df, cache_root_path)
        if dataset_symbol != sym:
            _write_cached_dataset(sym, timeframe, combined_df, cache_root_path)

    filtered_df = combined_df
    if not filtered_df.empty:
        upper_bound = requested_end + tolerance
        filtered_df = filtered_df.loc[(filtered_df.index >= requested_start) & (filtered_df.index <= upper_bound)]

    coverage = (
        filtered_df.index.max() - filtered_df.index.min()
        if not filtered_df.empty
        else pd.Timedelta(0)
    )
    reached_target = desired_span_ms <= 0 or coverage + tolerance >= target_coverage
    return OHLCVFetchResult(
        filtered_df,
        pages=pages,
        limit_per_page=limit_per_page,
        max_pages=page_budget,
        requested_start=requested_start,
        requested_end=requested_end,
        target_coverage=target_coverage,
        reached_target=reached_target,
    )

