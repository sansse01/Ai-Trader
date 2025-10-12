
import itertools
import json
import logging
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ccxt  # type: ignore

from ohlcv_fetcher import (
    OHLCVFetchResult,
    fetch_ohlcv as fetch_ohlcv_core,
    resolve_since,
)
from risk.engine import OrderContext, PortfolioState, RiskEngine, RiskEngineConfig
from strategies import DEFAULT_STRATEGY_KEY, STRATEGY_REGISTRY, register_strategy
from strategies.base import StrategyDefinition
from strategy_builder.backtest_svc import BacktestService
from strategy_builder.builder_graph import GraphValidator, OperatorCatalog, load_graph_from_json
from strategy_builder.codegen import GraphExecutor, build_param_schema, compile_and_load, render_strategy_code
from strategy_builder.datasvc import DataService
from strategy_builder.evaluator import accept as evaluator_accept
from strategy_builder.evaluator import choose_champion
from strategy_builder.llm_optimizer import LLMOptimizer
from strategy_builder.registry import StrategyRegistry
from strategy_builder.schemas import StrategyGraph

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="AI Trader Workspace", layout="wide")

GPT5_MODEL_CAPTION = (
    "**Model presets** — System card → API alias: "
    "gpt-5-thinking → gpt-5 · "
    "gpt-5-thinking-mini → gpt-5-mini · "
    "gpt-5-thinking-nano → gpt-5-nano · "
    "gpt-5-main → gpt-5-chat-latest. "
    "The `gpt-5-main-mini` card is not exposed via the API."
)


def is_gpt5_model(model: str) -> bool:
    """Return True when the API alias targets a GPT-5 family model."""

    return model.strip().lower().startswith("gpt-5")


def make_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serialisable primitives."""

    if isinstance(value, Mapping):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        return make_json_safe(value.model_dump())
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return make_json_safe(value.dict())
        except TypeError:
            pass
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).isoformat()
    return value


def _format_download_size(size: Optional[int]) -> str:
    if not size or size <= 0:
        return "an unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    formatted = f"{value:.0f}" if value >= 10 or unit == "B" else f"{value:.1f}"
    return f"{formatted} {unit}"


_KRAKEN_CONFIRM_STATE_KEY = "kraken_dataset_confirm::{symbol}::{timeframe}"


class KrakenDatasetDownloadDecisionRequired(RuntimeError):
    """Signal that a cached fetch needs user confirmation to continue."""

    def __init__(self, size: Optional[int]):
        super().__init__(
            "Kraken dataset download requires explicit approval before continuing."
        )
        self.size = size


def _kraken_download_state_key(symbol: str, timeframe: str) -> str:
    return _KRAKEN_CONFIRM_STATE_KEY.format(symbol=symbol, timeframe=timeframe)


def _prompt_kraken_dataset_download(
    symbol: str, timeframe: str, size: Optional[int]
) -> None:
    key = _kraken_download_state_key(symbol, timeframe)
    size_label = _format_download_size(size)
    prompt = (
        f"The Kraken OHLC dataset (~{size_label}) is required to backfill {symbol} "
        f"{timeframe} candles."
    )
    warning_box = st.container()
    warning_box.warning(prompt, icon="⬇️")
    warning_box.caption(
        "Downloading once stores the archive locally so future runs can reuse it."
    )
    col_accept, col_skip = warning_box.columns(2)
    with col_accept:
        if st.button(
            "Download dataset", key=f"{key}::approve", use_container_width=True
        ):
            st.session_state[key] = True
            st.rerun()
    with col_skip:
        if st.button(
            "Skip download", key=f"{key}::skip", use_container_width=True
        ):
            st.session_state[key] = False
            st.rerun()


@st.cache_data(show_spinner=False)
def _cached_fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    target_end_ms: Optional[int] = None,
    limit_per_page: int = 720,
    download_decision: Optional[bool] = None,
) -> OHLCVFetchResult:
    def _confirm(size: Optional[int]) -> bool:
        if download_decision is None:
            raise KrakenDatasetDownloadDecisionRequired(size)
        return download_decision

    return fetch_ohlcv_core(
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        target_end_ms=target_end_ms,
        limit_per_page=limit_per_page,
        confirm_download=_confirm,
    )


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    target_end_ms: Optional[int] = None,
    limit_per_page: int = 720,
) -> OHLCVFetchResult:
    key = _kraken_download_state_key(symbol, timeframe)
    stored_decision = st.session_state.get(key)
    download_decision: Optional[bool]
    if isinstance(stored_decision, bool):
        download_decision = stored_decision
    else:
        download_decision = None

    try:
        return _cached_fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            target_end_ms=target_end_ms,
            limit_per_page=limit_per_page,
            download_decision=download_decision,
        )
    except KrakenDatasetDownloadDecisionRequired as exc:
        _prompt_kraken_dataset_download(symbol, timeframe, exc.size)
        st.stop()


def _controls_from_schema(model: Any) -> tuple[OrderedDict[str, Dict[str, Any]], Dict[str, Any]]:
    controls: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    defaults: Dict[str, Any] = {}

    for field_name, field in getattr(model, "__fields__", {}).items():
        field_type = getattr(field, "type_", float)
        dtype = int if field_type in {int} else float
        default = field.default
        if default is None or default is ...:
            default = 1 if dtype is int else 1.0
        defaults[field_name] = default
        if dtype is int:
            span = max(abs(int(default)) // 2, 1)
            min_value = int(default) - span
            max_value = int(default) + span
            step = 1
            control = dict(label=field_name.replace("_", " ").title(), dtype=int, min=min_value, max=max_value, value=int(default), step=step)
        else:
            span = max(abs(float(default)) * 0.5, 0.5)
            min_value = float(default) - span
            max_value = float(default) + span
            if min_value == max_value:
                max_value = min_value + 1.0
            step = round(span / 10, 4) or 0.1
            control = dict(
                label=field_name.replace("_", " ").title(),
                dtype=float,
                min=min_value,
                max=max_value,
                value=float(default),
                step=step,
                format="%.4f",
            )
        controls[field_name] = control

    return controls, defaults


def create_generated_strategy(
    graph: StrategyGraph,
    name: str,
    key: str,
    description: str,
) -> StrategyDefinition:
    schema_model = build_param_schema(graph)
    controls, param_defaults = _controls_from_schema(schema_model)

    base_defaults = {
        "risk_fraction": 0.25,
        "contract_size": 0.001,
        "allow_shorts": True,
        "ignore_trend": False,
        "sizing_mode": "Fraction of equity (0-1)",
        "initial_cash": 10_000.0,
    }
    default_params = {**base_defaults, **param_defaults}

    def _prepare(df: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
        return df

    def _generate(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        merged = {**default_params, **dict(params)}
        executor = GraphExecutor(graph=graph, params=merged)
        outputs = executor.run(df)
        signals = pd.DataFrame(index=df.index)
        for col, series in outputs.items():
            signals[col] = series.reindex(df.index).astype(float)
        if "position_stream" in signals:
            position = signals["position_stream"].astype(float)
        elif not signals.empty:
            position = signals.iloc[:, 0].astype(float)
            signals["position_stream"] = position
        else:
            position = pd.Series(0.0, index=df.index)
            signals["position_stream"] = position
        signals["desired_position"] = position.fillna(0.0)
        signals["long_signal"] = signals["desired_position"] > 0
        signals["short_signal"] = signals["desired_position"] < 0
        return signals

    def _build_backtest(df: pd.DataFrame, df_sig: pd.DataFrame, params: Mapping[str, Any]):
        from backtesting import Strategy

        merged = {**default_params, **dict(params)}
        risk_fraction = float(merged.get("risk_fraction", 0.25))
        sizing_mode = str(merged.get("sizing_mode", "Fraction of equity (0-1)"))
        allow_shorts = bool(merged.get("allow_shorts", True))

        desired = df_sig.get("desired_position")
        if desired is None:
            desired = pd.Series(0.0, index=df.index)
        desired = desired.reindex(df.index).fillna(0.0)

        class GeneratedGraphStrategy(Strategy):
            def init(self):  # type: ignore[override]
                self._desired = desired

            def _position_size(self, price: float) -> float:
                if sizing_mode.startswith("Whole"):
                    cash = float(self.equity)
                    notional = cash * risk_fraction
                    units = max(1, int(notional / max(price, 1e-9)))
                    return units
                return max(0.001, min(0.9999, risk_fraction))

            def next(self):  # type: ignore[override]
                ts = self.data.index[-1]
                target = float(self._desired.loc[ts]) if ts in self._desired.index else 0.0
                if not allow_shorts and target < 0:
                    target = 0.0
                if self.position:
                    if target == 0 or (self.position.is_long and target < 0) or (self.position.is_short and target > 0):
                        self.position.close()
                if target > 0 and (not self.position or self.position.is_short):
                    if self.position and self.position.is_short:
                        self.position.close()
                    size = self._position_size(float(self.data.Close[-1]))
                    self.buy(size=size)
                elif target < 0 and allow_shorts and (not self.position or self.position.is_long):
                    if self.position and self.position.is_long:
                        self.position.close()
                    size = self._position_size(float(self.data.Close[-1]))
                    self.sell(size=size)

        return GeneratedGraphStrategy

    return StrategyDefinition(
        key=key,
        name=name,
        description=description,
        controls=controls,
        range_controls={},
        default_params=default_params,
        data_requirements={
            "signal_columns": {"long": "long_signal", "short": "short_signal"},
            "preview_columns": ["desired_position", "long_signal", "short_signal"],
        },
        prepare_data=_prepare,
        generate_signals=_generate,
        build_simple_backtest_strategy=_build_backtest,
    )


def slugify_strategy_key(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.lower())
    cleaned = cleaned.replace("-", "_")
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or f"generated_{uuid.uuid4().hex[:6]}"


def render_llm_controls(prefix: str) -> Tuple[str, Optional[float], Optional[str], Optional[str], Optional[int]]:
    """Collect shared LLM configuration controls for GPT-5 and legacy models."""

    model = st.text_input("Model", value="gpt-5", key=f"{prefix}_model").strip()
    st.caption(GPT5_MODEL_CAPTION)

    if is_gpt5_model(model):
        reasoning_key = f"{prefix}_reasoning"
        current_reasoning = st.session_state.get(reasoning_key)
        if isinstance(current_reasoning, str) and current_reasoning in {"minimal", "low", "medium", "high"}:
            st.session_state[reasoning_key] = current_reasoning.capitalize()
        reasoning_option = st.selectbox(
            "Reasoning effort",
            [
                "Use model default",
                "Minimal",
                "Low",
                "Medium",
                "High",
            ],
            index=0,
            key=reasoning_key,
        )
        reasoning_map = {
            "Use model default": None,
            "Minimal": "minimal",
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }
        reasoning = reasoning_map[reasoning_option]

        verbosity_key = f"{prefix}_verbosity"
        current_verbosity = st.session_state.get(verbosity_key)
        if isinstance(current_verbosity, str) and current_verbosity in {"low", "medium", "high"}:
            st.session_state[verbosity_key] = current_verbosity.capitalize()
        verbosity_option = st.selectbox(
            "Response verbosity",
            [
                "Use model default",
                "Low",
                "Medium",
                "High",
            ],
            index=0,
            key=verbosity_key,
        )
        verbosity_map = {
            "Use model default": None,
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }
        verbosity = verbosity_map[verbosity_option]
        max_tokens_raw = st.number_input(
            "Max output tokens (0 uses the API default)",
            min_value=0,
            max_value=8192,
            value=0,
            step=128,
            key=f"{prefix}_max_tokens",
        )
        max_tokens = int(max_tokens_raw) or None
        return model, None, reasoning, verbosity, max_tokens

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        key=f"{prefix}_temperature",
    )
    return model, float(temperature), None, None, None

def plot_chart(
    df: pd.DataFrame,
    overlays: Iterable[Tuple[str, pd.Series]] | None = None,
    trades: list | None = None,
    symbol: str = "",
):
    fig = go.Figure(
        [
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
            )
        ]
    )

    if overlays:
        for label, series in overlays:
            if series is None:
                continue
            aligned = series.reindex(df.index)
            fig.add_trace(go.Scatter(x=df.index, y=aligned, name=label))
    if trades is not None and len(trades):
        ent = [t for t in trades if t.get('etype') == 'entry']
        exi = [t for t in trades if t.get('etype') == 'exit']
        if ent:
            fig.add_trace(
                go.Scatter(
                    x=[t['time'] for t in ent],
                    y=[t['price'] for t in ent],
                    mode="markers",
                    name="Entries",
                    marker=dict(size=9, symbol="triangle-up"),
                )
            )
        if exi:
            fig.add_trace(
                go.Scatter(
                    x=[t['time'] for t in exi],
                    y=[t['price'] for t in exi],
                    mode="markers",
                    name="Exits",
                    marker=dict(size=9, symbol="x"),
                )
            )
    fig.update_layout(height=800, xaxis_title="Time", yaxis_title=symbol)
    return fig


def _to_native(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value
    return value


def _format_label(template: str, params: Dict[str, object]) -> str:
    try:
        return template.format(**params)
    except Exception:
        return template


_PARAM_STATE_KEY = "strategy_param_state"
_RANGE_STATE_KEY = "strategy_range_state"
_SYMBOL_STATE_KEY = "strategy_symbol_state"
_RESULTS_DIR = Path("strategy_results")


def _ensure_state_dict(key: str) -> Dict[Any, Any]:
    value = st.session_state.get(key)
    if not isinstance(value, dict):
        st.session_state[key] = {}
    return st.session_state[key]


def _get_param_store() -> Dict[tuple[str, str], Any]:
    return _ensure_state_dict(_PARAM_STATE_KEY)


def _get_range_store() -> Dict[tuple[str, str], Dict[str, Any]]:
    return _ensure_state_dict(_RANGE_STATE_KEY)


def _get_symbol_store() -> Dict[tuple[str, str], Any]:
    return _ensure_state_dict(_SYMBOL_STATE_KEY)


def _get_param_value(strategy_key: str, param_name: str, default: Any) -> Any:
    store = _get_param_store()
    return store.get((strategy_key, param_name), default)


def _set_param_value(strategy_key: str, param_name: str, value: Any) -> None:
    store = _get_param_store()
    store[(strategy_key, param_name)] = value


def _get_range_spec(strategy_key: str, param_name: str) -> Dict[str, Any] | None:
    store = _get_range_store()
    return store.get((strategy_key, param_name))


def _set_range_spec(strategy_key: str, param_name: str, spec: Dict[str, Any] | None) -> None:
    store = _get_range_store()
    key = (strategy_key, param_name)
    if spec is None:
        store.pop(key, None)
    else:
        store[key] = spec


def _set_symbol_value(strategy_key: str, symbol_id: str, value: Any) -> None:
    store = _get_symbol_store()
    store[(strategy_key, symbol_id)] = value


def _get_symbol_value(strategy_key: str, symbol_id: str, default: Any) -> Any:
    store = _get_symbol_store()
    return store.get((strategy_key, symbol_id), default)


def _build_chart_overlays(
    strategy: StrategyDefinition,
    df_sig: pd.DataFrame,
    params: Dict[str, object],
) -> list[Tuple[str, pd.Series]]:
    overlays: list[Tuple[str, pd.Series]] = []
    specs = strategy.data_requirements.get("chart_overlays", []) if strategy else []
    for spec in specs:
        column = spec.get("column")
        if not column or column not in df_sig:
            continue
        label_template = spec.get("label", column)
        label = _format_label(label_template, params)
        overlays.append((label, df_sig[column]))
    return overlays


def _compute_signal_summary(
    strategy: StrategyDefinition,
    df_sig: pd.DataFrame,
    ignore_trend: bool,
) -> Dict[str, int]:
    summary: Dict[str, int] = {"bars": int(len(df_sig))}
    signal_cols = strategy.data_requirements.get("signal_columns", {}) if strategy else {}

    long_key = signal_cols.get("long")
    short_key = signal_cols.get("short")
    trend_up_key = signal_cols.get("trend_up")
    trend_down_key = signal_cols.get("trend_down")

    if long_key and long_key in df_sig:
        long_mask = df_sig[long_key].astype(bool)
        if not ignore_trend and trend_up_key and trend_up_key in df_sig:
            long_mask = long_mask & df_sig[trend_up_key].astype(bool)
        summary["long_cond_count"] = int(long_mask.sum())
    if short_key and short_key in df_sig:
        short_mask = df_sig[short_key].astype(bool)
        if not ignore_trend and trend_down_key and trend_down_key in df_sig:
            short_mask = short_mask & df_sig[trend_down_key].astype(bool)
        summary["short_cond_count"] = int(short_mask.sum())
    return summary


def run_simple_backtest(
    df: pd.DataFrame,
    params: dict,
    strategy: StrategyDefinition | None = None,
    df_sig: pd.DataFrame | None = None,
    df_prepared: pd.DataFrame | None = None,
):
    from backtesting import Backtest

    if strategy is None:
        raise ValueError("Strategy definition required for backtesting.")

    base_params = dict(strategy.default_params or {})
    base_params.update(params or {})
    if "sizing_mode" not in base_params:
        base_params["sizing_mode"] = "Whole units (int)"

    contract_size = float(base_params.get("contract_size", 0.001))
    initial_cash = float(base_params.get("initial_cash", 10_000.0))
    fee_percent = float(base_params.get("fee_percent", 0.0))
    slippage_percent = float(base_params.get("slippage_percent", 0.0))

    prepared_df = df_prepared if df_prepared is not None else strategy.prepare_data(df, base_params)
    if df_sig is None:
        df_sig = strategy.generate_signals(prepared_df, base_params)

    builder = strategy.build_simple_backtest_strategy
    if builder is None:
        raise ValueError(f"Strategy '{strategy.key}' does not provide a simple backtest builder.")

    strat_cls = builder(prepared_df, df_sig, base_params)

    risk_cfg = RiskEngineConfig.from_params(base_params.get("risk_config"))
    risk_engine_template = RiskEngine(risk_cfg)
    risk_symbol = str(base_params.get("symbol", strategy.key))

    class RiskManagedStrategy(strat_cls):
        _risk_engine_template = risk_engine_template
        _risk_symbol = risk_symbol

        def init(self):
            self._risk_engine = self._risk_engine_template.clone()
            self._risk_engine.reset()
            self._risk_prev_equity = None
            self._risk_latest_state: PortfolioState | None = None
            super().init()

        def _risk_snapshot_state(self, update_return: bool) -> PortfolioState:
            idx = self.data.index[-1]
            price = float(self.data.Close[-1])
            equity = float(self.equity)
            broker_cash = float(getattr(self._broker, "_cash", 0.0))
            position_units = float(self.position.size)
            positions = {self._risk_symbol: position_units}
            prices = {self._risk_symbol: price}
            notional = abs(position_units) * price
            gross = notional / max(equity, 1e-9)
            net = (position_units * price) / max(equity, 1e-9)
            if update_return:
                prev_equity = getattr(self, "_risk_prev_equity", None)
                if prev_equity not in (None, 0.0):
                    ret = (equity - prev_equity) / prev_equity
                else:
                    ret = 0.0
                self._risk_prev_equity = equity
            else:
                ret = None
            state = PortfolioState(
                timestamp=idx,
                equity=equity,
                cash=broker_cash,
                gross_leverage=gross,
                net_leverage=net,
                positions=positions,
                prices=prices,
                return_pct=ret,
            )
            if update_return:
                self._risk_latest_state = state
            return state

        def _risk_prepare_order(self, raw_size: float, direction: int, price: float):
            state = self._risk_latest_state or self._risk_snapshot_state(update_return=False)
            fractional = -1 < raw_size < 1
            magnitude = abs(raw_size)
            if fractional:
                magnitude = magnitude * max(state.equity, 1e-9) / max(price, 1e-9)
            current_units = state.positions.get(self._risk_symbol, 0.0)
            projected_units = current_units + direction * magnitude
            is_exit = abs(projected_units) < abs(current_units)
            ctx = OrderContext(
                symbol=self._risk_symbol,
                direction=direction,
                size=magnitude,
                price=price,
                is_exit=is_exit,
            )
            return ctx, fractional, state

        def next(self):
            state = self._risk_snapshot_state(update_return=True)
            flatten = self._risk_engine.on_bar(state)
            if flatten and self.position:
                self.position.close()
            super().next()

        def buy(self, *, size: float = 0.9999, **kwargs):
            if not getattr(self, "_risk_engine", None) or not self._risk_engine.enabled:
                try:
                    return super().buy(size=size, **kwargs)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("Backtesting buy rejected: %s", exc)
                    return None
            price = float(self.data.Close[-1])
            ctx, fractional, state = self._risk_prepare_order(float(size), 1, price)
            decision = self._risk_engine.modify_order(ctx, state)
            if decision.cancel or decision.size <= 0:
                return None
            final_size = decision.size
            if fractional:
                equity = max(state.equity, 1e-9)
                final_size = float(final_size) * equity / max(price, 1e-9)
                if final_size <= 1e-9:
                    return None
            else:
                final_size = float(round(final_size))
                if final_size <= 0:
                    return None
            try:
                return super().buy(size=final_size, **kwargs)
            except ValueError as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Backtesting buy rejected: %s", exc)
                return None

        def sell(self, *, size: float = 0.9999, **kwargs):
            if not getattr(self, "_risk_engine", None) or not self._risk_engine.enabled:
                try:
                    return super().sell(size=size, **kwargs)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("Backtesting sell rejected: %s", exc)
                    return None
            price = float(self.data.Close[-1])
            ctx, fractional, state = self._risk_prepare_order(float(size), -1, price)
            decision = self._risk_engine.modify_order(ctx, state)
            if decision.cancel or decision.size <= 0:
                return None
            final_size = decision.size
            if fractional:
                equity = max(state.equity, 1e-9)
                final_size = float(final_size) * equity / max(price, 1e-9)
                if final_size <= 1e-9:
                    return None
            else:
                final_size = float(round(final_size))
                if final_size <= 0:
                    return None
            try:
                return super().sell(size=final_size, **kwargs)
            except ValueError as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Backtesting sell rejected: %s", exc)
                return None

    df_bt = prepared_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_bt[["Open", "High", "Low", "Close"]] = df_bt[["Open", "High", "Low", "Close"]] * contract_size

    bt = Backtest(
        df_bt,
        RiskManagedStrategy,
        cash=initial_cash,
        commission=(fee_percent + slippage_percent) / 100.0,
        exclusive_orders=False,
    )

    stats = bt.run()
    trades_df = getattr(stats, "_trades", pd.DataFrame()).copy()
    trade_markers = []
    if not trades_df.empty and {"EntryTime", "EntryPrice", "ExitTime", "ExitPrice", "Size"}.issubset(trades_df.columns):
        trades_df["Direction"] = np.where(trades_df["Size"] > 0, "Long", "Short")
        trades_df["Duration"] = pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"])
        denom = (trades_df["EntryPrice"].abs() * trades_df["Size"].abs()).replace(0, np.nan)
        trades_df["ReturnPct_est"] = trades_df["PnL"] / denom * 100.0
        for _, r in trades_df.iterrows():
            trade_markers.append({
                "time": r["EntryTime"],
                "price": r["EntryPrice"] / contract_size,
                "etype": "entry",
            })
            trade_markers.append({
                "time": r["ExitTime"],
                "price": r["ExitPrice"] / contract_size,
                "etype": "exit",
            })

    metrics = {k: _to_native(v) for k, v in stats.items()}
    metrics["__engine__"] = "Simple (backtesting.py)"

    return {
        "metrics": metrics,
        "trades": trades_df,
        "trade_markers": trade_markers,
    }


def run_true_stop_backtest(
    df: pd.DataFrame,
    params: dict,
    strategy: StrategyDefinition | None = None,
    df_sig: pd.DataFrame | None = None,
    df_prepared: pd.DataFrame | None = None,
):
    import backtrader as bt

    if strategy is None:
        raise ValueError("Strategy definition required for backtesting.")

    base_params = dict(strategy.default_params or {})
    base_params.update(params or {})

    risk_fraction = float(base_params.get("risk_fraction", 0.25))
    max_leverage = float(base_params.get("max_leverage", 5.0))
    fee_percent = float(base_params.get("fee_percent", 0.0))
    slippage_percent = float(base_params.get("slippage_percent", 0.0))
    initial_cash = float(base_params.get("initial_cash", 10_000.0))

    prepared_df = df_prepared if df_prepared is not None else strategy.prepare_data(df, base_params)
    if df_sig is None:
        df_sig = strategy.generate_signals(prepared_df, base_params)

    builder = strategy.build_true_stop_strategy
    if builder is None:
        raise ValueError(f"Strategy '{strategy.key}' does not provide a true stop builder.")

    bt_strategy_cls = builder(prepared_df, df_sig, base_params)

    risk_cfg = RiskEngineConfig.from_params(base_params.get("risk_config"))
    risk_engine_template = RiskEngine(risk_cfg)

    class RiskManagedBTStrategy(bt_strategy_cls):
        def __init__(self, *args, **kwargs):
            self._risk_engine = risk_engine_template.clone()
            self._risk_engine.reset()
            self._risk_prev_equity = None
            self._risk_latest_state: PortfolioState | None = None
            super().__init__(*args, **kwargs)

        def _risk_snapshot_state(self, update_return: bool) -> PortfolioState:
            equity = float(self.broker.getvalue())
            cash = float(self.broker.getcash())
            prices: Dict[str, float] = {}
            positions: Dict[str, float] = {}
            total_notional = 0.0
            net_notional = 0.0
            for idx, data in enumerate(self.datas):
                symbol_name = getattr(data, "_name", None) or getattr(data, "_dataname", f"data{idx}")
                price = float(data.close[0])
                pos = self.getposition(data)
                units = float(getattr(pos, "size", 0.0))
                prices[symbol_name] = price
                positions[symbol_name] = units
                total_notional += abs(units) * price
                net_notional += units * price
            gross = total_notional / max(equity, 1e-9)
            net = net_notional / max(equity, 1e-9)
            if update_return:
                prev_equity = getattr(self, "_risk_prev_equity", None)
                if prev_equity not in (None, 0.0):
                    ret = (equity - prev_equity) / prev_equity
                else:
                    ret = 0.0
                self._risk_prev_equity = equity
            else:
                ret = None
            state = PortfolioState(
                timestamp=self.data.datetime.datetime(0),
                equity=equity,
                cash=cash,
                gross_leverage=gross,
                net_leverage=net,
                positions=positions,
                prices=prices,
                return_pct=ret,
            )
            if update_return:
                self._risk_latest_state = state
            return state

        def next(self):
            state = self._risk_snapshot_state(update_return=True)
            flatten = self._risk_engine.on_bar(state)
            if flatten:
                for data in self.datas:
                    if self.getposition(data).size:
                        self.close(data=data)
            super().next()

    class RiskAwareCashSizer(bt.Sizer):
        params = (("risk_frac", risk_fraction), ("min_unit", 0.0001), ("max_lev", max_leverage), ("retint", False))

        def _getsizing(self, comminfo, cash, data, isbuy):
            price = float(data.close[0])
            notional = cash * self.p.risk_frac * self.p.max_lev
            units = notional / max(price, 1e-9)
            snapped = max(self.p.min_unit, round(units / self.p.min_unit) * self.p.min_unit)
            engine = getattr(self.strategy, "_risk_engine", None)
            if engine and engine.enabled:
                state = self.strategy._risk_snapshot_state(update_return=False)
                symbol_name = getattr(data, "_name", None) or getattr(data, "_dataname", "data0")
                current_units = state.positions.get(symbol_name, 0.0)
                direction = 1 if isbuy else -1
                projected_units = current_units + direction * snapped
                is_exit = abs(projected_units) < abs(current_units)
                ctx = OrderContext(
                    symbol=symbol_name,
                    direction=direction,
                    size=snapped,
                    price=price,
                    is_exit=is_exit,
                )
                decision = engine.modify_order(ctx, state)
                if decision.cancel or decision.size <= 0:
                    return 0.0
                snapped = max(self.p.min_unit, round(decision.size / self.p.min_unit) * self.p.min_unit)
            return snapped if not self.p.retint else int(snapped)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcommission(commission=(fee_percent + slippage_percent) / 100.0)
    cerebro.broker.set_slippage_perc(perc=slippage_percent / 100.0)
    cerebro.broker.setcash(initial_cash)

    timeframe = str(base_params.get("timeframe", "1h"))
    comp = 60 if timeframe == "1h" else (240 if timeframe == "4h" else 1440)
    data_bt = bt.feeds.PandasData(
        dataname=prepared_df, timeframe=bt.TimeFrame.Minutes, compression=comp
    )
    cerebro.adddata(data_bt)
    cerebro.addsizer(RiskAwareCashSizer)
    cerebro.addstrategy(RiskManagedBTStrategy)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')

    class TradeLogger(bt.Analyzer):
        def __init__(self):
            self.rows = []

        def notify_trade(self, trade):
            if not trade.isclosed:
                return

            entry_dt = bt.num2date(getattr(trade, 'dtopen', self.strategy.datas[0].datetime[0]))
            exit_dt = bt.num2date(getattr(trade, 'dtclose', self.strategy.datas[0].datetime[0]))
            entry_px = float(getattr(trade, 'price', float('nan')))
            exit_px = float('nan')
            dirn = None
            size_abs = abs(float(getattr(trade, 'size', 0.0)))

            hist = getattr(trade, 'history', None)
            if hist and len(hist) >= 2:
                h0, hN = hist[0], hist[-1]
                try:
                    dirn = "Long" if float(getattr(h0.event, 'size', 0.0)) > 0 else "Short"
                    entry_dt = bt.num2date(getattr(h0.status, 'dt', trade.dtopen))
                    exit_dt = bt.num2date(getattr(hN.status, 'dt', trade.dtclose))
                    entry_px = float(getattr(h0.event, 'price', entry_px))
                    exit_px = float(getattr(hN.event, 'price', float('nan')))
                    size_abs = abs(float(getattr(h0.event, 'size', size_abs)))
                except Exception:
                    pass

            if dirn is None:
                dirn = "Long" if getattr(trade, 'pnl', 0.0) >= 0 else "Short/unknown"

            self.rows.append(dict(
                Direction=dirn,
                EntryTime=entry_dt, ExitTime=exit_dt,
                EntryPrice=entry_px, ExitPrice=exit_px,
                Size=size_abs, PnL=float(getattr(trade, 'pnlcomm', getattr(trade, 'pnl', 0.0))),
            ))

        def get_analysis(self):
            return self.rows

    cerebro.addanalyzer(TradeLogger, _name='trades')

    results = cerebro.run()
    res = results[0]
    ta = res.analyzers.ta.get_analysis() if hasattr(res.analyzers, 'ta') else {}
    trade_rows = res.analyzers.trades.get_analysis() if hasattr(res.analyzers, 'trades') else []

    trades_df = pd.DataFrame(trade_rows)
    broker_value = cerebro.broker.getvalue()
    pnl = broker_value - initial_cash
    return_pct = (pnl / initial_cash) * 100 if initial_cash else 0.0
    total_closed = ta.get('total', {}).get('closed', 0) if isinstance(ta, dict) else len(trades_df)

    metrics = {
        "Final Equity": broker_value,
        "Equity Final [$]": broker_value,
        "PnL": pnl,
        "Return [%]": return_pct,
        "Closed Trades": total_closed,
        "Win Rate [%]": ta.get('won', {}).get('total', 0) / total_closed * 100 if total_closed else 0.0,
        "__engine__": "TRUE STOP (backtrader)",
    }

    return {
        "metrics": metrics,
        "trades": trades_df,
        "trade_markers": [],
    }


def render_scalar_or_range(
    label: str,
    key: str,
    mode: str,
    *,
    min_value,
    max_value,
    value,
    step,
    dtype=float,
    slider=False,
    format: str | None = None,
    range_defaults: Dict[str, Any] | None = None,
    stored_range: Dict[str, Any] | None = None,
):
    range_seed = stored_range or range_defaults or {}
    dtype = range_seed.get("dtype", dtype)
    if dtype is int:
        min_value = int(min_value)
        max_value = int(max_value)
        value = int(value)
        step = int(step)

    if mode == "Single Run":
        if slider:
            return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key), None
        kwargs = dict(min_value=min_value, max_value=max_value, value=value, step=step, key=key)
        if format is not None:
            kwargs["format"] = format
        return st.number_input(label, **kwargs), None

    current_min = range_seed.get("min", value)
    current_max = range_seed.get("max", value)
    current_step = range_seed.get("step", step)

    if dtype is int:
        current_min = int(current_min)
        current_max = int(current_max)
        current_step = max(1, int(current_step))
    else:
        current_min = float(current_min)
        current_max = float(current_max)
        current_step = float(current_step) if current_step else float(step)

    cols = st.columns(3)
    with cols[0]:
        min_kwargs = dict(
            min_value=min_value,
            max_value=max_value,
            value=max(min_value, min(current_min, max_value)),
            step=step,
            key=f"{key}__min",
        )
        if format is not None:
            min_kwargs["format"] = format
        min_val = st.number_input(f"{label} min", **min_kwargs)
    with cols[1]:
        max_default = max(min_val, min(current_max, max_value))
        max_kwargs = dict(
            min_value=min_val,
            max_value=max_value,
            value=max_default,
            step=step,
            key=f"{key}__max",
        )
        if format is not None:
            max_kwargs["format"] = format
        max_val = st.number_input(f"{label} max", **max_kwargs)
    with cols[2]:
        min_step = 1 if dtype is int else step
        step_default = max(min_step, current_step)
        step_kwargs = dict(min_value=min_step, value=step_default, key=f"{key}__step")
        if dtype is int:
            step_kwargs["step"] = 1
            step_kwargs["max_value"] = max(1, max_val - min_val)
        else:
            step_kwargs["step"] = step
            if format is not None:
                step_kwargs["format"] = format
        step_val = st.number_input(f"{label} step", **step_kwargs)

    return None, {
        "label": label,
        "key": key,
        "min": min_val,
        "max": max_val,
        "step": step_val,
        "dtype": dtype,
    }


def generate_range(range_spec: dict) -> list:
    start = range_spec["min"]
    stop = range_spec["max"]
    step = range_spec["step"]
    dtype = range_spec.get("dtype", float)
    if dtype is int:
        values = list(range(int(start), int(stop) + 1, max(1, int(step))))
        return values
    if step <= 0:
        return [float(start)]
    count = int(max(1, round((stop - start) / step))) + 1
    values = [float(start + i * step) for i in range(count)]
    # Ensure we do not exceed stop due to floating errors
    filtered = []
    for v in values:
        if v > stop + step / 2:
            break
        filtered.append(round(v, 10))
    if filtered and filtered[-1] < stop - step / 2:
        filtered.append(round(stop, 10))
    return filtered

def render_trading_console():
    st.title("📈 AI Trader — Trading Console")
    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.header("Mode & Engine")
        run_mode = st.radio("Run mode", ["Single Run", "Optimization"], index=0, key="run_mode")
        mode = st.radio("Mode", ["Backtest", "Dry-Run (paper)"] , index=0, key="session_mode")
        engine = st.selectbox("Backtest engine", ["Simple (backtesting.py)", "TRUE STOP (backtrader)"], index=0, key="engine")

        strategy_keys = list(STRATEGY_REGISTRY.keys())
        default_strategy_key = st.session_state.get("active_strategy", DEFAULT_STRATEGY_KEY)
        if default_strategy_key not in STRATEGY_REGISTRY:
            default_strategy_key = DEFAULT_STRATEGY_KEY
        if st.session_state.get("active_strategy") not in STRATEGY_REGISTRY:
            st.session_state["active_strategy"] = default_strategy_key
        selected_strategy_key = st.session_state.get("active_strategy", default_strategy_key)
        active_strategy = STRATEGY_REGISTRY.get(selected_strategy_key, STRATEGY_REGISTRY[DEFAULT_STRATEGY_KEY])

        pending_champion = st.session_state.get("builder_pending_apply")
        if pending_champion:
            target_strategy = pending_champion.get("strategy")
            if target_strategy == selected_strategy_key:
                st.info("Champion parameters from the builder are ready to apply.")
                if st.button("Apply champion parameters", key="apply_champion_params"):
                    for param_name, param_value in pending_champion.get("params", {}).items():
                        if param_name in active_strategy.controls:
                            _set_param_value(selected_strategy_key, param_name, param_value)
                            st.session_state[f"ctrl::{selected_strategy_key}::{param_name}"] = param_value
                        elif param_name in {"allow_shorts", "ignore_trend", "sizing_mode"}:
                            st.session_state[param_name] = param_value
                    st.session_state.pop("builder_pending_apply", None)
                    st.success("Champion parameters applied to the controls.")
                    st.rerun()
            else:
                st.caption(
                    f"Champion parameters available for strategy '{target_strategy}'. Switch strategies to apply them."
                )

        st.header("Data")
        symbol_groups_spec = active_strategy.data_requirements.get("symbols", []) if active_strategy else []
        symbol_groups: Dict[str, list[str]] = {}
        primary_symbol = None
        primary_group_id = "primary"
        if not symbol_groups_spec:
            default_symbol = _get_symbol_value(selected_strategy_key, "primary", st.session_state.get("symbol", "BTC/EUR"))
            if isinstance(default_symbol, (list, tuple)):
                default_symbol = default_symbol[0] if default_symbol else ""
            symbol_key = f"symbol_input::{selected_strategy_key}::primary"
            if symbol_key not in st.session_state:
                st.session_state[symbol_key] = str(default_symbol or "")
            symbol_value = st.text_input("Symbol (Kraken/ccxt)", key=symbol_key)
            symbol_clean = symbol_value.strip()
            if symbol_clean:
                symbol_groups["primary"] = [symbol_clean]
                primary_symbol = symbol_clean
            else:
                st.error("Please provide a symbol to proceed.")
                st.stop()
            _set_symbol_value(selected_strategy_key, "primary", [symbol_clean])
        else:
            for idx, spec in enumerate(symbol_groups_spec):
                spec_id = str(spec.get("id") or spec.get("key") or spec.get("label") or f"group_{idx}")
                if idx == 0:
                    primary_group_id = spec_id
                label = spec.get("label", spec_id.replace("_", " ").title())
                multiple = bool(spec.get("multiple", False))
                required = spec.get("required", True)
                default_value = spec.get("default", [] if multiple else "")
                stored_value = _get_symbol_value(selected_strategy_key, spec_id, default_value)
                if multiple:
                    if isinstance(stored_value, str):
                        stored_list = [stored_value]
                    else:
                        stored_list = list(stored_value or [])
                    default_text = ", ".join(stored_list)
                else:
                    if isinstance(stored_value, (list, tuple)):
                        stored_list = list(stored_value)
                        default_text = stored_list[0] if stored_list else ""
                    else:
                        default_text = str(stored_value or "")
                input_key = f"symbol_input::{selected_strategy_key}::{spec_id}"
                if input_key not in st.session_state:
                    st.session_state[input_key] = default_text
                raw_value = st.text_input(label, key=input_key)
                if multiple:
                    entries = [s.strip() for s in raw_value.split(",") if s.strip()]
                    _set_symbol_value(selected_strategy_key, spec_id, entries)
                else:
                    entries = [raw_value.strip()] if raw_value.strip() else []
                    _set_symbol_value(selected_strategy_key, spec_id, entries[0] if entries else "")
                if required and not entries:
                    st.error(f"Please provide at least one symbol for '{label}'.")
                    st.stop()
                symbol_groups[spec_id] = entries
                if primary_symbol is None and entries:
                    primary_symbol = entries[0]

        if not primary_symbol:
            st.error("No primary symbol resolved for the selected strategy.")
            st.stop()

        symbol = primary_symbol
        st.session_state["symbol"] = symbol
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(st.session_state.get("timeframe", "1h")), key="timeframe")
        since_str = st.text_input("Since (UTC ISO8601)", value=st.session_state.get("since_str", "2022-01-01T00:00:00Z"), key="since_str")
        ex = ccxt.kraken()
        try:
            since_ms = ex.parse8601(since_str)
        except Exception:
            since_ms = None
            st.error("Invalid 'Since' string (e.g. 2023-01-01T00:00:00Z)")

        st.header("Strategy")
        strategy_index = strategy_keys.index(selected_strategy_key)
        selected_strategy_key = st.selectbox(
            "Strategy",
            strategy_keys,
            index=strategy_index,
            format_func=lambda key: STRATEGY_REGISTRY[key].name,
            key="active_strategy",
        )
        active_strategy = STRATEGY_REGISTRY[selected_strategy_key]
        if active_strategy.description:
            st.caption(active_strategy.description)

        st.header("Strategy Parameters")
        scalar_params: Dict[str, Any] = {}
        range_specs: Dict[str, Dict[str, Any]] = {}
        range_defaults_map = getattr(active_strategy, "range_controls", {}) or {}
        for key, cfg in active_strategy.controls.items():
            default_value = cfg.get("value")
            stored_value = _get_param_value(selected_strategy_key, key, default_value)
            widget_key = f"ctrl::{selected_strategy_key}::{key}"
            range_defaults = range_defaults_map.get(key)
            stored_range = _get_range_spec(selected_strategy_key, key)
            value, range_spec = render_scalar_or_range(
                cfg["label"],
                widget_key,
                run_mode,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=stored_value,
                step=cfg["step"],
                dtype=cfg.get("dtype", float),
                slider=cfg.get("slider", False),
                format=cfg.get("format"),
                range_defaults=range_defaults,
                stored_range=stored_range,
            )
            if range_spec is None:
                scalar_params[key] = value
                _set_param_value(selected_strategy_key, key, value)
                _set_range_spec(selected_strategy_key, key, None)
            else:
                scalar_params[key] = range_spec["min"]
                _set_param_value(selected_strategy_key, key, range_spec.get("min", stored_value))
                _set_range_spec(selected_strategy_key, key, range_spec)
                range_specs[key] = range_spec

        for key, default_spec in range_defaults_map.items():
            if not default_spec:
                continue
            if key not in range_specs:
                stored_range = _get_range_spec(selected_strategy_key, key)
                if stored_range is not None:
                    range_specs[key] = dict(stored_range)
                else:
                    range_specs[key] = dict(default_spec)

        preset_params = {
            key: _get_param_value(selected_strategy_key, key, cfg.get("value"))
            for key, cfg in active_strategy.controls.items()
        }
        preset_ranges_raw = {
            key: spec
            for key, spec in ((_key, _get_range_spec(selected_strategy_key, _key)) for _key in active_strategy.controls.keys())
            if spec is not None
        }
        preset_payload = {
            "strategy": selected_strategy_key,
            "saved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "params": preset_params,
        }
        if preset_ranges_raw:
            serializable_ranges: Dict[str, Dict[str, Any]] = {}
            for param_name, spec in preset_ranges_raw.items():
                cleaned = dict(spec)
                dtype_value = cleaned.get("dtype")
                if dtype_value is not None:
                    cleaned["dtype"] = getattr(dtype_value, "__name__", str(dtype_value))
                serializable_ranges[param_name] = cleaned
            preset_payload["ranges"] = serializable_ranges
        preset_bytes = json.dumps(preset_payload, indent=2).encode("utf-8")
        preset_cols = st.columns([1, 1])
        with preset_cols[0]:
            st.download_button(
                "💾 Save preset",
                data=preset_bytes,
                file_name=f"{selected_strategy_key}_params.json",
                mime="application/json",
                use_container_width=True,
            )
        with preset_cols[1]:
            uploaded_preset = st.file_uploader(
                "Load preset",
                type="json",
                key=f"preset_uploader::{selected_strategy_key}",
                label_visibility="collapsed",
            )
            if uploaded_preset is not None:
                try:
                    loaded = json.load(uploaded_preset)
                except Exception as exc:
                    st.error(f"Failed to read preset: {exc}")
                else:
                    target_strategy = loaded.get("strategy", selected_strategy_key)
                    if target_strategy != selected_strategy_key:
                        st.warning(
                            f"Preset belongs to strategy '{target_strategy}'. Applying overlapping parameters only.",
                            icon="⚠️",
                        )
                    loaded_params = loaded.get("params", {}) or {}
                    loaded_ranges = loaded.get("ranges", {}) or {}
                    for param_name, param_value in loaded_params.items():
                        if param_name not in active_strategy.controls:
                            continue
                        _set_param_value(selected_strategy_key, param_name, param_value)
                        _set_range_spec(selected_strategy_key, param_name, None)
                        st.session_state[f"ctrl::{selected_strategy_key}::{param_name}"] = param_value
                    for param_name, range_spec in loaded_ranges.items():
                        if param_name not in active_strategy.controls:
                            continue
                        if isinstance(range_spec, dict) and "dtype" in range_spec:
                            dtype_token = range_spec["dtype"]
                            if dtype_token in {int, float}:
                                dtype_obj = dtype_token
                            elif str(dtype_token).lower() in {"int", "integer"}:
                                dtype_obj = int
                            elif str(dtype_token).lower() in {"float", "double"}:
                                dtype_obj = float
                            else:
                                dtype_obj = float
                            range_spec = {**range_spec, "dtype": dtype_obj}
                        _set_range_spec(selected_strategy_key, param_name, range_spec)
                        base_key = f"ctrl::{selected_strategy_key}::{param_name}"
                        st.session_state.pop(f"{base_key}__min", None)
                        st.session_state.pop(f"{base_key}__max", None)
                        st.session_state.pop(f"{base_key}__step", None)
                    st.success("Preset loaded. Refreshing controls...")
                    st.rerun()

        allow_shorts = st.checkbox("Allow shorts", value=st.session_state.get("allow_shorts", True), key="allow_shorts")
        ignore_trend = st.checkbox("Ignore trend filter (debug)", value=st.session_state.get("ignore_trend", False), key="ignore_trend")

        sizing_mode = st.selectbox("Sizing mode", ["Whole units (int)", "Fraction of equity (0-1)"] , index=["Whole units (int)", "Fraction of equity (0-1)"].index(st.session_state.get("sizing_mode", "Whole units (int)")), key="sizing_mode")

        with st.expander("Risk engine", expanded=False):
            risk_enabled = st.checkbox(
                "Enable risk engine",
                value=st.session_state.get("risk_enabled", True),
                key="risk_enabled",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                target_vol_pct = st.number_input(
                    "Target volatility (%)",
                    min_value=0.0,
                    max_value=200.0,
                    value=st.session_state.get("risk_target_vol", 25.0),
                    step=1.0,
                    key="risk_target_vol",
                )
                max_drawdown_pct = st.number_input(
                    "Max drawdown trigger (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get("risk_drawdown", 20.0),
                    step=1.0,
                    key="risk_drawdown",
                )
                cooldown_scale = st.number_input(
                    "Cooldown scaling (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("risk_cooldown_scale", 0.3),
                    step=0.05,
                    key="risk_cooldown_scale",
                )
                rerisk_step = st.number_input(
                    "Re-risk step",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("risk_rerisk_step", 0.1),
                    step=0.05,
                    key="risk_rerisk_step",
                )
            with col_b:
                vol_lookback = st.number_input(
                    "Volatility lookback (bars)",
                    min_value=2,
                    max_value=500,
                    value=st.session_state.get("risk_vol_lookback", 20),
                    step=1,
                    key="risk_vol_lookback",
                )
                cooldown_bars = st.number_input(
                    "Drawdown cooldown (bars)",
                    min_value=0,
                    max_value=500,
                    value=st.session_state.get("risk_cooldown_bars", 10),
                    step=1,
                    key="risk_cooldown_bars",
                )
                rerisk_threshold_pct = st.number_input(
                    "Re-risk threshold (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get("risk_rerisk_threshold", 5.0),
                    step=0.5,
                    key="risk_rerisk_threshold",
                )
                circuit_drawdown_pct = st.number_input(
                    "Circuit breaker drawdown (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get("risk_cb_drawdown", 35.0),
                    step=1.0,
                    key="risk_cb_drawdown",
                )
            col_c, col_d = st.columns(2)
            with col_c:
                circuit_cooldown = st.number_input(
                    "Circuit breaker cooldown (bars)",
                    min_value=0,
                    max_value=1000,
                    value=st.session_state.get("risk_cb_cooldown", 50),
                    step=1,
                    key="risk_cb_cooldown",
                )
                max_gross_exposure = st.number_input(
                    "Max gross exposure (× equity)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.get("risk_max_gross", 2.0),
                    step=0.1,
                    key="risk_max_gross",
                )
            with col_d:
                max_single_exposure = st.number_input(
                    "Max single exposure (× equity)",
                    min_value=0.1,
                    max_value=5.0,
                    value=st.session_state.get("risk_max_single", 1.0),
                    step=0.1,
                    key="risk_max_single",
                )
                correlation_threshold = st.number_input(
                    "Correlation cut threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("risk_corr_threshold", 0.85),
                    step=0.05,
                    key="risk_corr_threshold",
                )
                max_corr_exposure = st.number_input(
                    "Max correlated exposure (× equity)",
                    min_value=0.0,
                    max_value=5.0,
                    value=st.session_state.get("risk_corr_limit", 0.6),
                    step=0.1,
                    key="risk_corr_limit",
                )
            enable_corr_cuts = st.checkbox(
                "Enable correlation cuts",
                value=st.session_state.get("risk_corr_enabled", True),
                key="risk_corr_enabled",
            )

        risk_config = {
            "enabled": bool(risk_enabled),
            "target_volatility": float(target_vol_pct) / 100.0,
            "vol_lookback": int(vol_lookback),
            "drawdown_limit": float(max_drawdown_pct) / 100.0,
            "drawdown_cooldown_bars": int(cooldown_bars),
            "cooldown_scale": float(cooldown_scale),
            "rerisk_drawdown": float(rerisk_threshold_pct) / 100.0,
            "rerisk_step": float(rerisk_step),
            "circuit_breaker_drawdown": float(circuit_drawdown_pct) / 100.0,
            "circuit_breaker_cooldown_bars": int(circuit_cooldown),
            "max_gross_exposure": float(max_gross_exposure),
            "max_position_exposure": float(max_single_exposure),
            "correlation_threshold": float(correlation_threshold),
            "max_correlation_exposure": float(max_corr_exposure),
            "enable_correlation_cuts": bool(enable_corr_cuts),
        }

        st.header("Preview")
        preview_rows = st.slider("Preview last N rows", 10, 1000, st.session_state.get("preview_rows", 50), step=10, key="preview_rows")

    if since_ms is None:
        st.stop()

    requested_start = datetime.fromtimestamp(since_ms / 1000.0, tz=UTC)
    target_end_dt = datetime.now(UTC)
    target_end_ms = int(target_end_dt.timestamp() * 1000)

    with st.spinner("Fetching data..."):
        symbol_datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
        symbol_fetch_info: Dict[str, Dict[str, OHLCVFetchResult]] = {}
        empty_symbols: list[tuple[str, str]] = []
        for group_id, entries in symbol_groups.items():
            group_frames: Dict[str, pd.DataFrame] = {}
            group_meta: Dict[str, OHLCVFetchResult] = {}
            for entry in entries:
                result = fetch_ohlcv(
                    entry,
                    timeframe,
                    since_ms,
                    target_end_ms=target_end_ms,
                )
                df_entry = result.data
                if df_entry.empty:
                    empty_symbols.append((group_id, entry))
                else:
                    if df_entry.index.tz is None:
                        df_entry.index = df_entry.index.tz_localize('UTC')
                group_frames[entry] = df_entry
                group_meta[entry] = result
            symbol_datasets[group_id] = group_frames
            symbol_fetch_info[group_id] = group_meta

    if empty_symbols:
        missing_descriptions = ", ".join(f"{grp}:{sym}" for grp, sym in empty_symbols)
        st.error(
            "No data fetched for: " + missing_descriptions + ". Try adjusting the symbol or the date range.")
        st.stop()

    primary_group = symbol_datasets.get(primary_group_id, {})
    primary_meta = symbol_fetch_info.get(primary_group_id, {})
    primary_result = primary_meta.get(symbol)
    df = primary_group.get(symbol)
    if df is None or df.empty:
        st.error("No data fetched. Try: 1) symbol 'XBT/EUR', 2) earlier 'Since', 3) timeframe '1h'.")
        st.stop()

    # -----------------------------
    # Data Preview
    # -----------------------------
    st.subheader("Data Preview")
    if primary_result:
        coverage = primary_result.coverage
        target_span = primary_result.target_coverage
        coverage_text = ""
        if coverage is not None and coverage > pd.Timedelta(0):
            coverage_text = f" | Coverage: {coverage}"
        target_text = ""
        if target_span is not None and target_span > pd.Timedelta(0):
            target_text = f" | Requested: {target_span}"
        truncation = " (API budget reached)" if primary_result.truncated else ""
        st.caption(
            f"Bars: {primary_result.rows:,} | Range: {primary_result.start} → {primary_result.end}"
            f" | Timezone: {df.index.tz}{truncation}{coverage_text}{target_text}"
            f" | Requests: {primary_result.pages}/{primary_result.max_pages}"
        )
        if (
            not primary_result.reached_target
            and target_span is not None
            and coverage is not None
        ):
            st.warning(
                "Exchange history is shorter than requested."
                f" Received ≈{coverage} vs desired ≈{target_span}."
                " Additional automatic requests have been exhausted; consider reducing the window or timeframe.",
                icon="ℹ️",
            )
    else:
        st.caption(f"Bars: {len(df):,}  |  Range: {df.index.min()} → {df.index.max()}  |  Timezone: {df.index.tz}")
    st.dataframe(df.tail(preview_rows))
    with st.expander("Show first rows"):
        st.dataframe(df.head(preview_rows))
    csv = df.to_csv().encode("utf-8")
    st.download_button("Download OHLCV CSV", data=csv, file_name=f"{symbol.replace('/','_')}_{timeframe}.csv", mime="text/csv")

    if any(
        group_id != primary_group_id or any(sym != symbol for sym in group_frames.keys())
        for group_id, group_frames in symbol_datasets.items()
    ):
        with st.expander("Additional symbol datasets"):
            for group_id, group_frames in symbol_datasets.items():
                group_meta = symbol_fetch_info.get(group_id, {})
                for sym, sym_df in group_frames.items():
                    if group_id == primary_group_id and sym == symbol:
                        continue
                    meta = group_meta.get(sym)
                    if meta:
                        st.caption(
                            f"{group_id} · {sym} — Bars: {meta.rows:,} | Range: {meta.start} → {meta.end}"
                            f" | Pages: {meta.pages}/{meta.max_pages}"
                        )
                    else:
                        st.caption(
                            f"{group_id} · {sym} — Bars: {len(sym_df):,} | Range: {sym_df.index.min()} → {sym_df.index.max()}"
                        )
                    st.dataframe(sym_df.tail(preview_rows))

    # -----------------------------
    # Signals
    # -----------------------------
    strategy_params = {
        **scalar_params,
        "allow_shorts": allow_shorts,
        "ignore_trend": ignore_trend,
        "sizing_mode": sizing_mode,
        "timeframe": timeframe,
        "symbol_groups": symbol_groups,
        "symbol_datasets": symbol_datasets,
        "primary_symbol": symbol,
        "primary_symbol_group": primary_group_id,
    }

    is_valid, validation_message = active_strategy.validate_context(strategy_params)
    if not is_valid:
        if validation_message:
            st.warning(validation_message)
        else:
            st.warning("Strategy requirements are not met for the selected configuration.")
        st.stop()

    prepared_df = active_strategy.prepare_data(df, strategy_params)
    df_sig = active_strategy.generate_signals(prepared_df, strategy_params)
    chart_overlays = _build_chart_overlays(active_strategy, df_sig, strategy_params)

    with st.expander("🔎 Signal diagnostics"):
        summary = _compute_signal_summary(active_strategy, df_sig, ignore_trend)
        st.write(summary)
        preview_columns = active_strategy.data_requirements.get("preview_columns", [])
        if preview_columns:
            missing = [col for col in preview_columns if col not in df_sig.columns]
            if missing:
                st.dataframe(df_sig.tail(preview_rows))
            else:
                st.dataframe(df_sig[preview_columns].tail(preview_rows))
        else:
            st.dataframe(df_sig.tail(preview_rows))

    # -----------------------------
    # Backtest engines
    # -----------------------------
    if mode == "Backtest":
        st.subheader("Backtest")
        initial_cash = st.number_input("Initial cash", 100.0, 1_000_000.0, st.session_state.get("initial_cash", 10_000.0), step=100.0, key="initial_cash")

        base_params = {
            **scalar_params,
            "allow_shorts": allow_shorts,
            "ignore_trend": ignore_trend,
            "sizing_mode": sizing_mode,
            "initial_cash": initial_cash,
            "timeframe": timeframe,
            "risk_config": risk_config,
            "symbol": symbol,
            "primary_symbol": symbol,
            "symbol_groups": symbol_groups,
            "symbol_datasets": symbol_datasets,
            "primary_symbol_group": primary_group_id,
        }

        if run_mode == "Single Run":
            show_trades = st.checkbox("Show trades on chart", value=st.session_state.get("show_trades", True), key="show_trades")
            run_bt = st.button("Run Backtest", key="run_backtest")

            if run_bt:
                if engine == "Simple (backtesting.py)":
                    result = run_simple_backtest(
                        df,
                        base_params,
                        strategy=active_strategy,
                        df_sig=df_sig,
                        df_prepared=prepared_df,
                    )
                else:
                    result = run_true_stop_backtest(
                        df,
                        base_params,
                        strategy=active_strategy,
                        df_sig=df_sig,
                        df_prepared=prepared_df,
                    )

                metrics_series = pd.Series(result["metrics"])
                st.markdown("### Summary metrics")
                st.dataframe(metrics_series.to_frame("Value"))

                trades_df = result.get("trades", pd.DataFrame())
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    st.markdown("### Trades")
                    st.dataframe(trades_df)
                else:
                    st.info("No closed trades for the selected parameters.")

                trade_markers = result.get("trade_markers", []) if show_trades else None
                fig = plot_chart(df, chart_overlays, trade_markers, symbol)
                st.plotly_chart(fig, use_container_width=True)

        else:
            objective_options = [
                "Return [%]",
                "Return (Ann.) [%]",
                "Sharpe Ratio",
                "Win Rate [%]",
                "Equity Final [$]",
                "PnL",
            ]
            objective_metric = st.selectbox("Objective metric", objective_options, index=0, key="objective_metric")
            maximize_objective = st.checkbox("Maximize objective", value=True, key="maximize_objective")

            range_values = {key: generate_range(spec) for key, spec in range_specs.items()}
            total_combinations = int(np.prod([len(v) for v in range_values.values()])) if range_values else 1
            st.caption(f"Parameter combinations to evaluate: {total_combinations:,}")
            max_evaluations = st.number_input("Max evaluations", min_value=1, value=min(total_combinations, 500), step=1, key="max_evaluations")

            run_opt = st.button("Run Optimization", key="run_optimization")

            if run_opt:
                if not range_values:
                    st.warning("No parameter ranges provided. Add min/max/step values to at least one parameter.")
                else:
                    limit = min(total_combinations, int(max_evaluations))
                    combos_iter = itertools.product(*range_values.values())
                    combos_keys = list(range_values.keys())
                    status = st.status("Running optimization...", state="running")
                    progress = st.progress(0)
                    results_rows = []
                    evaluated = 0

                    for combo in combos_iter:
                        evaluated += 1
                        combo_params = base_params.copy()
                        for k, v in zip(combos_keys, combo):
                            combo_params[k] = v
                        if engine == "Simple (backtesting.py)":
                            result = run_simple_backtest(
                                df,
                                combo_params,
                                strategy=active_strategy,
                                df_prepared=None,
                            )
                        else:
                            result = run_true_stop_backtest(
                                df,
                                combo_params,
                                strategy=active_strategy,
                                df_prepared=None,
                            )

                        metrics = result.get("metrics", {})
                        objective_value = metrics.get(objective_metric)
                        if objective_value is None:
                            objective_value = float("-inf") if maximize_objective else float("inf")
                        row = {k: combo_params[k] for k in combos_keys}
                        row.update(metrics)
                        row["Objective"] = objective_value
                        results_rows.append(row)

                        progress.progress(min(1.0, evaluated / limit))
                        status.write(f"Evaluated {evaluated} / {limit} combinations")

                        if evaluated >= limit:
                            break

                    status.update(label="Optimization complete", state="complete")

                    if results_rows:
                        results_df = pd.DataFrame(results_rows)
                        ascending = not maximize_objective
                        results_df = results_df.sort_values("Objective", ascending=ascending).reset_index(drop=True)
                        st.markdown("### Optimization results")
                        st.dataframe(results_df)

                        try:
                            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                            output_path = _RESULTS_DIR / f"{selected_strategy_key}.json"
                            serial_ranges: Dict[str, Dict[str, Any]] = {}
                            for param_key, spec in range_specs.items():
                                cleaned: Dict[str, Any] = {}
                                for spec_key in ("min", "max", "step"):
                                    if spec_key in spec:
                                        cleaned[spec_key] = _to_native(spec[spec_key])
                                dtype_value = spec.get("dtype")
                                if dtype_value is not None:
                                    cleaned["dtype"] = getattr(dtype_value, "__name__", str(dtype_value))
                                serial_ranges[param_key] = cleaned

                            serialized_results = [
                                {k: _to_native(v) for k, v in row.items()}
                                for row in results_df.to_dict(orient="records")
                            ]
                            payload = {
                                "strategy": selected_strategy_key,
                                "saved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                                "objective": {
                                    "metric": objective_metric,
                                    "maximize": bool(maximize_objective),
                                },
                                "ranges": serial_ranges,
                                "evaluations": serialized_results,
                                "evaluated": evaluated,
                            }
                            output_path.write_text(json.dumps(payload, indent=2))
                            st.caption(f"Results saved to {output_path}")
                        except Exception as exc:
                            st.warning(f"Failed to persist optimization results: {exc}")

                        best_row = results_df.iloc[0]
                        best_params = base_params.copy()
                        for k in combos_keys:
                            best_params[k] = best_row[k]

                        st.session_state["best_run"] = {
                            "params": best_params,
                            "metrics": {col: best_row[col] for col in results_df.columns if col not in combos_keys},
                        }

                        if st.button("Apply top parameters to controls", key="apply_best"):
                            for k, v in best_params.items():
                                if k in active_strategy.controls:
                                    _set_param_value(selected_strategy_key, k, v)
                                    _set_range_spec(selected_strategy_key, k, None)
                                    st.session_state[f"ctrl::{selected_strategy_key}::{k}"] = v
                                elif k in {"allow_shorts", "ignore_trend", "sizing_mode"}:
                                    st.session_state[k] = v
                            st.rerun()
                    else:
                        st.warning("No results produced during optimization.")

    # -----------------------------
    # Paper mode (already fractional)
    # -----------------------------
    else:
        st.subheader("Dry-Run (Paper)")
        st.info("Paper trading uses fractional units already based on notional/price.")


def render_parameter_generator():
    st.title("🤖 AI Trader — Parameter Generator")
    st.caption("Generate LLM-backed parameter proposals and evaluate them without leaving the dashboard.")

    registry = StrategyRegistry()
    strategy_ids = sorted(registry._registry.keys())  # type: ignore[attr-defined]
    if not strategy_ids:
        st.error("No registered strategies found in the strategy builder registry.")
        return

    timeframe_options = ["1m", "15m", "1h", "3h", "1d"]

    with st.sidebar:
        st.divider()
        st.header("Parameter Generator")
        strategy_id = st.selectbox("Strategy", strategy_ids, key="builder_strategy")
        symbol = st.text_input("Symbol (Kraken/ccxt)", value="BTC/EUR", key="builder_symbol")
        timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_options.index("1h"), key="builder_timeframe")
        window_value = st.number_input(
            "Data window", min_value=1, max_value=1825, value=180, step=1, key="builder_window"
        )
        window_unit = st.selectbox(
            "Window unit",
            ["Days", "Months", "Years"],
            index=1,
            key="builder_window_unit",
        )
        since_override = st.text_input(
            "Override start (optional, UTC ISO8601)", value="", key="builder_since_override"
        )
        try:
            _, computed_start, _ = resolve_since(int(window_value), window_unit, since_override)
            st.caption(f"Approximate start: {computed_start.isoformat()} UTC")
        except ValueError:
            st.caption("Provide a valid ISO8601 override or leave blank to use the window.")
        st.subheader("LLM configuration")
        model, temperature, reasoning_effort, response_verbosity, max_output_tokens = render_llm_controls("builder")
        proposal_count = st.number_input("Proposals", min_value=1, max_value=16, value=8, step=1, key="builder_proposals")
        prompt_dir = st.text_input(
            "Prompt directory",
            value=str(Path("strategy_builder/prompts").resolve()),
            key="builder_prompt_dir",
        )
        openai_key = st.text_input(
            "OpenAI API key",
            value="",
            type="password",
            help="Optional. Provide to let the app call OpenAI directly.",
            key="builder_openai_key",
        )

    st.write(
        "Configure the data window and click *Generate parameter proposals* to ask the builder for new configurations. "
        "Each proposal is backtested locally and evaluated using the same acceptance criteria as the CLI."
    )

    trigger = st.button("Generate parameter proposals", key="builder_run")

    if trigger:
        st.session_state.pop("builder_error", None)
        st.session_state.pop("builder_results", None)
        try:
            if not symbol.strip():
                raise ValueError("Symbol is required for data fetch.")

            try:
                since_ms, since_dt, window_delta = resolve_since(int(window_value), window_unit, since_override)
            except ValueError as exc:  # pragma: no cover - user input validation
                raise ValueError(str(exc)) from exc

            with st.spinner("Fetching data and querying the LLM..."):
                target_end_dt = since_dt + window_delta
                now_dt = datetime.now(UTC)
                if target_end_dt > now_dt:
                    target_end_dt = now_dt
                target_end_ms = int(target_end_dt.timestamp() * 1000)
                fetch_result = fetch_ohlcv(
                    symbol.strip(),
                    timeframe,
                    since_ms,
                    target_end_ms=target_end_ms,
                )

                if fetch_result.data.empty:
                    raise ValueError("No OHLCV data retrieved for the requested window.")

                st.session_state["builder_last_fetch"] = fetch_result
                st.session_state["builder_last_start"] = since_dt
                st.session_state["builder_last_end"] = target_end_dt
                st.session_state["builder_last_symbol"] = symbol.strip()
                st.session_state["builder_last_timeframe"] = timeframe

                df_reset = fetch_result.data.reset_index().rename(columns={"index": "Timestamp"})
                df_reset["Timestamp"] = pd.to_datetime(df_reset["Timestamp"], utc=True)
                df_reset = df_reset[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

                data_service = DataService()
                summary = data_service.summarize(df_reset)

                prompt_path = Path(prompt_dir).expanduser()
                if not prompt_path.exists():
                    raise FileNotFoundError(f"Prompt directory not found: {prompt_path}")

                client = None
                if openai_key:
                    try:
                        from openai import OpenAI  # type: ignore

                        client = OpenAI(api_key=openai_key)
                    except Exception as exc:  # pragma: no cover - optional dependency failure
                        raise RuntimeError(f"Failed to initialise OpenAI client: {exc}") from exc

                optimizer = LLMOptimizer(
                    registry=registry,
                    data_service=data_service,
                    prompt_dir=prompt_path,
                    client=client,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    verbosity=response_verbosity,
                    max_output_tokens=max_output_tokens,
                )

                response = optimizer.optimize(
                    strategy_id=strategy_id,
                    timeframe=timeframe,
                    data_summary=summary,
                    prior_bests=None,
                    constraints=None,
                    n=int(proposal_count),
                )

                backtester = BacktestService()
                key_metrics: Dict[str, Any] = {}

                for proposal in response.proposals:
                    params = dict(proposal.params)
                    metrics = backtester.run(strategy_id, params, df_reset)
                    key_metrics[json.dumps(params, sort_keys=True)] = metrics

                evaluations = []
                card = {
                    "version": registry.version,
                    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "prompt_hash": DataService.summary_hash(summary),
                    "strategy": strategy_id,
                    "timeframe": timeframe,
                    "summary": summary,
                    "proposals": [],
                }

                table_rows: List[Dict[str, Any]] = []

                for proposal in response.proposals:
                    key = json.dumps(proposal.params, sort_keys=True)
                    metrics = key_metrics[key]
                    accepted, reason = evaluator_accept(proposal.predicted_metrics, metrics)
                    evaluations.append((proposal, metrics, accepted, reason))
                    card["proposals"].append(
                        {
                            "proposal": proposal.model_dump()
                            if hasattr(proposal, "model_dump")
                            else proposal.dict(),
                            "measured": metrics.model_dump()
                            if hasattr(metrics, "model_dump")
                            else metrics.dict(),
                            "accepted": bool(accepted),
                            "reason": reason,
                        }
                    )
                    table_rows.append(
                        {
                            "Params": json.dumps(proposal.params, sort_keys=True),
                            "CAGR": metrics.cagr,
                            "Sharpe": metrics.sharpe,
                            "Sortino": metrics.sortino,
                            "Max Drawdown": metrics.max_drawdown,
                            "Total Return": metrics.total_return,
                            "Accepted": bool(accepted),
                            "Reason": reason,
                        }
                    )

                try:
                    champion_params, champion_metrics, report = choose_champion(
                        [proposal for proposal, *_ in evaluations],
                        key_metrics,
                    )
                    champion_accepted, champion_reason = evaluator_accept({}, champion_metrics)
                    card["champion"] = {
                        "params": champion_params,
                        "metrics": champion_metrics.model_dump()
                        if hasattr(champion_metrics, "model_dump")
                        else champion_metrics.dict(),
                        "report": report,
                        "accepted": bool(champion_accepted),
                        "reason": champion_reason,
                    }
                    if champion_accepted:
                        st.session_state["builder_pending_apply"] = {
                            "strategy": strategy_id,
                            "params": champion_params,
                        }
                except ValueError as exc:
                    card["champion_error"] = str(exc)

                st.session_state["builder_results"] = {
                    "card": make_json_safe(card),
                    "rows": table_rows,
                    "symbol": symbol.strip(),
                    "timeframe": timeframe,
                    "strategy": strategy_id,
                }
        except Exception as exc:  # pragma: no cover - defensive user feedback
            st.session_state["builder_error"] = str(exc)

    fetch_info = st.session_state.get("builder_last_fetch")
    if isinstance(fetch_info, OHLCVFetchResult):
        symbol_hint = st.session_state.get("builder_last_symbol", symbol)
        timeframe_hint = st.session_state.get("builder_last_timeframe", timeframe)
        coverage = fetch_info.coverage
        target_span = fetch_info.target_coverage
        message = (
            f"Latest data fetch for {symbol_hint} @ {timeframe_hint}: {fetch_info.rows:,} bars"
            f" from {fetch_info.start} to {fetch_info.end}. Requests {fetch_info.pages}/{fetch_info.max_pages}."
        )
        if coverage is not None and coverage > pd.Timedelta(0):
            message += f" Coverage ≈{coverage}."
        if target_span is not None and target_span > pd.Timedelta(0):
            message += f" Requested ≈{target_span}."
        if not fetch_info.reached_target and target_span is not None:
            st.warning(
                message + " Exchange returned less data than requested; automatic retries were exhausted.",
                icon="ℹ️",
            )
        else:
            st.info(message, icon="ℹ️")

    error_message = st.session_state.get("builder_error")
    results_payload = st.session_state.get("builder_results")

    if error_message:
        st.error(error_message)
        return

    if not results_payload:
        st.info("Adjust the settings and run the generator to see proposals and evaluations here.")
        return

    card = results_payload["card"]
    table_rows = results_payload["rows"]
    results_df = pd.DataFrame(table_rows)
    st.markdown("### Evaluated proposals")
    st.dataframe(results_df)

    champion = card.get("champion")
    if champion:
        st.markdown("### Champion proposal")
        st.json(make_json_safe(champion))

    card_bytes = json.dumps(make_json_safe(card), indent=2).encode("utf-8")
    file_name = (
        f"strategy_card_{card['strategy']}_{card['timeframe']}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
    )
    st.download_button(
        "Download strategy card",
        data=card_bytes,
        file_name=file_name,
        mime="application/json",
        key="builder_download",
    )


def render_strategy_generator():
    st.title("🛠️ AI Trader — Strategy Generator")
    st.caption("Describe the behaviour you want and let the builder compose, validate, and compile a strategy graph.")

    timeframe_options = ["1m", "15m", "1h", "3h", "1d"]

    with st.sidebar:
        st.divider()
        st.header("Strategy Generator")
        prompt_dir = st.text_input(
            "Prompt directory",
            value=str(Path("strategy_builder/prompts").resolve()),
            key="compose_prompt_dir",
        )
        catalog_path = st.text_input(
            "Operator catalog",
            value=str(Path("strategy_builder/configs/operators_catalog.yaml").resolve()),
            key="compose_catalog",
        )
        model, temperature, reasoning_effort, response_verbosity, max_output_tokens = render_llm_controls("compose")
        symbol = st.text_input("Validation symbol", value="BTC/EUR", key="compose_symbol")
        timeframe = st.selectbox("Validation timeframe", timeframe_options, index=timeframe_options.index("1h"), key="compose_timeframe")
        window_value = st.number_input(
            "Data window", min_value=1, max_value=1825, value=365, step=1, key="compose_window"
        )
        window_unit = st.selectbox(
            "Window unit",
            ["Days", "Months", "Years"],
            index=2,
            key="compose_window_unit",
        )
        since_override = st.text_input(
            "Override start (optional, UTC ISO8601)", value="", key="compose_since_override"
        )
        try:
            _, compose_start, _ = resolve_since(int(window_value), window_unit, since_override)
            st.caption(f"Approximate start: {compose_start.isoformat()} UTC")
        except ValueError:
            st.caption("Provide a valid ISO8601 override or leave blank to use the window.")
        openai_key = st.text_input(
            "OpenAI API key",
            value="",
            type="password",
            help="Optional. Provide to let the app call OpenAI directly.",
            key="compose_openai_key",
        )

    spec_default = """Create a breakout strategy that enters on 20-bar highs, \
uses ATR based stops and trails, and reduces position size during sideways regimes."""
    spec_text = st.text_area("Strategy specification", value=spec_default, height=220, key="compose_spec")

    trigger = st.button("Generate strategy module", key="compose_run")

    if trigger:
        st.session_state.pop("compose_error", None)
        st.session_state.pop("compose_result", None)
        try:
            if not spec_text.strip():
                raise ValueError("Please provide a natural language specification for the strategy.")

            prompt_path = Path(prompt_dir).expanduser()
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt directory not found: {prompt_path}")

            catalog_file = Path(catalog_path).expanduser()
            if not catalog_file.exists():
                raise FileNotFoundError(f"Operator catalog not found: {catalog_file}")

            try:
                since_ms, since_dt, window_delta = resolve_since(int(window_value), window_unit, since_override)
            except ValueError as exc:  # pragma: no cover - user input validation
                raise ValueError(str(exc)) from exc

            client = None
            if openai_key:
                try:
                    from openai import OpenAI  # type: ignore

                    client = OpenAI(api_key=openai_key)
                except Exception as exc:  # pragma: no cover - optional dependency failure
                    raise RuntimeError(f"Failed to initialise OpenAI client: {exc}") from exc

            with st.spinner("Calling the LLM and validating the generated graph..."):
                optimizer = LLMOptimizer(
                    registry=StrategyRegistry(),
                    data_service=DataService(),
                    prompt_dir=prompt_path,
                    client=client,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    verbosity=response_verbosity,
                    max_output_tokens=max_output_tokens,
                )

                template = (prompt_path / "compose_template.md").read_text()
                prompt = template.replace("{{graph_schema_json}}", StrategyGraph.schema_json(indent=2))
                prompt = prompt.replace("{{nl_spec}}", spec_text.strip())
                raw = optimizer._call_llm(prompt)
                graph = load_graph_from_json(raw)

                catalog = OperatorCatalog(catalog_file)
                validator = GraphValidator(catalog)
                validator.validate(graph)

                compile_and_load(graph)

                target_end_dt = since_dt + window_delta
                now_dt = datetime.now(UTC)
                if target_end_dt > now_dt:
                    target_end_dt = now_dt
                target_end_ms = int(target_end_dt.timestamp() * 1000)
                fetch_result = fetch_ohlcv(
                    symbol.strip(),
                    timeframe,
                    since_ms,
                    target_end_ms=target_end_ms,
                )
                if fetch_result.data.empty:
                    raise ValueError("No OHLCV data retrieved for validation window.")

                st.session_state["compose_last_fetch"] = fetch_result
                st.session_state["compose_last_start"] = since_dt
                st.session_state["compose_last_end"] = target_end_dt
                st.session_state["compose_last_symbol"] = symbol.strip()
                st.session_state["compose_last_timeframe"] = timeframe

                df_reset = fetch_result.data.reset_index().rename(columns={"index": "Timestamp"})
                df_reset["Timestamp"] = pd.to_datetime(df_reset["Timestamp"], utc=True)
                df_reset = df_reset[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

                executor = GraphExecutor(graph=graph, params={})
                executor.run(df_reset)

                code = render_strategy_code(graph)

                st.session_state["compose_result"] = {
                    "graph": graph.model_dump() if hasattr(graph, "model_dump") else graph.dict(),
                    "code": code,
                }
        except Exception as exc:  # pragma: no cover - defensive user feedback
            st.session_state["compose_error"] = str(exc)

    compose_fetch = st.session_state.get("compose_last_fetch")
    if isinstance(compose_fetch, OHLCVFetchResult):
        symbol_hint = st.session_state.get("compose_last_symbol", symbol)
        timeframe_hint = st.session_state.get("compose_last_timeframe", timeframe)
        coverage = compose_fetch.coverage
        target_span = compose_fetch.target_coverage
        fetch_message = (
            f"Validation data for {symbol_hint} @ {timeframe_hint}: {compose_fetch.rows:,} bars"
            f" from {compose_fetch.start} to {compose_fetch.end}. Requests {compose_fetch.pages}/{compose_fetch.max_pages}."
        )
        if coverage is not None and coverage > pd.Timedelta(0):
            fetch_message += f" Coverage ≈{coverage}."
        if target_span is not None and target_span > pd.Timedelta(0):
            fetch_message += f" Requested ≈{target_span}."
        if not compose_fetch.reached_target and target_span is not None:
            st.warning(
                fetch_message + " Exchange returned less data than requested; automatic retries were exhausted.",
                icon="ℹ️",
            )
        else:
            st.info(fetch_message, icon="ℹ️")

    compose_error = st.session_state.get("compose_error")
    compose_result = st.session_state.get("compose_result")

    if compose_error:
        st.error(compose_error)
        return

    if not compose_result:
        st.info("Provide a specification and run the generator to preview generated strategies here.")
        return

    st.markdown("### Generated strategy code")
    st.code(compose_result["code"], language="python")

    st.markdown("### Strategy graph")
    st.json(compose_result["graph"])

    code_bytes = compose_result["code"].encode("utf-8")
    st.download_button(
        "Download strategy module",
        data=code_bytes,
        file_name="generated_strategy.py",
        mime="text/x-python",
        key="compose_download",
    )

    st.markdown("### Integrate generated strategy")
    default_name = st.session_state.get("compose_register_name") or "Generated Strategy"
    name_input = st.text_input("Strategy name", value=default_name, key="compose_register_name")
    default_key = st.session_state.get("compose_register_key") or slugify_strategy_key(name_input)
    key_input = st.text_input("Strategy key", value=default_key, key="compose_register_key")
    default_description = (
        st.session_state.get("compose_register_description")
        or f"Generated from spec: {spec_text.strip()[:140]}"
    )
    description_input = st.text_area(
        "Description",
        value=default_description,
        height=120,
        key="compose_register_description",
    )

    if st.button("Register strategy", key="compose_register_button"):
        try:
            name_clean = name_input.strip() or "Generated Strategy"
            raw_key = key_input.strip() or name_clean
            key_clean = slugify_strategy_key(raw_key)
            if key_clean in STRATEGY_REGISTRY:
                raise ValueError(f"Strategy key '{key_clean}' already exists.")
            graph_obj = StrategyGraph.parse_obj(compose_result["graph"])
            description_clean = description_input.strip() or f"Generated from spec: {spec_text.strip()[:140]}"
            definition = create_generated_strategy(graph_obj, name_clean, key_clean, description_clean)
            register_strategy(definition)
            st.session_state["active_strategy"] = key_clean
            st.session_state["builder_strategy"] = key_clean
            st.session_state["workspace_mode"] = "Trading Console"
            st.session_state["compose_register_key"] = key_clean
            st.session_state["compose_register_name"] = name_clean
            st.session_state["compose_register_description"] = description_clean
            if key_clean != raw_key:
                st.info(f"Strategy key normalised to '{key_clean}'.")
            st.success(
                f"Strategy '{name_clean}' registered as '{key_clean}'. It is now available in the trading console and optimizer."
            )
        except Exception as exc:
            st.error(f"Failed to register strategy: {exc}")


with st.sidebar:
    workspace_options = ["Trading Console", "Parameter Generator", "Strategy Generator"]
    default_workspace = st.session_state.get("workspace_mode", workspace_options[0])
    if default_workspace not in workspace_options:
        default_workspace = workspace_options[0]
    workspace_index = workspace_options.index(default_workspace)
    workspace = st.selectbox("Workspace", workspace_options, index=workspace_index, key="workspace_mode")

if workspace == "Trading Console":
    render_trading_console()
elif workspace == "Parameter Generator":
    render_parameter_generator()
else:
    render_strategy_generator()
