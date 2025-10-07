
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk.engine import OrderContext, PortfolioState, RiskEngine, RiskEngineConfig
from strategies import DEFAULT_STRATEGY_KEY, STRATEGY_REGISTRY
from strategies.base import StrategyDefinition

st.set_page_config(page_title="AI Trader — Contract Size", layout="wide")
st.title("📈 AI Trader — Kraken (Contract Size)")

# -----------------------------
# Kraken symbol normalization (BTC↔XBT) + robust fetch
# -----------------------------
def normalize_symbol(exchange: 'ccxt.Exchange', symbol: str) -> str:
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

@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, since_ms: int, limit_per_page: int = 720, max_pages: int = 60) -> pd.DataFrame:
    ex = ccxt.kraken()
    sym = normalize_symbol(ex, symbol)
    all_rows, cursor, pages = [], since_ms, 0
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe, since=cursor, limit=limit_per_page)
        except ccxt.RateLimitExceeded:
            time.sleep(1.25); continue
        except Exception:
            break
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        cursor = ohlcv[-1][0] + 1
        pages += 1
        if pages >= max_pages or len(ohlcv) < 2:
            break
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    df = pd.DataFrame(all_rows, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['Timestamp']).set_index('Timestamp').sort_index()
    return df

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
                return super().buy(size=size, **kwargs)
            price = float(self.data.Close[-1])
            ctx, fractional, state = self._risk_prepare_order(float(size), 1, price)
            decision = self._risk_engine.modify_order(ctx, state)
            if decision.cancel or decision.size <= 0:
                return None
            final_size = decision.size
            if fractional:
                equity = max(state.equity, 1e-9)
                final_size = min(0.9999, final_size * price / equity)
                if final_size <= 0:
                    return None
            else:
                final_size = float(round(final_size))
                if final_size <= 0:
                    return None
            return super().buy(size=final_size, **kwargs)

        def sell(self, *, size: float = 0.9999, **kwargs):
            if not getattr(self, "_risk_engine", None) or not self._risk_engine.enabled:
                return super().sell(size=size, **kwargs)
            price = float(self.data.Close[-1])
            ctx, fractional, state = self._risk_prepare_order(float(size), -1, price)
            decision = self._risk_engine.modify_order(ctx, state)
            if decision.cancel or decision.size <= 0:
                return None
            final_size = decision.size
            if fractional:
                equity = max(state.equity, 1e-9)
                final_size = min(0.9999, final_size * price / equity)
                if final_size <= 0:
                    return None
            else:
                final_size = float(round(final_size))
                if final_size <= 0:
                    return None
            return super().sell(size=final_size, **kwargs)

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
        "saved_at": datetime.utcnow().isoformat() + "Z",
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
                st.experimental_rerun()

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

with st.spinner("Fetching data..."):
    symbol_datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
    empty_symbols: list[tuple[str, str]] = []
    for group_id, entries in symbol_groups.items():
        group_frames: Dict[str, pd.DataFrame] = {}
        for entry in entries:
            df_entry = fetch_ohlcv(entry, timeframe, since_ms)
            if df_entry.empty:
                empty_symbols.append((group_id, entry))
            else:
                if df_entry.index.tz is None:
                    df_entry.index = df_entry.index.tz_localize('UTC')
            group_frames[entry] = df_entry
        symbol_datasets[group_id] = group_frames

if empty_symbols:
    missing_descriptions = ", ".join(f"{grp}:{sym}" for grp, sym in empty_symbols)
    st.error(
        "No data fetched for: " + missing_descriptions + ". Try adjusting the symbol or the date range.")
    st.stop()

primary_group = symbol_datasets.get(primary_group_id, {})
df = primary_group.get(symbol)
if df is None or df.empty:
    st.error("No data fetched. Try: 1) symbol 'XBT/EUR', 2) earlier 'Since', 3) timeframe '1h'.")
    st.stop()

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("Data Preview")
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
            for sym, sym_df in group_frames.items():
                if group_id == primary_group_id and sym == symbol:
                    continue
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
                            "saved_at": datetime.utcnow().isoformat() + "Z",
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
                        st.experimental_rerun()
                else:
                    st.warning("No results produced during optimization.")

# -----------------------------
# Paper mode (already fractional)
# -----------------------------
else:
    st.subheader("Dry-Run (Paper)")
    st.info("Paper trading uses fractional units already based on notional/price.")
