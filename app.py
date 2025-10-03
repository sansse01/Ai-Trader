
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import ccxt
import time

st.set_page_config(page_title="EMA9 Trend Trader — Contract Size (Option 2)", layout="wide")
st.title("📈 EMA9 Trend Trader — Kraken (Contract Size)")

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

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def slope_pct(series: pd.Series, lookback: int) -> pd.Series:
    ref = series.shift(lookback)
    return (series.sub(ref)).div(ref).mul(100.0)

def signals(df: pd.DataFrame, ema_period: int, lb: int, min_slope: float) -> pd.DataFrame:
    s = df.copy()
    s['EMA'] = ema(s['Close'], ema_period)
    s['SlopePct'] = slope_pct(s['EMA'], lb).fillna(0.0)
    s['trend_up'] = s['SlopePct'] > min_slope
    s['trend_dn'] = s['SlopePct'] < -min_slope
    s['prev_close'] = s['Close'].shift(1)
    s['prev_ema'] = s['EMA'].shift(1)
    s['bull_cross'] = (s['prev_close'] <= s['prev_ema']) & (s['Close'] > s['EMA'])
    s['bear_cross'] = (s['prev_close'] >= s['prev_ema']) & (s['Close'] < s['EMA'])
    return s

def plot_chart(df, ema_period, trades=None, symbol=""):
    ema_line = ema(df['Close'], int(ema_period))
    fig = go.Figure([go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")])
    fig.add_trace(go.Scatter(x=df.index, y=ema_line, name=f"EMA({int(ema_period)})"))
    if trades is not None and len(trades):
        ent = [t for t in trades if t.get('etype')=='entry']
        exi = [t for t in trades if t.get('etype')=='exit']
        if ent:
            fig.add_trace(go.Scatter(x=[t['time'] for t in ent], y=[t['price'] for t in ent], mode="markers", name="Entries",
                                     marker=dict(size=9, symbol="triangle-up")))
        if exi:
            fig.add_trace(go.Scatter(x=[t['time'] for t in exi], y=[t['price'] for t in exi], mode="markers", name="Exits",
                                     marker=dict(size=9, symbol="x")))
    fig.update_layout(height=800, xaxis_title="Time", yaxis_title=symbol)
    return fig


def _to_native(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value
    return value


def run_simple_backtest(df: pd.DataFrame, params: dict, df_sig: pd.DataFrame | None = None):
    from backtesting import Backtest, Strategy

    ema_period = int(params.get('ema_period', 9))
    slope_lookback = int(params.get('slope_lookback', 5))
    min_slope_percent = float(params.get('min_slope_percent', 0.1))
    stop_loss_percent = float(params.get('stop_loss_percent', 0.0))
    trail_percent = float(params.get('trail_percent', 0.0))
    allow_shorts = bool(params.get('allow_shorts', True))
    ignore_trend = bool(params.get('ignore_trend', False))
    sizing_mode = params.get('sizing_mode', "Whole units (int)")
    risk_fraction = float(params.get('risk_fraction', 0.25))
    contract_size = float(params.get('contract_size', 0.001))
    initial_cash = float(params.get('initial_cash', 10_000.0))
    fee_percent = float(params.get('fee_percent', 0.0))
    slippage_percent = float(params.get('slippage_percent', 0.0))

    if df_sig is None:
        df_sig = signals(df, ema_period, slope_lookback, min_slope_percent)

    df_bt = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_bt[['Open', 'High', 'Low', 'Close']] = df_bt[['Open', 'High', 'Low', 'Close']] * contract_size

    class Strat(Strategy):
        def init(self):
            self.current_sl = None
            self.last_entry_ts = None

        def next(self):
            ts = self.data.index[-1]
            close = self.data.Close[-1]
            row = df_sig.loc[ts]
            min_bars = max(ema_period + 1, slope_lookback + 1)
            if len(self.data) < min_bars or pd.isna(row['prev_close']) or pd.isna(row['prev_ema']):
                return
            if not self.position:
                self.current_sl = None
            if self.position:
                start_ts = self.last_entry_ts or self.data.index[-1]
                if self.position.is_long:
                    recent_high = df.loc[start_ts: ts]['High'].max() * contract_size
                    trail = recent_high * (1 - trail_percent / 100.0)
                    if self.current_sl is None or trail > self.current_sl:
                        self.current_sl = trail
                    if self.data.Low[-1] <= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return
                else:
                    recent_low = df.loc[start_ts: ts]['Low'].min() * contract_size
                    trail = recent_low * (1 + trail_percent / 100.0)
                    if self.current_sl is None or trail < self.current_sl:
                        self.current_sl = trail
                    if self.data.High[-1] >= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return

            go_long = row['bull_cross'] and (ignore_trend or row['trend_up'])
            go_short = allow_shorts and row['bear_cross'] and (ignore_trend or row['trend_dn'])

            if sizing_mode.startswith("Whole"):
                cash = self.equity
                notional = cash * risk_fraction
                units = max(1, int(notional / max(1e-9, close)))
                size = units
            else:
                size = max(0.001, min(0.99, risk_fraction))

            if not self.position:
                if go_long:
                    sl_unscaled = df.loc[ts]['Close'] * (1 - stop_loss_percent / 100.0) if stop_loss_percent > 0 else None
                    sl = sl_unscaled * contract_size if sl_unscaled is not None else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.buy(size=size, sl=sl)
                elif go_short:
                    sl_unscaled = df.loc[ts]['Close'] * (1 + stop_loss_percent / 100.0) if stop_loss_percent > 0 else None
                    sl = sl_unscaled * contract_size if sl_unscaled is not None else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.sell(size=size, sl=sl)

    bt = Backtest(
        df_bt,
        Strat,
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


def run_true_stop_backtest(df: pd.DataFrame, params: dict, df_sig: pd.DataFrame | None = None):
    import backtrader as bt

    ema_period = int(params.get('ema_period', 9))
    slope_lookback = int(params.get('slope_lookback', 5))
    min_slope_percent = float(params.get('min_slope_percent', 0.1)) / 100.0
    stop_loss_percent = float(params.get('stop_loss_percent', 0.0)) / 100.0
    trail_percent = float(params.get('trail_percent', 0.0)) / 100.0
    allow_shorts = bool(params.get('allow_shorts', True))
    ignore_trend = bool(params.get('ignore_trend', False))
    risk_fraction = float(params.get('risk_fraction', 0.25))
    fee_percent = float(params.get('fee_percent', 0.0))
    slippage_percent = float(params.get('slippage_percent', 0.0))
    initial_cash = float(params.get('initial_cash', 10_000.0))

    class CashSizer(bt.Sizer):
        params = (('risk_frac', risk_fraction), ('min_unit', 0.0001), ('max_lev', float(params.get('max_leverage', 5.0))), ('retint', False))

        def _getsizing(self, comminfo, cash, data, isbuy):
            price = data.close[0]
            notional = cash * self.p.risk_frac * min(self.p.max_lev, float(params.get('max_leverage', 5.0)))
            units = notional / max(price, 1e-9)
            snapped = max(self.p.min_unit, round(units / self.p.min_unit) * self.p.min_unit)
            return snapped if not self.p.retint else int(snapped)

    class BTStrategy(bt.Strategy):
        params = dict(
            ema_period=ema_period,
            slope_lb=slope_lookback,
            min_slope=min_slope_percent,
            stop_loss=stop_loss_percent,
            trail=trail_percent,
            allow_shorts=allow_shorts,
            ignore_trend=ignore_trend,
        )

        def __init__(self):
            self.close = self.datas[0].close
            self.high = self.datas[0].high
            self.low = self.datas[0].low
            self.ema = bt.ind.EMA(self.close, period=self.p.ema_period)
            self.prev_close = self.close(-1)
            self.prev_ema = self.ema(-1)
            self.ema_shift = self.ema - self.ema(-int(self.p.slope_lb))
            self.slope_pct = self.ema_shift / bt.If(self.ema(-int(self.p.slope_lb)) == 0, 1e-9, self.ema(-int(self.p.slope_lb)))
            self.ord = None

        def next(self):
            if self.ord:
                return
            bull_cross = (self.prev_close[0] <= self.prev_ema[0]) and (self.close[0] > self.ema[0])
            bear_cross = (self.prev_close[0] >= self.prev_ema[0]) and (self.close[0] < self.ema[0])
            trend_up = (self.slope_pct[0]) > self.p.min_slope or self.p.ignore_trend
            trend_dn = (self.slope_pct[0]) < -self.p.min_slope or self.p.ignore_trend
            if not self.position:
                if bull_cross and trend_up:
                    self.ord = self.buy()
                    price = self.close[0]
                    if self.p.stop_loss > 0:
                        sl = price * (1 - self.p.stop_loss)
                        self.sell(exectype=bt.Order.Stop, price=sl)
                    if self.p.trail > 0:
                        self.sell(exectype=bt.Order.StopTrail, trailpercent=self.p.trail)
                elif self.p.allow_shorts and bear_cross and trend_dn:
                    self.ord = self.sell()
                    price = self.close[0]
                    if self.p.stop_loss > 0:
                        sl = price * (1 + self.p.stop_loss)
                        self.buy(exectype=bt.Order.Stop, price=sl)
                    if self.p.trail > 0:
                        self.buy(exectype=bt.Order.StopTrail, trailpercent=self.p.trail)

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.ord = None

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcommission(commission=(fee_percent + slippage_percent) / 100.0)
    cerebro.broker.set_slippage_perc(perc=slippage_percent / 100.0)
    cerebro.broker.setcash(initial_cash)

    comp = 60 if params.get('timeframe') == '1h' else (240 if params.get('timeframe') == '4h' else 1440)
    data_bt = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=comp)
    cerebro.adddata(data_bt)
    cerebro.addsizer(CashSizer)
    cerebro.addstrategy(BTStrategy)
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


def render_scalar_or_range(label: str, key: str, mode: str, *, min_value, max_value, value, step, dtype=float, slider=False, format: str | None = None):
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

    cols = st.columns(3)
    with cols[0]:
        min_kwargs = dict(min_value=min_value, max_value=max_value, value=value, step=step, key=f"{key}_min")
        if format is not None:
            min_kwargs["format"] = format
        min_val = st.number_input(f"{label} min", **min_kwargs)
    with cols[1]:
        max_kwargs = dict(min_value=min_val, max_value=max_value, value=max(value, min_val), step=step, key=f"{key}_max")
        if format is not None:
            max_kwargs["format"] = format
        max_val = st.number_input(f"{label} max", **max_kwargs)
    with cols[2]:
        range_step_default = step
        step_kwargs = dict(min_value=step, value=range_step_default, key=f"{key}_step")
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
PARAM_CONFIG = {
    "ema_period": dict(label="EMA period", dtype=int, min=1, max=200, value=9, step=1),
    "slope_lookback": dict(label="Slope lookback (bars)", dtype=int, min=1, max=200, value=5, step=1),
    "min_slope_percent": dict(label="Min slope %", dtype=float, min=0.0, max=100.0, value=0.1, step=0.1, format="%.2f"),
    "stop_loss_percent": dict(label="Stop loss %", dtype=float, min=0.0, max=50.0, value=2.0, step=0.1, format="%.2f"),
    "trail_percent": dict(label="Trailing stop %", dtype=float, min=0.0, max=50.0, value=1.5, step=0.1, format="%.2f"),
    "fee_percent": dict(label="Fee % per fill", dtype=float, min=0.0, max=2.0, value=0.26, step=0.01, format="%.4f"),
    "slippage_percent": dict(label="Slippage % per fill", dtype=float, min=0.0, max=2.0, value=0.05, step=0.01, format="%.4f"),
    "max_leverage": dict(label="Max leverage", dtype=float, min=1.0, max=10.0, value=5.0, step=0.5, format="%.2f"),
    "risk_fraction": dict(label="Risk fraction of equity", dtype=float, min=0.01, max=1.0, value=0.25, step=0.01, slider=True),
    "contract_size": dict(label="Contract size (BTC per unit)", dtype=float, min=0.000001, max=10.0, value=0.001, step=0.0001, format="%.6f"),
}

with st.sidebar:
    st.header("Mode & Engine")
    run_mode = st.radio("Run mode", ["Single Run", "Optimization"], index=0, key="run_mode")
    mode = st.radio("Mode", ["Backtest", "Dry-Run (paper)"] , index=0, key="session_mode")
    engine = st.selectbox("Backtest engine", ["Simple (backtesting.py)", "TRUE STOP (backtrader)"], index=0, key="engine")

    st.header("Data")
    symbol = st.text_input("Symbol (Kraken/ccxt)", value=st.session_state.get("symbol", "BTC/EUR"), key="symbol")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(st.session_state.get("timeframe", "1h")), key="timeframe")
    since_str = st.text_input("Since (UTC ISO8601)", value=st.session_state.get("since_str", "2022-01-01T00:00:00Z"), key="since_str")
    ex = ccxt.kraken()
    try:
        since_ms = ex.parse8601(since_str)
    except Exception:
        since_ms = None
        st.error("Invalid 'Since' string (e.g. 2023-01-01T00:00:00Z)")

    st.header("Strategy")
    scalar_params = {}
    range_specs = {}
    for key, cfg in PARAM_CONFIG.items():
        default_value = st.session_state.get(key, cfg.get("value"))
        value, range_spec = render_scalar_or_range(
            cfg["label"],
            key,
            run_mode,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=default_value,
            step=cfg["step"],
            dtype=cfg.get("dtype", float),
            slider=cfg.get("slider", False),
            format=cfg.get("format"),
        )
        if range_spec is None:
            scalar_params[key] = value
        else:
            scalar_params[key] = range_spec["min"]
            range_specs[key] = range_spec

    allow_shorts = st.checkbox("Allow shorts", value=st.session_state.get("allow_shorts", True), key="allow_shorts")
    ignore_trend = st.checkbox("Ignore trend filter (debug)", value=st.session_state.get("ignore_trend", False), key="ignore_trend")

    sizing_mode = st.selectbox("Sizing mode", ["Whole units (int)", "Fraction of equity (0-1)"] , index=["Whole units (int)", "Fraction of equity (0-1)"].index(st.session_state.get("sizing_mode", "Whole units (int)")), key="sizing_mode")

    st.header("Preview")
    preview_rows = st.slider("Preview last N rows", 10, 1000, st.session_state.get("preview_rows", 50), step=10, key="preview_rows")

if since_ms is None:
    st.stop()

with st.spinner("Fetching data..."):
    df = fetch_ohlcv(symbol, timeframe, since_ms)
if df.empty:
    st.error("No data fetched. Try: 1) symbol 'XBT/EUR', 2) earlier 'Since', 3) timeframe '1h'.")
    st.stop()
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

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

# -----------------------------
# Signals
# -----------------------------
ema_period_val = int(scalar_params["ema_period"])
slope_lookback_val = int(scalar_params["slope_lookback"])
min_slope_val = float(scalar_params["min_slope_percent"])

df_sig = signals(df, ema_period_val, slope_lookback_val, min_slope_val)

with st.expander("🔎 Signal diagnostics"):
    longc = int((df_sig['bull_cross'] & (df_sig['trend_up'] | ignore_trend)).sum())
    shortc = int((df_sig['bear_cross'] & (df_sig['trend_dn'] | ignore_trend)).sum())
    st.write({"bars": int(len(df)), "long_cond_count": longc, "short_cond_count": shortc})

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
    }

    if run_mode == "Single Run":
        show_trades = st.checkbox("Show trades on chart", value=st.session_state.get("show_trades", True), key="show_trades")
        run_bt = st.button("Run Backtest", key="run_backtest")

        if run_bt:
            if engine == "Simple (backtesting.py)":
                result = run_simple_backtest(df, base_params, df_sig=df_sig)
            else:
                result = run_true_stop_backtest(df, base_params)

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
            fig = plot_chart(df, ema_period_val, trade_markers, symbol)
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
                        result = run_simple_backtest(df, combo_params)
                    else:
                        result = run_true_stop_backtest(df, combo_params)

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
                            if k in PARAM_CONFIG:
                                st.session_state[k] = v
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
