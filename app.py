
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
import ccxt

st.set_page_config(page_title="EMA9 Trend Trader", layout="wide")
st.title("📈 EMA9 Trend Trader — Kraken")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, since_ms: int, limit: int = 5000) -> pd.DataFrame:
    ex = ccxt.kraken()
    all_rows = []
    since = since_ms
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=min(limit, 720))
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 2 or len(all_rows) > 25000:
            break
    if not all_rows:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    df = pd.DataFrame(all_rows, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Timestamp', inplace=True)
    return df

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def slope_pct(series: pd.Series, lookback: int) -> pd.Series:
    ref = series.shift(lookback)
    return (series.sub(ref)).div(ref).mul(100.0)

def calc_size(equity: float, price: float, risk_fraction: float, max_leverage: float) -> float:
    notional = equity * risk_fraction * max(1.0, min(max_leverage, 100.0))
    return 0.0 if price <= 0 else notional / price

def apply_slip(price: float, is_buy: bool, slippage_percent: float) -> float:
    s = float(slippage_percent) / 100.0
    return price * (1 + s) if is_buy else price * (1 - s)

def bt_fraction_size(risk_fraction: float, max_leverage: float) -> float:
    """Backtesting.py requires size in (0,1) for equity-fraction sizing.
    We cap to 0.99 because margin isn't modeled; leverage>1 has no effect in Backtest.
    """
    frac = float(risk_fraction) * min(1.0, float(max_leverage))
    return max(0.01, min(0.99, frac))

# -----------------------------
# Sidebar — Global Inputs
# -----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Select mode", ["Backtest", "Dry-Run (paper)"], index=0)

    st.header("Data")
    symbol = st.text_input("Symbol (Kraken/ccxt)", value="BTC/EUR")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
    since_str = st.text_input("Since (UTC ISO8601)", value="2022-01-01T00:00:00Z")
    ex = ccxt.kraken()
    try:
        since_ms = ex.parse8601(since_str)
    except Exception:
        since_ms = None
        st.error("Invalid 'Since' string (use e.g. 2023-01-01T00:00:00Z)")

    st.header("Strategy")
    ema_period = st.number_input("EMA period", 1, 200, 9)
    slope_lookback = st.number_input("Slope lookback (bars)", 1, 200, 5)
    min_slope_percent = st.number_input("Min slope %", 0.0, 100.0, 0.1, step=0.1)
    stop_loss_percent = st.number_input("Stop loss %", 0.0, 50.0, 2.0, step=0.1)
    trail_percent = st.number_input("Trailing stop %", 0.0, 50.0, 1.5, step=0.1)
    allow_shorts = st.checkbox("Allow shorts", value=True)
    ignore_trend = st.checkbox("Ignore trend filter (debug)", value=False)

    st.header("Fees & Risk")
    fee_percent = st.number_input("Fee % per fill (Kraken spot ~0.26%)", 0.0, 2.0, 0.26, step=0.01)
    slippage_percent = st.number_input("Slippage % per fill", 0.0, 2.0, 0.05, step=0.01)
    max_leverage = st.number_input("Max leverage (paper only)", 1.0, 10.0, 5.0, step=0.5)
    risk_fraction = st.slider("Risk fraction of equity per trade", 0.05, 1.0, 1.0, 0.05)

# -----------------------------
# Fetch data
# -----------------------------
if since_ms is None:
    st.stop()

with st.spinner("Fetching data from Kraken..."):
    df = fetch_ohlcv(symbol, timeframe, since_ms)
if df.empty:
    st.warning("No data fetched. Try an earlier 'Since' or a different timeframe.")
    st.stop()

if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

st.subheader("Data Preview")
st.dataframe(df.tail(10))

# -----------------------------
# Strategy core (shared logic)
# -----------------------------
def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out['EMA'] = ema(out['Close'], int(ema_period))
    out['SlopePct'] = slope_pct(out['EMA'], int(slope_lookback)).fillna(0.0)
    out['trend_up'] = out['SlopePct'] > float(min_slope_percent)
    out['trend_dn'] = out['SlopePct'] < -float(min_slope_percent)
    out['prev_close'] = out['Close'].shift(1)
    out['prev_ema'] = out['EMA'].shift(1)
    out['bull_cross'] = (out['prev_close'] <= out['prev_ema']) & (out['Close'] > out['EMA'])
    out['bear_cross'] = (out['prev_close'] >= out['prev_ema']) & (out['Close'] < out['EMA'])
    return out

df_sig = generate_signals(df)

with st.expander("🔎 Signal diagnostics"):
    st.write({
        'bars': int(len(df_sig)),
        'bull_cross_count': int(df_sig['bull_cross'].sum()),
        'bear_cross_count': int(df_sig['bear_cross'].sum()),
        'trend_up_count': int(df_sig['trend_up'].sum()),
        'trend_dn_count': int(df_sig['trend_dn'].sum()),
        'long_cond_count': int((df_sig['bull_cross'] & df_sig['trend_up']).sum()),
        'short_cond_count': int((df_sig['bear_cross'] & df_sig['trend_dn']).sum()),
    })

# -----------------------------
# Backtest mode
# -----------------------------
if mode == "Backtest":
    st.subheader("Backtest")
    initial_cash = st.number_input("Initial cash", 100.0, 1_000_000.0, 10_000.0, step=100.0)
    show_trades_on_chart = st.checkbox("Show trades on chart", value=True)
    run_bt = st.button("Run Backtest")

    class Strat(Strategy):
        def init(self):
            self.current_sl = None  # we track the SL ourselves
            self.last_entry_ts = None

        def next(self):
            ts = self.data.index[-1]
            close = self.data.Close[-1]
            row = df_sig.loc[ts]

            # Warm-up guard
            min_bars = max(int(ema_period) + 1, int(slope_lookback) + 1)
            if len(self.data) < min_bars or pd.isna(row['prev_close']) or pd.isna(row['prev_ema']):
                return

            # If flat, clear trailing tracker
            if not self.position:
                self.current_sl = None

            # Trailing exit without Position.set_sl / .sl
            if self.position:
                start_ts = self.last_entry_ts if self.last_entry_ts is not None else self.data.index[-1]
                if self.position.is_long:
                    recent_high = df.loc[start_ts: ts]['High'].max()
                    trail_level = recent_high * (1 - float(trail_percent)/100.0)
                    if self.current_sl is None or trail_level > self.current_sl:
                        self.current_sl = trail_level
                    if self.data.Low[-1] <= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return
                else:
                    recent_low = df.loc[start_ts: ts]['Low'].min()
                    trail_level = recent_low * (1 + float(trail_percent)/100.0)
                    if self.current_sl is None or trail_level < self.current_sl:
                        self.current_sl = trail_level
                    if self.data.High[-1] >= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return

            go_long  = row['bull_cross'] and (bool(ignore_trend) or row['trend_up'])
            go_short = bool(allow_shorts) and row['bear_cross'] and (bool(ignore_trend) or row['trend_dn'])

            # Backtesting.py wants fraction of equity in (0,1)
            size = bt_fraction_size(float(risk_fraction), float(max_leverage))

            if not self.position:
                if go_long:
                    entry = apply_slip(close, True, float(slippage_percent))
                    sl = entry * (1 - float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.buy(size=size, sl=sl)
                elif go_short:
                    entry = apply_slip(close, False, float(slippage_percent))
                    sl = entry * (1 + float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.sell(size=size, sl=sl)

    if run_bt:
        bt = Backtest(
            df[['Open','High','Low','Close','Volume']], Strat,
            cash=float(initial_cash),
            commission=(float(fee_percent)+float(slippage_percent))/100.0,
            exclusive_orders=False
        )
        stats = bt.run()
        st.write(stats.to_frame().style.format(precision=5))

        # Plotly chart
        ema_series = ema(df['Close'], int(ema_period))
        fig = go.Figure(data=[
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")
        ])
        fig.add_trace(go.Scatter(x=df.index, y=ema_series, name=f"EMA({int(ema_period)})"))
        if show_trades_on_chart and hasattr(stats, "_trades") and isinstance(stats._trades, pd.DataFrame) and not stats._trades.empty:
            tdf = stats._trades.copy()
            if {"EntryTime","EntryPrice"}.issubset(tdf.columns):
                fig.add_trace(go.Scatter(x=tdf["EntryTime"], y=tdf["EntryPrice"],
                                         mode="markers", name="Entries",
                                         marker=dict(size=9, symbol="triangle-up")))
            if {"ExitTime","ExitPrice"}.issubset(tdf.columns):
                fig.add_trace(go.Scatter(x=tdf["ExitTime"], y=tdf["ExitPrice"],
                                         mode="markers", name="Exits",
                                         marker=dict(size=9, symbol="x")))
        fig.update_layout(height=800, xaxis_title="Time", yaxis_title=symbol)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Dry-Run (paper) mode
# -----------------------------
else:
    st.subheader("Dry-Run (Paper Trading)")
    if "paper_state" not in st.session_state:
        st.session_state.paper_state = {
            "equity": 10_000.0,
            "position": None,
            "trades": [],
        }

    equity = st.number_input("Paper initial equity", 100.0, 1_000_000.0, float(st.session_state.paper_state["equity"]), step=100.0)
    apply_equity = st.button("Apply Equity")
    if apply_equity:
        st.session_state.paper_state["equity"] = equity

    def process_bar(ts: pd.Timestamp):
        row = df_sig.loc[ts]
        close = df.loc[ts]['Close']
        high = df.loc[ts]['High']
        low  = df.loc[ts]['Low']
        state = st.session_state.paper_state

        # Update trailing (paper)
        if state["position"]:
            pos = state["position"]
            if pos["side"] == "long":
                recent_high = df.loc[pos["entry_time"]: ts]['High'].max()
                new_sl = recent_high * (1 - float(trail_percent)/100.0)
                if pos.get("sl") is None or new_sl > pos["sl"]:
                    pos["sl"] = new_sl
                if low <= pos["sl"]:
                    exit_px = pos["sl"]
                    notional = exit_px * pos["size"]
                    fee = notional * (float(fee_percent)/100.0)
                    pnl = (exit_px - pos["entry"]) * pos["size"] - fee
                    state["equity"] += pnl
                    state["trades"].append({"time": ts, "side": "sell", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "trail/SL"})
                    state["position"] = None
            else:
                recent_low = df.loc[pos["entry_time"]: ts]['Low'].min()
                new_sl = recent_low * (1 + float(trail_percent)/100.0)
                if pos.get("sl") is None or new_sl < pos["sl"]:
                    pos["sl"] = new_sl
                if high >= pos["sl"]:
                    exit_px = pos["sl"]
                    notional = exit_px * pos["size"]
                    fee = notional * (float(fee_percent)/100.0)
                    pnl = (pos["entry"] - exit_px) * pos["size"] - fee
                    state["equity"] += pnl
                    state["trades"].append({"time": ts, "side": "buy", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "trail/SL"})
                    state["position"] = None

        # Entries if flat
        if st.session_state.paper_state["position"] is None:
            size = calc_size(state["equity"], close, float(risk_fraction), float(max_leverage))
            if size > 0:
                if (row["bull_cross"] and (bool(ignore_trend) or row["trend_up"])):
                    px = apply_slip(close, True, float(slippage_percent))
                    notional = px * size
                    fee = notional * (float(fee_percent)/100.0)
                    state["equity"] -= fee
                    sl = px * (1 - float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    state["position"] = {"side":"long","size":size,"entry":px,"sl":sl,"entry_time":ts}
                    state["trades"].append({"time": ts, "side": "buy", "price": px, "size": size, "pnl": 0.0, "reason": "entry"})
                elif bool(allow_shorts) and (row["bear_cross"] and (bool(ignore_trend) or row["trend_dn"])):
                    px = apply_slip(close, False, float(slippage_percent))
                    notional = px * size
                    fee = notional * (float(fee_percent)/100.0)
                    state["equity"] -= fee
                    sl = px * (1 + float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    state["position"] = {"side":"short","size":size,"entry":px,"sl":sl,"entry_time":ts}
                    state["trades"].append({"time": ts, "side": "sell", "price": px, "size": size, "pnl": 0.0, "reason": "entry"})

        # Hard SL intrabar (paper)
        if st.session_state.paper_state["position"]:
            pos = st.session_state.paper_state["position"]
            if pos["side"] == "long" and low <= pos["sl"]:
                exit_px = pos["sl"]
                fee = exit_px * pos["size"] * (float(fee_percent)/100.0)
                pnl = (exit_px - pos["entry"]) * pos["size"] - fee
                st.session_state.paper_state["equity"] += pnl
                st.session_state.paper_state["trades"].append({"time": ts, "side": "sell", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "SL"})
                st.session_state.paper_state["position"] = None
            elif pos["side"] == "short" and high >= pos["sl"]:
                exit_px = pos["sl"]
                fee = exit_px * pos["size"] * (float(fee_percent)/100.0)
                pnl = (pos["entry"] - exit_px) * pos["size"] - fee
                st.session_state.paper_state["equity"] += pnl
                st.session_state.paper_state["trades"].append({"time": ts, "side": "buy", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "SL"})
                st.session_state.paper_state["position"] = None

    col1, col2 = st.columns([1,1])
    with col1:
        step_btn = st.button("Step 1 latest bar")
    with col2:
        reset_btn = st.button("Reset paper state")

    if reset_btn:
        st.session_state.paper_state = {"equity": float(equity), "position": None, "trades": []}

    if step_btn:
        ts = df_sig.index[-1]
        process_bar(ts)

    st.markdown("### Portfolio")
    st.write(f"**Equity:** {st.session_state.paper_state['equity']:.2f}")
    pos = st.session_state.paper_state["position"]
    if pos:
        st.write(f"**Position:** {pos['side']} | size {pos['size']:.6f} | entry {pos['entry']:.2f} | SL {pos.get('sl')}")

    if st.session_state.paper_state["trades"]:
        trades_df = pd.DataFrame(st.session_state.paper_state["trades"])
        st.markdown("### Trades")
        st.dataframe(trades_df)
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades CSV", data=csv, file_name="paper_trades.csv", mime="text/csv")
    else:
        st.info("No trades yet.")

st.caption("Built for Sani • Backtest & Dry-Run with manual trailing exits (no Position.set_sl).")
