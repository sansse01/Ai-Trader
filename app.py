
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# -----------------------------
# Sidebar — Global Inputs
# -----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Select mode", ["Backtest", "Dry-Run (paper)"], index=0)

    st.header("Data")
    symbol = st.text_input("Symbol (Kraken/ccxt)", value="BTC/EUR")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
    since_str = st.text_input("Since (UTC ISO8601)", value="2023-01-01T00:00:00Z")
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

    st.header("Fees & Risk")
    fee_percent = st.number_input("Fee % per fill (Kraken spot ~0.26%)", 0.0, 2.0, 0.26, step=0.01)
    slippage_percent = st.number_input("Slippage % per fill", 0.0, 2.0, 0.05, step=0.01)
    max_leverage = st.number_input("Max leverage", 1.0, 10.0, 5.0, step=0.5)
    risk_fraction = st.slider("Risk fraction of equity per trade", 0.05, 1.0, 1.0, 0.05)

# -----------------------------
# Fetch data
# -----------------------------
if since_ms is None:
    st.stop()

with st.spinner("Fetching data from Kraken..."):
    df = fetch_ohlcv(symbol, timeframe, since_ms)
if df.empty:
    st.warning("No data fetched. Try a different start date or timeframe.")
    st.stop()

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
    # Cross conditions use prev close vs prev EMA, and current close vs current EMA
    out['prev_close'] = out['Close'].shift(1)
    out['prev_ema'] = out['EMA'].shift(1)
    out['bull_cross'] = (out['prev_close'] < out['prev_ema']) & (out['Close'] > out['EMA'])
    out['bear_cross'] = (out['prev_close'] > out['prev_ema']) & (out['Close'] < out['EMA'])
    return out

df_sig = generate_signals(df)

def apply_slip(price: float, is_buy: bool) -> float:
    s = slippage_percent / 100.0
    return price * (1 + s) if is_buy else price * (1 - s)

def fill_fee(notional: float) -> float:
    return notional * (fee_percent / 100.0)

# -----------------------------
# Backtest mode
# -----------------------------
if mode == "Backtest":
    st.subheader("Backtest")
    initial_cash = st.number_input("Initial cash", 100.0, 1_000_000.0, 10_000.0, step=100.0)
    run_bt = st.button("Run Backtest")

    class Strat(Strategy):
        def init(self):
            pass
        def next(self):
            i = len(self.data) - 1
            close = self.data.Close[-1]
            # derive EMA & signals from df_sig by index alignment
            ts = self.data.index[-1]
            row = df_sig.loc[ts]
            if pd.isna(row['prev_close']) or pd.isna(row['prev_ema']):
                return

            # trailing: backtesting.py supports sl via order args; we'll maintain via set_sl on Position
            if self.position:
                if self.position.is_long:
                    recent_high = df.loc[getattr(self, 'last_entry_ts', df.index[-1]): ts]['High'].max()
                    new_sl = recent_high * (1 - trail_percent/100.0)
                    if self.position.sl is None or new_sl > self.position.sl:
                        self.position.set_sl(new_sl)
                else:
                    recent_low = df.loc[getattr(self, 'last_entry_ts', df.index[-1]): ts]['Low'].min()
                    new_sl = recent_low * (1 + trail_percent/100.0)
                    if self.position.sl is None or new_sl < self.position.sl:
                        self.position.set_sl(new_sl)

            if not self.position:
                size = calc_size(self.equity, close, float(risk_fraction), float(max_leverage))
                if size <= 0:
                    return
                if row['bull_cross'] and row['trend_up']:
                    entry = apply_slip(close, True)
                    fee = fill_fee(entry * size)
                    sl = entry * (1 - float(stop_loss_percent)/100.0) if stop_loss_percent>0 else None
                    self.last_entry_ts = ts
                    self.buy(size=size, sl=sl)
                elif allow_shorts and row['bear_cross'] and row['trend_dn']:
                    entry = apply_slip(close, False)
                    fee = fill_fee(entry * size)
                    sl = entry * (1 + float(stop_loss_percent)/100.0) if stop_loss_percent>0 else None
                    self.last_entry_ts = ts
                    self.sell(size=size, sl=sl)

    if run_bt:
        bt = Backtest(df[['Open','High','Low','Close','Volume']], Strat,
                      cash=float(initial_cash), commission=(float(fee_percent)+float(slippage_percent))/100.0, exclusive_orders=False)
        stats = bt.run()
        st.write(stats.to_frame().style.format(precision=5))
        html = bt.plot(open_browser=False)
        st.components.v1.html(html, height=900, scrolling=True)

# -----------------------------
# Dry-Run (paper) mode
# -----------------------------
else:
    st.subheader("Dry-Run (Paper Trading)")
    # Session state for portfolio
    if "paper_state" not in st.session_state:
        st.session_state.paper_state = {
            "equity": 10_000.0,
            "position": None,     # dict with {side, size, entry, sl, entry_time}
            "trades": [],         # list of dicts
        }

    equity = st.number_input("Paper initial equity", 100.0, 1_000_000.0, float(st.session_state.paper_state["equity"]), step=100.0)
    apply_equity = st.button("Apply Equity")
    if apply_equity:
        st.session_state.paper_state["equity"] = equity

    def render_position(pos):
        if not pos:
            st.info("No open position.")
            return
        st.write(f"**Side**: {pos['side']}  |  **Size**: {pos['size']:.6f}  |  **Entry**: {pos['entry']:.2f}  |  **SL**: {pos.get('sl')}  |  **Since**: {pos['entry_time']}")

    # Controls
    step_btn = st.button("Step 1 bar (latest)")
    auto = st.checkbox("Auto-refresh every N seconds")
    interval = st.number_input("Refresh seconds", 2, 120, 10)
    if auto:
        st.caption("Streamlit will refresh after the interval via the browser's rerun.")

    # One step evaluation
    def process_bar(ts: pd.Timestamp):
        row = df_sig.loc[ts]
        close = df.loc[ts]['Close']
        high = df.loc[ts]['High']
        low  = df.loc[ts]['Low']
        state = st.session_state.paper_state

        # Update trailing on open position
        if state["position"]:
            pos = state["position"]
            if pos["side"] == "long":
                recent_high = df.loc[pos["entry_time"]: ts]['High'].max()
                new_sl = recent_high * (1 - float(trail_percent)/100.0)
                if pos.get("sl") is None or new_sl > pos["sl"]:
                    pos["sl"] = new_sl
                # Stop-out check
                if low <= pos["sl"]:
                    exit_px = pos["sl"]
                    notional = exit_px * pos["size"]
                    fee = fill_fee(notional)
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
                    fee = fill_fee(notional)
                    pnl = (pos["entry"] - exit_px) * pos["size"] - fee
                    state["equity"] += pnl
                    state["trades"].append({"time": ts, "side": "buy", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "trail/SL"})
                    state["position"] = None

        # Entries if flat
        if st.session_state.paper_state["position"] is None:
            size = calc_size(state["equity"], close, float(risk_fraction), float(max_leverage))
            if size > 0:
                if row["bull_cross"] and row["trend_up"]:
                    px = apply_slip(close, True)
                    notional = px * size
                    fee = fill_fee(notional)
                    state["equity"] -= fee
                    sl = px * (1 - float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    state["position"] = {"side":"long","size":size,"entry":px,"sl":sl,"entry_time":ts}
                    state["trades"].append({"time": ts, "side": "buy", "price": px, "size": size, "pnl": 0.0, "reason": "entry"})
                elif bool(allow_shorts) and row["bear_cross"] and row["trend_dn"]:
                    px = apply_slip(close, False)
                    notional = px * size
                    fee = fill_fee(notional)
                    state["equity"] -= fee
                    sl = px * (1 + float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                    state["position"] = {"side":"short","size":size,"entry":px,"sl":sl,"entry_time":ts}
                    state["trades"].append({"time": ts, "side": "sell", "price": px, "size": size, "pnl": 0.0, "reason": "entry"})

        # Hard stop-loss if still in position and SL breached intrabar
        if st.session_state.paper_state["position"]:
            pos = st.session_state.paper_state["position"]
            if pos["side"] == "long" and low <= pos["sl"]:
                exit_px = pos["sl"]
                fee = fill_fee(exit_px * pos["size"])
                pnl = (exit_px - pos["entry"]) * pos["size"] - fee
                st.session_state.paper_state["equity"] += pnl
                st.session_state.paper_state["trades"].append({"time": ts, "side": "sell", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "SL"})
                st.session_state.paper_state["position"] = None
            elif pos["side"] == "short" and high >= pos["sl"]:
                exit_px = pos["sl"]
                fee = fill_fee(exit_px * pos["size"])
                pnl = (pos["entry"] - exit_px) * pos["size"] - fee
                st.session_state.paper_state["equity"] += pnl
                st.session_state.paper_state["trades"].append({"time": ts, "side": "buy", "price": exit_px, "size": pos["size"], "pnl": pnl, "reason": "SL"})
                st.session_state.paper_state["position"] = None

    # Run one step or auto
    if step_btn:
        ts = df_sig.index[-1]
        process_bar(ts)

    if auto:
        ts = df_sig.index[-1]
        process_bar(ts)
        time.sleep(int(interval))
        st.experimental_rerun()

    # Render portfolio state
    st.markdown("### Portfolio")
    st.write(f"**Equity:** {st.session_state.paper_state['equity']:.2f}")
    render_position(st.session_state.paper_state["position"])

    # Trades table
    if st.session_state.paper_state["trades"]:
        trades_df = pd.DataFrame(st.session_state.paper_state["trades"])
        st.markdown("### Trades")
        st.dataframe(trades_df)
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades CSV", data=csv, file_name="paper_trades.csv", mime="text/csv")
    else:
        st.info("No trades yet.")

st.caption("Built for Sani • Backtest & Dry-Run with configurable fees, slippage, leverage cap, and UI controls.")
