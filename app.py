
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import ccxt

st.set_page_config(page_title="EMA9 Trend Trader — Trade Analysis", layout="wide")
st.title("📈 EMA9 Trend Trader — Kraken (Trade Analysis)")

# -----------------------------
# Data helpers
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
        if len(ohlcv) < 2 or len(all_rows) > 30000:
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

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("configuration")
    preview_rows = st.slider("Preview last N rows", 0, 100, 1, step=5)

    st.header("Mode & Engine")
    mode = st.radio("Mode", ["Backtest", "Dry-Run (paper)"], index=0)
    engine = st.selectbox("Backtest engine", ["Simple (backtesting.py)", "TRUE STOP (backtrader)"], index=0)

    st.header("Data")
    symbol = st.text_input("Symbol (Kraken/ccxt)", value="BTC/EUR")
    timeframe = st.selectbox("Timeframe", ["1h","4h","1d"], index=0)
    since_str = st.text_input("Since (UTC ISO8601)", value="2021-01-01T00:00:00Z")
    ex = ccxt.kraken()
    try:
        since_ms = ex.parse8601(since_str)
    except Exception:
        since_ms = None
        st.error("Invalid 'Since' string (e.g. 2023-01-01T00:00:00Z)")

    st.header("Strategy")
    ema_period = st.number_input("EMA period", 1, 200, 9)
    slope_lookback = st.number_input("Slope lookback (bars)", 1, 200, 5)
    min_slope_percent = st.number_input("Min slope %", 0.0, 100.0, 0.1, step=0.1)
    stop_loss_percent = st.number_input("Stop loss %", 0.0, 50.0, 2.0, step=0.1)
    trail_percent = st.number_input("Trailing stop %", 0.0, 50.0, 1.5, step=0.1)
    allow_shorts = st.checkbox("Allow shorts", value=True)
    ignore_trend = st.checkbox("Ignore trend filter (debug)", value=False)

    st.header("Fees & Risk")
    fee_percent = st.number_input("Fee % per fill", 0.0, 2.0, 0.26, step=0.01)
    slippage_percent = st.number_input("Slippage % per fill", 0.0, 2.0, 0.05, step=0.01)
    max_leverage = st.number_input("Max leverage", 1.0, 10.0, 5.0, step=0.5)
    risk_fraction = st.slider("Risk fraction of equity", 0.05, 1.0, 1.0, 0.05)


if since_ms is None:
    st.stop()

with st.spinner("Fetching data..."):
    df = fetch_ohlcv(symbol, timeframe, since_ms)
if df.empty:
    st.warning("No data fetched; try earlier 'Since' / different timeframe.")
    st.stop()
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

df_sig = signals(df, int(ema_period), int(slope_lookback), float(min_slope_percent))

with st.expander("🔎 Signal diagnostics"):
    longc = int((df_sig['bull_cross'] & (df_sig['trend_up'] | ignore_trend)).sum())
    shortc = int((df_sig['bear_cross'] & (df_sig['trend_dn'] | ignore_trend)).sum())
    st.write({"bars": int(len(df)), "long_cond_count": longc, "short_cond_count": shortc})

# -----------------------------
# Backtest engines
# -----------------------------
if mode == "Backtest":
    st.subheader("Data Preview")

    # Meta
    st.caption(f"Bars: {len(df):,}  |  Range: {df.index.min()} → {df.index.max()}  |  Timezone: {df.index.tz}")

    # Tail preview
    st.dataframe(df.tail(preview_rows))

    # Optional: show first few too (helps catch “only 1 row” UI confusion)
    with st.expander("Show first rows"):
        st.dataframe(df.head(preview_rows))

    # Download CSV
    csv = df.to_csv().encode("utf-8")
    st.download_button("Download OHLCV CSV", data=csv, file_name=f"{symbol.replace('/', '_')}_{timeframe}.csv",
                       mime="text/csv")

    st.subheader("Backtest")
    initial_cash = st.number_input("Initial cash", 100.0, 1_000_000.0, 10_000.0, step=100.0)
    run_bt = st.button("Run Backtest")
    show_trades = st.checkbox("Show trades on chart", value=True)

    if run_bt and engine == "Simple (backtesting.py)":
        from backtesting import Backtest, Strategy

        def bt_fraction_size(risk_fraction: float, max_leverage: float) -> float:
            frac = float(risk_fraction) * min(1.0, float(max_leverage))
            return max(0.01, min(0.99, frac))

        class Strat(Strategy):
            def init(self):
                self.current_sl = None
                self.last_entry_ts = None
            def next(self):
                ts = self.data.index[-1]
                close = self.data.Close[-1]
                row = df_sig.loc[ts]
                min_bars = max(int(ema_period) + 1, int(slope_lookback) + 1)
                if len(self.data) < min_bars or pd.isna(row['prev_close']) or pd.isna(row['prev_ema']):
                    return
                if not self.position:
                    self.current_sl = None
                if self.position:
                    start_ts = self.last_entry_ts or self.data.index[-1]
                    if self.position.is_long:
                        recent_high = df.loc[start_ts: ts]['High'].max()
                        trail = recent_high * (1 - float(trail_percent)/100.0)
                        if self.current_sl is None or trail > self.current_sl:
                            self.current_sl = trail
                        if self.data.Low[-1] <= self.current_sl:
                            self.position.close(); self.current_sl=None; return
                    else:
                        recent_low = df.loc[start_ts: ts]['Low'].min()
                        trail = recent_low * (1 + float(trail_percent)/100.0)
                        if self.current_sl is None or trail < self.current_sl:
                            self.current_sl = trail
                        if self.data.High[-1] >= self.current_sl:
                            self.position.close(); self.current_sl=None; return
                go_long  = row['bull_cross'] and (bool(ignore_trend) or row['trend_up'])
                go_short = bool(allow_shorts) and row['bear_cross'] and (bool(ignore_trend) or row['trend_dn'])
                size = bt_fraction_size(float(risk_fraction), float(max_leverage))
                if not self.position:
                    if go_long:
                        sl = close * (1 - float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                        self.last_entry_ts = ts; self.current_sl = sl; self.buy(size=size, sl=sl)
                    elif go_short:
                        sl = close * (1 + float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                        self.last_entry_ts = ts; self.current_sl = sl; self.sell(size=size, sl=sl)

        bt = Backtest(df[['Open','High','Low','Close','Volume']], Strat,
                      cash=float(initial_cash),
                      commission=(float(fee_percent)+float(slippage_percent))/100.0,
                      exclusive_orders=False)
        stats = bt.run()
        st.write(stats.to_frame().style.format(precision=5))

        # ---- Trade analysis table (backtesting.py)
        trades_df = pd.DataFrame()
        if hasattr(stats, "_trades") and isinstance(stats._trades, pd.DataFrame) and not stats._trades.empty:
            trades_df = stats._trades.copy()
            # Direction from Size
            if "Size" in trades_df.columns:
                trades_df["Direction"] = np.where(trades_df["Size"] > 0, "Long", "Short")
            # Duration
            if {"EntryTime","ExitTime"}.issubset(trades_df.columns):
                trades_df["Duration"] = (pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"]))
            # Return %
            if "PnL" in trades_df.columns and "EntryPrice" in trades_df.columns:
                trades_df["ReturnPct_est"] = trades_df["PnL"] / (trades_df["EntryPrice"].abs() * trades_df["Size"].abs()) * 100.0
        if trades_df.empty:
            st.info("No completed trades to analyze.")
        else:
            st.markdown("### Trades (Simple engine)")
            st.dataframe(trades_df)
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download trades CSV", data=csv, file_name="trades_simple.csv", mime="text/csv")

        # Chart
        trades = []
        if not trades_df.empty and {"EntryTime","EntryPrice","ExitTime","ExitPrice"}.issubset(trades_df.columns):
            for _, r in trades_df.iterrows():
                trades.append({"time": r["EntryTime"], "price": r["EntryPrice"], "etype":"entry"})
                trades.append({"time": r["ExitTime"], "price": r["ExitPrice"], "etype":"exit"})
        fig = plot_chart(df, int(ema_period), trades if show_trades else None, symbol)
        st.plotly_chart(fig, use_container_width=True)

    if run_bt and engine == "TRUE STOP (backtrader)":
        import backtrader as bt

        class TradeLogger(bt.Analyzer):
            def __init__(self):
                self.trades = []
            def notify_trade(self, trade):
                if trade.isclosed:
                    dirn = "Long" if trade.history[0].event.size > 0 else "Short"
                    entry_dt = bt.num2date(trade.history[0].status.dt).replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
                    exit_dt  = bt.num2date(trade.history[-1].status.dt).replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
                    self.trades.append(dict(
                        Direction=dirn,
                        EntryTime=entry_dt,
                        ExitTime=exit_dt,
                        EntryPrice=trade.history[0].event.price,
                        ExitPrice=trade.history[-1].event.price,
                        Size=abs(trade.size),
                        PnL=trade.pnlcomm,
                        ReturnPct=np.nan  # can compute later if desired
                    ))
            def get_analysis(self):
                return self.trades

        class BTStrategy(bt.Strategy):
            params = dict(
                ema_period=int(ema_period),
                slope_lb=int(slope_lookback),
                min_slope=float(min_slope_percent)/100.0,
                stop_loss=float(stop_loss_percent)/100.0,
                trail=float(trail_percent)/100.0,
                risk_frac=float(risk_fraction),
                allow_shorts=bool(allow_shorts),
                ignore_trend=bool(ignore_trend),
                max_lev=float(max_leverage),
            )
            def __init__(self):
                self.close = self.datas[0].close
                self.high  = self.datas[0].high
                self.low   = self.datas[0].low
                self.ema   = bt.ind.EMA(self.close, period=self.p.ema_period)
                self.prev_close = self.close(-1)
                self.prev_ema   = self.ema(-1)
                self.ema_shift = self.ema - self.ema(-int(self.p.slope_lb))
                self.slope_pct = self.ema_shift / bt.If(self.ema(-int(self.p.slope_lb)) == 0, 1e-9, self.ema(-int(self.p.slope_lb)))
                self.order = None
                self.entry_size = 0
            def next(self):
                if self.order:
                    return
                bull_cross = (self.prev_close[0] <= self.prev_ema[0]) and (self.close[0] > self.ema[0])
                bear_cross = (self.prev_close[0] >= self.prev_ema[0]) and (self.close[0] < self.ema[0])
                trend_up = (self.slope_pct[0] * 100.0) > (float({min_slope_percent})) or self.p.ignore_trend
                trend_dn = (self.slope_pct[0] * 100.0) < -(float({min_slope_percent})) or self.p.ignore_trend
                if self.position:
                    return
                cash = self.broker.getcash()
                price = self.close[0]
                notional = cash * self.p.risk_frac * min(self.p.max_lev, 5.0)
                size = int(max(1, notional / price))
                if not self.position:
                    if bull_cross and trend_up:
                        self.order = self.buy(size=size)
                        self.entry_size = size
                        if self.p.stop_loss > 0:
                            sl_price = price * (1 - self.p.stop_loss)
                            self.sell(exectype=bt.Order.Stop, price=sl_price, size=size)
                        if self.p.trail > 0:
                            self.sell(exectype=bt.Order.StopTrail, trailpercent=self.p.trail, size=size)
                    elif self.p.allow_shorts and bear_cross and trend_dn:
                        self.order = self.sell(size=size)
                        self.entry_size = -size
                        if self.p.stop_loss > 0:
                            sl_price = price * (1 + self.p.stop_loss)
                            self.buy(exectype=bt.Order.Stop, price=sl_price, size=size)
                        if self.p.trail > 0:
                            self.buy(exectype=bt.Order.StopTrail, trailpercent=self.p.trail, size=size)
            def notify_order(self, order):
                if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                    self.order = None

        # Cerebro
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcommission(commission=(float(fee_percent)+float(slippage_percent))/100.0)
        cerebro.broker.set_slippage_perc(perc=float(slippage_percent)/100.0)
        cerebro.broker.setcash(float(initial_cash))

        # Feed
        comp = 60 if timeframe=='1h' else (240 if timeframe=='4h' else 1440)
        data_bt = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=comp)
        cerebro.adddata(data_bt)
        cerebro.addstrategy(BTStrategy)
        trade_an = cerebro.addanalyzer(TradeLogger, _name='trades')

        results = cerebro.run()
        analysis = results[0].analyzers.trades.get_analysis() if hasattr(results[0].analyzers, 'trades') else []

        trades_df = pd.DataFrame(analysis)
        if trades_df.empty:
            st.info("No completed trades to analyze (TRUE STOP engine).")
        else:
            st.markdown("### Trades (TRUE STOP engine)")
            # Compute Duration and Return%
            if {"EntryTime","ExitTime"}.issubset(trades_df.columns):
                trades_df["Duration"] = pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"])
            if {"EntryPrice","ExitPrice","Size"}.issubset(trades_df.columns):
                denom = (trades_df["EntryPrice"].abs() * trades_df["Size"].abs()).replace(0, np.nan)
                trades_df["ReturnPct_est"] = trades_df["PnL"] / denom * 100.0
            st.dataframe(trades_df)
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download trades CSV", data=csv, file_name="trades_true_stop.csv", mime="text/csv")

        fig = plot_chart(df, int(ema_period), None, symbol)  # markers could be added if we collect entry/exit times separately
        st.plotly_chart(fig, use_container_width=True)

elif mode == "Dry-Run (paper)":
    st.subheader("Dry-Run (Paper)")
    st.info("Use Backtest mode to view trade analysis for each engine.")
