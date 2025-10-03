
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

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode & Engine")
    mode = st.radio("Mode", ["Backtest", "Dry-Run (paper)"], index=0)
    engine = st.selectbox("Backtest engine", ["Simple (backtesting.py)", "TRUE STOP (backtrader)"], index=0)

    st.header("Data")
    symbol = st.text_input("Symbol (Kraken/ccxt)", value="BTC/EUR")
    timeframe = st.selectbox("Timeframe", ["1h","4h","1d"], index=0)
    since_str = st.text_input("Since (UTC ISO8601)", value="2022-01-01T00:00:00Z")
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
    risk_fraction = st.slider("Risk fraction of equity", 0.01, 1.0, 0.25, 0.01)

    st.header("Sizing (Simple engine)")
    sizing_mode = st.selectbox("Sizing mode", ["Whole units (int)", "Fraction of equity (0-1)"], index=0)
    contract_size = st.number_input("Contract size (BTC per unit)", 0.000001, 10.0, 0.001, step=0.0001, format="%.6f")

    st.header("Preview")
    preview_rows = st.slider("Preview last N rows", 10, 1000, 50, step=10)

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
df_sig = signals(df, int(ema_period), int(slope_lookback), float(min_slope_percent))

with st.expander("🔎 Signal diagnostics"):
    longc = int((df_sig['bull_cross'] & (df_sig['trend_up'] | ignore_trend)).sum())
    shortc = int((df_sig['bear_cross'] & (df_sig['trend_dn'] | ignore_trend)).sum())
    st.write({"bars": int(len(df)), "long_cond_count": longc, "short_cond_count": shortc})

# -----------------------------
# Backtest engines
# -----------------------------
if mode == "Backtest":
    st.subheader("Backtest")
    initial_cash = st.number_input("Initial cash", 100.0, 1_000_000.0, 10_000.0, step=100.0)
    run_bt = st.button("Run Backtest")
    show_trades = st.checkbox("Show trades on chart", value=True)

    if run_bt and engine == "Simple (backtesting.py)":
        from backtesting import Backtest, Strategy

        # Build a scaled DataFrame for the Simple engine (synthetic contracts)
        df_bt = df[['Open','High','Low','Close','Volume']].copy()
        df_bt[['Open','High','Low','Close']] = df_bt[['Open','High','Low','Close']] * float(contract_size)

        class Strat(Strategy):
            def init(self):
                self.current_sl = None
                self.last_entry_ts = None
            def next(self):
                ts = self.data.index[-1]
                close = self.data.Close[-1]  # NOTE: scaled price
                # Align to original signals (df_sig uses original prices)
                row = df_sig.loc[ts]
                min_bars = max(int(ema_period) + 1, int(slope_lookback) + 1)
                if len(self.data) < min_bars or pd.isna(row['prev_close']) or pd.isna(row['prev_ema']):
                    return
                if not self.position:
                    self.current_sl = None
                if self.position:
                    start_ts = self.last_entry_ts or self.data.index[-1]
                    if self.position.is_long:
                        recent_high = df.loc[start_ts: ts]['High'].max() * float(contract_size)  # scale for SL compare
                        trail = recent_high * (1 - float(trail_percent)/100.0)
                        if self.current_sl is None or trail > self.current_sl:
                            self.current_sl = trail
                        if self.data.Low[-1] <= self.current_sl:
                            self.position.close(); self.current_sl=None; return
                    else:
                        recent_low = df.loc[start_ts: ts]['Low'].min() * float(contract_size)
                        trail = recent_low * (1 + float(trail_percent)/100.0)
                        if self.current_sl is None or trail < self.current_sl:
                            self.current_sl = trail
                        if self.data.High[-1] >= self.current_sl:
                            self.position.close(); self.current_sl=None; return
                go_long  = row['bull_cross'] and (bool(ignore_trend) or row['trend_up'])
                go_short = bool(allow_shorts) and row['bear_cross'] and (bool(ignore_trend) or row['trend_dn'])

                if sizing_mode.startswith("Whole"):
                    cash = self.equity
                    notional = cash * float(risk_fraction)
                    units = max(1, int(notional / max(1e-9, close)))  # integer *contracts*
                    size = units
                else:
                    size = max(0.001, min(0.99, float(risk_fraction)))  # fraction of equity path still available

                if not self.position:
                    if go_long:
                        sl_unscaled = df.loc[ts]['Close'] * (1 - float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                        sl = sl_unscaled * float(contract_size) if sl_unscaled is not None else None
                        self.last_entry_ts = ts; self.current_sl = sl; self.buy(size=size, sl=sl)
                    elif go_short:
                        sl_unscaled = df.loc[ts]['Close'] * (1 + float(stop_loss_percent)/100.0) if float(stop_loss_percent)>0 else None
                        sl = sl_unscaled * float(contract_size) if sl_unscaled is not None else None
                        self.last_entry_ts = ts; self.current_sl = sl; self.sell(size=size, sl=sl)

        bt = Backtest(df_bt, Strat,
                      cash=float(initial_cash),
                      commission=(float(fee_percent)+float(slippage_percent))/100.0,
                      exclusive_orders=False)
        stats = bt.run()
        st.write(stats.to_frame().style.format(precision=5))
        st.caption(f"Note: In Simple engine, 1 unit = {contract_size} BTC (synthetic contract). Prices were scaled by this factor only inside the engine.")

        # Trades table
        trades_df = getattr(stats, "_trades", pd.DataFrame())
        trades = []
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and {"EntryTime","EntryPrice","ExitTime","ExitPrice","Size"}.issubset(trades_df.columns):
            trades_df = trades_df.copy()
            trades_df["Direction"] = np.where(trades_df["Size"] > 0, "Long", "Short")
            trades_df["Duration"] = pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"])
            denom = (trades_df["EntryPrice"].abs() * trades_df["Size"].abs()).replace(0, np.nan)
            trades_df["ReturnPct_est"] = trades_df["PnL"] / denom * 100.0
            for _, r in trades_df.iterrows():
                trades.append({"time": r["EntryTime"], "price": r["EntryPrice"] / float(contract_size), "etype":"entry"})
                trades.append({"time": r["ExitTime"], "price": r["ExitPrice"] / float(contract_size), "etype":"exit"})
            st.markdown("### Trades")
            st.dataframe(trades_df)

        # Plot against ORIGINAL price chart
        fig = plot_chart(df, int(ema_period), trades if show_trades else None, symbol)
        st.plotly_chart(fig, use_container_width=True)

    if run_bt and engine == "TRUE STOP (backtrader)":
        import backtrader as bt

        class CashSizer(bt.Sizer):
            params = (('risk_frac', 0.25), ('min_unit', 0.0001), ('max_lev', 5.0), ('retint', False))
            def _getsizing(self, comminfo, cash, data, isbuy):
                price = data.close[0]
                notional = cash * self.p.risk_frac * min(self.p.max_lev, 5.0)
                units = notional / max(price, 1e-9)
                # allow fractional, snap to min_unit
                snapped = max(self.p.min_unit, round(units / self.p.min_unit) * self.p.min_unit)
                return snapped if not self.p.retint else int(snapped)

        class BTStrategy(bt.Strategy):
            params = dict(
                ema_period=int(ema_period),
                slope_lb=int(slope_lookback),
                min_slope=float(min_slope_percent)/100.0,
                stop_loss=float(stop_loss_percent)/100.0,
                trail=float(trail_percent)/100.0,
                allow_shorts=bool(allow_shorts),
                ignore_trend=bool(ignore_trend),
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
        cerebro.broker.setcommission(commission=(float(fee_percent)+float(slippage_percent))/100.0)
        cerebro.broker.set_slippage_perc(perc=float(slippage_percent)/100.0)
        cerebro.broker.setcash(float(initial_cash))

        comp = 60 if timeframe=='1h' else (240 if timeframe=='4h' else 1440)
        data_bt = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=comp)
        cerebro.adddata(data_bt)
        cerebro.addsizer(CashSizer, risk_frac=float(risk_fraction), min_unit=0.0001)
        cerebro.addstrategy(BTStrategy)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')


        class TradeLogger(bt.Analyzer):
            def __init__(self):
                self.rows = []

            def notify_trade(self, trade):
                if not trade.isclosed:
                    return

                # Stable fallbacks
                entry_dt = bt.num2date(getattr(trade, 'dtopen', self.strategy.datas[0].datetime[0]))
                exit_dt = bt.num2date(getattr(trade, 'dtclose', self.strategy.datas[0].datetime[0]))
                entry_px = float(getattr(trade, 'price', float('nan')))  # avg entry
                exit_px = float('nan')
                dirn = None
                size_abs = abs(float(getattr(trade, 'size', 0.0)))

                # Try history if present
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

        import pandas as pd

        trades_df = pd.DataFrame(trade_rows)
        st.markdown("### Trades (TRUE STOP)")
        if trades_df.empty:
            # surface useful counts so you can tell if signals fired
            total_closed = ta.get('total', {}).get('closed', 0) if isinstance(ta, dict) else 0
            st.info(f"No closed trades. Closed={total_closed}. "
                    f"Try earlier 'Since', lower min slope, or check min unit/risk.")
        else:
            # optional: Duration/Return%
            if {"EntryTime", "ExitTime"}.issubset(trades_df.columns):
                trades_df["Duration"] = pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"])
            if {"EntryPrice", "ExitPrice", "Size"}.issubset(trades_df.columns):
                denom = (trades_df["EntryPrice"].abs() * trades_df["Size"].abs()).replace(0, np.nan)
                trades_df["ReturnPct_est"] = trades_df["PnL"] / denom * 100.0
            st.dataframe(trades_df)

        fig = plot_chart(df, int(ema_period), None, symbol)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Paper mode (already fractional)
# -----------------------------
else:
    st.subheader("Dry-Run (Paper)")
    st.info("Paper trading uses fractional units already based on notional/price.")
