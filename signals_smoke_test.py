import ccxt, pandas as pd

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def slope_pct(s, lb): return (s - s.shift(lb)) / s.shift(lb) * 100

ex = ccxt.kraken()
since = ex.parse8601("2024-01-01T00:00:00Z")
ohlcv = ex.fetch_ohlcv("BTC/EUR", "1h", since=since, limit=5000)
df = pd.DataFrame(ohlcv, columns=["ts","Open","High","Low","Close","Volume"])
df["ts"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
df.set_index("ts", inplace=True)

ema_period=9; lb=5; min_slope=0.1
df["EMA"]=ema(df["Close"], ema_period)
df["SlopePct"]=slope_pct(df["EMA"], lb)
df["trend_up"]=df["SlopePct"]>min_slope
df["trend_dn"]=df["SlopePct"]<-min_slope
df["prev_close"]=df["Close"].shift(1); df["prev_ema"]=df["EMA"].shift(1)
df["bull_cross"]=(df["prev_close"]<=df["prev_ema"]) & (df["Close"]>df["EMA"])
df["bear_cross"]=(df["prev_close"]>=df["prev_ema"]) & (df["Close"]<df["EMA"])
df["long_cond"]=df["bull_cross"] & df["trend_up"]
df["short_cond"]=df["bear_cross"] & df["trend_dn"]

print("Bars:", len(df))
print("Long signals:", int(df["long_cond"].sum()))
print("Short signals:", int(df["short_cond"].sum()))
print("First longs:\n", df[df["long_cond"]][["Close","EMA","SlopePct"]].head(5))
print("First shorts:\n", df[df["short_cond"]][["Close","EMA","SlopePct"]].head(5))
