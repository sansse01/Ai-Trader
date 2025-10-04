from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

import pandas as pd

from .base import StrategyDefinition


DEFAULT_PARAMS: Dict[str, Any] = {
    "ema_period": 9,
    "slope_lookback": 5,
    "min_slope_percent": 0.1,
    "stop_loss_percent": 2.0,
    "trail_percent": 1.5,
    "fee_percent": 0.26,
    "slippage_percent": 0.05,
    "max_leverage": 5.0,
    "risk_fraction": 0.25,
    "contract_size": 0.001,
    "allow_shorts": True,
    "ignore_trend": False,
    "initial_cash": 10_000.0,
}


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def slope_pct(series: pd.Series, lookback: int) -> pd.Series:
    ref = series.shift(lookback)
    return (series.sub(ref)).div(ref).mul(100.0)


def _ema_prepare(df: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return df


def _ema_generate_signals(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    merged_params = {**DEFAULT_PARAMS, **dict(params)}
    ema_period = int(merged_params.get("ema_period", DEFAULT_PARAMS["ema_period"]))
    slope_lookback = int(merged_params.get("slope_lookback", DEFAULT_PARAMS["slope_lookback"]))
    min_slope_percent = float(
        merged_params.get("min_slope_percent", DEFAULT_PARAMS["min_slope_percent"])
    )

    signals_df = df.copy()
    signals_df["EMA"] = ema(signals_df["Close"], ema_period)
    signals_df["SlopePct"] = slope_pct(signals_df["EMA"], slope_lookback).fillna(0.0)
    signals_df["trend_up"] = signals_df["SlopePct"] > min_slope_percent
    signals_df["trend_dn"] = signals_df["SlopePct"] < -min_slope_percent
    signals_df["prev_close"] = signals_df["Close"].shift(1)
    signals_df["prev_ema"] = signals_df["EMA"].shift(1)
    signals_df["bull_cross"] = (signals_df["prev_close"] <= signals_df["prev_ema"]) & (
        signals_df["Close"] > signals_df["EMA"]
    )
    signals_df["bear_cross"] = (signals_df["prev_close"] >= signals_df["prev_ema"]) & (
        signals_df["Close"] < signals_df["EMA"]
    )
    return signals_df


def _ema_build_orders(
    _df: pd.DataFrame,
    _signals: pd.DataFrame,
    _params: Mapping[str, Any],
) -> Dict[str, Any]:
    return {}


def _build_simple_backtest_strategy(
    df: pd.DataFrame,
    df_sig: pd.DataFrame,
    params: Mapping[str, Any],
):
    from backtesting import Strategy

    merged_params = {**DEFAULT_PARAMS, **dict(params)}
    ema_period = int(merged_params.get("ema_period", DEFAULT_PARAMS["ema_period"]))
    slope_lookback = int(merged_params.get("slope_lookback", DEFAULT_PARAMS["slope_lookback"]))
    stop_loss_percent = float(merged_params.get("stop_loss_percent", 0.0))
    trail_percent = float(merged_params.get("trail_percent", 0.0))
    allow_shorts = bool(merged_params.get("allow_shorts", True))
    ignore_trend = bool(merged_params.get("ignore_trend", False))
    contract_size = float(merged_params.get("contract_size", 1.0))
    risk_fraction = float(merged_params.get("risk_fraction", 0.25))
    sizing_mode = str(merged_params.get("sizing_mode", "Whole units (int)"))

    class EMASimpleStrategy(Strategy):
        def init(self):
            self.current_sl = None
            self.last_entry_ts = None

        def _position_size(self, close_price: float) -> float:
            if sizing_mode.startswith("Whole"):
                cash = self.equity
                notional = cash * risk_fraction
                units = max(1, int(notional / max(1e-9, close_price)))
                return units
            return max(0.001, min(0.99, risk_fraction))

        def next(self):
            ts = self.data.index[-1]
            close = self.data.Close[-1]
            row = df_sig.loc[ts]
            min_bars = max(ema_period + 1, slope_lookback + 1)
            if len(self.data) < min_bars or pd.isna(row["prev_close"]) or pd.isna(row["prev_ema"]):
                return

            if not self.position:
                self.current_sl = None

            if self.position:
                start_ts = self.last_entry_ts or ts
                if self.position.is_long:
                    recent_high = df.loc[start_ts:ts]["High"].max() * contract_size
                    trail = recent_high * (1 - trail_percent / 100.0)
                    if self.current_sl is None or trail > self.current_sl:
                        self.current_sl = trail
                    if self.data.Low[-1] <= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return
                else:
                    recent_low = df.loc[start_ts:ts]["Low"].min() * contract_size
                    trail = recent_low * (1 + trail_percent / 100.0)
                    if self.current_sl is None or trail < self.current_sl:
                        self.current_sl = trail
                    if self.data.High[-1] >= self.current_sl:
                        self.position.close()
                        self.current_sl = None
                        return

            go_long = bool(row["bull_cross"]) and (ignore_trend or bool(row["trend_up"]))
            go_short = allow_shorts and bool(row["bear_cross"]) and (ignore_trend or bool(row["trend_dn"]))

            size = self._position_size(close)

            if not self.position:
                if go_long:
                    sl_unscaled = (
                        df.loc[ts]["Close"] * (1 - stop_loss_percent / 100.0)
                        if stop_loss_percent > 0
                        else None
                    )
                    sl = sl_unscaled * contract_size if sl_unscaled is not None else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.buy(size=size, sl=sl)
                elif go_short:
                    sl_unscaled = (
                        df.loc[ts]["Close"] * (1 + stop_loss_percent / 100.0)
                        if stop_loss_percent > 0
                        else None
                    )
                    sl = sl_unscaled * contract_size if sl_unscaled is not None else None
                    self.last_entry_ts = ts
                    self.current_sl = sl
                    self.sell(size=size, sl=sl)

    return EMASimpleStrategy


def _build_true_stop_strategy(
    _df: pd.DataFrame,
    _df_sig: pd.DataFrame,
    params: Mapping[str, Any],
):
    import backtrader as bt

    merged_params = {**DEFAULT_PARAMS, **dict(params)}
    ema_period = int(merged_params.get("ema_period", DEFAULT_PARAMS["ema_period"]))
    slope_lookback = int(merged_params.get("slope_lookback", DEFAULT_PARAMS["slope_lookback"]))
    min_slope_percent = float(
        merged_params.get("min_slope_percent", DEFAULT_PARAMS["min_slope_percent"])
    ) / 100.0
    stop_loss_percent = float(merged_params.get("stop_loss_percent", 0.0)) / 100.0
    trail_percent = float(merged_params.get("trail_percent", 0.0)) / 100.0
    allow_shorts = bool(merged_params.get("allow_shorts", True))
    ignore_trend = bool(merged_params.get("ignore_trend", False))

    class EMATrueStopStrategy(bt.Strategy):
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
            self.slope_pct = self.ema_shift / bt.If(
                self.ema(-int(self.p.slope_lb)) == 0, 1e-9, self.ema(-int(self.p.slope_lb))
            )
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

    return EMATrueStopStrategy


EMA_STRATEGY = StrategyDefinition(
    key="ema_trend",
    name="EMA Trend",
    description="Trend-following strategy using EMA crossovers and slope filters.",
    controls=OrderedDict(
        {
            "ema_period": dict(label="EMA period", dtype=int, min=1, max=200, value=9, step=1),
            "slope_lookback": dict(label="Slope lookback (bars)", dtype=int, min=1, max=200, value=5, step=1),
            "min_slope_percent": dict(
                label="Min slope %", dtype=float, min=0.0, max=100.0, value=0.1, step=0.1, format="%.2f"
            ),
            "stop_loss_percent": dict(
                label="Stop loss %", dtype=float, min=0.0, max=50.0, value=2.0, step=0.1, format="%.2f"
            ),
            "trail_percent": dict(
                label="Trailing stop %", dtype=float, min=0.0, max=50.0, value=1.5, step=0.1, format="%.2f"
            ),
            "fee_percent": dict(
                label="Fee % per fill", dtype=float, min=0.0, max=2.0, value=0.26, step=0.01, format="%.4f"
            ),
            "slippage_percent": dict(
                label="Slippage % per fill", dtype=float, min=0.0, max=2.0, value=0.05, step=0.01, format="%.4f"
            ),
            "max_leverage": dict(
                label="Max leverage", dtype=float, min=1.0, max=10.0, value=5.0, step=0.5, format="%.2f"
            ),
            "risk_fraction": dict(
                label="Risk fraction of equity", dtype=float, min=0.01, max=1.0, value=0.25, step=0.01, slider=True
            ),
            "contract_size": dict(
                label="Contract size (BTC per unit)", dtype=float, min=0.000001, max=10.0, value=0.001, step=0.0001, format="%.6f"
            ),
        }
    ),
    range_controls={},
    default_params=DEFAULT_PARAMS,
    data_requirements={
        "chart_overlays": [
            {"column": "EMA", "label": "EMA ({ema_period})"},
        ],
        "signal_columns": {
            "long": "bull_cross",
            "short": "bear_cross",
            "trend_up": "trend_up",
            "trend_down": "trend_dn",
        },
        "preview_columns": [
            "Close",
            "EMA",
            "SlopePct",
            "bull_cross",
            "bear_cross",
            "trend_up",
            "trend_dn",
        ],
    },
    prepare_data=_ema_prepare,
    generate_signals=_ema_generate_signals,
    build_orders=_ema_build_orders,
    build_simple_backtest_strategy=_build_simple_backtest_strategy,
    build_true_stop_strategy=_build_true_stop_strategy,
)


__all__ = [
    "DEFAULT_PARAMS",
    "EMA_STRATEGY",
    "ema",
    "slope_pct",
]
