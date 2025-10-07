import numpy as np
import pandas as pd
import pytest
import backtrader as bt

from ai_trader_strats.config_schemas import FeesConfig, OverlayHourlyConfig, RiskConfig
from ai_trader_strats.orchestrator import Context
from ai_trader_strats.overlay_hourly import HourlyTrendOverlay
from ai_trader_strats.risk_hooks import RiskEngineAdapter


def _run_overlay_strategy(prices: np.ndarray, context: Context) -> HourlyTrendOverlay:
    dates = pd.date_range("2022-01-01", periods=len(prices), freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.full_like(prices, 5.0),
        },
        index=dates,
    )
    daily_df = df.resample("1D").last().copy()
    daily_df["Open"] = daily_df["Close"]
    daily_df["High"] = daily_df["Close"]
    daily_df["Low"] = daily_df["Close"]
    daily_df["Volume"] = 10.0
    cfg = OverlayHourlyConfig(
        ema_fast=4,
        ema_slow=12,
        donchian_high=6,
        overlay_notional=0.2,
        max_total_leverage=1.5,
        fees=FeesConfig(fees_bps_per_side=0),
    )
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(0.0)
    cerebro.adddata(bt.feeds.PandasData(dataname=daily_df))
    cerebro.adddata(bt.feeds.PandasData(dataname=df), name="hourly")
    risk_cfg = RiskConfig(dd_lock_pct=-0.2, daily_loss_lock_pct=-0.2, single_asset_cap_pct=5.0, gross_cap_pct=5.0)
    risk = RiskEngineAdapter(cerebro.getbroker(), risk_cfg)
    cerebro.addstrategy(HourlyTrendOverlay, config=cfg, context=context, risk=risk)
    strategies = cerebro.run()
    return strategies[0]


def test_overlay_engages_when_daily_strong():
    context = Context()
    context.update_core("strong", 1.3)
    prices = np.linspace(20, 40, 400)
    strat = _run_overlay_strategy(prices, context)
    assert strat._last_target == pytest.approx(1.5)
    assert context.overlay_notional() == pytest.approx(0.2)


def test_overlay_idle_when_daily_not_strong():
    context = Context()
    context.update_core("weak", 0.8)
    prices = np.linspace(20, 40, 400)
    strat = _run_overlay_strategy(prices, context)
    assert strat._last_target == pytest.approx(0.8)
    assert context.overlay_notional() == pytest.approx(0.0)
