import numpy as np
import pandas as pd
import pytest
import backtrader as bt

from ai_trader_strats.booster_daily import DailyTrendBooster
from ai_trader_strats.config_schemas import BoosterDailyConfig, FeesConfig, RiskConfig
from ai_trader_strats.orchestrator import Context
from ai_trader_strats.risk_hooks import RiskEngineAdapter


def _run_daily_strategy(prices: np.ndarray) -> DailyTrendBooster:
    dates = pd.date_range("2022-01-01", periods=len(prices), freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices * 0.999,
            "Close": prices,
            "Volume": np.full_like(prices, 1.0),
        },
        index=dates,
    )
    cfg = BoosterDailyConfig(
        filters_sma_len=10,
        filters_donchian_len=10,
        pos_strong=1.3,
        pos_weak=0.3,
        max_leverage=1.5,
        fees=FeesConfig(fees_bps_per_side=0),
        risk=RiskConfig(dd_lock_pct=-0.2, daily_loss_lock_pct=-0.2, single_asset_cap_pct=5.0, gross_cap_pct=5.0),
    )
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(0.0)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    context = Context()
    risk = RiskEngineAdapter(cerebro.getbroker(), cfg.risk)
    cerebro.addstrategy(DailyTrendBooster, config=cfg, context=context, risk=risk)
    strategies = cerebro.run()
    return strategies[0]


def test_daily_strong_signal_targets_boost():
    prices = np.linspace(10, 30, 220)
    strat = _run_daily_strategy(prices)
    assert strat._last_target_notional == pytest.approx(1.3)
    assert strat.context.daily_is_strong()


def test_daily_weak_signal_targets_low_exposure():
    prices = np.full(220, 20.0)
    strat = _run_daily_strategy(prices)
    assert strat._last_target_notional == pytest.approx(0.3)
    assert strat.context.core_notional() == pytest.approx(0.3)
