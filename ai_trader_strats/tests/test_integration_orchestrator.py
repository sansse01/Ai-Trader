from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_trader_strats.booster_daily import DailyTrendBooster
from ai_trader_strats.overlay_hourly import HourlyTrendOverlay
from ai_trader_strats.orchestrator import MultiTimeframeOrchestrator


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def test_orchestrator_integration(tmp_path: Path):
    daily_dates = pd.date_range("2022-01-01", periods=260, freq="D", tz="UTC")
    daily_prices = np.concatenate([
        np.linspace(100, 180, 220),
        np.linspace(180, 60, 40),
    ])
    daily_df = pd.DataFrame(
        {
            "Timestamp": daily_dates,
            "Open": daily_prices,
            "High": daily_prices * 1.01,
            "Low": daily_prices * 0.99,
            "Close": daily_prices,
            "Volume": np.full_like(daily_prices, 10.0),
        }
    )
    hourly_dates = pd.date_range(daily_dates[0], periods=260 * 24, freq="h", tz="UTC")
    hourly_prices = np.linspace(100, 60, len(hourly_dates))
    hourly_df = pd.DataFrame(
        {
            "Timestamp": hourly_dates,
            "Open": hourly_prices,
            "High": hourly_prices * 1.005,
            "Low": hourly_prices * 0.995,
            "Close": hourly_prices,
            "Volume": np.full_like(hourly_prices, 5.0),
        }
    )

    daily_path = tmp_path / "daily.csv"
    hourly_path = tmp_path / "hourly.csv"
    _write_csv(daily_path, daily_df)
    _write_csv(hourly_path, hourly_df)

    root = Path(__file__).resolve().parents[1]
    orchestrator = MultiTimeframeOrchestrator(
        config_daily=root / "configs" / "btc_eur_daily_trend_booster.json",
        config_hourly=root / "configs" / "btc_eur_hourly_overlay.json",
    )
    cerebro = orchestrator.run(data_daily=daily_path, data_hourly=hourly_path, cash=250_000)

    strategies = orchestrator.last_run_strategies
    assert any(isinstance(s, DailyTrendBooster) for s in strategies)
    assert any(isinstance(s, HourlyTrendOverlay) for s in strategies)

    for strat in strategies:
        if isinstance(strat, DailyTrendBooster):
            assert abs(strat._last_target_notional) <= strat.config.max_leverage + 1e-6
            assert strat.metrics.max_drawdown >= 0.095
            assert abs(strat._last_target_notional) <= strat.config.risk.single_asset_cap_pct + 1e-6
            assert strat.risk.enforce_caps(strat.config.max_leverage) <= strat.config.risk.single_asset_cap_pct
        if isinstance(strat, HourlyTrendOverlay):
            assert strat._last_target <= strat.config.max_total_leverage + 1e-6
            assert strat.context.overlay_notional() == pytest.approx(0.0)
