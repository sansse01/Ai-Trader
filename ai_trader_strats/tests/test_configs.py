from pathlib import Path

from ai_trader_strats.config_schemas import BoosterDailyConfig, OverlayHourlyConfig


def test_daily_config_ranges():
    root = Path(__file__).resolve().parents[1]
    cfg = BoosterDailyConfig.from_json(root / "configs" / "btc_eur_daily_trend_booster.json")
    assert 0.0 <= cfg.pos_weak < cfg.pos_strong <= cfg.max_leverage <= 2.0


def test_hourly_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = OverlayHourlyConfig.from_json(root / "configs" / "btc_eur_hourly_overlay.json")
    assert cfg.overlay_notional > 0
    assert cfg.ema_fast < cfg.ema_slow
