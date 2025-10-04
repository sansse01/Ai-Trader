from __future__ import annotations

from typing import Dict

import pytest

from risk.engine import OrderContext, PortfolioState, RiskEngine, RiskEngineConfig


def _make_state(
    *,
    equity: float,
    return_pct: float,
    prices: Dict[str, float] | None = None,
    positions: Dict[str, float] | None = None,
) -> PortfolioState:
    return PortfolioState(
        timestamp=len(prices or {}),
        equity=equity,
        cash=equity,
        gross_leverage=0.0,
        net_leverage=0.0,
        positions=positions or {},
        prices=prices or {"BTC": 1.0},
        return_pct=return_pct,
    )


def test_drawdown_cooldown_and_rerisk_transitions() -> None:
    config = RiskEngineConfig(
        drawdown_limit=0.1,
        drawdown_cooldown_bars=1,
        cooldown_scale=0.2,
        rerisk_drawdown=0.02,
        rerisk_step=0.3,
        min_scaler=0.05,
        max_scaler=1.0,
        max_gross_exposure=10.0,
        max_position_exposure=10.0,
    )
    engine = RiskEngine(config)

    base_state = _make_state(equity=100.0, return_pct=0.0)
    engine.on_bar(base_state)

    drawdown_state = _make_state(equity=85.0, return_pct=-0.15)
    engine.on_bar(drawdown_state)

    decision = engine.modify_order(
        OrderContext(symbol="BTC", direction=1, size=1.0, price=1.0), drawdown_state
    )
    assert decision.size == pytest.approx(0.2, rel=1e-3)
    assert not decision.cancel

    recovery_state = _make_state(equity=99.0, return_pct=0.1647)
    engine.on_bar(recovery_state)

    recovery_decision = engine.modify_order(
        OrderContext(symbol="BTC", direction=1, size=1.0, price=1.0), recovery_state
    )
    assert recovery_decision.size == pytest.approx(0.5, rel=1e-3)
    assert not recovery_decision.cancel


def test_circuit_breaker_blocks_orders_until_cooldown_expires() -> None:
    config = RiskEngineConfig(
        drawdown_limit=0.1,
        drawdown_cooldown_bars=1,
        cooldown_scale=0.3,
        circuit_breaker_drawdown=0.25,
        circuit_breaker_cooldown_bars=2,
        max_gross_exposure=10.0,
        max_position_exposure=10.0,
    )
    engine = RiskEngine(config)

    engine.on_bar(_make_state(equity=100.0, return_pct=0.0))

    crash_state = _make_state(equity=60.0, return_pct=-0.4)
    assert engine.on_bar(crash_state)

    blocked = engine.modify_order(
        OrderContext(symbol="BTC", direction=1, size=1.0, price=1.0), crash_state
    )
    assert blocked.cancel
    assert blocked.size == 0.0
    assert blocked.reason == "circuit_breaker"

    cooldown_state = _make_state(equity=80.0, return_pct=0.3333)
    assert engine.on_bar(cooldown_state)

    post_cooldown_state = _make_state(equity=85.0, return_pct=0.0625)
    assert not engine.on_bar(post_cooldown_state)
    assert not engine.is_flattening

    restored = engine.modify_order(
        OrderContext(symbol="BTC", direction=1, size=1.0, price=1.0), post_cooldown_state
    )
    assert restored.size > 0.0
    assert not restored.cancel


def test_volatility_target_scales_position_size() -> None:
    config = RiskEngineConfig(
        target_volatility=0.05,
        vol_lookback=5,
        min_scaler=0.01,
        max_scaler=2.0,
        drawdown_limit=1.0,
        circuit_breaker_drawdown=1.0,
        max_gross_exposure=10.0,
        max_position_exposure=10.0,
    )
    engine = RiskEngine(config)

    equity = 100.0
    prices = {"BTC": 1.0}
    engine.on_bar(_make_state(equity=equity, return_pct=0.0, prices=prices))

    returns = [0.2, -0.15, 0.18, -0.12, 0.22]
    for r in returns:
        equity *= 1 + r
        engine.on_bar(_make_state(equity=equity, return_pct=r, prices=prices))

    latest_state = _make_state(equity=equity, return_pct=0.0, prices=prices)
    scale = engine.volatility_target()
    decision = engine.modify_order(
        OrderContext(symbol="BTC", direction=1, size=1.0, price=1.0), latest_state
    )
    assert scale < 1.0
    assert decision.size == pytest.approx(scale, rel=1e-3)
    assert not decision.cancel
