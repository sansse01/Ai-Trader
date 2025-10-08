from __future__ import annotations

from datetime import datetime

from strategy_builder.evaluator import accept
from strategy_builder.schemas import Metrics


def make_metrics(**overrides):
    defaults = dict(
        total_return=0.3,
        cagr=0.25,
        max_drawdown=0.08,
        sharpe=1.4,
        sortino=1.6,
        calmar=3.0,
        hit_rate=0.55,
        avg_win=0.02,
        avg_loss=-0.01,
        trades=40,
        turnover=2.0,
        fees_paid=0.01,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        timeframe="1h",
    )
    defaults.update(overrides)
    return Metrics(**defaults)


def test_accept_within_tolerances() -> None:
    predicted = {"cagr": 0.22, "maxdd": 0.09, "sharpe": 1.2}
    measured = make_metrics()
    ok, reason = accept(predicted, measured)
    assert ok
    assert reason == "Accepted"


def test_reject_on_drawdown() -> None:
    predicted = {"cagr": 0.22, "maxdd": 0.09, "sharpe": 1.2}
    measured = make_metrics(max_drawdown=0.11)
    ok, reason = accept(predicted, measured)
    assert not ok
    assert "Max drawdown" in reason


def test_reject_on_stress() -> None:
    measured = make_metrics(max_drawdown=0.09)
    predicted = {}
    ok, reason = accept(predicted, measured)
    assert not ok
    assert "Stress" in reason
