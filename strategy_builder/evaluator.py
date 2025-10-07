"""Metric evaluation utilities."""
from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Tuple

from .schemas import Metrics, ParamProposal

LOGGER = logging.getLogger(__name__)


def _stress_metrics(metrics: Metrics, stress: Dict[str, float]) -> Metrics:
    """Generate stressed metrics adjusting returns, fees, and drawdowns."""

    scale = stress.get("return_scale", 1.0)
    dd_scale = stress.get("dd_scale", 1.0)
    fee_scale = stress.get("fee_scale", 1.0)
    stressed = metrics.copy(update={
        "total_return": metrics.total_return * scale,
        "cagr": metrics.cagr * scale,
        "sharpe": metrics.sharpe * scale,
        "sortino": metrics.sortino * scale,
        "calmar": metrics.calmar * scale,
        "max_drawdown": min(metrics.max_drawdown * dd_scale, 1.0),
        "fees_paid": metrics.fees_paid * fee_scale,
    })
    return stressed


def _stress_scenarios() -> List[Dict[str, float]]:
    return [
        {"return_scale": 0.9, "dd_scale": 1.05, "fee_scale": 2.0},
        {"return_scale": 0.8, "dd_scale": 1.5, "fee_scale": 1.0},
        {"return_scale": 0.85, "dd_scale": 1.2, "fee_scale": 1.2},
    ]


def accept(
    predicted: Dict[str, float],
    measured: Metrics,
    hard_maxdd: float = 0.10,
    cagr_tol: float = 0.08,
    dd_tol: float = 0.015,
    sharpe_tol: float = 0.2,
) -> Tuple[bool, str]:
    """Decide whether the measured metrics are acceptable given predictions."""

    LOGGER.info("Evaluating measured metrics against tolerances")
    if measured.max_drawdown > hard_maxdd:
        return False, f"Max drawdown {measured.max_drawdown:.2%} exceeds hard limit {hard_maxdd:.2%}"

    if predicted:
        pred_cagr = predicted.get("cagr", 0.0)
        if measured.cagr + cagr_tol < pred_cagr:
            return False, "Measured CAGR materially below prediction"
        pred_dd = predicted.get("maxdd", 0.0)
        if measured.max_drawdown - dd_tol > pred_dd:
            return False, "Measured MaxDD worse than predicted"
        pred_sharpe = predicted.get("sharpe", 0.0)
        if measured.sharpe + sharpe_tol < pred_sharpe:
            return False, "Measured Sharpe below expectation"

    for stress in _stress_scenarios():
        stressed = _stress_metrics(measured, stress)
        if stressed.max_drawdown > 0.12:
            return False, "Stress test drawdown above 12%"

    return True, "Accepted"


def choose_champion(
    proposals: Iterable[ParamProposal],
    measured_metrics: Dict[str, Metrics],
) -> Tuple[Dict[str, float], Metrics, Dict[str, float]]:
    """Pick the winning parameter set with the highest CAGR."""

    best_params: Dict[str, float] | None = None
    best_metrics: Metrics | None = None
    report: Dict[str, float] = {}

    for proposal in proposals:
        key = json.dumps(proposal.params, sort_keys=True)
        metrics = measured_metrics.get(key)
        if metrics is None:
            continue
        if best_metrics is None or metrics.cagr > best_metrics.cagr:
            best_metrics = metrics
            best_params = dict(proposal.params)
            report = {
                "cagr": metrics.cagr,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
            }

    if best_params is None or best_metrics is None:
        raise ValueError("No metrics were supplied for champion selection")

    return best_params, best_metrics, report
