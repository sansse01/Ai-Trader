"""Position sizing helpers shared across strategies."""
from __future__ import annotations

import logging
from typing import Optional

import backtrader as bt

_LOGGER = logging.getLogger(__name__)


def rebalance_to_notional(
    strategy: bt.Strategy, target_notional: float, fees_bps: int
) -> Optional[bt.Order]:
    """Rebalance the instrument to the desired notional exposure.

    Parameters
    ----------
    strategy:
        Strategy issuing the rebalance.
    target_notional:
        Desired notional exposure expressed as a multiple of account equity.
    fees_bps:
        Trading fee in basis points per side.

    Returns
    -------
    Optional[bt.Order]
        The generated order, ``None`` when no trade is required.
    """

    data = strategy.data0
    price = float(data.close[0])
    equity = float(strategy.broker.getvalue())
    if equity <= 0 or price <= 0:
        _LOGGER.debug("Skipping rebalance due to non-positive equity=%.4f price=%.4f", equity, price)
        return None

    position = strategy.getposition(data)
    current_size = float(getattr(position, "size", 0.0))
    current_value = current_size * price
    current_notional = current_value / equity if equity else 0.0
    if abs(target_notional - current_notional) < 1e-4:
        return None

    target_value = target_notional * equity
    target_size = target_value / price
    delta_size = target_size - current_size
    if abs(delta_size) * price < 1e-6:
        return None

    fees_fraction = fees_bps / 10000.0
    if delta_size > 0:
        target_size = current_size + delta_size * (1.0 + fees_fraction)
    else:
        target_size = current_size + delta_size * (1.0 + fees_fraction)
    _LOGGER.debug(
        "Rebalance request: current_notional=%.4f target=%.4f delta_size=%.4f",
        current_notional,
        target_notional,
        delta_size,
    )
    order = strategy.order_target_size(data=data, target=target_size)
    return order


__all__ = ["rebalance_to_notional"]
