from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(slots=True)
class RiskEngineConfig:
    """Configuration for the risk engine."""

    enabled: bool = True
    target_volatility: float = 0.25
    vol_lookback: int = 20
    min_volatility: float = 1e-6
    max_scaler: float = 2.0
    min_scaler: float = 0.05
    drawdown_limit: float = 0.2
    drawdown_cooldown_bars: int = 10
    cooldown_scale: float = 0.3
    rerisk_drawdown: float = 0.05
    rerisk_step: float = 0.1
    circuit_breaker_drawdown: float = 0.35
    circuit_breaker_cooldown_bars: int = 50
    max_gross_exposure: float = 2.0
    max_position_exposure: float = 1.0
    min_position_size: float = 0.0
    correlation_threshold: float = 0.85
    max_correlation_exposure: float = 0.6
    enable_correlation_cuts: bool = True

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> "RiskEngineConfig":
        if not params:
            return cls()
        return cls(
            enabled=bool(params.get("enabled", True)),
            target_volatility=float(params.get("target_volatility", 0.25)),
            vol_lookback=int(params.get("vol_lookback", 20)),
            min_volatility=float(params.get("min_volatility", 1e-6)),
            max_scaler=float(params.get("max_scaler", 2.0)),
            min_scaler=float(params.get("min_scaler", 0.05)),
            drawdown_limit=float(params.get("drawdown_limit", 0.2)),
            drawdown_cooldown_bars=int(params.get("drawdown_cooldown_bars", 10)),
            cooldown_scale=float(params.get("cooldown_scale", 0.3)),
            rerisk_drawdown=float(params.get("rerisk_drawdown", 0.05)),
            rerisk_step=float(params.get("rerisk_step", 0.1)),
            circuit_breaker_drawdown=float(params.get("circuit_breaker_drawdown", 0.35)),
            circuit_breaker_cooldown_bars=int(params.get("circuit_breaker_cooldown_bars", 50)),
            max_gross_exposure=float(params.get("max_gross_exposure", 2.0)),
            max_position_exposure=float(params.get("max_position_exposure", 1.0)),
            min_position_size=float(params.get("min_position_size", 0.0)),
            correlation_threshold=float(params.get("correlation_threshold", 0.85)),
            max_correlation_exposure=float(params.get("max_correlation_exposure", 0.6)),
            enable_correlation_cuts=bool(params.get("enable_correlation_cuts", True)),
        )


@dataclass(slots=True)
class PortfolioState:
    """Snapshot of the portfolio used by the risk engine."""

    timestamp: Any
    equity: float
    cash: float
    gross_leverage: float
    net_leverage: float
    positions: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    return_pct: Optional[float] = None
    correlations: Dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass(slots=True)
class OrderContext:
    """Describes an order proposal that should be checked by the risk engine."""

    symbol: str
    direction: int  # +1 long, -1 short
    size: float
    price: float
    is_exit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signed_size(self) -> float:
        return self.direction * self.size


@dataclass(slots=True)
class RiskDecision:
    """Decision returned by the risk engine for a proposed order."""

    size: float
    cancel: bool = False
    reason: Optional[str] = None


class RiskEngine:
    """Risk engine coordinating dynamic position sizing and risk controls."""

    def __init__(self, config: RiskEngineConfig | None = None):
        self.config = config or RiskEngineConfig()
        self.enabled = self.config.enabled
        self._returns: deque[float] = deque(maxlen=max(2, self.config.vol_lookback * 4))
        self._equity_peak: Optional[float] = None
        self._drawdown: float = 0.0
        self._cooldown_remaining: int = 0
        self._circuit_remaining: int = 0
        self._current_scaler: float = 1.0
        self._force_flatten: bool = False
        self._latest_state: Optional[PortfolioState] = None

    def clone(self) -> "RiskEngine":
        """Return a fresh risk engine instance with the same configuration."""

        clone = RiskEngine(self.config)
        return clone

    def reset(self) -> None:
        """Reset all dynamic state."""

        self._returns.clear()
        self._equity_peak = None
        self._drawdown = 0.0
        self._cooldown_remaining = 0
        self._circuit_remaining = 0
        self._current_scaler = 1.0
        self._force_flatten = False
        self._latest_state = None

    @property
    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_remaining > 0

    @property
    def is_flattening(self) -> bool:
        return self._force_flatten or self.is_circuit_breaker_active

    def on_bar(self, state: PortfolioState) -> bool:
        """Update engine state for the new bar and determine if positions should flatten."""

        self._latest_state = state
        if not self.enabled:
            return False

        if state.return_pct is not None:
            self._returns.append(float(state.return_pct))

        if self._equity_peak is None:
            self._equity_peak = state.equity
        else:
            self._equity_peak = max(self._equity_peak, state.equity)

        peak = max(self._equity_peak or 0.0, 1e-9)
        drawdown = max(0.0, 1.0 - (state.equity / peak)) if peak > 0 else 0.0
        self._drawdown = drawdown

        if drawdown >= self.config.circuit_breaker_drawdown:
            self._circuit_remaining = max(self._circuit_remaining, self.config.circuit_breaker_cooldown_bars)
        elif drawdown >= self.config.drawdown_limit:
            self._cooldown_remaining = max(self._cooldown_remaining, self.config.drawdown_cooldown_bars)

        flatten = False
        if self._circuit_remaining > 0:
            flatten = True
            self._circuit_remaining -= 1
        else:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                self._current_scaler = min(self._current_scaler, self.config.cooldown_scale)
            else:
                if drawdown <= self.config.rerisk_drawdown:
                    self._current_scaler = min(1.0, self._current_scaler + self.config.rerisk_step)

        self._force_flatten = flatten
        if flatten:
            self._current_scaler = 0.0
        else:
            self._current_scaler = min(self.config.max_scaler, max(self.config.min_scaler, self._current_scaler))
        return flatten

    def volatility_target(self) -> float:
        """Return a scaling factor derived from realized volatility."""

        if not self.enabled or self.config.target_volatility <= 0:
            return 1.0
        window = list(self._returns)[-self.config.vol_lookback :]
        if len(window) < 2:
            return 1.0
        realized = float(np.std(window, ddof=1))
        if realized < self.config.min_volatility:
            return 1.0
        target = self.config.target_volatility
        scale = target / realized if realized else 1.0
        return float(max(self.config.min_scaler, min(self.config.max_scaler, scale)))

    def modify_order(self, order: OrderContext, state: Optional[PortfolioState] = None) -> RiskDecision:
        """Adjust or block a proposed order based on risk rules."""

        if not self.enabled:
            return RiskDecision(size=order.size)

        state = state or self._latest_state
        if state is None:
            return RiskDecision(size=order.size)

        if order.is_exit:
            return RiskDecision(size=order.size)

        if self.is_flattening:
            return RiskDecision(size=0.0, cancel=True, reason="circuit_breaker")

        scaler = self._current_scaler
        scaler *= self.volatility_target()
        if scaler <= 0:
            return RiskDecision(size=0.0, cancel=True, reason="risk_scaler_zero")

        adjusted = order.size * scaler
        adjusted = self.apply_exposure_caps(order, state, adjusted)
        if adjusted <= 0:
            return RiskDecision(size=0.0, cancel=True, reason="exposure_cap")

        adjusted = self.apply_correlation_cuts(order, state, adjusted)
        if adjusted <= 0:
            return RiskDecision(size=0.0, cancel=True, reason="correlation_cap")

        if adjusted < self.config.min_position_size:
            return RiskDecision(size=0.0, cancel=True, reason="min_position_size")

        return RiskDecision(size=adjusted)

    # Exposure helpers -------------------------------------------------
    def apply_exposure_caps(self, order: OrderContext, state: PortfolioState, proposed_size: float) -> float:
        """Limit order size so that exposure caps are not breached."""

        if proposed_size <= 0:
            return 0.0
        price = float(order.price or state.prices.get(order.symbol, 0.0) or 0.0)
        equity = max(state.equity, 1e-9)

        # Cap gross exposure
        if self.config.max_gross_exposure > 0 and price > 0:
            total_notional = self._total_notional(state.positions, state.prices)
            current_gross_notional = total_notional
            limit_notional = self.config.max_gross_exposure * equity
            desired_positions = dict(state.positions)
            desired_positions[order.symbol] = desired_positions.get(order.symbol, 0.0) + order.direction * proposed_size
            desired_notional = self._total_notional(desired_positions, self._merge_prices(state.prices, order.symbol, price))
            if desired_notional > limit_notional:
                additional_allow = max(0.0, limit_notional - current_gross_notional)
                if additional_allow <= 0:
                    return 0.0
                proposed_size = min(proposed_size, additional_allow / price)

        if proposed_size <= 0:
            return 0.0

        # Cap single position exposure
        if self.config.max_position_exposure > 0 and price > 0:
            current_units = state.positions.get(order.symbol, 0.0)
            limit_notional = self.config.max_position_exposure * equity
            current_notional = abs(current_units) * price
            desired_units = current_units + order.direction * proposed_size
            desired_notional = abs(desired_units) * price
            if desired_notional > limit_notional:
                additional_allow = max(0.0, limit_notional - current_notional)
                if additional_allow <= 0:
                    return 0.0
                proposed_size = min(proposed_size, additional_allow / price)

        return max(0.0, proposed_size)

    def apply_correlation_cuts(self, order: OrderContext, state: PortfolioState, proposed_size: float) -> float:
        """Reduce order size when highly correlated exposure is already present."""

        if not self.config.enable_correlation_cuts or proposed_size <= 0:
            return proposed_size
        correlations: Mapping[str, float] = {}
        if "correlations" in order.metadata:
            raw = order.metadata["correlations"]
            if isinstance(raw, Mapping):
                correlations = {str(k): float(v) for k, v in raw.items()}
        elif state.correlations:
            temp: Dict[str, float] = {}
            for (left, right), corr in state.correlations.items():
                if order.symbol == left:
                    temp[right] = corr
                elif order.symbol == right:
                    temp[left] = corr
            correlations = temp

        if not correlations:
            return proposed_size

        equity = max(state.equity, 1e-9)
        price = float(order.price or state.prices.get(order.symbol, 0.0) or 0.0)
        for other_symbol, corr in correlations.items():
            if abs(corr) < self.config.correlation_threshold:
                continue
            other_units = state.positions.get(other_symbol, 0.0)
            other_price = state.prices.get(other_symbol, price)
            if other_price <= 0:
                continue
            exposure = abs(other_units) * other_price / equity
            if exposure >= self.config.max_correlation_exposure:
                if price <= 0:
                    return 0.0
                allowed_exposure = max(0.0, self.config.max_correlation_exposure - exposure)
                proposed_size = min(proposed_size, allowed_exposure * equity / price)
                if proposed_size <= 0:
                    return 0.0
        return proposed_size

    # Utility helpers --------------------------------------------------
    @staticmethod
    def _total_notional(positions: Mapping[str, float], prices: Mapping[str, float]) -> float:
        total = 0.0
        for symbol, units in positions.items():
            price = float(prices.get(symbol, 0.0) or 0.0)
            total += abs(units) * price
        return total

    @staticmethod
    def _merge_prices(prices: Mapping[str, float], symbol: str, price: float) -> Dict[str, float]:
        merged: Dict[str, float] = dict(prices)
        merged[symbol] = price
        return merged


__all__ = [
    "OrderContext",
    "PortfolioState",
    "RiskDecision",
    "RiskEngine",
    "RiskEngineConfig",
]
