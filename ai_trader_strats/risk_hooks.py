"""Adapters to bridge strategy logic with the platform risk engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
from typing import Optional

from .config_schemas import RiskConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _RiskState:
    """Internal bookkeeping for risk locks."""

    equity_hwm: float
    day_start_equity: float
    day: date
    drawdown_lock: bool = False
    daily_lock: bool = False
    last_timestamp: Optional[datetime] = None


class RiskEngineAdapter:
    """Minimal adapter enforcing drawdown and daily loss locks."""

    def __init__(self, broker, risk_cfg: RiskConfig):
        self._broker = broker
        self._cfg = risk_cfg
        initial_equity = float(getattr(self._broker, "getvalue", lambda: 0.0)())
        today = datetime.utcnow().date()
        self._state = _RiskState(
            equity_hwm=initial_equity,
            day_start_equity=initial_equity,
            day=today,
        )
        self._lock_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_clock(self, timestamp: datetime) -> None:
        """Update the internal clock to support daily lock resets."""

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        self._state.last_timestamp = timestamp.astimezone(timezone.utc)

    def locked(self) -> bool:
        """Return ``True`` when drawdown or daily loss locks are engaged."""

        self._refresh_state()
        return self._state.drawdown_lock or self._state.daily_lock

    def enforce_caps(self, target_notional: float) -> float:
        """Clamp exposure requests to the configured caps."""

        if self.locked():
            return 0.0
        cap = min(abs(self._cfg.single_asset_cap_pct), abs(self._cfg.gross_cap_pct))
        if cap <= 0:
            return 0.0
        adjusted = max(-cap, min(cap, target_notional))
        if adjusted != target_notional:
            _LOGGER.debug(
                "Risk caps reduced notional from %.4f to %.4f", target_notional, adjusted
            )
        return adjusted

    def on_fill(self, order) -> None:
        """Update state upon order execution."""

        executed = getattr(order, "executed", None)
        if executed is not None and getattr(executed, "dt", None) is not None:
            try:
                from backtrader.utils.date import num2date

                dt = num2date(executed.dt)
                if isinstance(dt, datetime):
                    self.update_clock(dt)
            except Exception:  # pragma: no cover - defensive
                _LOGGER.debug("Failed to parse order execution timestamp", exc_info=True)
        self._refresh_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        equity = float(getattr(self._broker, "getvalue", lambda: 0.0)())
        current_dt = self._current_day()

        previous_hwm = self._state.equity_hwm
        if equity > previous_hwm:
            self._state.equity_hwm = equity
            if self._state.drawdown_lock:
                _LOGGER.info(
                    "Drawdown lock released after equity recovered to high water mark"
                )
                self._state.drawdown_lock = False
                if self._lock_reason == "drawdown":
                    self._lock_reason = None

        drawdown_pct = (equity / self._state.equity_hwm - 1.0) if self._state.equity_hwm else 0.0
        if drawdown_pct <= self._cfg.dd_lock_pct:
            if not self._state.drawdown_lock:
                _LOGGER.warning(
                    "Drawdown lock engaged at %.2f%%", drawdown_pct * 100.0
                )
            self._state.drawdown_lock = True
            self._lock_reason = "drawdown"

        if current_dt != self._state.day:
            self._state.day = current_dt
            self._state.day_start_equity = equity
            if self._state.daily_lock:
                _LOGGER.info("Daily loss lock reset for new trading day")
            self._state.daily_lock = False
            if self._lock_reason == "daily":
                self._lock_reason = None

        if self._state.day_start_equity == 0:
            daily_change = 0.0
        else:
            daily_change = equity / self._state.day_start_equity - 1.0
        if daily_change <= self._cfg.daily_loss_lock_pct:
            if not self._state.daily_lock:
                _LOGGER.warning(
                    "Daily loss lock engaged at %.2f%%", daily_change * 100.0
                )
            self._state.daily_lock = True
            self._lock_reason = "daily"
        elif not self._state.drawdown_lock:
            self._lock_reason = None

    def _current_day(self) -> date:
        timestamp = self._state.last_timestamp
        if timestamp is None:
            return datetime.utcnow().date()
        return timestamp.astimezone(timezone.utc).date()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def lock_reason(self) -> Optional[str]:
        """Return the active lock reason if any."""

        if not self.locked():
            return None
        return self._lock_reason


__all__ = ["RiskEngineAdapter"]
