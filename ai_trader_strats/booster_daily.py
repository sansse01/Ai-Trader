"""Daily trend booster strategy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

import backtrader as bt

from .config_schemas import BoosterDailyConfig
from .risk_hooks import RiskEngineAdapter
from .sizing import rebalance_to_notional

_LOGGER = logging.getLogger(__name__)


class PreviousDonchianHigh(bt.Indicator):
    """Donchian channel high that ignores the current bar."""

    lines = ("donch",)
    params = (("period", 55),)

    def __init__(self):
        super().__init__()
        self.addminperiod(self.p.period + 1)

    def next(self) -> None:  # type: ignore[override]
        window = list(self.data.get(size=self.p.period + 1))
        if len(window) <= 1:
            self.lines.donch[0] = float("nan")
            return
        # Exclude current bar (last element)
        self.lines.donch[0] = max(window[:-1])


@dataclass(slots=True)
class StrategyMetrics:
    trades: int = 0
    fees_paid: float = 0.0
    equity_peak: float = 0.0
    max_drawdown: float = 0.0
    position_sum: float = 0.0
    bars: int = 0

    def update_equity(self, equity: float) -> None:
        if equity > self.equity_peak:
            self.equity_peak = equity
        if self.equity_peak > 0:
            drawdown = 1.0 - equity / self.equity_peak
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def track_position(self, position_notional: float) -> None:
        self.position_sum += position_notional
        self.bars += 1

    @property
    def average_position(self) -> float:
        return self.position_sum / self.bars if self.bars else 0.0


class DailyTrendBooster(bt.Strategy):
    """Daily trend strategy that boosts exposure in strong markets."""

    params = dict(
        sma_len=200,
        don_len=55,
        strong=1.3,
        weak=0.3,
        max_lev=1.5,
        fees_bps=5,
        config=None,
        context=None,
        risk=None,
    )

    def __init__(self, *args, **kwargs):  # type: ignore[override]
        super().__init__(*args, **kwargs)
        cfg: BoosterDailyConfig | None = self.p.config
        if cfg is None:
            cfg = BoosterDailyConfig()
        self.config = cfg
        if self.p.context is None:
            raise ValueError("DailyTrendBooster requires a shared context")
        self.context = self.p.context
        self.risk: RiskEngineAdapter = self.p.risk or RiskEngineAdapter(self.broker, cfg.risk)

        # Override parameter defaults with configuration values
        self.p.sma_len = cfg.filters_sma_len
        self.p.don_len = cfg.filters_donchian_len
        self.p.strong = cfg.pos_strong
        self.p.weak = cfg.pos_weak
        self.p.max_lev = cfg.max_leverage
        self.p.fees_bps = cfg.fees.fees_bps_per_side

        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_len)
        self.donch = PreviousDonchianHigh(self.data.high, period=self.p.don_len)

        self.metrics = StrategyMetrics()
        self._last_target_notional: float = 0.0

    # ------------------------------------------------------------------
    # Backtrader hooks
    # ------------------------------------------------------------------
    def log(self, msg: str, *args) -> None:
        _LOGGER.info("[DailyTrendBooster] " + msg, *args)

    def notify_order(self, order) -> None:  # type: ignore[override]
        if order.status not in (order.Completed, order.Partial):
            return
        self.metrics.trades += 1
        self.metrics.fees_paid += float(getattr(order.executed, "comm", 0.0))
        self.risk.on_fill(order)
        _LOGGER.debug(
            "Order filled: size=%.4f price=%.2f commission=%.4f",
            order.executed.size,
            order.executed.price,
            order.executed.comm,
        )

    def next(self) -> None:  # type: ignore[override]
        dt: datetime = self.data.datetime.datetime(0)
        self.risk.update_clock(dt)

        equity = float(self.broker.getvalue())
        self.metrics.update_equity(equity)

        price = float(self.data.close[0])
        position = self.getposition(self.data)
        notional = (position.size * price) / equity if equity else 0.0
        self.metrics.track_position(notional)

        if self.risk.locked():
            if abs(position.size) > 1e-8:
                _LOGGER.info("Risk lock active; flattening position")
                rebalance_to_notional(self, 0.0, self.p.fees_bps)
            if self.context is not None:
                self.context.update_core("flat", 0.0)
            self._last_target_notional = 0.0
            return

        strong_signal = False
        if len(self.data) >= max(self.p.sma_len, self.p.don_len) + 1:
            strong_signal = bool(price > float(self.sma[0]) and price > float(self.donch[0]))

        target = self.p.strong if strong_signal else self.p.weak
        target = min(target, self.p.max_lev)
        target = self.risk.enforce_caps(target)

        order = rebalance_to_notional(self, target, self.p.fees_bps)
        if order is not None:
            _LOGGER.info(
                "Rebalancing towards %.2fx (signal=%s)",
                target,
                "strong" if strong_signal else "weak",
            )
        self._last_target_notional = target
        if self.context is not None:
            state = "strong" if strong_signal else "weak"
            self.context.update_core(state, target)

    def stop(self) -> None:  # type: ignore[override]
        avg_pos = self.metrics.average_position
        self.log(
            "Trades=%d fees=%.4f maxDD=%.2f%% avg_notional=%.3f",
            self.metrics.trades,
            self.metrics.fees_paid,
            self.metrics.max_drawdown * 100.0,
            avg_pos,
        )


__all__ = ["DailyTrendBooster"]
