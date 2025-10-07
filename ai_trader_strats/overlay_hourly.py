"""Hourly overlay strategy that adds exposure during strong trends."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

import backtrader as bt

from .booster_daily import PreviousDonchianHigh
from .config_schemas import OverlayHourlyConfig
from .risk_hooks import RiskEngineAdapter
from .sizing import rebalance_to_notional

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OverlayMetrics:
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

    def track_position(self, notional: float) -> None:
        self.position_sum += notional
        self.bars += 1

    @property
    def average_position(self) -> float:
        return self.position_sum / self.bars if self.bars else 0.0


class HourlyTrendOverlay(bt.Strategy):
    """Hourly momentum overlay to piggyback on the daily core signal."""

    params = dict(
        ema_fast=8,
        ema_slow=34,
        don=10,
        overlay=0.2,
        max_total=1.5,
        fees_bps=5,
        config=None,
        context=None,
        risk=None,
    )

    def __init__(self, *args, **kwargs):  # type: ignore[override]
        super().__init__(*args, **kwargs)
        cfg: OverlayHourlyConfig | None = self.p.config
        if cfg is None:
            cfg = OverlayHourlyConfig()
        self.config = cfg
        if self.p.context is None:
            raise ValueError("HourlyTrendOverlay requires a shared context")
        self.context = self.p.context
        if self.p.risk is None:
            raise ValueError("HourlyTrendOverlay requires a shared RiskEngineAdapter")
        self.risk: RiskEngineAdapter = self.p.risk

        self.p.ema_fast = cfg.ema_fast
        self.p.ema_slow = cfg.ema_slow
        self.p.don = cfg.donchian_high
        self.p.overlay = cfg.overlay_notional
        self.p.max_total = cfg.max_total_leverage
        self.p.fees_bps = cfg.fees.fees_bps_per_side

        try:
            hourly_data = self.getdatabyname("hourly")
        except KeyError:
            if len(self.datas) >= 2:
                hourly_data = self.datas[1]
            else:
                raise ValueError(
                    "HourlyTrendOverlay requires an hourly data feed added after the daily feed"
                )
        self._hourly_data: bt.LineIterator = hourly_data

        self.fast = bt.indicators.ExponentialMovingAverage(
            self._hourly_data.close, period=self.p.ema_fast
        )
        self.slow = bt.indicators.ExponentialMovingAverage(
            self._hourly_data.close, period=self.p.ema_slow
        )
        self.donch = PreviousDonchianHigh(self._hourly_data.high, period=self.p.don)

        self.metrics = OverlayMetrics()
        self._last_target: float = 0.0

    def notify_order(self, order) -> None:  # type: ignore[override]
        if order.status not in (order.Completed, order.Partial):
            return
        self.metrics.trades += 1
        self.metrics.fees_paid += float(getattr(order.executed, "comm", 0.0))
        self.risk.on_fill(order)
        _LOGGER.debug(
            "Overlay fill: size=%.4f price=%.2f comm=%.4f",
            order.executed.size,
            order.executed.price,
            order.executed.comm,
        )

    def next(self) -> None:  # type: ignore[override]
        dt: datetime = self._hourly_data.datetime.datetime(0)
        self.risk.update_clock(dt)
        equity = float(self.broker.getvalue())
        self.metrics.update_equity(equity)

        price = float(self._hourly_data.close[0])
        position = self.getposition(self._hourly_data)
        notional = (position.size * price) / equity if equity else 0.0
        self.metrics.track_position(notional)

        if self.risk.locked():
            if abs(position.size) > 1e-8:
                _LOGGER.info("Risk lock active; overlay flattening")
                rebalance_to_notional(self, 0.0, self.p.fees_bps, data=self._hourly_data)
            if self.context is not None:
                self.context.update_overlay(0.0)
            self._last_target = 0.0
            return

        overlay_allowed = bool(self.context and self.context.daily_is_strong())
        overlay_signal = False
        if overlay_allowed and len(self._hourly_data) >= max(
            self.p.ema_fast, self.p.ema_slow, self.p.don
        ) + 1:
            overlay_signal = bool(
                float(self.fast[0]) > float(self.slow[0]) and price > float(self.donch[0])
            )

        base_notional = self.context.core_notional() if self.context else 0.0
        target = base_notional
        if overlay_allowed and overlay_signal:
            additional = min(self.p.overlay, max(0.0, self.p.max_total - base_notional))
            target = base_notional + additional
        target = min(target, self.p.max_total)
        target = self.risk.enforce_caps(target)

        order = rebalance_to_notional(self, target, self.p.fees_bps, data=self._hourly_data)
        if order is not None:
            _LOGGER.info(
                "Overlay targeting %.2fx (base=%.2f overlay=%.2f)",
                target,
                base_notional,
                target - base_notional,
            )
        self._last_target = target
        if self.context is not None:
            self.context.update_overlay(max(0.0, target - base_notional))

    def stop(self) -> None:  # type: ignore[override]
        _LOGGER.info(
            "[HourlyOverlay] Trades=%d fees=%.4f maxDD=%.2f%% avg_notional=%.3f",
            self.metrics.trades,
            self.metrics.fees_paid,
            self.metrics.max_drawdown * 100.0,
            self.metrics.average_position,
        )


__all__ = ["HourlyTrendOverlay"]
