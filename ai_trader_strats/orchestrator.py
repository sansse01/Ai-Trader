"""Orchestration helpers for running the multi-timeframe strategy stack."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import backtrader as bt
import pandas as pd

from .booster_daily import DailyTrendBooster
from .config_schemas import BoosterDailyConfig, OverlayHourlyConfig, load_config
from .overlay_hourly import HourlyTrendOverlay
from .risk_hooks import RiskEngineAdapter

_LOGGER = logging.getLogger(__name__)


class _DailyPandasData(bt.feeds.PandasData):
    params = (("timeframe", bt.TimeFrame.Days), ("compression", 1))


class _HourlyPandasData(bt.feeds.PandasData):
    params = (("timeframe", bt.TimeFrame.Minutes), ("compression", 60))


@dataclass(slots=True)
class Context:
    """Shared context between strategies."""

    core_state: Literal["strong", "weak", "flat"] = "flat"
    _core_notional: float = 0.0
    _overlay_notional: float = 0.0

    def daily_is_strong(self) -> bool:
        return self.core_state == "strong"

    def core_notional(self) -> float:
        return self._core_notional

    def overlay_notional(self) -> float:
        return self._overlay_notional

    def update_core(self, state: Literal["strong", "weak", "flat"], notional: float) -> None:
        self.core_state = state
        self._core_notional = notional

    def update_overlay(self, notional: float) -> None:
        self._overlay_notional = notional


class MultiTimeframeOrchestrator:
    """Coordinate the daily booster and hourly overlay strategies."""

    def __init__(
        self,
        config_daily: str | Path,
        config_hourly: str | Path | None = None,
        broker: str = "paper",
    ) -> None:
        self.daily_config = load_config(config_daily, BoosterDailyConfig)
        self.hourly_config = (
            load_config(config_hourly, OverlayHourlyConfig) if config_hourly else None
        )
        self.broker_mode = broker
        self._last_run: List[bt.Strategy] = []

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def _load_dataframe(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        rename_map = {col.lower(): col for col in df.columns}
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        lower_cols = {col.lower() for col in df.columns}
        missing = required - lower_cols
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        df = df.rename(columns={rename_map[k]: k for k in required})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("UTC")
        ordered = df[["open", "high", "low", "close", "volume"]].copy()
        ordered.columns = [col.capitalize() for col in ordered.columns]
        return ordered

    def _build_data_feed(self, df: pd.DataFrame, interval: str) -> bt.feeds.PandasData:
        if interval == "1d":
            return _DailyPandasData(dataname=df)
        if interval == "1h":
            return _HourlyPandasData(dataname=df)
        raise ValueError(f"Unsupported interval: {interval}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        data_daily: str | Path,
        data_hourly: str | Path | None = None,
        *,
        cash: float = 1_000_000.0,
    ) -> bt.Cerebro:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(self.daily_config.fees.commission)
        context = Context()
        risk = RiskEngineAdapter(cerebro.getbroker(), self.daily_config.risk)

        daily_df = self._load_dataframe(data_daily)
        cerebro.adddata(self._build_data_feed(daily_df, self.daily_config.bar_interval), name="daily")

        if self.hourly_config and data_hourly:
            hourly_df = self._load_dataframe(data_hourly)
            cerebro.adddata(
                self._build_data_feed(hourly_df, self.hourly_config.bar_interval), name="hourly"
            )

        cerebro.addstrategy(
            DailyTrendBooster,
            config=self.daily_config,
            context=context,
            risk=risk,
        )

        if self.hourly_config and data_hourly:
            cerebro.addstrategy(
                HourlyTrendOverlay,
                config=self.hourly_config,
                context=context,
                risk=risk,
            )

        _LOGGER.info("Starting backtest broker=%s cash=%.2f", self.broker_mode, cash)
        self._last_run = cerebro.run()
        return cerebro

    @property
    def last_run_strategies(self) -> List[bt.Strategy]:
        return list(self._last_run)


def _parse_args(argv: Optional[Iterable[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI trader strategy stack")
    parser.add_argument("--config-daily", required=True)
    parser.add_argument("--data-daily", required=True)
    parser.add_argument("--config-hourly")
    parser.add_argument("--data-hourly")
    parser.add_argument("--broker", default="paper")
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    orchestrator = MultiTimeframeOrchestrator(
        config_daily=args.config_daily,
        config_hourly=args.config_hourly,
        broker=args.broker,
    )
    orchestrator.run(data_daily=args.data_daily, data_hourly=args.data_hourly, cash=args.cash)


if __name__ == "__main__":
    main()
