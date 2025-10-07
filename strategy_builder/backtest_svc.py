"""Local backtesting adapters and fallbacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from .schemas import Metrics

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from yourapp.risk import RiskEngine  # type: ignore
except Exception:  # pragma: no cover
    class RiskEngine:  # type: ignore
        """Fallback risk engine applying drawdown locks."""

        def __init__(self, max_drawdown: float = 0.095, daily_loss: float = 0.02) -> None:
            self.max_drawdown = max_drawdown
            self.daily_loss = daily_loss

        def enforce(self, equity_curve: pd.Series) -> pd.Series:
            peak = equity_curve.cummax()
            drawdown = equity_curve / peak - 1
            locked = drawdown < -self.max_drawdown
            if locked.any():
                first_lock = locked.idxmax()
                equity_curve.loc[first_lock:] = equity_curve.loc[first_lock]
            daily = equity_curve.resample("1D").last().pct_change().fillna(0)
            daily_lock = daily < -self.daily_loss
            if daily_lock.any():
                first = daily_lock.idxmax()
                mask = equity_curve.index >= first
                equity_curve.loc[mask] = equity_curve.loc[first]
            return equity_curve

try:  # pragma: no cover
    from yourapp.exec import BacktestRunner  # type: ignore
except Exception:  # pragma: no cover
    BacktestRunner = None  # type: ignore


@dataclass(slots=True)
class SimpleBacktestRunner:
    """Deterministic vectorised fallback runner."""

    risk_engine: RiskEngine

    def run(self, strategy_id: str, params: Dict[str, Any], data: pd.DataFrame) -> Metrics:
        df = data.copy()
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        if strategy_id == "trend_atr":
            equity = self._run_trend_atr(params, df)
        elif strategy_id == "breakout_pullback":
            equity = self._run_breakout_pullback(params, df)
        else:
            equity = self._buy_and_hold(df)
        equity = self.risk_engine.enforce(equity)
        return _metrics_from_equity(equity, df)

    def _run_trend_atr(self, params: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        fast = int(params.get("ema_fast", 20))
        slow = int(params.get("ema_slow", 100))
        donchian = int(params.get("donchian", 55))
        atr_len = int(params.get("atr_len", 14))
        stop_mult = float(params.get("stop_atr", 2.0))
        trail_mult = float(params.get("trail_atr", 3.0))

        close = data["Close"].astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        donchian_high = data["High"].rolling(donchian, min_periods=1).max()
        atr = _true_range(data).rolling(atr_len, min_periods=1).mean()
        long_signal = (ema_fast > ema_slow) & (close >= donchian_high)
        stop = close - atr * stop_mult
        trail = (close - atr * trail_mult).cummax()
        exit_signal = close < stop.combine(trail, max)
        position = pd.Series(0, index=data.index)
        holding = False
        for idx in range(len(data)):
            if not holding and long_signal.iat[idx]:
                holding = True
            elif holding and exit_signal.iat[idx]:
                holding = False
            position.iat[idx] = 1 if holding else 0
        returns = position.shift(1).fillna(0) * close.pct_change().fillna(0)
        equity = (1 + returns).cumprod()
        return _align_equity_index(equity, data)

    def _run_breakout_pullback(self, params: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        breakout = int(params.get("breakout_len", 40))
        pullback = float(params.get("pullback_pct", 0.05))
        atr_len = int(params.get("atr_len", 14))
        stop_mult = float(params.get("stop_atr", 2.0))

        close = data["Close"].astype(float)
        high_roll = data["High"].rolling(breakout, min_periods=1).max()
        entry = close >= high_roll.shift(1)
        atr = _true_range(data).rolling(atr_len, min_periods=1).mean()
        stop = close - atr * stop_mult
        pullback_exit = close.pct_change().rolling(3).min() <= -pullback
        position = pd.Series(0, index=data.index)
        holding = False
        for idx in range(len(data)):
            if not holding and entry.iat[idx]:
                holding = True
            elif holding and (pullback_exit.iat[idx] or close.iat[idx] < stop.iat[idx]):
                holding = False
            position.iat[idx] = 1 if holding else 0
        returns = position.shift(1).fillna(0) * close.pct_change().fillna(0)
        equity = (1 + returns).cumprod()
        return _align_equity_index(equity, data)

    def _buy_and_hold(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"].astype(float)
        returns = close.pct_change().fillna(0)
        return _align_equity_index((1 + returns).cumprod(), data)


def _true_range(data: pd.DataFrame) -> pd.Series:
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close_prev = data["Close"].shift(1).astype(float)
    tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.fillna(0)


def _align_equity_index(equity: pd.Series, data: pd.DataFrame) -> pd.Series:
    if "Timestamp" in data.columns:
        index = pd.to_datetime(data["Timestamp"], utc=True)
        equity.index = index
    return equity


def _metrics_from_equity(equity: pd.Series, data: pd.DataFrame) -> Metrics:
    returns = equity.pct_change().fillna(0)
    total_return = float(equity.iloc[-1] - 1)
    periods = len(returns)
    years = max(periods / 252, 1e-9)
    cagr = float((1 + total_return) ** (1 / years) - 1)
    drawdown = (equity / equity.cummax() - 1).min()
    downside = returns[returns < 0]
    sharpe = float(returns.mean() / (returns.std(ddof=0) + 1e-9) * np.sqrt(252))
    sortino = float(returns.mean() / (downside.std(ddof=0) + 1e-9) * np.sqrt(252)) if len(downside) else sharpe
    calmar = float(cagr / abs(drawdown) if drawdown else 0)
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    avg_win = float(returns[returns > 0].mean() if wins else 0.0)
    avg_loss = float(returns[returns < 0].mean() if losses else 0.0)
    turnover = float((returns.abs()).sum())
    fees_paid = float(turnover * 0.0005)
    timeframe = _infer_timeframe(data)
    metrics = Metrics(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=float(abs(drawdown)),
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        hit_rate=float(wins / max(wins + losses, 1)),
        avg_win=avg_win,
        avg_loss=avg_loss,
        trades=int((position_changes := (returns != 0).sum())),
        turnover=turnover,
        fees_paid=fees_paid,
        start=data["Timestamp"].iloc[0] if "Timestamp" in data else datetime.utcnow(),
        end=data["Timestamp"].iloc[-1] if "Timestamp" in data else datetime.utcnow(),
        timeframe=timeframe,
    )
    return metrics


def _infer_timeframe(data: pd.DataFrame) -> str:
    if "Timestamp" not in data:
        return "1d"
    diffs = data["Timestamp"].diff().dropna().dt.total_seconds() / 60
    avg = diffs.median() if not diffs.empty else 1440
    mapping = {1: "1m", 15: "15m", 60: "1h", 180: "3h", 1440: "1d"}
    closest = min(mapping, key=lambda minutes: abs(minutes - avg))
    return mapping[closest]


@dataclass(slots=True)
class BacktestService:
    """Wrapper that presents a consistent interface."""

    runner: Any | None = None
    risk_engine: RiskEngine | None = None

    def __post_init__(self) -> None:
        if self.risk_engine is None:
            self.risk_engine = RiskEngine()
        if self.runner is None:
            if BacktestRunner is not None:
                self.runner = BacktestRunner()
            else:
                self.runner = SimpleBacktestRunner(self.risk_engine)

    def run(self, strategy_id: str, params: Dict[str, Any], data: pd.DataFrame) -> Metrics:
        LOGGER.info("Running backtest for %s", strategy_id)
        if isinstance(self.runner, SimpleBacktestRunner):
            return self.runner.run(strategy_id, params, data)
        result = self.runner.run(strategy_id=strategy_id, params=params, data=data)
        if isinstance(result, Metrics):
            return result
        return Metrics.parse_obj(result)

    def buy_and_hold(self, data: pd.DataFrame) -> Metrics:
        equity = SimpleBacktestRunner(self.risk_engine)._buy_and_hold(data)
        return _metrics_from_equity(equity, data)
