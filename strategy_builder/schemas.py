"""Pydantic schemas used across the strategy builder package."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .pydantic_shim import BaseModel, Field, condecimal, validator


class TrendATRParams(BaseModel):
    """Parameter schema for the Trend ATR strategy."""

    ema_fast: int = Field(..., ge=2, lt=400)
    ema_slow: int = Field(..., gt=5, le=600)
    donchian: int = Field(..., ge=10, le=200)
    atr_len: int = Field(default=14, ge=5, le=50)
    stop_atr: float = Field(..., gt=0, le=10)
    trail_atr: float = Field(..., gt=0, le=10)

    @validator("ema_slow")
    def _ema_order(cls, v: int, values: Dict[str, Any]) -> int:  # pragma: no cover - simple validation
        fast = values.get("ema_fast")
        if fast is not None and v <= fast:
            raise ValueError("ema_slow must be greater than ema_fast")
        return v


class BreakoutPullbackParams(BaseModel):
    """Parameter schema for the breakout + pullback strategy."""

    breakout_len: int = Field(..., ge=10, le=200)
    pullback_pct: float = Field(..., ge=0.0, le=0.2)
    atr_len: int = Field(default=14, ge=5, le=50)
    stop_atr: float = Field(..., gt=0, le=10)
    trail_atr: float = Field(..., gt=0, le=10)


class RegimeParams(BaseModel):
    """Strategy composed of trend-following and mean-reversion components."""

    trend: TrendATRParams
    mean_reversion: BreakoutPullbackParams
    regime_switch_threshold: float = Field(..., ge=0.0, le=1.0)


class Metrics(BaseModel):
    """Performance metrics computed locally."""

    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    trades: int
    turnover: float
    fees_paid: float
    start: datetime
    end: datetime
    timeframe: str


class ParamProposal(BaseModel):
    """LLM parameter proposal payload."""

    params: Dict[str, Any]
    predicted_metrics: Dict[str, Any]
    rationale: str
    confidence: condecimal(ge=Decimal("0"), le=Decimal("1"))


class OptimizeResponse(BaseModel):
    """Response envelope returned by the optimization prompt."""

    proposals: List[ParamProposal]
    notes: Optional[str] = ""


class NodeType(str, Enum):
    """Supported node types within a strategy graph."""

    INDICATOR = "indicator"
    FEATURE = "feature"
    SIGNAL = "signal"
    POSITION = "position"
    RISK = "risk"


class Node(BaseModel):
    """A vertex within the strategy computation graph."""

    id: str
    type: NodeType
    op: str
    inputs: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class StrategyGraph(BaseModel):
    """Collection of nodes describing computation order."""

    nodes: List[Node]
    outputs: List[str]
