from __future__ import annotations

from typing import Dict

from .base import StrategyDefinition
from .ema_trend import DEFAULT_PARAMS as EMA_DEFAULT_PARAMS
from .ema_trend import EMA_STRATEGY, ema, slope_pct


STRATEGY_REGISTRY: Dict[str, StrategyDefinition] = {EMA_STRATEGY.key: EMA_STRATEGY}
DEFAULT_STRATEGY_KEY = EMA_STRATEGY.key

__all__ = [
    "StrategyDefinition",
    "STRATEGY_REGISTRY",
    "DEFAULT_STRATEGY_KEY",
    "EMA_STRATEGY",
    "EMA_DEFAULT_PARAMS",
    "ema",
    "slope_pct",
]
