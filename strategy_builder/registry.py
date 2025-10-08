"""Registry mapping strategy identifiers to parameter schemas and bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Type

from .pydantic_shim import BaseModel

from . import __version__
from .schemas import BreakoutPullbackParams, RegimeParams, TrendATRParams


@dataclass(slots=True)
class RegistryEntry:
    schema: Type[BaseModel]
    default_bounds: Mapping[str, tuple[float, float]]
    code_path_hint: str


class StrategyRegistry:
    """Central catalogue of strategies known by the builder."""

    def __init__(self) -> None:
        self._registry: MutableMapping[str, RegistryEntry] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        self.register(
            "trend_atr",
            TrendATRParams,
            {
                "ema_fast": (5, 80),
                "ema_slow": (50, 400),
                "donchian": (20, 120),
                "atr_len": (7, 28),
                "stop_atr": (1.0, 5.0),
                "trail_atr": (1.0, 6.0),
            },
            "strategies.trend_atr",
        )
        self.register(
            "breakout_pullback",
            BreakoutPullbackParams,
            {
                "breakout_len": (20, 120),
                "pullback_pct": (0.01, 0.12),
                "atr_len": (7, 28),
                "stop_atr": (1.0, 5.0),
                "trail_atr": (1.0, 6.0),
            },
            "strategies.breakout_pullback",
        )
        self.register(
            "regime_switch",
            RegimeParams,
            {
                "trend.ema_fast": (5, 60),
                "trend.ema_slow": (100, 400),
                "trend.donchian": (20, 80),
                "mean_reversion.breakout_len": (15, 90),
                "mean_reversion.pullback_pct": (0.01, 0.08),
                "regime_switch_threshold": (0.2, 0.8),
            },
            "strategies.regime_switch",
        )

    def register(
        self,
        strategy_id: str,
        schema: Type[BaseModel],
        bounds: Mapping[str, tuple[float, float]],
        code_path_hint: str,
    ) -> None:
        if strategy_id in self._registry:
            raise ValueError(f"Strategy '{strategy_id}' already registered")
        self._registry[strategy_id] = RegistryEntry(schema, bounds, code_path_hint)

    def get(self, strategy_id: str) -> RegistryEntry:
        try:
            return self._registry[strategy_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown strategy '{strategy_id}'") from exc

    def schema_json(self, strategy_id: str) -> str:
        entry = self.get(strategy_id)
        return entry.schema.schema_json(indent=2)

    def clamp(self, strategy_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.get(strategy_id)
        clamped = params.copy()
        for key, (lo, hi) in entry.default_bounds.items():
            path = key.split(".")
            self._clamp_path(clamped, path, lo, hi)
        return clamped

    def _clamp_path(self, data: Dict[str, Any], path: list[str], lo: float, hi: float) -> None:
        if len(path) == 1:
            key = path[0]
            if key in data:
                val = data[key]
                if isinstance(val, (int, float)):
                    data[key] = float(min(max(val, lo), hi))
        else:
            head, tail = path[0], path[1:]
            child = data.get(head)
            if isinstance(child, dict):
                self._clamp_path(child, tail, lo, hi)

    @property
    def version(self) -> str:
        return __version__


def get_registry() -> StrategyRegistry:
    return StrategyRegistry()
