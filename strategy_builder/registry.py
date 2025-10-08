"""Registry mapping strategy identifiers to parameter schemas and bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Type

from strategies import STRATEGY_REGISTRY as APP_STRATEGIES
from strategies import StrategyDefinition

from .pydantic_shim import BaseModel, Field, create_model

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
        self._register_legacy_strategies()
        for strategy_id, definition in APP_STRATEGIES.items():
            if strategy_id in self._registry:
                continue
            try:
                schema, bounds = _model_from_definition(definition)
            except Exception:
                # Skip strategies that cannot be adapted into schema form.
                continue
            self.register(strategy_id, schema, bounds, definition.__module__)

    def _register_legacy_strategies(self) -> None:
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


def _model_from_definition(
    definition: StrategyDefinition,
) -> tuple[Type[BaseModel], Dict[str, tuple[float, float]]]:
    controls = definition.controls or {}
    defaults = dict(definition.default_params or {})
    fields: Dict[str, tuple[type, Any]] = {}
    bounds: Dict[str, tuple[float, float]] = {}

    for name, spec in controls.items():
        dtype = spec.get("dtype", float)
        if dtype in {bool, str}:
            # Skip non-numeric controls for the optimizer schema.
            continue
        annotation: type = float
        if dtype in {int}:
            annotation = int
        default = defaults.get(name, spec.get("value"))
        if default is None:
            default = 0
        field_kwargs: Dict[str, Any] = {}
        if spec.get("min") is not None:
            field_kwargs["ge"] = spec["min"]
        if spec.get("max") is not None:
            field_kwargs["le"] = spec["max"]
        fields[name] = (annotation, Field(default=default, **field_kwargs))
        bounds[name] = _bounds_from_spec(spec, float(default))

    if not fields:
        # Provide at least one numeric parameter so prompts remain valid.
        fields["tuning_factor"] = (float, Field(default=1.0, ge=0.0, le=5.0))
        bounds["tuning_factor"] = (0.0, 5.0)

    model_name = f"{definition.key.title().replace('_', '')}Params"
    model = create_model(model_name, **fields)  # type: ignore[arg-type]
    return model, bounds


def _bounds_from_spec(spec: Mapping[str, Any], default: float) -> tuple[float, float]:
    min_value = spec.get("min")
    max_value = spec.get("max")
    if min_value is not None and max_value is not None:
        return float(min_value), float(max_value)
    spread = max(abs(default) * 0.5, 1.0)
    return float(default - spread), float(default + spread)
