"""Configuration schemas for the AI trader strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_type_hints

T = TypeVar("T", bound="_ConfigBase")


@dataclass(slots=True)
class _ConfigBase:
    """Base class for simple dataclass driven configuration objects."""

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an instance from a dictionary, recursively instantiating nested configs."""

        field_values: Dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if field_name not in data:
                continue
            value = data[field_name]
            field_type = type_hints.get(field_name, field_def.type)
            if isinstance(value, dict) and hasattr(field_type, "from_dict"):
                field_values[field_name] = field_type.from_dict(value)  # type: ignore[arg-type]
            else:
                field_values[field_name] = value
        return cls(**field_values)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls: Type[T], path: Path | str) -> T:
        """Load configuration from a JSON file."""

        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = json.load(handle)
        return cls.from_dict(payload)


@dataclass(slots=True)
class RiskConfig(_ConfigBase):
    """Risk constraints mirroring the centralized risk engine configuration."""

    dd_lock_pct: float = -0.095
    daily_loss_lock_pct: float = -0.02
    single_asset_cap_pct: float = 0.25
    gross_cap_pct: float = 1.5


@dataclass(slots=True)
class FeesConfig(_ConfigBase):
    """Trading fee configuration expressed in basis points per side."""

    fees_bps_per_side: int = 5

    @property
    def commission(self) -> float:
        """Return the per-side commission in decimal form."""

        return self.fees_bps_per_side / 10000.0


@dataclass(slots=True)
class BoosterDailyConfig(_ConfigBase):
    """Configuration for the daily trend booster core strategy."""

    name: str = "btc_eur_daily_trend_booster"
    symbol: str = "BTC/EUR"
    bar_interval: str = "1d"
    filters_sma_len: int = 200
    filters_donchian_len: int = 55
    pos_strong: float = 1.3
    pos_weak: float = 0.3
    max_leverage: float = 1.5
    fees: FeesConfig = field(default_factory=FeesConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


@dataclass(slots=True)
class OverlayHourlyConfig(_ConfigBase):
    """Configuration for the hourly overlay strategy."""

    name: str = "btc_eur_hourly_overlay"
    symbol: str = "BTC/EUR"
    bar_interval: str = "1h"
    ema_fast: int = 8
    ema_slow: int = 34
    donchian_high: int = 10
    overlay_notional: float = 0.2
    max_total_leverage: float = 1.5
    fees: FeesConfig = field(default_factory=FeesConfig)


def load_config(path: Path | str, cls: Type[T]) -> T:
    """Load any of the configuration dataclasses from JSON."""

    return cls.from_json(path)


__all__ = [
    "RiskConfig",
    "FeesConfig",
    "BoosterDailyConfig",
    "OverlayHourlyConfig",
    "load_config",
]
