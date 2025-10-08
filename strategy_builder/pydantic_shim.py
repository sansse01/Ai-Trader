"""Compatibility layer that prefers the real pydantic package when available."""
from __future__ import annotations

try:  # pragma: no cover - runtime import guard
    from pydantic import BaseModel, Field, ValidationError, condecimal, create_model, validator
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    from ._pydantic_stub import (  # type: ignore F401
        BaseModel,
        Field,
        ValidationError,
        condecimal,
        create_model,
        validator,
    )

__all__ = [
    "BaseModel",
    "Field",
    "ValidationError",
    "condecimal",
    "create_model",
    "validator",
]
