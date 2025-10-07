"""Lightweight stand-in for pydantic used in the strategy builder tests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
import sys
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, get_args, get_origin


class ValidationError(Exception):
    """Raised when validation fails."""


class _UndefinedType:
    pass


_UNDEFINED = _UndefinedType()


@dataclass
class FieldInfo:
    default: Any
    default_factory: Callable[[], Any] | None
    metadata: Dict[str, Any]


def Field(
    default: Any = _UNDEFINED,
    *,
    default_factory: Callable[[], Any] | None = None,
    **metadata: Any,
) -> FieldInfo:
    return FieldInfo(default=default, default_factory=default_factory, metadata=metadata)


T = TypeVar("T", bound="BaseModel")


class BaseModel:
    __field_defaults__: Dict[str, Any] = {}
    __validators__: Dict[str, List[Callable[[Type["BaseModel"], Any, Dict[str, Any]], Any]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__field_defaults__ = {}
        cls.__validators__ = {}
        annotations = getattr(cls, "__annotations__", {})
        for name in annotations:
            default = getattr(cls, name, _UNDEFINED)
            if isinstance(default, FieldInfo):
                if default.default is not _UNDEFINED:
                    cls.__field_defaults__[name] = default.default
                    setattr(cls, name, default.default)
                elif default.default_factory is not None:
                    cls.__field_defaults__[name] = default.default_factory
                    setattr(cls, name, default.default_factory)
                else:
                    cls.__field_defaults__[name] = _UNDEFINED
                    setattr(cls, name, None)
            else:
                cls.__field_defaults__[name] = default
        for attr in dir(cls):
            func = getattr(cls, attr)
            fields = getattr(func, "__validator_fields__", None)
            if fields:
                for field in fields:
                    cls.__validators__.setdefault(field, []).append(func)

    def __init__(self, **data: Any) -> None:
        self._data: Dict[str, Any] = {}
        for field, default in self.__field_defaults__.items():
            if field in data:
                value = data[field]
            elif default is not _UNDEFINED:
                value = default() if callable(default) else default
            else:
                raise ValidationError(f"Field '{field}' required")
            value = self._coerce(field, value)
            self._data[field] = value
            object.__setattr__(self, field, value)
        for field, validators in self.__validators__.items():
            if field in self._data:
                for func in validators:
                    self._data[field] = func(self.__class__, self._data[field], self._data)
                    object.__setattr__(self, field, self._data[field])

    def _coerce(self, field: str, value: Any) -> Any:
        annotation = getattr(self.__class__, "__annotations__", {}).get(field)
        if annotation and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if isinstance(value, dict):
                return annotation(**value)
        origin = get_origin(annotation)
        inner_type: Type[BaseModel] | None = None
        if origin in (list, List):
            args = get_args(annotation)
            if args:
                inner = args[0]
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    inner_type = inner
        elif isinstance(annotation, str) and annotation.startswith("List[") and annotation.endswith("]"):
            inner_name = annotation[5:-1]
            inner_type = getattr(sys.modules[self.__class__.__module__], inner_name, None)
            if not isinstance(inner_type, type) or not issubclass(inner_type, BaseModel):
                inner_type = None
        if inner_type and isinstance(value, list):
            return [inner_type(**item) if isinstance(item, dict) else item for item in value]
        return value

    def dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, BaseModel):
                result[key] = value.dict()
            else:
                result[key] = value
        return result

    def copy(self: T, update: Dict[str, Any] | None = None) -> T:
        payload = self.dict()
        if update:
            payload.update(update)
        return self.__class__(**payload)

    @classmethod
    def parse_obj(cls: Type[T], obj: Any) -> T:
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            raise ValidationError("Expected dict for parse_obj")
        return cls(**obj)

    @classmethod
    def parse_raw(cls: Type[T], raw: str) -> T:
        return cls.parse_obj(json.loads(raw))

    @classmethod
    def schema_json(cls, indent: int | None = None) -> str:
        schema = {
            "title": cls.__name__,
            "type": "object",
            "properties": {name: {"type": "string"} for name in getattr(cls, "__annotations__", {})},
        }
        return json.dumps(schema, indent=indent)

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            return self._data[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in getattr(self.__class__, "__annotations__", {}):
            self._data[key] = value
        object.__setattr__(self, key, value)


def validator(*fields: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__validator_fields__", fields)
        return func

    return decorator


def condecimal(**kwargs: Any) -> Type[Decimal]:  # pragma: no cover - trivial helper
    return Decimal


def create_model(name: str, **fields: Tuple[type, Any]) -> Type[BaseModel]:
    annotations = {key: field_type for key, (field_type, _) in fields.items()}
    namespace: Dict[str, Any] = {"__annotations__": annotations}
    for key, (_, default) in fields.items():
        namespace[key] = default
    return type(name, (BaseModel,), namespace)
