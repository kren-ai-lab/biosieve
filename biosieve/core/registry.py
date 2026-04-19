"""Strategy registry and lookup helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import import_module
from typing import Any

StrategyLike = str | type[Any]


def lazy_import_class(import_path: str) -> type[Any]:
    """Import a class from an import path in the form ``pkg.mod:ClassName``."""
    if ":" not in import_path:
        msg = f"Invalid import_path '{import_path}'. Expected format 'module:ClassName'."
        raise ValueError(msg)
    mod_name, cls_name = import_path.split(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate_strategy(cls: type[object], params: dict[str, object]) -> object:
    """Instantiate a strategy class with strict parameter validation."""
    params = params or {}

    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        unknown = set(params) - allowed
        if unknown:
            msg = f"Unknown parameters for {cls.__name__}: {sorted(unknown)}. Allowed: {sorted(allowed)}"
            raise ValueError(msg)
        return cls(**params)

    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())
    unknown = set(params) - allowed
    if unknown:
        msg = f"Unknown parameters for {cls.__name__}: {sorted(unknown)}. Allowed: {sorted(allowed)}"
        raise ValueError(msg)
    return cls(**params)


@dataclass
class StrategyRegistry:
    """Registry of available reducer and splitter strategies."""

    reducers: dict[str, StrategyLike] = field(default_factory=dict)
    splitters: dict[str, StrategyLike] = field(default_factory=dict)

    def has_reducer(self, name: str) -> bool:
        """Return whether a reducer strategy exists."""
        return name in self.reducers

    def has_splitter(self, name: str) -> bool:
        """Return whether a splitter strategy exists."""
        return name in self.splitters

    def list_reducers(self) -> dict[str, StrategyLike]:
        """Return a copy of registered reducers."""
        return dict(self.reducers)

    def list_splitters(self) -> dict[str, StrategyLike]:
        """Return a copy of registered splitters."""
        return dict(self.splitters)

    def get_reducer_import_path(self, name: str) -> str | None:
        """Return the registered reducer import path when still unresolved."""
        obj = self.reducers.get(name)
        return obj if isinstance(obj, str) else None

    def get_splitter_import_path(self, name: str) -> str | None:
        """Return the registered splitter import path when still unresolved."""
        obj = self.splitters.get(name)
        return obj if isinstance(obj, str) else None

    def get_reducer_class(self, name: str) -> type[Any]:
        """Resolve and return a reducer class by name."""
        if name not in self.reducers:
            msg = f"Unknown reducer strategy '{name}'. Available: {sorted(self.reducers)}"
            raise KeyError(msg)
        obj = self.reducers[name]
        if isinstance(obj, str):
            cls = lazy_import_class(obj)
            self.reducers[name] = cls
            return cls
        return obj

    def get_splitter_class(self, name: str) -> type[Any]:
        """Resolve and return a splitter class by name."""
        if name not in self.splitters:
            msg = f"Unknown splitter strategy '{name}'. Available: {sorted(self.splitters)}"
            raise KeyError(msg)
        obj = self.splitters[name]
        if isinstance(obj, str):
            cls = lazy_import_class(obj)
            self.splitters[name] = cls
            return cls
        return obj

    def create_reducer(self, name: str, params: dict[str, object] | None = None) -> object:
        """Resolve and instantiate a reducer by name."""
        return instantiate_strategy(self.get_reducer_class(name), params or {})

    def create_splitter(self, name: str, params: dict[str, object] | None = None) -> object:
        """Resolve and instantiate a splitter by name."""
        return instantiate_strategy(self.get_splitter_class(name), params or {})
