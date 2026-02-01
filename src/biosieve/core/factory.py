from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")


def instantiate_strategy(cls: Type[T], params: Dict[str, Any]) -> T:
    """
    Instantiate a reducer/splitter class using params, but:
      - only allow kwargs that exist in the __init__ signature
      - raise on unknown keys (prevents silent typos)
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())

    # For normal classes, signature includes 'self' only in bound methods,
    # but for class constructors it's already excluded. Still safe:
    allowed.discard("self")

    unknown = [k for k in params.keys() if k not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for {cls.__name__}: {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )

    return cls(**params)  # type: ignore[arg-type]
