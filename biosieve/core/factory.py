"""Factory helpers for strategy instantiation."""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass


def instantiate_strategy(cls: type[object], params: dict[str, object]) -> object:
    """Instantiate a strategy class with strict parameter validation.

    Unknown parameter keys raise ValueError to prevent silent typos.

    Args:
        cls: Strategy class (dataclass preferred).
        params: Parameter dictionary.

    Returns:
        Instantiated strategy object.

    Raises:
        ValueError: If unknown keys are provided.

    """
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
