from __future__ import annotations

import inspect
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


def instantiate_strategy(cls: Type[T], params: Dict[str, Any]) -> T:
    """
    Instantiate a strategy class (splitter/reducer) from validated parameters.

    This helper enforces strict parameter validation to prevent silent typos:
    unknown keys in `params` will raise an error.

    Parameters
    ----------
    cls:
        Strategy class (typically a dataclass-based splitter/reducer).
    params:
        Dictionary of keyword arguments used to instantiate `cls`.

    Returns
    -------
    T
        Instantiated strategy object.

    Raises
    ------
    ValueError
        If `params` is not a dict, or if it contains unknown keys for the class
        constructor signature.

    Notes
    -----
    - This function validates keys against the `__init__` signature (via `inspect.signature`).
    - It does not validate value ranges; that is the responsibility of each strategy
      (e.g., ensuring `test_size` is in (0,1), thresholds are valid, etc.).
    - Designed to be used by runners/CLI to enforce reproducible, fail-fast behaviour.

    Examples
    --------
    >>> splitter = instantiate_strategy(RandomSplitter, {"test_size": 0.2, "seed": 13})
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    unknown = [k for k in params.keys() if k not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for {cls.__name__}: {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )

    return cls(**params)  # type: ignore[arg-type]
