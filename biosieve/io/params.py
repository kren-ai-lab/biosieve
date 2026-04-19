"""Parameter loading and override utilities for strategy configuration."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

log = logging.getLogger(__name__)

MIN_OVERRIDE_KEY_PARTS = 2


def _try_import_yaml() -> ModuleType | None:
    try:
        return importlib.import_module("yaml")
    except ImportError:
        return None


def _load_file(path: Path) -> dict[str, object]:
    if not path.exists():
        msg = f"Params file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            msg = (
                "YAML params requested but PyYAML is not installed in this environment. "
                "Install `pyyaml` in the same environment as biosieve, or use JSON params instead."
            )
            raise ImportError(msg)
        data = yaml.safe_load(text) or {}
    else:
        msg = f"Unsupported params file extension: {suffix}. Use .json or .yaml/.yml"
        raise ValueError(msg)

    if not isinstance(data, dict):
        msg = "Params root must be a dict. Example: {strategy_name: {param: value}}"
        raise TypeError(msg)

    return cast("dict[str, object]", data)


def _parse_value(raw: str) -> object:
    """Parse a CLI override value into a Python object.

    The parser first tries JSON decoding, then falls back to a raw string.
    """
    # try json first (handles numbers, bool, null, lists, dicts, quoted strings)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: plain string
        return raw


def _split_override(s: str) -> tuple[str, object]:
    if "=" not in s:
        msg = f"Invalid --set override (missing '='): {s}"
        raise ValueError(msg)
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        msg = f"Invalid --set override (empty key): {s}"
        raise ValueError(msg)
    return key, _parse_value(raw.strip())


def _set_nested(d: dict[str, object], dotted_key: str, value: object) -> None:
    """Set a nested dictionary value from a dotted key path.

    Example: ``embedding_cosine.threshold=0.97``.
    """
    parts = dotted_key.split(".")
    if len(parts) < MIN_OVERRIDE_KEY_PARTS:
        msg = (
            "Override key must include strategy and parameter, "
            f"e.g. embedding_cosine.threshold. Got: {dotted_key}"
        )
        raise ValueError(msg)

    cur: dict[str, object] = d
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            msg = f"Cannot set nested key under non-dict path: {p} in {dotted_key}"
            raise TypeError(msg)
        cur = cast("dict[str, object]", cur[p])
    cur[parts[-1]] = value


def load_params(
    params_path: str | None,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    """Load strategy parameters from JSON/YAML and apply CLI overrides."""
    base: dict[str, object] = {}
    if params_path:
        base = _load_file(Path(params_path))

    overrides = overrides or []
    for ov in overrides:
        k, v = _split_override(ov)
        _set_nested(base, k, v)

    return base


def params_for_strategy(all_params: Mapping[str, object], strategy_name: str) -> dict[str, object]:
    """Return the parameter mapping for one strategy, or an empty mapping."""
    if strategy_name not in all_params:
        if all_params:
            known = sorted(all_params.keys())
            log.warning(
                "No params found for strategy '%s' in params file. "
                "Keys present: %s. Check that the top-level key matches the strategy name.",
                strategy_name,
                known,
            )
        return {}
    v = all_params[strategy_name]
    if v is None:
        return {}
    if not isinstance(v, dict):
        msg = f"Params for strategy '{strategy_name}' must be a dict, got {type(v).__name__}."
        raise TypeError(msg)
    return cast("dict[str, object]", v)
