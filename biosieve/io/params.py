"""Parameter loading and override utilities for strategy configuration."""

from __future__ import annotations

import dataclasses
import importlib
import json
import types
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from biosieve.core.common import PATH_FIELDS

if TYPE_CHECKING:
    import os
    from types import ModuleType

Issue = tuple[str, str]  # (dotted path, message); an empty path means the file root

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
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {path}: {exc}"
            raise ValueError(msg) from exc
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
    params_path: str | os.PathLike[str] | Mapping[str, object] | None,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    """Load strategy parameters from JSON/YAML (or an in-memory mapping) and apply CLI overrides."""
    base: dict[str, object] = {}
    if isinstance(params_path, Mapping):
        base = dict(cast("Mapping[str, object]", params_path))
    elif params_path:
        base = _load_file(Path(params_path))

    overrides = overrides or []
    for ov in overrides:
        k, v = _split_override(ov)
        _set_nested(base, k, v)

    return base


def _type_ok(value: object, hint: Any) -> bool:  # noqa: ANN401, PLR0911
    """Return whether a parsed config value fits a dataclass field annotation."""
    origin = get_origin(hint)
    if hint is Any or hint is object:
        return True
    if origin in {Union, types.UnionType}:
        return any(_type_ok(value, h) for h in get_args(hint))
    if origin is Literal:
        return value in get_args(hint)
    if hint is type(None):
        return value is None
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if origin is list:
        (item,) = get_args(hint)
        return isinstance(value, list) and all(_type_ok(v, item) for v in value)
    if origin is dict:
        return isinstance(value, dict)
    return isinstance(value, hint) if isinstance(hint, type) else True


def read_config(
    path: str | os.PathLike[str] | Mapping[str, object] | None, overrides: list[str] | None = None
) -> tuple[dict[str, object], list[Issue]]:
    """Load a params file plus ``--set`` overrides; a problem is returned as an issue, not raised."""
    try:
        return load_params(path, overrides), []
    except (OSError, ValueError, TypeError, ImportError) as exc:
        return {}, [("", str(exc))]


def check_params(all_params: Mapping[str, object], strategy: str, cls: type[Any]) -> list[Issue]:
    """Validate ``{strategy: {param: value}}`` against the strategy dataclass; return every problem."""
    issues: list[Issue] = [
        (name, f"block for strategy '{name}' but --strategy is '{strategy}'")
        for name in all_params
        if name != strategy
    ]
    block = all_params.get(strategy)
    if block is None:
        return issues
    if not isinstance(block, Mapping):
        return [*issues, (strategy, f"expected a mapping, got {type(block).__name__}")]

    hints: dict[str, Any] = get_type_hints(cls)
    allowed = {f.name for f in dataclasses.fields(cls)}
    for key, value in cast("Mapping[str, object]", block).items():
        where = f"{strategy}.{key}"
        if key in PATH_FIELDS:
            issues.append((where, f"input paths are not config keys; use {PATH_FIELDS[key]}"))
        elif key not in allowed:
            issues.append((where, f"unknown parameter; allowed: {sorted(allowed - PATH_FIELDS.keys())}"))
        elif not _type_ok(value, hints[key]):
            issues.append((where, f"expected {hints[key]}, got {value!r}"))
    return issues
