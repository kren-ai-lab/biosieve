from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType


def _try_import_yaml() -> ModuleType | None:
    try:
        return importlib.import_module("yaml")
    except Exception:
        return None


def _load_file(path: Path) -> dict[str, object]:
    if not path.exists():
        msg = f"Params file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".json"}:
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            msg = (
                "YAML params requested but PyYAML is not installed. "
                "Install with: pip install pyyaml  (or conda install -c conda-forge pyyaml)"
            )
            raise ImportError(
                msg
            )
        data = yaml.safe_load(text) or {}
    else:
        msg = f"Unsupported params file extension: {suffix}. Use .json or .yaml/.yml"
        raise ValueError(msg)

    if not isinstance(data, dict):
        msg = "Params root must be a dict. Example: {strategy_name: {param: value}}"
        raise ValueError(msg)

    return cast("dict[str, object]", data)


def _parse_value(raw: str) -> object:
    """Parse CLI override values.
    Accepts JSON-like scalars: true/false/null, numbers, strings.
    Also accepts quoted strings.

    Examples:
      0.95 -> float
      16 -> int
      true -> bool
      "abc" -> str
      [1,2] -> list

    """
    # try json first (handles numbers, bool, null, lists, dicts, quoted strings)
    try:
        return json.loads(raw)
    except Exception:
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
    """dotted_key format: strategy.param or strategy.sub.param (we allow nesting).
    Example: embedding_cosine.threshold=0.97
    """
    parts = dotted_key.split(".")
    if len(parts) < 2:
        msg = (
            "Override key must include strategy and parameter, "
            f"e.g. embedding_cosine.threshold. Got: {dotted_key}"
        )
        raise ValueError(
            msg
        )

    cur: dict[str, object] = d
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            msg = f"Cannot set nested key under non-dict path: {p} in {dotted_key}"
            raise ValueError(msg)
        cur = cast("dict[str, object]", cur[p])
    cur[parts[-1]] = value


def load_params(
    params_path: str | None,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    """Load {strategy_name: {param: value}} from YAML/JSON.
    Apply overrides like: ["embedding_cosine.threshold=0.97", "mmseqs2.threads=32"].
    """
    base: dict[str, object] = {}
    if params_path:
        base = _load_file(Path(params_path))

    overrides = overrides or []
    for ov in overrides:
        k, v = _split_override(ov)
        _set_nested(base, k, v)

    return base


def params_for_strategy(all_params: Mapping[str, object], strategy_name: str) -> dict[str, object]:
    """Return parameter dict for a given strategy, or {} if not present.
    Enforces it must be dict if present.
    """
    if strategy_name not in all_params:
        return {}
    v = all_params[strategy_name]
    if v is None:
        return {}
    if not isinstance(v, dict):
        msg = f"Params for strategy '{strategy_name}' must be a dict, got {type(v).__name__}."
        raise ValueError(msg)
    return cast("dict[str, object]", v)
