from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _try_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _load_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".json"}:
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            raise ImportError(
                "YAML params requested but PyYAML is not installed. "
                "Install with: pip install pyyaml  (or conda install -c conda-forge pyyaml)"
            )
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported params file extension: {suffix}. Use .json or .yaml/.yml")

    if not isinstance(data, dict):
        raise ValueError("Params root must be a dict. Example: {strategy_name: {param: value}}")

    return data


def _parse_value(raw: str) -> Any:
    """
    Parse CLI override values.
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


def _split_override(s: str) -> Tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --set override (missing '='): {s}")
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set override (empty key): {s}")
    return key, _parse_value(raw.strip())


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    dotted_key format: strategy.param or strategy.sub.param (we allow nesting).
    Example: embedding_cosine.threshold=0.97
    """
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Override key must include strategy and parameter, e.g. embedding_cosine.threshold. Got: {dotted_key}"
        )

    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            raise ValueError(f"Cannot set nested key under non-dict path: {p} in {dotted_key}")
        cur = cur[p]
    cur[parts[-1]] = value


def load_params(
    params_path: Optional[str],
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load {strategy_name: {param: value}} from YAML/JSON.
    Apply overrides like: ["embedding_cosine.threshold=0.97", "mmseqs2.threads=32"].
    """
    base: Dict[str, Any] = {}
    if params_path:
        base = _load_file(Path(params_path))

    overrides = overrides or []
    for ov in overrides:
        k, v = _split_override(ov)
        _set_nested(base, k, v)

    return base


def params_for_strategy(all_params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Return parameter dict for a given strategy, or {} if not present.
    Enforces it must be dict if present.
    """
    if strategy_name not in all_params:
        return {}
    v = all_params[strategy_name]
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"Params for strategy '{strategy_name}' must be a dict, got {type(v).__name__}.")
    return v
