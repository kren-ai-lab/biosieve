"""Shared helpers for CLI and runner plumbing."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl

JSONScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]  # noqa: UP040


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with a trailing Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def ensure_parent(path: str | Path | None) -> None:
    """Create a parent directory for a file path if needed."""
    if path is None:
        return
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload using UTF-8 and pretty indentation."""
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_jsonable(x: object) -> JSONValue:
    """Convert objects into a JSON-serializable representation."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [safe_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe_jsonable(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        return safe_jsonable(asdict(x))
    return str(x)


def normalize_read_csv_kwargs(read_csv_kwargs: dict[str, object] | None) -> dict[str, object]:
    """Normalize user-facing CSV kwargs into the Polars shape used internally."""
    kwargs = dict(read_csv_kwargs or {})
    if "sep" in kwargs and "separator" not in kwargs:
        kwargs["separator"] = kwargs.pop("sep")
    return kwargs


def validate_unique_ids(df: pl.DataFrame, id_col: str) -> None:
    """Fail fast when the dataset id column is missing or not unique."""
    if id_col not in df.columns:
        msg = f"Missing id column '{id_col}' in input data. Columns: {df.columns}"
        raise ValueError(msg)

    n_in = df.height
    unique_ids = df[id_col].cast(pl.String).n_unique()
    if unique_ids != n_in:
        msg = (
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{id_col}'."
        )
        raise ValueError(msg)
