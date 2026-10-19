"""Shared helpers for CLI and runner plumbing."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl

from biosieve.errors import ConfigError

JSONScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]  # noqa: UP040

# strategy field -> CLI flag that supplies it
PATH_FIELDS = {
    "embeddings_path": "--embeddings",
    "ids_path": "--embedding-ids",
    "edges_path": "--edges",
    "clusters_path": "--clusters",
}


def provenance_inputs(in_path: str | Path, paths: dict[str, str]) -> list[Path]:
    """List every file a run reads, including a sibling ``<stem>.ids.txt`` picked up implicitly."""
    inputs = [Path(in_path), *map(Path, paths.values())]
    embeddings = paths.get("embeddings_path")
    if embeddings and "ids_path" not in paths:
        sibling = Path(embeddings).with_suffix(".ids.txt")
        # a .npz carries its own ids, so the sibling is only read for .npy
        if Path(embeddings).suffix != ".npz" and sibling.exists():
            inputs.append(sibling)
    return inputs


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp, seconds precision, with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def read_table(
    path: str | Path, id_col: str, read_csv_kwargs: dict[str, object] | None = None
) -> pl.DataFrame:
    """Read a CSV or Parquet table (by extension); the id column is always a string."""
    if Path(path).suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
        return df.with_columns(pl.col(id_col).cast(pl.String)) if id_col in df.columns else df
    kwargs = normalize_read_csv_kwargs(read_csv_kwargs)
    kwargs["schema_overrides"] = {id_col: pl.String}
    return pl.read_csv(path, **cast("dict[str, Any]", kwargs))


def write_table(df: pl.DataFrame, path: str | Path, id_col: str) -> None:
    """Write a CSV or Parquet table (by extension) with the id column first."""
    ensure_parent(path)
    if id_col in df.columns:
        df = df.select(id_col, pl.exclude(id_col))
    if Path(path).suffix.lower() == ".parquet":
        df.write_parquet(path)
    else:
        df.write_csv(path)


def effective_params(strategy: object) -> dict[str, object]:
    """Return the strategy's parameters without the input paths the CLI supplies."""
    return {k: v for k, v in asdict(cast("Any", strategy)).items() if k not in PATH_FIELDS}


def check_paths(cls: type[Any], paths: dict[str, str]) -> None:
    """Reject input-path flags that the selected strategy does not use."""
    unused = [PATH_FIELDS[k] for k in paths if k not in {f.name for f in fields(cls)}]
    if unused:
        msg = f"{', '.join(unused)} not used by strategy '{cls.__name__}'"
        raise ConfigError(msg)


def validate_unique_ids(df: pl.DataFrame, id_col: str) -> None:
    """Fail fast when the dataset id column is missing or not unique."""
    if id_col not in df.columns:
        msg = f"Missing id column '{id_col}' in input data; set it with --id-column. Columns: {df.columns}"
        raise ValueError(msg)

    n_in = df.height
    unique_ids = df[id_col].cast(pl.String).n_unique()
    if unique_ids != n_in:
        msg = (
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{id_col}'."
        )
        raise ValueError(msg)
