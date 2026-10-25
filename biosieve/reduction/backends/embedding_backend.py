"""Embedding loading helpers shared by embedding-based reducers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from biosieve.errors import ConfigError


@dataclass(frozen=True)
class EmbeddingStore:
    """Embedding matrix and aligned row identifiers."""

    ids: list[str]
    X: np.ndarray  # shape (N, D), float32/float64


def _read_ids(ids_path: Path, id_col: str) -> list[str]:
    """Read ids from ``.txt`` (one per line), ``.npz`` (``ids`` array) or a CSV with ``id_col``."""
    suffix = ids_path.suffix.lower()
    if suffix == ".txt":
        return ids_path.read_text(encoding="utf-8").split()
    if suffix == ".npz":
        with np.load(ids_path, allow_pickle=False) as z:
            return [str(x) for x in z["ids"]]
    df = pl.read_csv(ids_path, schema_overrides={id_col: pl.String})
    if id_col not in df.columns:
        msg = f"Embedding ids column '{id_col}' not found in {ids_path}. Columns: {df.columns}"
        raise ValueError(msg)
    return df[id_col].to_list()


def _load_array(ep: Path) -> tuple[np.ndarray, list[str] | None]:
    """Load the matrix from ``.npy`` or ``.npz`` (``values``); also return the ``.npz`` ``ids`` if present."""
    if ep.suffix.lower() != ".npz":
        return np.load(ep, allow_pickle=False), None
    with np.load(ep, allow_pickle=False) as z:
        return z["values"], ([str(x) for x in z["ids"]] if "ids" in z.files else None)


def _resolve_ids(ep: Path, ids_path: str | None, npz_ids: list[str] | None, id_col: str) -> list[str]:
    if ids_path is not None:
        ip = Path(ids_path)
        if not ip.exists():
            msg = f"Embedding ids file not found: {ip}"
            raise FileNotFoundError(msg)
        return _read_ids(ip, id_col)
    if npz_ids is not None:
        return npz_ids
    sibling = ep.with_suffix(".ids.txt")
    if sibling.exists():
        return _read_ids(sibling, id_col)
    msg = "Embedding ids not found: pass --embedding-ids (or use a .npz with 'ids' / <stem>.ids.txt)."
    raise ConfigError(msg)


def load_embeddings(
    embeddings_path: str | None,
    ids_path: str | None,
    ids_col: str = "id",
    dtype: str | None = None,
) -> EmbeddingStore:
    """Load embeddings and aligned ids; ids default to the ``.npz`` ``ids`` array or ``<stem>.ids.txt``."""
    if embeddings_path is None:
        msg = "This strategy requires --embeddings."
        raise ConfigError(msg)
    ep = Path(embeddings_path)
    if not ep.exists():
        msg = f"Embeddings file not found: {ep}"
        raise FileNotFoundError(msg)

    X, npz_ids = _load_array(ep)
    ids = _resolve_ids(ep, ids_path, npz_ids, ids_col)

    if X.ndim != 2:  # noqa: PLR2004
        msg = f"Embeddings must be 2D array (N,D). Got shape {X.shape}"
        raise ValueError(msg)
    if len(ids) != X.shape[0]:
        msg = (
            f"Mismatch: ids ({len(ids)}) vs embeddings rows ({X.shape[0]}). "
            "They must align 1-to-1 in the same order."
        )
        raise ValueError(msg)
    if len(set(ids)) != len(ids):
        msg = "Duplicate embedding ids; ids must be unique."
        raise ValueError(msg)

    return EmbeddingStore(ids=ids, X=X.astype(dtype, copy=False) if dtype is not None else X)
