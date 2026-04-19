"""Shared helpers for reduction strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

EMPTY_MAPPING_SCHEMA = {
    "removed_id": pl.String,
    "representative_id": pl.String,
    "cluster_id": pl.String,
    "score": pl.Float64,
}
SKLEARN_REQUIRED_MESSAGE = (
    "scikit-learn could not be imported, but it is a required biosieve dependency. "
    "Check that biosieve was installed correctly in this environment."
)


def empty_mapping_df() -> pl.DataFrame:
    """Return an empty mapping DataFrame with the stable schema used by reducers."""
    return pl.DataFrame(schema=EMPTY_MAPPING_SCHEMA)


class _NearestNeighborsModel(Protocol):
    def fit(self, X: object) -> object: ...

    def radius_neighbors(
        self,
        X: np.ndarray,
        *,
        radius: float,
        return_distance: bool,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _NearestNeighborsFactory(Protocol):
    def __call__(self, *, metric: str, algorithm: str, n_jobs: int) -> _NearestNeighborsModel: ...


def require_sklearn_neighbors(feature: str) -> _NearestNeighborsFactory:
    """Return sklearn.neighbors.NearestNeighbors or raise a consistent ImportError."""
    try:
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    except ImportError as e:
        msg = f"{feature} requires scikit-learn. {SKLEARN_REQUIRED_MESSAGE}"
        raise ImportError(msg) from e
    return cast("_NearestNeighborsFactory", NearestNeighbors)


def prepare_reduction_work(df: pl.DataFrame, id_col: str) -> tuple[pl.DataFrame, list[str]]:
    """Sort by id and validate uniqueness for deterministic reduction behavior."""
    work = df.clone().sort(id_col, maintain_order=True)
    ids = work[id_col].cast(pl.String).to_list()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


def build_mapping(
    removed_rows: Sequence[tuple[str, str, float | None]],
    *,
    cluster_prefix: str,
) -> pl.DataFrame:
    """Build the standard removed->representative mapping DataFrame."""
    if not removed_rows:
        return empty_mapping_df()

    rows = [
        {
            "removed_id": removed_id,
            "representative_id": representative_id,
            "score": score,
        }
        for removed_id, representative_id, score in removed_rows
    ]
    return (
        pl.DataFrame(rows)
        .with_columns(score=pl.col("score").cast(pl.Float64))
        .with_columns(cluster_id=pl.lit(f"{cluster_prefix}:") + pl.col("representative_id").cast(pl.String))
        .select(["removed_id", "representative_id", "cluster_id", "score"])
    )


def attach_cluster_ids(
    df: pl.DataFrame,
    *,
    id_col: str,
    column_name: str,
    cluster_prefix: str,
) -> pl.DataFrame:
    """Attach a cluster id column to a representative-only DataFrame."""
    return df.with_columns(**{column_name: pl.lit(f"{cluster_prefix}:") + pl.col(id_col).cast(pl.String)})


def build_reduction_stats(
    *,
    n_total: int,
    n_kept: int,
    **extra: object,
) -> dict[str, object]:
    """Build the common reduction stats payload used across reducers."""
    n_removed = int(n_total - n_kept)
    return {
        "n_total": int(n_total),
        "n_kept": int(n_kept),
        "n_removed": n_removed,
        "reduction_ratio": float(n_kept / n_total) if n_total else 0.0,
        **extra,
    }
