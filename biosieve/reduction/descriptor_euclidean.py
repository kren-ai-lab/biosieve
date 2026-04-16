"""Descriptor-space reduction strategy based on Euclidean neighborhood pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import polars as pl

from biosieve.reduction.backends.descriptor_backend import (
    extract_descriptor_matrix,
    infer_descriptor_columns,
)
from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)
DESCRIPTOR_PREVIEW_LIMIT = 10


class _NearestNeighborsModel(Protocol):
    def fit(self, X: np.ndarray) -> object: ...

    def radius_neighbors(
        self, X: np.ndarray, *, radius: float, return_distance: bool
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _NearestNeighborsFactory(Protocol):
    def __call__(self, *, metric: str, algorithm: str, n_jobs: int) -> _NearestNeighborsModel: ...


def _try_import_sklearn_nn() -> _NearestNeighborsFactory | None:
    try:
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

        return cast("_NearestNeighborsFactory", NearestNeighbors)
    except ImportError:
        return None


def _zscore_fit_transform(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def _validate_inputs(
    *,
    df: pl.DataFrame,
    cols: Columns,
    threshold: float,
    metric: str,
    n_jobs: int,
) -> None:
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns}"
        raise ValueError(msg)
    if threshold < 0:
        msg = "threshold must be >= 0"
        raise ValueError(msg)
    if metric != "euclidean":
        msg = "v0.1 supports metric='euclidean' only"
        raise ValueError(msg)
    if n_jobs < 1:
        msg = "n_jobs must be >= 1"
        raise ValueError(msg)


def _build_work_ids(df: pl.DataFrame, id_col: str) -> tuple[pl.DataFrame, list[str]]:
    work = df.clone().sort(id_col, maintain_order=True)
    ids = work[id_col].cast(pl.String).to_list()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


def _reduce_with_sklearn(
    *,
    ids: list[str],
    X: np.ndarray,
    threshold: float,
    n_jobs: int,
) -> tuple[set[str], dict[str, str], dict[str, float], dict[str, str]] | None:
    removed: set[str] = set()
    rep_of: dict[str, str] = {}
    score_of: dict[str, float] = {}
    cluster_of: dict[str, str] = {}
    nn_factory = _try_import_sklearn_nn()
    if nn_factory is None:
        return None
    nn = nn_factory(metric="euclidean", algorithm="auto", n_jobs=n_jobs)
    nn.fit(X)
    for i, rep_id in enumerate(ids):
        if rep_id in removed:
            continue
        rep_cluster = f"deuc:{rep_id}"
        dist, ind = nn.radius_neighbors(X[i : i + 1], radius=float(threshold), return_distance=True)
        pairs = sorted(zip(dist[0].tolist(), ind[0].tolist(), strict=False), key=lambda x: (x[0], x[1]))
        for d, j in pairs:
            if j == i:
                continue
            nbr_id = ids[j]
            if nbr_id in removed:
                continue
            removed.add(nbr_id)
            rep_of[nbr_id] = rep_id
            score_of[nbr_id] = float(d)
            cluster_of[nbr_id] = rep_cluster
    return removed, rep_of, score_of, cluster_of


def _reduce_bruteforce(
    *,
    ids: list[str],
    X: np.ndarray,
    threshold: float,
    removed: set[str],
    rep_of: dict[str, str],
    score_of: dict[str, float],
    cluster_of: dict[str, str],
) -> None:
    for i, rep_id in enumerate(ids):
        if rep_id in removed:
            continue
        rep_cluster = f"deuc:{rep_id}"
        dists = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        order = np.argsort(dists, kind="mergesort")
        for j in order:
            if j == i:
                continue
            if float(dists[j]) > threshold:
                break
            nbr_id = ids[int(j)]
            if nbr_id in removed:
                continue
            removed.add(nbr_id)
            rep_of[nbr_id] = rep_id
            score_of[nbr_id] = float(dists[j])
            cluster_of[nbr_id] = rep_cluster


@dataclass(frozen=True)
class DescriptorEuclideanReducer:
    r"""Greedy redundancy reduction in descriptor space using Euclidean distance.

    This reducer removes near-duplicate samples based on Euclidean distance between
    descriptor vectors (e.g., tabular physicochemical descriptors). It uses a deterministic
    greedy policy:

    Greedy policy:
    1) Sort rows by `cols.id_col` (stable).
    2) Iterate in that order. First unseen id becomes representative.
    3) Remove any samples within radius `threshold` (distance <= threshold).

    Descriptor selection:
    - If `descriptor_cols` is provided, those exact columns are used.
    - Otherwise, columns starting with `descriptor_prefix` are used.

    Standardization:
    If `standardize=True`, descriptors are z-scored before distance calculations.
    This is recommended when descriptors have heterogeneous scales.

    Backend:
    - Uses sklearn `NearestNeighbors(radius_neighbors)` when available.
    - Falls back to O(N^2) brute force when sklearn is unavailable.

    Args:
        threshold: Euclidean radius. Samples at distance <= threshold are considered
            redundant. Must be >= 0.
        descriptor_prefix: Prefix used to infer descriptor columns (e.g., "desc_").
        descriptor_cols: Explicit list of descriptor columns to use (overrides prefix inference).
        standardize: If True, z-score descriptors prior to distance computations.
        metric: Distance metric. v0.1 supports only "euclidean".
        n_jobs: Parallel jobs for sklearn backend. Must be >= 1 (ignored in brute-force).
        dtype: Floating dtype for descriptor matrix ("float32" recommended).

    Returns:
        ReductionResult:
            Result containing representative-only data, a removed-to-representative
            mapping (with distance score), strategy name, and effective parameters.
            The reduced dataframe includes `descriptor_euclidean_cluster_id`
            (`deuc:<rep_id>`), and params include `stats` with
            `n_total/n_kept/n_removed/reduction_ratio/n_descriptors`.

    Raises:
        ValueError: If id column is missing, ids are duplicated, threshold < 0, n_jobs < 1,
        descriptor columns cannot be inferred, or descriptor matrix contains NaNs/non-numerics.
        ImportError: Not raised directly. If sklearn is missing, the reducer uses brute-force fallback.

    Notes:
        - This is a greedy algorithm: results depend on representative ordering
        (here: sorted by id for determinism).
        - `score` is a distance (not similarity). Lower values indicate more redundancy.
        - This reducer does not enforce biological leakage constraints (homology/structure);
        it only reduces redundancy in descriptor space.

    Examples:
        >>> biosieve reduce \\
        ...   --in dataset.csv \\
        ...   --out data_nr_desc.csv \\
        ...   --strategy descriptor_euclidean \\
        ...   --map map_desc.csv \\
        ...   --report report_desc.json \\
        ...   --params params.yaml

    """

    threshold: float = 1.0
    descriptor_prefix: str = "desc_"
    descriptor_cols: list[str] | None = None

    standardize: bool = True
    metric: str = "euclidean"  # v0.1: euclidean only
    n_jobs: int = 1
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "descriptor_euclidean"

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:
        """Reduce descriptor redundancy and return representatives plus mapping."""
        _validate_inputs(
            df=df,
            cols=cols,
            threshold=self.threshold,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        work, ids = _build_work_ids(df, cols.id_col)

        dcols = infer_descriptor_columns(
            work,
            prefix=self.descriptor_prefix,
            explicit_cols=self.descriptor_cols,
        )
        mat = extract_descriptor_matrix(work, dcols, dtype=self.dtype)
        X = mat.X

        # standardize if requested
        mu = None
        sd = None
        if self.standardize:
            X, mu, sd = _zscore_fit_transform(X)

        reduced = _reduce_with_sklearn(
            ids=ids,
            X=X,
            threshold=self.threshold,
            n_jobs=self.n_jobs,
        )
        if reduced is None:
            removed: set[str] = set()
            rep_of: dict[str, str] = {}
            score_of: dict[str, float] = {}
            cluster_of: dict[str, str] = {}
            _reduce_bruteforce(
                ids=ids,
                X=X,
                threshold=self.threshold,
                removed=removed,
                rep_of=rep_of,
                score_of=score_of,
                cluster_of=cluster_of,
            )
        else:
            removed, rep_of, score_of, cluster_of = reduced

        keep_ids = [sid for sid in ids if sid not in removed]
        kept_df = work.filter(pl.col(cols.id_col).cast(pl.String).is_in(set(keep_ids)))

        rows = []
        for rid, rep in rep_of.items():
            rows.append(
                {
                    "removed_id": rid,
                    "representative_id": rep,
                    "cluster_id": cluster_of.get(rid, f"deuc:{rep}"),
                    "score": score_of.get(rid),  # distance
                }
            )
        mapping = (
            pl.DataFrame(rows)
            if rows
            else pl.DataFrame(
                schema={
                    "removed_id": pl.String,
                    "representative_id": pl.String,
                    "cluster_id": pl.String,
                    "score": pl.Float64,
                }
            )
        )

        kept_df = kept_df.with_columns(
            (pl.lit("deuc:") + pl.col(cols.id_col).cast(pl.String)).alias("descriptor_euclidean_cluster_id")
        )

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_kept": kept_df.height,
            "n_removed": mapping.height,
            "reduction_ratio": float(kept_df.height / work.height) if work.height else 0.0,
            "n_descriptors": len(dcols),
        }

        return ReductionResult(
            df=kept_df,
            mapping=mapping,
            strategy=self.strategy,
            params={
                "threshold": self.threshold,
                "descriptor_prefix": self.descriptor_prefix,
                "descriptor_cols": self.descriptor_cols,
                "standardize": self.standardize,
                "metric": self.metric,
                "n_jobs": self.n_jobs,
                "dtype": self.dtype,
                "n_descriptors": len(dcols),
                "descriptor_cols_used": dcols[:DESCRIPTOR_PREVIEW_LIMIT]
                + (["..."] if len(dcols) > DESCRIPTOR_PREVIEW_LIMIT else []),
                "zscore_mean_saved": bool(mu is not None),
                "zscore_std_saved": bool(sd is not None),
                "stats": stats,
            },
        )
