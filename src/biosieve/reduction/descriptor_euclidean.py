from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.reduction.backends.descriptor_backend import (
    infer_descriptor_columns,
    extract_descriptor_matrix,
)

from biosieve.utils.logging import get_logger
log = get_logger(__name__)

def _try_import_sklearn_nn():
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        return NearestNeighbors
    except Exception:
        return None


def _zscore_fit_transform(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


@dataclass(frozen=True)
class DescriptorEuclideanReducer:
    """
    Greedy redundancy reduction in descriptor space using Euclidean distance.

    This reducer removes near-duplicate samples based on Euclidean distance between
    descriptor vectors (e.g., tabular physicochemical descriptors). It uses a deterministic
    greedy policy:

    Greedy policy
    -------------
    1) Sort rows by `cols.id_col` (stable).
    2) Iterate in that order. First unseen id becomes representative.
    3) Remove any samples within radius `threshold` (distance <= threshold).

    Descriptor selection
    --------------------
    - If `descriptor_cols` is provided, those exact columns are used.
    - Otherwise, columns starting with `descriptor_prefix` are used.

    Standardization
    ---------------
    If `standardize=True`, descriptors are z-scored before distance calculations.
    This is recommended when descriptors have heterogeneous scales.

    Backend
    -------
    - Uses sklearn `NearestNeighbors(radius_neighbors)` when available.
    - Falls back to O(N^2) brute force when sklearn is unavailable.

    Parameters
    ----------
    threshold:
        Euclidean radius. Samples at distance <= threshold are considered redundant.
        Must be >= 0.
    descriptor_prefix:
        Prefix used to infer descriptor columns (e.g., "desc_").
    descriptor_cols:
        Explicit list of descriptor columns to use (overrides prefix inference).
    standardize:
        If True, z-score descriptors prior to distance computations.
    metric:
        Distance metric. v0.1 supports only "euclidean".
    n_jobs:
        Parallel jobs for sklearn backend. Must be >= 1 (ignored in brute-force).
    dtype:
        Floating dtype for descriptor matrix ("float32" recommended).

    Returns
    -------
    ReductionResult
        - df:
            Reduced dataframe containing representatives only.
            Adds column `descriptor_euclidean_cluster_id` for convenience (`deuc:<rep_id>`).
        - mapping:
            DataFrame with columns:
              * removed_id
              * representative_id
              * cluster_id (rep-based, `deuc:<rep_id>`)
              * score (distance; smaller means closer/more similar)
        - strategy:
            "descriptor_euclidean"
        - params:
            Effective parameters plus `stats`:
              * n_total, n_kept, n_removed, reduction_ratio, n_descriptors

    Raises
    ------
    ValueError
        If id column is missing, ids are duplicated, threshold < 0, n_jobs < 1,
        descriptor columns cannot be inferred, or descriptor matrix contains NaNs/non-numerics.
    ImportError
        Not raised directly. If sklearn is missing, the reducer uses brute-force fallback.

    Notes
    -----
    - This is a greedy algorithm: results depend on representative ordering
      (here: sorted by id for determinism).
    - `score` is a distance (not similarity). Lower values indicate more redundancy.
    - This reducer does not enforce biological leakage constraints (homology/structure);
      it only reduces redundancy in descriptor space.

    Examples
    --------
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
    descriptor_cols: Optional[List[str]] = None

    standardize: bool = True
    metric: str = "euclidean"  # v0.1: euclidean only
    n_jobs: int = 1
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        return "descriptor_euclidean"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if cols.id_col not in df.columns:
            raise ValueError(f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}")

        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.metric != "euclidean":
            raise ValueError("v0.1 supports metric='euclidean' only")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")

        # deterministic order
        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        ids = work[cols.id_col].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids detected. IDs must be unique for deterministic reduction mapping.")

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

        NearestNeighbors = _try_import_sklearn_nn()

        removed: set[str] = set()
        rep_of: Dict[str, str] = {}
        score_of: Dict[str, float] = {}  # distance (lower = closer)
        cluster_of: Dict[str, str] = {}

        if NearestNeighbors is not None:
            nn = NearestNeighbors(metric="euclidean", algorithm="auto", n_jobs=self.n_jobs)
            nn.fit(X)

            for i, rep_id in enumerate(ids):
                if rep_id in removed:
                    continue

                rep_cluster = f"deuc:{rep_id}"
                dist, ind = nn.radius_neighbors(
                    X[i : i + 1], radius=float(self.threshold), return_distance=True
                )
                dist = dist[0]
                ind = ind[0]

                # deterministic: sort by distance asc, then by index
                pairs = sorted(zip(dist.tolist(), ind.tolist()), key=lambda x: (x[0], x[1]))

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

        else:
            # fallback brute force O(N^2)
            for i, rep_id in enumerate(ids):
                if rep_id in removed:
                    continue
                rep_cluster = f"deuc:{rep_id}"
                diffs = X - X[i]
                dists = np.sqrt((diffs * diffs).sum(axis=1))
                order = np.argsort(dists, kind="mergesort")
                for j in order:
                    if j == i:
                        continue
                    if float(dists[j]) > self.threshold:
                        break
                    nbr_id = ids[int(j)]
                    if nbr_id in removed:
                        continue
                    removed.add(nbr_id)
                    rep_of[nbr_id] = rep_id
                    score_of[nbr_id] = float(dists[j])
                    cluster_of[nbr_id] = rep_cluster

        keep_ids = [sid for sid in ids if sid not in removed]
        kept_df = work[work[cols.id_col].astype(str).isin(set(keep_ids))].copy()
        kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        rows = []
        for rid, rep in rep_of.items():
            rows.append(
                {
                    "removed_id": rid,
                    "representative_id": rep,
                    "cluster_id": cluster_of.get(rid, f"deuc:{rep}"),
                    "score": score_of.get(rid, None),  # distance
                }
            )
        mapping = pd.DataFrame(rows, columns=["removed_id", "representative_id", "cluster_id", "score"])

        kept_df["descriptor_euclidean_cluster_id"] = kept_df[cols.id_col].astype(str).apply(lambda x: f"deuc:{x}")

        stats: Dict[str, Any] = {
            "n_total": int(len(work)),
            "n_kept": int(len(kept_df)),
            "n_removed": int(len(mapping)),
            "reduction_ratio": float(len(kept_df) / len(work)) if len(work) else 0.0,
            "n_descriptors": int(len(dcols)),
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
                "descriptor_cols_used": dcols[:10] + (["..."] if len(dcols) > 10 else []),
                "zscore_mean_saved": bool(mu is not None),
                "zscore_std_saved": bool(sd is not None),
                "stats": stats,
            },
        )
