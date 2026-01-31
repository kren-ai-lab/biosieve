from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.reduction.backends.descriptor_backend import (
    infer_descriptor_columns,
    extract_descriptor_matrix,
)


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

    Typical usage:
      - descriptors are columns in the dataset (e.g., desc_000, desc_001, ...)

    Greedy policy:
      - sort by id deterministically
      - keep first-seen representative
      - remove samples within radius (distance <= threshold)

    Notes:
      - If standardize=True, Euclidean distances are in z-score space (recommended).
      - Uses sklearn NearestNeighbors radius_neighbors for efficiency.
    """

    threshold: float = 1.0              # Euclidean radius
    descriptor_prefix: str = "desc_"
    descriptor_cols: Optional[List[str]] = None

    standardize: bool = True
    metric: str = "euclidean"           # in v1: euclidean only (clean). could extend to manhattan later.
    n_jobs: int = 1
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        return "descriptor_euclidean"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.metric != "euclidean":
            raise ValueError("v1 supports metric='euclidean' only")

        # deterministic order
        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

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

        # greedy bookkeeping
        ids = work[cols.id_col].astype(str).tolist()
        removed: set[str] = set()
        rep_of: Dict[str, str] = {}
        score_of: Dict[str, float] = {}      # store distance (lower is closer)
        cluster_of: Dict[str, str] = {}

        # backend: sklearn radius neighbors
        if NearestNeighbors is not None:
            nn = NearestNeighbors(metric="euclidean", algorithm="auto", n_jobs=self.n_jobs)
            nn.fit(X)

            for i, rep_id in enumerate(ids):
                if rep_id in removed:
                    continue

                rep_cluster = f"deuc:{rep_id}"
                # all neighbors within radius (including itself)
                dist, ind = nn.radius_neighbors(X[i : i + 1], radius=float(self.threshold), return_distance=True)
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
            # fallback brute force O(N^2) – ok only for small N
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

        # kept = those not removed
        keep_ids = [sid for sid in ids if sid not in removed]
        kept_df = work[work[cols.id_col].astype(str).isin(set(keep_ids))].copy()
        kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        # mapping
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

        # optional: store cluster id for kept reps
        kept_df["descriptor_euclidean_cluster_id"] = kept_df[cols.id_col].astype(str).apply(lambda x: f"deuc:{x}")

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
            },
        )
