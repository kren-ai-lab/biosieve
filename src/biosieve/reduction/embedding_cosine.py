from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.reduction.backends.embedding_backend import load_embeddings, EmbeddingStore


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _try_import_sklearn_nn():
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        return NearestNeighbors
    except Exception:
        return None


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


@dataclass(frozen=True)
class EmbeddingCosineReducer:
    """
    Greedy redundancy reduction in embedding space using cosine similarity.

    Requires:
      - embeddings .npy (N,D)
      - ids .csv with same order (N)

    Greedy policy:
      - stable sort by id (deterministic)
      - keep first-seen representative
      - remove neighbors with cosine_sim >= threshold

    Neighbor backend:
      - FAISS (if installed) for fast inner-product search
      - else sklearn NearestNeighbors radius search (cosine distance)
    """

    embeddings_path: str = "embeddings.npy"
    ids_path: str = "embedding_ids.csv"
    ids_col: str = "id"

    threshold: float = 0.95  # cosine similarity
    use_faiss: bool = True   # auto-fallback if not available
    n_jobs: int = 1          # used for sklearn if available
    dtype: str = "float32"   # good default for speed/memory

    @property
    def strategy(self) -> str:
        return "embedding_cosine"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        store = load_embeddings(
            embeddings_path=self.embeddings_path,
            ids_path=self.ids_path,
            ids_col=self.ids_col,
            dtype=self.dtype,
        )

        # Map dataset ids -> embedding row index
        id_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(store.ids)}

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        work_ids = work[cols.id_col].astype(str).tolist()

        present = [sid for sid in work_ids if sid in id_to_idx]
        missing = [sid for sid in work_ids if sid not in id_to_idx]
        if len(present) == 0:
            raise ValueError(
                "None of the dataset ids were found in embedding ids file. "
                f"Example dataset id: {work_ids[0] if work_ids else 'EMPTY'}, "
                f"example embedding id: {store.ids[0]}"
            )

        # Subset embeddings to only rows present in dataset, keeping work order (deterministic)
        idxs = np.array([id_to_idx[sid] for sid in present], dtype=int)
        X = store.X[idxs]
        X = _l2_normalize(X)

        # We'll operate on the 'present' subset only; missing ids will be kept as standalone reps
        # (you can change this policy later; this is safest and reproducible)
        present_set = set(present)

        # Build neighbor search
        faiss = _try_import_faiss() if self.use_faiss else None
        NearestNeighbors = _try_import_sklearn_nn()

        # We use cosine distance radius r = 1 - threshold on normalized vectors.
        # For normalized vectors, cosine_sim = dot(u,v), cosine_dist = 1 - dot(u,v)
        radius = float(1.0 - self.threshold)

        # Prepare a mapping from present-id -> local index [0..M-1]
        present_id_to_local = {sid: i for i, sid in enumerate(present)}

        # Greedy bookkeeping
        removed: set[str] = set()
        representative_of: Dict[str, str] = {}  # removed_id -> rep_id
        score_of: Dict[str, float] = {}         # removed_id -> cosine_sim
        cluster_of: Dict[str, str] = {}         # removed_id -> cluster_id (rep-based)

        # Query neighbors for each representative and mark removals
        # Backend A: FAISS inner-product search
        if faiss is not None:
            # Inner product on normalized vectors equals cosine similarity.
            # We'll search top-k and filter by threshold. Need a k: choose a reasonable cap.
            # For v1: k = min(256, M) (can be tuned later)
            X_f = X.astype("float32", copy=False)
            index = faiss.IndexFlatIP(X_f.shape[1])
            index.add(X_f)

            k = min(256, X_f.shape[0])

            for sid in present:
                if sid in removed:
                    continue
                rep_local = present_id_to_local[sid]
                rep_vec = X_f[rep_local : rep_local + 1]

                sims, nbrs = index.search(rep_vec, k)
                sims = sims[0]
                nbrs = nbrs[0]

                rep_cluster = f"embcos:{sid}"
                # iterate neighbors sorted by similarity
                for sim, nbr_local in zip(sims, nbrs):
                    if nbr_local < 0:
                        continue
                    nbr_id = present[nbr_local]
                    if nbr_id == sid:
                        continue
                    if sim < self.threshold:
                        continue
                    if nbr_id in removed:
                        continue

                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = float(sim)
                    cluster_of[nbr_id] = rep_cluster

        # Backend B: sklearn radius neighbors with cosine distance
        elif NearestNeighbors is not None:
            nn = NearestNeighbors(metric="cosine", algorithm="auto", n_jobs=self.n_jobs)
            nn.fit(X)

            # radius_neighbors returns indices within cosine distance <= radius
            # (cosine distance = 1 - cosine similarity)
            for sid in present:
                if sid in removed:
                    continue
                rep_local = present_id_to_local[sid]
                rep_cluster = f"embcos:{sid}"

                dist, ind = nn.radius_neighbors(X[rep_local : rep_local + 1], radius=radius, return_distance=True)
                dist = dist[0]
                ind = ind[0]

                # Convert to similarity and sort (optional but makes mapping stable)
                pairs = []
                for d, j in zip(dist, ind):
                    if j == rep_local:
                        continue
                    sim = float(1.0 - d)
                    if sim >= self.threshold:
                        pairs.append((sim, j))
                pairs.sort(key=lambda x: x[0], reverse=True)

                for sim, j in pairs:
                    nbr_id = present[j]
                    if nbr_id in removed:
                        continue
                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = float(sim)
                    cluster_of[nbr_id] = rep_cluster

        else:
            # Fallback brute force (safe but slow)
            # Compute full similarity row-by-row (O(M^2))
            for sid in present:
                if sid in removed:
                    continue
                rep_local = present_id_to_local[sid]
                rep_cluster = f"embcos:{sid}"

                sims = X @ X[rep_local]
                # Remove neighbors above threshold
                # Deterministic: process in decreasing similarity then by index
                order = np.argsort(-sims, kind="mergesort")
                for j in order:
                    if j == rep_local:
                        continue
                    sim = float(sims[j])
                    if sim < self.threshold:
                        break
                    nbr_id = present[j]
                    if nbr_id in removed:
                        continue
                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = sim
                    cluster_of[nbr_id] = rep_cluster

        # Build kept df:
        # - keep all missing ids (no embeddings) as their own representatives
        # - keep all present ids that were not removed
        keep_ids = []
        for sid in work_ids:
            if sid in present_set:
                if sid not in removed:
                    keep_ids.append(sid)
            else:
                keep_ids.append(sid)

        kept_df = work[work[cols.id_col].astype(str).isin(set(keep_ids))].copy()
        kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        # mapping dataframe
        removed_rows = []
        for rid, rep in representative_of.items():
            removed_rows.append(
                {
                    "removed_id": rid,
                    "representative_id": rep,
                    "cluster_id": cluster_of.get(rid, f"embcos:{rep}"),
                    "score": score_of.get(rid, None),
                }
            )
        mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "cluster_id", "score"])

        # Optional: attach cluster to kept_df for convenience
        kept_df["embedding_cosine_cluster_id"] = kept_df[cols.id_col].astype(str).apply(
            lambda x: f"embcos:{x}" if x in present_set and x not in removed else None
        )

        return ReductionResult(
            df=kept_df,
            mapping=mapping,
            strategy=self.strategy,
            params={
                "embeddings_path": self.embeddings_path,
                "ids_path": self.ids_path,
                "ids_col": self.ids_col,
                "threshold": self.threshold,
                "use_faiss": self.use_faiss,
                "n_jobs": self.n_jobs,
                "dtype": self.dtype,
                "note_missing_policy": "ids without embeddings are kept as standalone representatives",
            },
        )
