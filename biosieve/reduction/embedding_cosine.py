"""Embedding-space reduction strategy based on cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from biosieve.reduction.backends.embedding_backend import load_embeddings
from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


class _FaissIndex(Protocol):
    def add(self, X: np.ndarray) -> None: ...

    def search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class _FaissModule(Protocol):
    def IndexFlatIP(self, d: int) -> _FaissIndex: ...


class _NearestNeighborsModel(Protocol):
    def fit(self, X: np.ndarray) -> object: ...

    def radius_neighbors(
        self, X: np.ndarray, *, radius: float, return_distance: bool
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _NearestNeighborsFactory(Protocol):
    def __call__(
        self, *, metric: str, algorithm: str, n_jobs: int
    ) -> _NearestNeighborsModel: ...


def _try_import_faiss() -> _FaissModule | None:
    try:
        import faiss  # noqa: PLC0415

        return cast("_FaissModule", faiss)
    except ImportError:
        return None


def _try_import_sklearn_nn() -> _NearestNeighborsFactory | None:
    try:
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

        return cast("_NearestNeighborsFactory", NearestNeighbors)
    except ImportError:
        return None


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


@dataclass(frozen=True)
class EmbeddingCosineReducer:
    r"""Greedy redundancy reduction in embedding space using cosine similarity.

    This reducer removes near-duplicate samples according to cosine similarity between
    their embedding vectors. It follows a deterministic greedy policy:

    Greedy policy:
    1) Sort dataset rows by `cols.id_col` (stable).
    2) Iterate ids in that order. The first unseen id becomes a representative.
    3) Remove any neighbors with cosine similarity >= `threshold`.

    Inputs required (embeddings mode):
    - embeddings_path: .npy array of shape (N, D)
    - ids_path: CSV containing embedding row ids in the same order as embeddings

    Neighbor search backends:
    - FAISS (if available and use_faiss=True): inner-product search on normalized vectors.
    - sklearn NearestNeighbors (if available): radius neighbors with cosine distance.
    - brute-force fallback: O(M^2), safe but slow.

    Missing embeddings policy:
    Dataset ids that do not exist in the embedding id list are kept as standalone
    representatives (safest default), and are never removed by this strategy.

    Args:
        embeddings_path: Path to embeddings .npy file (N, D).
        ids_path: Path to CSV with embedding row ids (length N).
        ids_col:
            Column name inside ids_path CSV containing ids. If the CSV has a single column,
            it will be used automatically by the embedding backend.
        threshold: Cosine similarity threshold in [0, 1]. Neighbors with sim >= threshold are removed.
        use_faiss: If True, tries to use FAISS if installed; otherwise falls back to sklearn/brute-force.
        n_jobs: Number of parallel jobs for sklearn NearestNeighbors (ignored for FAISS).
        dtype: Floating dtype to load/cast embeddings (recommended: "float32").

    Returns:
        ReductionResult:
            Result containing representative-only data, a removed-to-representative
            mapping (with cosine similarity score), strategy name, and effective
            parameters. Representatives include `embedding_cosine_cluster_id`
            (`embcos:<rep_id>`), and params include coverage and reduction stats.

    Raises:
        ValueError: If `threshold` is out of range, dataset id column is missing, or no dataset ids
        are present in the embedding ids file.
        FileNotFoundError: If embeddings_path or ids_path are missing (raised by embedding backend).
        ImportError: Only raised implicitly if optional dependencies are partially broken; this reducer
        uses safe fallbacks when FAISS/sklearn are missing.

    Notes:
        - FAISS path uses a capped top-k search (`k=min(256, M)`) for speed. This is a heuristic:
        in extremely dense neighborhoods it may miss some neighbors above threshold.
        For guaranteed exactness, use sklearn radius mode or brute-force.
        - This is a greedy algorithm: results depend on the representative ordering
        (here: sorted by id for determinism).
        - This strategy removes redundancy in embedding space only; it does not ensure
        biological leakage control unless embeddings encode that information.

    Examples:
        >>> biosieve reduce \\
        ...   --in biosieve_example_dataset_1000.csv \\
        ...   --out data_nr_embcos.csv \\
        ...   --strategy embedding_cosine \\
        ...   --map map_embcos.csv \\
        ...   --report reduction_embcos.json \\
        ...   --params params.yaml

    """

    embeddings_path: str = "embeddings.npy"
    ids_path: str = "embedding_ids.csv"
    ids_col: str = "id"

    threshold: float = 0.95
    use_faiss: bool = True
    n_jobs: int = 1
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "embedding_cosine"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:  # noqa: C901,PLR0912,PLR0915
        """Reduce embedding redundancy and return representatives plus mapping."""
        if not (0.0 <= self.threshold <= 1.0):
            msg = "threshold must be in [0, 1]"
            raise ValueError(msg)

        if cols.id_col not in df.columns:
            msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}"
            raise ValueError(msg)

        store = load_embeddings(
            embeddings_path=self.embeddings_path,
            ids_path=self.ids_path,
            ids_col=self.ids_col,
            dtype=self.dtype,
        )

        # Map dataset ids -> embedding row index
        id_to_idx: dict[str, int] = {str(sid): i for i, sid in enumerate(store.ids)}

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        work_ids = work[cols.id_col].astype(str).tolist()

        present = [sid for sid in work_ids if sid in id_to_idx]
        missing = [sid for sid in work_ids if sid not in id_to_idx]

        if len(present) == 0:
            msg = (
                "None of the dataset ids were found in embedding ids file. "
                f"Example dataset id: {work_ids[0] if work_ids else 'EMPTY'}, "
                f"example embedding id: {store.ids[0] if store.ids else 'EMPTY'}"
            )
            raise ValueError(
                msg
            )

        # Subset embeddings to only rows present in dataset, keeping work order (deterministic)
        idxs = np.array([id_to_idx[sid] for sid in present], dtype=int)
        X = store.X[idxs]
        X = _l2_normalize(X)

        present_set = set(present)

        faiss = _try_import_faiss() if self.use_faiss else None
        NearestNeighbors = _try_import_sklearn_nn()

        # cosine distance radius r = 1 - threshold (for normalized vectors)
        radius = float(1.0 - self.threshold)

        # local index map
        present_id_to_local = {sid: i for i, sid in enumerate(present)}

        removed: set[str] = set()
        representative_of: dict[str, str] = {}
        score_of: dict[str, float] = {}

        # Backend A: FAISS (inner product)
        if faiss is not None:
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

                for sim, nbr_local in zip(sims, nbrs, strict=False):
                    if nbr_local < 0:
                        continue
                    nbr_id = present[nbr_local]
                    if nbr_id == sid:
                        continue
                    if float(sim) < self.threshold:
                        continue
                    if nbr_id in removed:
                        continue

                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = float(sim)

        # Backend B: sklearn radius neighbors
        elif NearestNeighbors is not None:
            nn = NearestNeighbors(metric="cosine", algorithm="auto", n_jobs=self.n_jobs)
            nn.fit(X)

            for sid in present:
                if sid in removed:
                    continue

                rep_local = present_id_to_local[sid]
                dist, ind = nn.radius_neighbors(
                    X[rep_local : rep_local + 1], radius=radius, return_distance=True
                )
                dist = dist[0]
                ind = ind[0]

                pairs = []
                for d, j in zip(dist, ind, strict=False):
                    if j == rep_local:
                        continue
                    sim = float(1.0 - d)
                    if sim >= self.threshold:
                        pairs.append((sim, int(j)))
                pairs.sort(key=lambda x: x[0], reverse=True)

                for sim, j in pairs:
                    nbr_id = present[j]
                    if nbr_id in removed:
                        continue
                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = float(sim)

        # Backend C: brute force fallback
        else:
            for sid in present:
                if sid in removed:
                    continue

                rep_local = present_id_to_local[sid]
                sims = X @ X[rep_local]

                order = np.argsort(-sims, kind="mergesort")
                for j in order:
                    if j == rep_local:
                        continue
                    sim = float(sims[j])
                    if sim < self.threshold:
                        break
                    nbr_id = present[int(j)]
                    if nbr_id in removed:
                        continue
                    removed.add(nbr_id)
                    representative_of[nbr_id] = sid
                    score_of[nbr_id] = sim

        # Build kept ids:
        # - keep all missing ids (no embeddings) as standalone reps
        # - keep all present ids that were not removed
        keep_ids: list[str] = []
        for sid in work_ids:
            if sid in present_set:
                if sid not in removed:
                    keep_ids.append(sid)
            else:
                keep_ids.append(sid)

        keep_set = set(keep_ids)

        kept_df = work[work[cols.id_col].astype(str).isin(keep_set)].copy()
        kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        # mapping df
        removed_rows = []
        for rid, rep in representative_of.items():
            removed_rows.append(
                {
                    "removed_id": rid,
                    "representative_id": rep,
                    "cluster_id": f"embcos:{rep}",
                    "score": score_of.get(rid),
                }
            )
        mapping = pd.DataFrame(
            removed_rows, columns=["removed_id", "representative_id", "cluster_id", "score"]
        )

        # Attach cluster id for representatives (convenience)
        kept_df["embedding_cosine_cluster_id"] = (
            kept_df[cols.id_col]
            .astype(str)
            .apply(
                lambda x: (
                    f"embcos:{x}"
                    if (x in present_set and x not in removed)
                    else (f"singleton:{x}" if x in missing else None)
                )
            )
        )

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_present_embeddings": len(present),
            "n_missing_embeddings": len(missing),
            "coverage_embeddings": float(len(present) / len(work)) if len(work) else 0.0,
            "n_kept": len(kept_df),
            "n_removed": len(mapping),
            "reduction_ratio": float(len(kept_df) / len(work)) if len(work) else 0.0,
        }

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
                "stats": stats,
            },
        )
