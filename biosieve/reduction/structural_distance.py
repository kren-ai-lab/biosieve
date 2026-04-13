from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from biosieve.reduction.backends.structure_backend import load_edges_csv
from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


@dataclass(frozen=True)
class StructuralDistanceReducer:
    """Greedy redundancy reduction using precomputed structural distances or similarities.

    This reducer removes redundant samples based on *precomputed structural relationships*
    between entities (e.g., protein structures, docking poses, folds). It consumes an
    edge list describing pairwise structural distances or similarities and applies a
    deterministic greedy selection of representatives.

    Input format
    ------------
    A CSV edge list with at least three columns:

    - id1: first entity id
    - id2: second entity id
    - value: distance or similarity score between id1 and id2

    Only edges where **both ids are present in the dataset** are considered.

    Modes
    -----
    - mode="distance":
        A neighbor is considered redundant if `value <= threshold`.
        Typical examples: RMSD, TM-score distance, graph distance.
    - mode="similarity":
        A neighbor is considered redundant if `value >= threshold`.
        Typical examples: TM-score similarity, contact overlap, docking similarity.

    Greedy policy
    -------------
    1) Sort dataset rows by `cols.id_col` (stable, deterministic).
    2) Iterate in that order.
    3) First unseen id becomes a representative.
    4) Remove all neighbors satisfying the redundancy criterion.

    Parameters
    ----------
    edges_path:
        Path to CSV file containing structural edges.
    mode:
        Redundancy mode:
        - "distance": redundant if value <= threshold
        - "similarity": redundant if value >= threshold
    threshold:
        Distance or similarity threshold defining redundancy.
    id1_col:
        Column name for first id in edge list.
    id2_col:
        Column name for second id in edge list.
    value_col:
        Column name for distance/similarity value in edge list.

    Returns
    -------
    ReductionResult
        - df:
            Reduced dataframe containing only structural representatives.
            Adds column `structural_cluster_id` with values `struct:<rep_id>`.
        - mapping:
            DataFrame with columns:
              * removed_id
              * representative_id
              * cluster_id (`struct:<rep_id>`)
              * score (raw distance or similarity value)
        - strategy:
            "structural_distance"
        - params:
            Effective parameters plus basic statistics.

    Raises
    ------
    ValueError
        If threshold is invalid for the selected mode, required columns are missing,
        or mode is not supported.
    FileNotFoundError
        If `edges_path` does not exist (raised by backend).
    RuntimeError
        If edge list cannot be parsed correctly (raised by backend).

    Notes
    -----
    - This reducer **assumes structural relationships are precomputed**.
      It does not perform any structural alignment or docking itself.
    - Missing edges are treated as "no redundancy" (i.e., conservative).
    - Redundancy is **not transitive** beyond greedy propagation.
      If A~B and B~C but A~C is missing, C may survive depending on order.
    - This strategy is ideal for:
        * pose clustering
        * fold-level deduplication
        * structure-aware dataset pruning
    - For strict structural leakage control, consider combining this reducer
      with `cluster_aware` or `homology_aware` splits.

    Examples
    --------
    >>> biosieve reduce \\
    ...   --in dataset.csv \\
    ...   --out data_nr_struct.csv \\
    ...   --strategy structural_distance \\
    ...   --map map_struct.csv \\
    ...   --report report_struct.json \\
    ...   --params params.yaml

    """

    edges_path: str = "struct_edges.csv"
    mode: str = "distance"  # "distance" | "similarity"
    threshold: float = 0.5

    id1_col: str = "id1"
    id2_col: str = "id2"
    value_col: str = "distance"

    @property
    def strategy(self) -> str:
        return "structural_distance"

    def _is_redundant(self, value: float) -> bool:
        if self.mode == "distance":
            return value <= self.threshold
        if self.mode == "similarity":
            return value >= self.threshold
        msg = "mode must be 'distance' or 'similarity'"
        raise ValueError(msg)

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if self.mode not in {"distance", "similarity"}:
            msg = "mode must be 'distance' or 'similarity'"
            raise ValueError(msg)

        if self.mode == "distance" and self.threshold < 0:
            msg = "threshold must be >= 0 for distance mode"
            raise ValueError(msg)

        if cols.id_col not in df.columns:
            msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}"
            raise ValueError(msg)

        # deterministic ordering
        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        ids = work[cols.id_col].astype(str).tolist()
        id_set = set(ids)

        edges = load_edges_csv(
            self.edges_path,
            id1_col=self.id1_col,
            id2_col=self.id2_col,
            value_col=self.value_col,
        )

        removed: set[str] = set()
        rep_of: dict[str, str] = {}
        score_of: dict[str, float] = {}
        cluster_of: dict[str, str] = {}

        # Greedy representative selection
        for rep_id in ids:
            if rep_id in removed:
                continue

            rep_cluster = f"struct:{rep_id}"

            for nbr_id, val in edges.adj.get(rep_id, []):
                if nbr_id not in id_set:
                    continue
                if nbr_id == rep_id:
                    continue
                if nbr_id in removed:
                    continue

                if self._is_redundant(float(val)):
                    removed.add(nbr_id)
                    rep_of[nbr_id] = rep_id
                    score_of[nbr_id] = float(val)
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
                    "cluster_id": cluster_of.get(rid, f"struct:{rep}"),
                    "score": score_of.get(rid),
                }
            )
        mapping = pd.DataFrame(
            rows,
            columns=["removed_id", "representative_id", "cluster_id", "score"],
        )

        kept_df["structural_cluster_id"] = kept_df[cols.id_col].astype(str).apply(lambda x: f"struct:{x}")

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_kept": len(kept_df),
            "n_removed": len(mapping),
            "n_edges_loaded": int(edges.n_edges),
            "reduction_ratio": float(len(kept_df) / len(work)) if len(work) else 0.0,
            "mode": self.mode,
        }

        return ReductionResult(
            df=kept_df,
            mapping=mapping,
            strategy=self.strategy,
            params={
                "edges_path": self.edges_path,
                "mode": self.mode,
                "threshold": self.threshold,
                "id1_col": self.id1_col,
                "id2_col": self.id2_col,
                "value_col": self.value_col,
                "stats": stats,
            },
        )
