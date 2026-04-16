"""Structural-edge reduction strategy based on distance or similarity thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from biosieve.reduction.backends.structure_backend import load_edges_csv
from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_inputs(df: pl.DataFrame, cols: Columns, mode: str, threshold: float) -> None:
    if mode not in {"distance", "similarity"}:
        msg = "mode must be 'distance' or 'similarity'"
        raise ValueError(msg)
    if mode == "distance" and threshold < 0:
        msg = "threshold must be >= 0 for distance mode"
        raise ValueError(msg)
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns}"
        raise ValueError(msg)


@dataclass(frozen=True)
class StructuralDistanceReducer:
    r"""Greedy redundancy reduction using precomputed structural distances or similarities.

    This reducer removes redundant samples based on *precomputed structural relationships*
    between entities (e.g., protein structures, docking poses, folds). It consumes an
    edge list describing pairwise structural distances or similarities and applies a
    deterministic greedy selection of representatives.

    Input format:
    A CSV edge list with at least three columns:

    - id1: first entity id
    - id2: second entity id
    - value: distance or similarity score between id1 and id2

    Only edges where **both ids are present in the dataset** are considered.

    Modes:
    - mode="distance":
        A neighbor is considered redundant if `value <= threshold`.
        Typical examples: RMSD, TM-score distance, graph distance.
    - mode="similarity":
        A neighbor is considered redundant if `value >= threshold`.
        Typical examples: TM-score similarity, contact overlap, docking similarity.

    Greedy policy:
    1) Sort dataset rows by `cols.id_col` (stable, deterministic).
    2) Iterate in that order.
    3) First unseen id becomes a representative.
    4) Remove all neighbors satisfying the redundancy criterion.

    Args:
        edges_path: Path to CSV file containing structural edges.
        mode: Redundancy mode: `"distance"` marks neighbors redundant when
            `value <= threshold`, and `"similarity"` marks neighbors redundant
            when `value >= threshold`.
        threshold: Distance or similarity threshold defining redundancy.
        id1_col: Column name for first id in edge list.
        id2_col: Column name for second id in edge list.
        value_col: Column name for distance/similarity value in edge list.

    Returns:
        ReductionResult:
            Result containing structural representatives, removed-to-
            representative mapping with edge value score, strategy name, and
            effective parameters with basic stats. The reduced dataframe includes
            `structural_cluster_id` (`struct:<rep_id>`).

    Raises:
        ValueError: If threshold is invalid for the selected mode, required columns are missing,
        or mode is not supported.
        FileNotFoundError: If `edges_path` does not exist (raised by backend).
        RuntimeError: If edge list cannot be parsed correctly (raised by backend).

    Notes:
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

    Examples:
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
        """Return the strategy identifier."""
        return "structural_distance"

    def _is_redundant(self, value: float) -> bool:
        if self.mode == "distance":
            return value <= self.threshold
        if self.mode == "similarity":
            return value >= self.threshold
        msg = "mode must be 'distance' or 'similarity'"
        raise ValueError(msg)

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:
        """Reduce redundancy using precomputed structural edge relationships."""
        _validate_inputs(df, cols, self.mode, self.threshold)

        # deterministic ordering
        work = df.clone().sort(cols.id_col, maintain_order=True)
        ids = work[cols.id_col].cast(pl.String).to_list()
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
        kept_df = work.filter(pl.col(cols.id_col).cast(pl.String).is_in(set(keep_ids)))

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
            (pl.lit("struct:") + pl.col(cols.id_col).cast(pl.String)).alias("structural_cluster_id")
        )

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_kept": kept_df.height,
            "n_removed": mapping.height,
            "n_edges_loaded": int(edges.n_edges),
            "reduction_ratio": float(kept_df.height / work.height) if work.height else 0.0,
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
