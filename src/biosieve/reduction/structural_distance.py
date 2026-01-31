from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.reduction.backends.structure_backend import load_edges_csv


@dataclass(frozen=True)
class StructuralDistanceReducer:
    """
    Greedy redundancy reduction using precomputed structural distances or similarities.

    Input (v1):
      - an edge list CSV with columns: id1, id2, distance (or similarity)

    Modes:
      - mode="distance": consider redundant if value <= threshold
      - mode="similarity": consider redundant if value >= threshold

    Deterministic:
      - sort dataset by id
      - greedy keep-first representative
      - remove all neighbors of representative that satisfy criterion
    """

    edges_path: str = "struct_edges.csv"
    mode: str = "distance"         # "distance" or "similarity"
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
        raise ValueError("mode must be 'distance' or 'similarity'")

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if self.threshold < 0 and self.mode == "distance":
            raise ValueError("threshold must be >= 0 for distance mode")

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
        rep_of: Dict[str, str] = {}
        score_of: Dict[str, float] = {}   # store raw value (distance or similarity)
        cluster_of: Dict[str, str] = {}

        # Greedy: for each id (in sorted order), if not removed => representative
        for rep_id in ids:
            if rep_id in removed:
                continue

            rep_cluster = f"struct:{rep_id}"

            # neighbors from precomputed graph
            for nbr_id, val in edges.adj.get(rep_id, []):
                # only act on ids that are in the dataset
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
                    "score": score_of.get(rid, None),
                }
            )
        mapping = pd.DataFrame(rows, columns=["removed_id", "representative_id", "cluster_id", "score"])

        kept_df["structural_cluster_id"] = kept_df[cols.id_col].astype(str).apply(lambda x: f"struct:{x}")

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
                "n_edges_loaded": edges.n_edges,
                "note": "This reducer consumes precomputed structural distances/similarities (edge list).",
            },
        )
