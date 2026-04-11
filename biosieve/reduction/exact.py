from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ExactDedupReducer:
    """
    Exact redundancy reduction by identical sequences (exact match).

    This reducer removes duplicates where sequences are exactly identical (string match)
    based on `cols.seq_col`. The first occurrence (after deterministic sorting by id) is
    kept as the representative; subsequent duplicates are removed.

    Greedy policy
    -------------
    1) Sort rows by `cols.id_col` (stable, deterministic).
    2) Mark duplicates by exact sequence equality in `cols.seq_col` (`keep="first"`).
    3) Keep the first occurrence as representative and remove others.

    Parameters
    ----------
    None
        This reducer has no configurable parameters.

    Returns
    -------
    ReductionResult
        - df:
            DataFrame containing only unique sequences (representatives).
            Adds `exact_cluster_id` as `exact:<rep_id>` for convenience.
        - mapping:
            DataFrame with columns:
              * removed_id
              * representative_id
              * cluster_id (`exact:<rep_id>`)
              * score (always 1.0 for exact duplicates)
        - strategy:
            "exact"
        - params:
            Contains `stats` summary.

    Raises
    ------
    ValueError
        If required columns (`cols.id_col`, `cols.seq_col`) are missing.

    Notes
    -----
    - Exact deduplication is the safest and fastest redundancy reducer.
    - It does not remove near-duplicates (single mutations, indels, etc.). Use k-mer,
      embedding, descriptor, or homology reducers for that.

    Examples
    --------
    >>> biosieve reduce \\
    ...   --in dataset.csv \\
    ...   --out data_nr_exact.csv \\
    ...   --strategy exact \\
    ...   --map map_exact.csv \\
    ...   --report report_exact.json
    """

    @property
    def strategy(self) -> str:
        return "exact"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if cols.id_col not in df.columns:
            raise ValueError(f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}")
        if cols.seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns.tolist()}")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        dup_mask = work.duplicated(subset=[cols.seq_col], keep="first")

        kept = work.loc[~dup_mask].reset_index(drop=True)
        removed = work.loc[dup_mask, [cols.id_col, cols.seq_col]].copy()

        # Map: sequence -> representative_id
        rep_by_seq = kept.set_index(cols.seq_col)[cols.id_col].astype(str).to_dict()

        mapping = pd.DataFrame(
            {
                "removed_id": removed[cols.id_col].astype(str).values,
                "representative_id": removed[cols.seq_col].map(rep_by_seq).astype(str).values,
            }
        )

        if len(mapping) > 0:
            mapping["cluster_id"] = mapping["representative_id"].astype(str).apply(lambda x: f"exact:{x}")
            mapping["score"] = 1.0
            mapping = mapping[["removed_id", "representative_id", "cluster_id", "score"]]
        else:
            mapping = pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"])

        kept["exact_cluster_id"] = kept[cols.id_col].astype(str).apply(lambda x: f"exact:{x}")

        stats: Dict[str, Any] = {
            "n_total": int(len(work)),
            "n_kept": int(len(kept)),
            "n_removed": int(len(mapping)),
            "reduction_ratio": float(len(kept) / len(work)) if len(work) else 0.0,
            "note": "Exact sequence duplicates removed (string equality).",
        }

        return ReductionResult(
            df=kept,
            mapping=mapping,
            strategy=self.strategy,
            params={"stats": stats},
        )
