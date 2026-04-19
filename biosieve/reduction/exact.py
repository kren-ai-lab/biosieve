"""Exact sequence deduplication strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.common import attach_cluster_ids, build_reduction_stats, empty_mapping_df
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_inputs(df: pl.DataFrame, cols: Columns) -> None:
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns}"
        raise ValueError(msg)
    if cols.seq_col not in df.columns:
        msg = f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ExactDedupReducer:
    r"""Exact redundancy reduction by identical sequences (exact match).

    This reducer removes duplicates where sequences are exactly identical (string match)
    based on `cols.seq_col`. The first occurrence (after deterministic sorting by id) is
    kept as the representative; subsequent duplicates are removed.

    Greedy policy:
    1) Sort rows by `cols.id_col` (stable, deterministic).
    2) Mark duplicates by exact sequence equality in `cols.seq_col` (`keep="first"`).
    3) Keep the first occurrence as representative and remove others.

    Returns:
        ReductionResult:
            Result containing unique-sequence representatives, duplicate-to-
            representative mapping, strategy name, and params with summary stats.
            The reduced dataframe includes `exact_cluster_id` (`exact:<rep_id>`).

    Raises:
        ValueError: If required columns (`cols.id_col`, `cols.seq_col`) are missing.

    Notes:
        - Exact deduplication is the safest and fastest redundancy reducer.
        - It does not remove near-duplicates (single mutations, indels, etc.). Use k-mer,
        embedding, descriptor, or homology reducers for that.

    Examples:
        >>> biosieve reduce \\
        ...   --in dataset.csv \\
        ...   --out data_nr_exact.csv \\
        ...   --strategy exact \\
        ...   --map map_exact.csv \\
        ...   --report report_exact.json

    """

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "exact"

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:
        """Remove exact sequence duplicates and return representative mapping."""
        _validate_inputs(df, cols)

        work = df.clone().sort(cols.id_col, maintain_order=True)
        kept = work.unique(subset=[cols.seq_col], keep="first", maintain_order=True)
        kept_ids = set(kept[cols.id_col].cast(pl.String).to_list())
        removed = work.filter(~pl.col(cols.id_col).cast(pl.String).is_in(kept_ids)).select(
            [cols.id_col, cols.seq_col]
        )

        rep_by_seq = dict(
            zip(kept[cols.seq_col].to_list(), kept[cols.id_col].cast(pl.String).to_list(), strict=False)
        )

        if removed.height > 0:
            mapping = (
                removed.select(
                    pl.col(cols.id_col).cast(pl.String).alias("removed_id"),
                    pl.col(cols.seq_col)
                    .replace_strict(rep_by_seq, default=None)
                    .cast(pl.String)
                    .alias("representative_id"),
                )
                .with_columns(
                    score=pl.lit(1.0),
                    cluster_id=pl.lit("exact:") + pl.col("representative_id"),
                )
                .select(["removed_id", "representative_id", "cluster_id", "score"])
            )
        else:
            mapping = empty_mapping_df()

        kept = attach_cluster_ids(
            kept, id_col=cols.id_col, column_name="exact_cluster_id", cluster_prefix="exact"
        )

        stats: dict[str, Any] = build_reduction_stats(
            n_total=work.height,
            n_kept=kept.height,
            note="Exact sequence duplicates removed (string equality).",
        )

        return ReductionResult(
            df=kept,
            mapping=mapping,
            strategy=self.strategy,
            params={},
            stats=stats,
        )
