"""Shared k-mer helpers used by kmer_jaccard and minhash_jaccard reducers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from biosieve.reduction.common import build_mapping, prepare_reduction_work

if TYPE_CHECKING:
    import polars as pl


def _kmer_set(seq: str, k: int) -> set[str]:
    """Convert a sequence into a set of k-mers.

    Args:
        seq: Input sequence string.
        k: K-mer size (>= 1).

    Returns:
        Set of unique k-mer tokens. If len(seq) < k, returns {seq}.

    Raises:
        ValueError: If k < 1.

    """
    if k <= 0:
        msg = "k must be >= 1"
        raise ValueError(msg)
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _prepare_work(df: pl.DataFrame, id_col: str) -> tuple[pl.DataFrame, list[str]]:
    """Sort by id_col for determinism and validate uniqueness."""
    return prepare_reduction_work(df, id_col)


def _build_mapping(
    removed_rows: list[tuple[str, str, float]],
    cluster_prefix: str = "kmer",
) -> pl.DataFrame:
    """Build a mapping DataFrame from (removed_id, representative_id, score) triples."""
    return build_mapping(
        [
            (removed_id, representative_id, float(score))
            for removed_id, representative_id, score in removed_rows
        ],
        cluster_prefix=cluster_prefix,
    )
