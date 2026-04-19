"""Shared k-mer helpers used by kmer_jaccard and minhash_jaccard reducers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from biosieve.reduction.common import build_mapping

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


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


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
