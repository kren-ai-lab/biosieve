"""Shared k-mer helpers used by kmer_jaccard and minhash_jaccard reducers."""

from __future__ import annotations

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
    """Sort by id_col for determinism, reset index, validate uniqueness."""
    work = df.clone().sort(id_col, maintain_order=True)
    ids = work[id_col].cast(pl.String).to_list()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


def _build_mapping(
    removed_rows: list[tuple[str, str, float]],
    cluster_prefix: str = "kmer",
) -> pl.DataFrame:
    """Build a mapping DataFrame from (removed_id, representative_id, score) triples."""
    if not removed_rows:
        return pl.DataFrame(
            schema={
                "removed_id": pl.String,
                "representative_id": pl.String,
                "cluster_id": pl.String,
                "score": pl.Float64,
            }
        )
    mapping = pl.DataFrame(removed_rows, schema=["removed_id", "representative_id", "score"], orient="row")
    return mapping.with_columns(
        (pl.lit(f"{cluster_prefix}:") + pl.col("representative_id").cast(pl.String)).alias("cluster_id")
    ).select(["removed_id", "representative_id", "cluster_id", "score"])
