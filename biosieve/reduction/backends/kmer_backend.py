"""Shared k-mer helpers used by kmer_jaccard and minhash_jaccard reducers."""

from __future__ import annotations

import pandas as pd


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


def _prepare_work(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Sort by id_col for determinism, reset index, validate uniqueness."""
    work = df.copy().sort_values(id_col, kind="mergesort").reset_index(drop=True)
    ids = work[id_col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


def _build_mapping(
    removed_rows: list[tuple[str, str, float]],
    cluster_prefix: str = "kmer",
) -> pd.DataFrame:
    """Build a mapping DataFrame from (removed_id, representative_id, score) triples."""
    mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "score"])
    if len(mapping) == 0:
        return pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"])
    mapping["cluster_id"] = mapping["representative_id"].astype(str).apply(lambda x: f"{cluster_prefix}:{x}")
    return mapping[["removed_id", "representative_id", "cluster_id", "score"]]
