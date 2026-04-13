from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Columns"]


@dataclass(frozen=True, slots=True)
class Columns:
    """Column specification for BioSieve datasets.

    BioSieve works with tabular datasets (typically CSV) that contain at minimum a
    unique identifier column and (optionally) sequence/label/grouping fields.

    Parameters
    ----------
    id_col:
        Column name containing unique sample identifiers.
        BioSieve assumes this column is unique (1 row = 1 id).
    seq_col:
        Column name containing sequence strings. Required by strategies that operate
        on sequences (e.g., exact deduplication, k-mer redundancy, etc.).
    label_col:
        Optional classification label column name (e.g., "label").
        Use when applying stratified splits for classification.
    group_col:
        Optional group identifier column name (e.g., subject_id, taxid).
        Used for group-aware splits and group-aware k-fold.
    cluster_col:
        Optional cluster identifier column name (e.g., precomputed homology clusters).
        Used for cluster-aware splits, and can be passed to group splitters by setting
        group_col=cluster_col.
    date_col:
        Optional temporal column name (e.g., collection date, deposition date).
        Used for time-based splits. Must be parseable to datetime by the consuming strategy.

    Notes
    -----
    - `label_col`, `group_col`, `cluster_col`, and `date_col` are optional because not all
      datasets contain these fields and not all strategies require them.
    - When a strategy requires a column and it is missing, the strategy should raise
      `ValueError` with a clear message including the missing column name.

    Examples
    --------
    Typical peptide/protein dataset columns:

    >>> cols = Columns(
    ...     id_col="id",
    ...     seq_col="sequence",
    ...     label_col="label",
    ...     group_col="taxid",
    ...     date_col="collection_date",
    ... )

    """

    id_col: str = "id"
    seq_col: str = "sequence"
    label_col: str | None = "label"
    group_col: str | None = None
    cluster_col: str | None = None
    date_col: str | None = None
