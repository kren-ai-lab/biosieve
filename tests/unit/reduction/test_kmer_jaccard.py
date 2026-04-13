"""Tests for KmerJaccardReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import pandas as pd
    import pytest



import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.kmer_jaccard import KmerJaccardReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pd.DataFrame) -> None:
    reducer = KmerJaccardReducer(threshold=0.3, k=3)
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "kmer_jaccard"
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_mapping_schema(df_base: pd.DataFrame) -> None:
    reducer = KmerJaccardReducer(threshold=0.3, k=3)
    res = reducer.run(df_base, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        assert "removed_id" in res.mapping.columns
        assert "representative_id" in res.mapping.columns


def test_no_ids_lost(df_base: pd.DataFrame) -> None:
    reducer = KmerJaccardReducer(threshold=0.3, k=3)
    res = reducer.run(df_base, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        kept = set(res.df["id"].astype(str))
        removed = set(res.mapping["removed_id"].astype(str))
        assert kept & removed == set()
        assert kept | removed == set(df_base["id"].astype(str))


def test_high_threshold_removes_nothing(df_base: pd.DataFrame) -> None:
    """threshold=1.0 → only perfect k-mer overlap → random seqs all kept."""
    reducer = KmerJaccardReducer(threshold=1.0, k=3)
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_missing_sequence_col(df_base: pd.DataFrame) -> None:
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = KmerJaccardReducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
