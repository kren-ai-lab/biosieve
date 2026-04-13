"""Tests for ExactDedupReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import pandas as pd
    import pytest



import pandas as pd
import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.exact import ExactDedupReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


@pytest.fixture
def df_with_duplicates(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base with 3 extra duplicate rows appended (same sequence, new ids)."""
    dupes = df_base.head(3).copy()
    dupes["id"] = ["dup_000", "dup_001", "dup_002"]
    return pd.concat([df_base, dupes], ignore_index=True)


def test_happy_path(df_with_duplicates: pd.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "exact"
    assert len(res.df) < len(df_with_duplicates)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_with_duplicates["id"]))


def test_duplicates_removed(df_with_duplicates: pd.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)
    # 3 duplicates were added, so output should have exactly 3 fewer rows
    assert len(res.df) == len(df_with_duplicates) - 3


def test_no_duplicates_unchanged(df_base: pd.DataFrame) -> None:
    """Input with no duplicates should return all rows."""
    reducer = ExactDedupReducer()
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_mapping_schema(df_with_duplicates: pd.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        assert "removed_id" in res.mapping.columns
        assert "representative_id" in res.mapping.columns


def test_no_ids_lost(df_with_duplicates: pd.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        kept = set(res.df["id"].astype(str))
        removed = set(res.mapping["removed_id"].astype(str))
        all_ids = set(df_with_duplicates["id"].astype(str))
        assert kept & removed == set()
        assert kept | removed == all_ids


def test_missing_sequence_col(df_base: pd.DataFrame) -> None:
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = ExactDedupReducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
