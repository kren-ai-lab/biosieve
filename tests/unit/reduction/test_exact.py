"""Tests for ExactDedupReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pytest

import polars as pl
import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.exact import ExactDedupReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


@pytest.fixture
def df_with_duplicates(df_base: pl.DataFrame) -> pl.DataFrame:
    """df_base with 3 extra duplicate rows appended (same sequence, new ids)."""
    dupes = df_base.head(3).with_columns(pl.Series("id", ["dup_000", "dup_001", "dup_002"]))
    return pl.concat([df_base, dupes], how="vertical")


def test_happy_path(df_with_duplicates: pl.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "exact"
    assert res.df.height < df_with_duplicates.height
    assert res.df.height > 0
    assert set(res.df["id"].to_list()).issubset(set(df_with_duplicates["id"].to_list()))


def test_duplicates_removed(df_with_duplicates: pl.DataFrame) -> None:
    reducer = ExactDedupReducer()
    res = reducer.run(df_with_duplicates, COLS)
    # 3 duplicates were added, so output should have exactly 3 fewer rows
    assert res.df.height == df_with_duplicates.height - 3


def test_no_duplicates_unchanged(df_base: pl.DataFrame) -> None:
    """Input with no duplicates should return all rows."""
    reducer = ExactDedupReducer()
    res = reducer.run(df_base, COLS)
    assert res.df.height == df_base.height


def test_missing_sequence_col(df_base: pl.DataFrame) -> None:
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = ExactDedupReducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
