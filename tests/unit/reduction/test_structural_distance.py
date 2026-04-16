"""Tests for StructuralDistanceReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    import pytest


import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.structural_distance import StructuralDistanceReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pl.DataFrame, edges_file: Path) -> None:
    reducer = StructuralDistanceReducer(
        edges_path=str(edges_file),
        threshold=0.4,
    )
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "structural_distance"
    assert res.df.height <= df_base.height
    assert res.df.height > 0
    assert set(res.df["id"].to_list()).issubset(set(df_base["id"].to_list()))


def test_zero_threshold_removes_nothing(df_base: pl.DataFrame, edges_file: Path) -> None:
    """threshold=0.0 → no pair passes → nothing removed."""
    reducer = StructuralDistanceReducer(edges_path=str(edges_file), threshold=0.0)
    res = reducer.run(df_base, COLS)
    assert res.df.height == df_base.height


def test_missing_edges_file_raises(df_base: pl.DataFrame, tmp_path: Path) -> None:
    reducer = StructuralDistanceReducer(
        edges_path=str(tmp_path / "nonexistent_edges.csv"),
    )
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        reducer.run(df_base, COLS)
