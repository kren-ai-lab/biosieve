"""Tests for StructuralDistanceReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest


import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.structural_distance import StructuralDistanceReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pd.DataFrame, edges_file: Path) -> None:
    reducer = StructuralDistanceReducer(
        edges_path=str(edges_file),
        threshold=0.4,
    )
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "structural_distance"
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_mapping_schema(df_base: pd.DataFrame, edges_file: Path) -> None:
    reducer = StructuralDistanceReducer(edges_path=str(edges_file), threshold=0.4)
    res = reducer.run(df_base, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        assert "removed_id" in res.mapping.columns
        assert "representative_id" in res.mapping.columns


def test_no_ids_lost(df_base: pd.DataFrame, edges_file: Path) -> None:
    reducer = StructuralDistanceReducer(edges_path=str(edges_file), threshold=0.4)
    res = reducer.run(df_base, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        kept = set(res.df["id"].astype(str))
        removed = set(res.mapping["removed_id"].astype(str))
        assert kept & removed == set()
        assert kept | removed == set(df_base["id"].astype(str))


def test_zero_threshold_removes_nothing(df_base: pd.DataFrame, edges_file: Path) -> None:
    """threshold=0.0 → no pair passes → nothing removed."""
    reducer = StructuralDistanceReducer(edges_path=str(edges_file), threshold=0.0)
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_missing_edges_file_raises(df_base: pd.DataFrame, tmp_path: Path) -> None:
    reducer = StructuralDistanceReducer(
        edges_path=str(tmp_path / "nonexistent_edges.csv"),
    )
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        reducer.run(df_base, COLS)
