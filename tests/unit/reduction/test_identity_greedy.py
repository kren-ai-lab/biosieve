"""Tests for IdentityGreedyReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.identity_greedy import IdentityGreedyReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pd.DataFrame) -> None:
    reducer = IdentityGreedyReducer(threshold=0.5)
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "identity_greedy"
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_high_threshold_removes_nothing(df_base: pd.DataFrame) -> None:
    """threshold=1.0 means only exact identity → random seqs should all be kept."""
    reducer = IdentityGreedyReducer(threshold=1.0)
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_missing_sequence_col(df_base: pd.DataFrame) -> None:
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = IdentityGreedyReducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
