"""Tests for StratifiedNumericSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import pandas as pd
    import pytest



import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.stratified_numeric import StratifiedNumericSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericSplitter(label_col="target", test_size=0.2, n_bins=5, seed=13)
    res = splitter.run(df_full, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "stratified_numeric"
    assert len(res.train) + len(res.test) == len(df_full)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericSplitter(label_col="target", test_size=0.2, n_bins=5, seed=13)
    res = splitter.run(df_full, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_stats_have_bins(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericSplitter(label_col="target", test_size=0.2, n_bins=5, seed=13)
    res = splitter.run(df_full, COLS)
    assert "n_bins_effective" in res.stats


def test_missing_label_col_raises(df_base: pd.DataFrame) -> None:
    splitter = StratifiedNumericSplitter(label_col="NONEXISTENT", n_bins=5)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
