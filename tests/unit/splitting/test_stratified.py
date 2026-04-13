"""Tests for StratifiedSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.stratified import StratifiedSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_labeled: pd.DataFrame) -> None:
    splitter = StratifiedSplitter(label_col="label", test_size=0.2, seed=13)
    res = splitter.run(df_labeled, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "stratified"
    assert len(res.train) + len(res.test) == len(df_labeled)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap(df_labeled: pd.DataFrame) -> None:
    splitter = StratifiedSplitter(label_col="label", test_size=0.2, seed=13)
    res = splitter.run(df_labeled, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_val_when_requested(df_labeled: pd.DataFrame) -> None:
    splitter = StratifiedSplitter(label_col="label", test_size=0.2, val_size=0.1, seed=13)
    res = splitter.run(df_labeled, COLS)
    assert res.val is not None
    assert len(res.val) > 0
    n_total = len(res.train) + len(res.test) + len(res.val)
    assert n_total == len(df_labeled)


def test_stats_have_label_counts(df_labeled: pd.DataFrame) -> None:
    splitter = StratifiedSplitter(label_col="label", test_size=0.2, seed=13)
    res = splitter.run(df_labeled, COLS)
    assert "train_label_counts" in res.stats
    assert "test_label_counts" in res.stats


def test_missing_label_col_raises(df_base: pd.DataFrame) -> None:
    splitter = StratifiedSplitter(label_col="NONEXISTENT", test_size=0.2, seed=13)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
