"""Tests for GroupSplitter — includes leakage invariant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pytest


import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.group import GroupSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_grouped: pl.DataFrame) -> None:
    splitter = GroupSplitter(group_col="group", test_size=0.2, seed=13)
    res = splitter.run(df_grouped, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "group"
    assert res.train.height + res.test.height == df_grouped.height
    assert res.train.height > 0
    assert res.test.height > 0


def test_leakage_zero(df_grouped: pl.DataFrame) -> None:
    """Core invariant: no group appears in both train and test."""
    splitter = GroupSplitter(group_col="group", test_size=0.2, seed=13)
    res = splitter.run(df_grouped, COLS)

    assert res.stats["leak_groups_train_test"] == 0

    train_groups = set(res.train["group"].to_list())
    test_groups = set(res.test["group"].to_list())
    assert train_groups & test_groups == set()


def test_val_leakage_zero(df_grouped: pl.DataFrame) -> None:
    splitter = GroupSplitter(group_col="group", test_size=0.2, val_size=0.2, seed=13)
    res = splitter.run(df_grouped, COLS)
    assert res.stats["leak_groups_train_val"] == 0
    assert res.stats["leak_groups_val_test"] == 0


def test_missing_group_col_raises(df_base: pl.DataFrame) -> None:
    splitter = GroupSplitter(group_col="NONEXISTENT", test_size=0.2, seed=13)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
