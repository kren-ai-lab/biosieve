"""Tests for RandomSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


from biosieve.splitting.base import SplitResult
from biosieve.splitting.random import RandomSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pl.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, seed=13)
    res = splitter.run(df_base, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "random"
    n_total = res.train.height + res.test.height
    assert n_total == df_base.height
    assert res.train.height > 0
    assert res.test.height > 0
    assert res.val is None


def test_val_when_requested(df_base: pl.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, val_size=0.1, seed=13)
    res = splitter.run(df_base, COLS)
    assert res.val is not None
    assert res.val.height > 0
    n_total = res.train.height + res.test.height + res.val.height
    assert n_total == df_base.height
    assert set(res.train["id"].to_list()) & set(res.val["id"].to_list()) == set()
    assert set(res.test["id"].to_list()) & set(res.val["id"].to_list()) == set()


def test_determinism(df_base: pl.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, seed=42)
    res1 = splitter.run(df_base, COLS)
    res2 = splitter.run(df_base, COLS)
    assert res1.train["id"].to_list() == res2.train["id"].to_list()
    assert res1.test["id"].to_list() == res2.test["id"].to_list()


def test_different_seeds_give_different_splits(df_base: pl.DataFrame) -> None:
    res1 = RandomSplitter(test_size=0.2, seed=1).run(df_base, COLS)
    res2 = RandomSplitter(test_size=0.2, seed=2).run(df_base, COLS)
    # With 50 rows it's astronomically unlikely these are equal
    assert set(res1.test["id"].to_list()) != set(res2.test["id"].to_list())
