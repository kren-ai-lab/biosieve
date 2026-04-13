"""Tests for RandomSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


from biosieve.splitting.base import SplitResult
from biosieve.splitting.random import RandomSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pd.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, seed=13)
    res = splitter.run(df_base, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "random"
    n_total = len(res.train) + len(res.test)
    assert n_total == len(df_base)
    assert len(res.train) > 0
    assert len(res.test) > 0
    assert res.val is None


def test_no_overlap(df_base: pd.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, seed=13)
    res = splitter.run(df_base, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_val_when_requested(df_base: pd.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, val_size=0.1, seed=13)
    res = splitter.run(df_base, COLS)
    assert res.val is not None
    assert len(res.val) > 0
    n_total = len(res.train) + len(res.test) + len(res.val)
    assert n_total == len(df_base)
    assert set(res.train["id"]) & set(res.val["id"]) == set()
    assert set(res.test["id"]) & set(res.val["id"]) == set()


def test_determinism(df_base: pd.DataFrame) -> None:
    splitter = RandomSplitter(test_size=0.2, seed=42)
    res1 = splitter.run(df_base, COLS)
    res2 = splitter.run(df_base, COLS)
    assert list(res1.train["id"]) == list(res2.train["id"])
    assert list(res1.test["id"]) == list(res2.test["id"])


def test_different_seeds_give_different_splits(df_base: pd.DataFrame) -> None:
    res1 = RandomSplitter(test_size=0.2, seed=1).run(df_base, COLS)
    res2 = RandomSplitter(test_size=0.2, seed=2).run(df_base, COLS)
    # With 50 rows it's astronomically unlikely these are equal
    assert set(res1.test["id"]) != set(res2.test["id"])
