"""Tests for MinHashJaccardReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import pytest

from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")

datasketch = pytest.importorskip("datasketch", reason="datasketch not installed; skip minhash tests")


def test_happy_path(df_base: pd.DataFrame) -> None:
    from biosieve.reduction.base import ReductionResult
    from biosieve.reduction.minhash_jaccard import MinHashJaccardReducer

    reducer = MinHashJaccardReducer(threshold=0.3, k=3, num_perm=64)
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "minhash_jaccard"
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_high_threshold_removes_nothing(df_base: pd.DataFrame) -> None:
    """threshold=1.0 → no LSH candidates match → all sequences kept."""
    from biosieve.reduction.minhash_jaccard import MinHashJaccardReducer

    reducer = MinHashJaccardReducer(threshold=1.0, k=3, num_perm=64)
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_missing_sequence_col(df_base: pd.DataFrame) -> None:
    from biosieve.reduction.minhash_jaccard import MinHashJaccardReducer

    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = MinHashJaccardReducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)


def test_importerror_without_datasketch(df_base: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate missing datasketch — run() must raise ImportError."""
    import biosieve.reduction.minhash_jaccard as mod
    from biosieve.reduction.minhash_jaccard import MinHashJaccardReducer

    monkeypatch.setattr(mod, "_try_import_datasketch", lambda: (None, None))
    reducer = MinHashJaccardReducer()
    with pytest.raises(ImportError, match="datasketch"):
        reducer.run(df_base, COLS)
