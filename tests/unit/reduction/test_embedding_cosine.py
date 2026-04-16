"""Tests for EmbeddingCosineReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest


import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.embedding_cosine import EmbeddingCosineReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_base: pd.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
    emb_path, ids_path = embeddings_files
    reducer = EmbeddingCosineReducer(
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        threshold=0.5,
        use_faiss=False,
        n_jobs=1,
    )
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "embedding_cosine"
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_high_threshold_removes_nothing(df_base: pd.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
    """threshold=1.0 (max cosine similarity) → nothing removed."""
    emb_path, ids_path = embeddings_files
    reducer = EmbeddingCosineReducer(
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        threshold=1.0,
        use_faiss=False,
    )
    res = reducer.run(df_base, COLS)
    assert len(res.df) == len(df_base)


def test_missing_embeddings_file_raises(df_base: pd.DataFrame, tmp_path: Path) -> None:
    reducer = EmbeddingCosineReducer(
        embeddings_path=str(tmp_path / "nonexistent.npy"),
        ids_path=str(tmp_path / "nonexistent_ids.csv"),
        use_faiss=False,
    )
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        reducer.run(df_base, COLS)
