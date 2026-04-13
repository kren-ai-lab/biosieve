"""Tests for DistanceAwareSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest


import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.distance_aware import DistanceAwareSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path_embeddings(df_base: pd.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
    emb_path, ids_path = embeddings_files
    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "distance_aware"
    assert len(res.train) + len(res.test) == len(df_base)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap_embeddings(df_base: pd.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
    emb_path, ids_path = embeddings_files
    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_stats_have_distance_info(df_base: pd.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
    emb_path, ids_path = embeddings_files
    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert "distance_stats_global" in res.stats or "feature_meta" in res.stats


def test_happy_path_descriptors(df_descriptors: pd.DataFrame) -> None:
    splitter = DistanceAwareSplitter(
        feature_mode="descriptors",
        descriptor_prefix="desc_",
        metric="euclidean",
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_descriptors, COLS)

    assert len(res.train) + len(res.test) == len(df_descriptors)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_missing_embeddings_raises(df_base: pd.DataFrame, tmp_path: Path) -> None:
    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(tmp_path / "nonexistent.npy"),
        ids_path=str(tmp_path / "nonexistent_ids.csv"),
        test_size=0.2,
    )
    with pytest.raises(FileNotFoundError):
        splitter.run(df_base, COLS)
