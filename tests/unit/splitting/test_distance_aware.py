"""Tests for DistanceAwareSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    import pytest

import numpy as np
import polars as pl
import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.distance_aware import DistanceAwareSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path_embeddings(df_base: pl.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
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


def test_stats_have_distance_info(df_base: pl.DataFrame, embeddings_files: tuple[Path, Path]) -> None:
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


def test_happy_path_descriptors(df_descriptors: pl.DataFrame) -> None:
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


def test_missing_embeddings_raises(df_base: pl.DataFrame, tmp_path: Path) -> None:
    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(tmp_path / "nonexistent.npy"),
        ids_path=str(tmp_path / "nonexistent_ids.csv"),
        test_size=0.2,
    )
    with pytest.raises(FileNotFoundError):
        splitter.run(df_base, COLS)


def test_partial_embedding_coverage_raises_for_undersized_feature_pool(
    df_base: pl.DataFrame, tmp_path: Path
) -> None:
    covered = df_base.head(2)
    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "embedding_ids.csv"
    np.save(emb_path, np.zeros((covered.height, 4), dtype=np.float32))
    pl.DataFrame({"id": covered["id"].cast(pl.String).to_list()}).write_csv(ids_path)

    splitter = DistanceAwareSplitter(
        feature_mode="embeddings",
        embeddings_path=str(emb_path),
        ids_path=str(ids_path),
        test_size=0.2,
        val_size=0.1,
        seed=13,
    )
    with pytest.raises(ValueError, match="Not enough feature-covered samples"):
        splitter.run(df_base, COLS)
