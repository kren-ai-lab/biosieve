"""Shared structural invariants tested across every splitter.

Non-k-fold invariants:
  - train & test == empty set  (no sample leaks between splits)
  - train + test (+ val) covers every input sample exactly once

K-fold invariants:
  - exactly n_splits folds returned
  - every fold has non-empty train and test, and records fold_index in stats
  - each sample appears in the test fold exactly once  (complete coverage)
  - train & test == empty set per fold

These are framework-level contracts. Testing them once via parametrize avoids
repeating the same assertions in every individual strategy test module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    import pytest

    from biosieve.splitting.base import KFoldSplitter, SplitResult, Splitter

import pytest

from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


# ---------------------------------------------------------------------------
# Non-k-fold: (splitter, df) fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "random",
        "stratified",
        "group",
        "time",
        "cluster_aware",
        "stratified_numeric",
        "distance_aware",
        "homology_aware",
    ]
)
def splitter_and_df(
    request: pytest.FixtureRequest,
    df_base: pl.DataFrame,
    df_labeled: pl.DataFrame,
    df_grouped: pl.DataFrame,
    df_timed: pl.DataFrame,
    df_clustered: pl.DataFrame,
    df_full: pl.DataFrame,
    embeddings_files: tuple[Path, Path],
    cluster_map_file: Path,
) -> tuple[Splitter, pl.DataFrame]:
    from biosieve.splitting.cluster import ClusterAwareSplitter
    from biosieve.splitting.distance_aware import DistanceAwareSplitter
    from biosieve.splitting.group import GroupSplitter
    from biosieve.splitting.homology_aware import HomologyAwareSplitter
    from biosieve.splitting.random import RandomSplitter
    from biosieve.splitting.stratified import StratifiedSplitter
    from biosieve.splitting.stratified_numeric import StratifiedNumericSplitter
    from biosieve.splitting.time_based import TimeSplitter

    emb_path, ids_path = embeddings_files

    cases: dict[str, Any] = {
        "random": (RandomSplitter(test_size=0.2, seed=13), df_base),
        "stratified": (StratifiedSplitter(label_col="label", test_size=0.2, seed=13), df_labeled),
        "group": (GroupSplitter(group_col="group", test_size=0.2, seed=13), df_grouped),
        "time": (TimeSplitter(time_col="date", test_size=0.2), df_timed),
        "cluster_aware": (
            ClusterAwareSplitter(cluster_col="cluster_id", test_size=0.2, seed=13),
            df_clustered,
        ),
        "stratified_numeric": (
            StratifiedNumericSplitter(label_col="target", test_size=0.2, n_bins=5, seed=13),
            df_full,
        ),
        "distance_aware": (
            DistanceAwareSplitter(
                feature_mode="embeddings",
                embeddings_path=str(emb_path),
                ids_path=str(ids_path),
                test_size=0.2,
                seed=13,
            ),
            df_base,
        ),
        "homology_aware": (
            HomologyAwareSplitter(
                mode="precomputed",
                clusters_path=str(cluster_map_file),
                clusters_format="csv",
                member_col="id",
                cluster_col="cluster_id",
                test_size=0.2,
                seed=13,
            ),
            df_base,
        ),
    }
    return cases[request.param]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Non-k-fold invariant tests
# ---------------------------------------------------------------------------


def test_no_overlap(splitter_and_df: tuple[Splitter, pl.DataFrame]) -> None:
    """No sample id appears in both train and test."""
    splitter, df = splitter_and_df
    res: SplitResult = splitter.run(df, COLS)
    assert set(res.train["id"].to_list()) & set(res.test["id"].to_list()) == set()


def test_all_ids_covered(splitter_and_df: tuple[Splitter, pl.DataFrame]) -> None:
    """train + test (+ val) accounts for every input sample exactly once."""
    splitter, df = splitter_and_df
    res: SplitResult = splitter.run(df, COLS)
    total = res.train.height + res.test.height + (res.val.height if res.val is not None else 0)
    assert total == df.height


# ---------------------------------------------------------------------------
# K-fold: (kfold_splitter, df) fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "random_kfold",
        "stratified_kfold",
        "group_kfold",
        "stratified_numeric_kfold",
        "distance_aware_kfold",
    ]
)
def kfold_splitter_and_df(
    request: pytest.FixtureRequest,
    df_base: pl.DataFrame,
    df_labeled: pl.DataFrame,
    df_grouped: pl.DataFrame,
    df_full: pl.DataFrame,
    embeddings_files: tuple[Path, Path],
) -> tuple[KFoldSplitter, pl.DataFrame]:
    from biosieve.splitting.distance_aware_kfold import DistanceAwareKFoldSplitter
    from biosieve.splitting.group_kfold import GroupKFoldSplitter
    from biosieve.splitting.random_kfold import RandomKFoldSplitter
    from biosieve.splitting.stratified_kfold import StratifiedKFoldSplitter
    from biosieve.splitting.stratified_numeric_kfold import StratifiedNumericKFoldSplitter

    emb_path, ids_path = embeddings_files

    cases: dict[str, Any] = {
        "random_kfold": (RandomKFoldSplitter(n_splits=N_SPLITS, seed=13), df_base),
        "stratified_kfold": (
            StratifiedKFoldSplitter(n_splits=N_SPLITS, label_col="label", seed=13),
            df_labeled,
        ),
        "group_kfold": (GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13), df_grouped),
        "stratified_numeric_kfold": (
            StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="target", n_bins=5, seed=13),
            df_full,
        ),
        "distance_aware_kfold": (
            DistanceAwareKFoldSplitter(
                n_splits=N_SPLITS,
                feature_mode="embeddings",
                embeddings_path=str(emb_path),
                ids_path=str(ids_path),
                seed=13,
            ),
            df_base,
        ),
    }
    return cases[request.param]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# K-fold invariant tests
# ---------------------------------------------------------------------------


def test_kfold_returns_n_folds(kfold_splitter_and_df: tuple[KFoldSplitter, pl.DataFrame]) -> None:
    splitter, df = kfold_splitter_and_df
    folds = splitter.run_folds(df, COLS)
    assert len(folds) == N_SPLITS


def test_kfold_each_fold_nonempty(kfold_splitter_and_df: tuple[KFoldSplitter, pl.DataFrame]) -> None:
    splitter, df = kfold_splitter_and_df
    folds = splitter.run_folds(df, COLS)
    for res in folds:
        assert res.train.height > 0
        assert res.test.height > 0
        assert "fold_index" in res.stats


def test_kfold_all_ids_appear_in_test_once(kfold_splitter_and_df: tuple[KFoldSplitter, pl.DataFrame]) -> None:
    """Complete coverage: every sample is in the test set of exactly one fold."""
    splitter, df = kfold_splitter_and_df
    folds = splitter.run_folds(df, COLS)
    all_test_ids = [res.test["id"].to_list() for res in folds]
    flat = [i for fold_ids in all_test_ids for i in fold_ids]
    assert len(flat) == df.height
    assert set(flat) == set(df["id"].to_list())


def test_kfold_no_overlap_per_fold(kfold_splitter_and_df: tuple[KFoldSplitter, pl.DataFrame]) -> None:
    splitter, df = kfold_splitter_and_df
    folds = splitter.run_folds(df, COLS)
    for res in folds:
        assert set(res.train["id"].to_list()) & set(res.test["id"].to_list()) == set()
