"""Cluster-aware splitting strategy with strict cluster leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import derive_val_fraction, validate_sizes
from biosieve.splitting.group import _split_groups
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)

_INTERNAL_CLUSTER_COL = "_biosieve_cluster_id__"
MIN_CLUSTERS_FOR_SPLIT = 2


def _load_cluster_map_csv(path: str, id_col: str, cluster_col: str) -> dict[str, str]:
    """Load a CSV mapping sample ids to cluster ids.

    Args:
        path: Path to the mapping CSV.
        id_col: Column name in mapping CSV containing sample ids.
        cluster_col: Column name in mapping CSV containing cluster ids.

    Returns:
        dict[str, str]: Dictionary mapping sample id -> cluster id.
        If the mapping CSV contains duplicates for the same id, the last occurrence wins.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If required columns are missing from the mapping CSV.

    """
    p = Path(path)
    if not p.exists():
        msg = f"cluster_map_path not found: {p}"
        raise FileNotFoundError(msg)

    df = pl.read_csv(p)
    if id_col not in df.columns or cluster_col not in df.columns:
        msg = f"cluster map must contain columns '{id_col}' and '{cluster_col}'. Found: {df.columns}"
        raise ValueError(msg)

    return dict(
        zip(df[id_col].cast(pl.String).to_list(), df[cluster_col].cast(pl.String).to_list(), strict=False)
    )


def _validate_inputs(
    df: pl.DataFrame,
    cols: Columns,
    *,
    test_size: float,
    val_size: float,
    cluster_col: str,
    cluster_map_path: str | None,
    map_id_col: str,
    map_cluster_col: str,
    assign_singletons_for_missing: bool,
) -> tuple[pl.DataFrame, str, int, int]:
    validate_sizes(test_size, val_size)
    work = df.clone()

    if cluster_col in work.columns:
        work = work.with_columns(**{_INTERNAL_CLUSTER_COL: work[cluster_col].cast(pl.String)})
        used_source = "dataset_column"
        missing = 0
    elif cluster_map_path:
        cmap = _load_cluster_map_csv(cluster_map_path, map_id_col, map_cluster_col)
        ids = work[cols.id_col].cast(pl.String).to_list()

        assigned = []
        missing = 0
        for sid in ids:
            cid = cmap.get(sid)
            if cid is None:
                missing += 1
                if assign_singletons_for_missing:
                    cid = f"singleton:{sid}"
                else:
                    msg = (
                        f"Missing cluster assignment for id={sid}. "
                        "Either provide full mapping or set assign_singletons_for_missing=true."
                    )
                    raise ValueError(msg)
            assigned.append(cid)

        work = work.with_columns(pl.Series(_INTERNAL_CLUSTER_COL, assigned))
        used_source = "mapping_file"
    else:
        msg = (
            "ClusterAwareSplitter requires either: "
            f"(i) a '{cluster_col}' column in the dataset OR "
            "(ii) cluster_map_path pointing to a CSV mapping id->cluster_id."
        )
        raise ValueError(msg)

    n_clusters = int(work[_INTERNAL_CLUSTER_COL].cast(pl.String).n_unique())
    if n_clusters < MIN_CLUSTERS_FOR_SPLIT:
        msg = f"Need at least 2 clusters to split. Found {n_clusters} unique clusters."
        raise ValueError(msg)

    return work, used_source, missing, n_clusters


@dataclass(frozen=True)
class ClusterAwareSplitter:
    r"""Cluster-aware split (group-based) to prevent cluster leakage across splits.

    This strategy is a thin wrapper around group-based splitting: it ensures that a
    cluster identifier never appears in more than one of train/test/val.

    Cluster sources:
    1) Dataset column: if `cluster_col` exists in the dataset, it is used directly.
    2) Mapping file: if `cluster_map_path` is provided, cluster ids are joined by `cols.id_col`
       using a mapping CSV (id -> cluster_id).

    Missing cluster assignments (mapping mode):
    If a sample id is not present in the mapping file:
    - If `assign_singletons_for_missing=True`, it is assigned to a singleton cluster `singleton:<id>`
      which is safe (does not create leakage).
    - Otherwise, an error is raised.

    Args:
        test_size: Fraction of samples assigned to the test split (cluster-disjoint from train/val).
        val_size: Fraction of samples assigned to validation (0 disables validation).
            Internally converted to a fraction of the remaining trainval split.
        seed: Random seed for deterministic group splitting.
        cluster_col: Column name in the dataset containing cluster ids (used if present).
        cluster_map_path: Optional path to a CSV mapping ids to cluster ids.
        map_id_col: Column in the mapping CSV with ids (typically "id").
        map_cluster_col: Column in the mapping CSV with cluster ids (typically "cluster_id").
        assign_singletons_for_missing: If True, assign singleton clusters for missing ids in the mapping file.

    Returns:
        Container with train/test/val splits plus:
        - params: effective parameters and mapping configuration
        - stats: counts, number of clusters, and leakage checks

        Leakage keys (must be zero):
        - leak_clusters_train_test
        - leak_clusters_val_test
        And if val exists:
        - leak_clusters_train_val

    Raises:
        ImportError: If scikit-learn is not installed (required by group splitting backend).
        ValueError: If split sizes are invalid, if no cluster info is provided, if too few clusters
        exist to split, or if mapping coverage is incomplete and singletons are disabled.
        FileNotFoundError: If `cluster_map_path` does not exist.

    Notes:
        - This splitter enforces cluster-disjoint splits, but does not guarantee balanced class
        distributions. If you need both “no leakage” and balancing, a hybrid (future) strategy
        is recommended (e.g., cluster-level balancing with stratified_numeric).
        - The leakage checks are computed using an internal cluster column that is always present
        during splitting, ensuring correctness for both dataset-column and mapping modes.

    Examples:
        Using a cluster id already present in the dataset:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_cluster_aware \\
    ...   --strategy cluster_aware \\
    ...   --params params.yaml

    Using an external id->cluster mapping:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_cluster_aware_map \\
    ...   --strategy cluster_aware \\
    ...   --params params.yaml

    """

    # sizes
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    # cluster source
    cluster_col: str = "cluster_id"  # if present in dataset
    cluster_map_path: str | None = None  # optional mapping file id->cluster_id
    map_id_col: str = "id"
    map_cluster_col: str = "cluster_id"

    # behavior
    assign_singletons_for_missing: bool = True

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "cluster_aware"

    def run(self, df: pl.DataFrame, cols: Columns) -> SplitResult:
        """Split data by cluster assignments into disjoint partitions."""
        work, used_source, missing, n_clusters = _validate_inputs(
            df,
            cols,
            test_size=self.test_size,
            val_size=self.val_size,
            cluster_col=self.cluster_col,
            cluster_map_path=self.cluster_map_path,
            map_id_col=self.map_id_col,
            map_cluster_col=self.map_cluster_col,
            assign_singletons_for_missing=self.assign_singletons_for_missing,
        )
        cluster_ids = work[_INTERNAL_CLUSTER_COL].cast(pl.String)

        # 1) split off test by clusters
        trainval, test = _split_groups(work, cluster_ids, test_size=self.test_size, seed=self.seed)

        val = None
        train = trainval

        # 2) optional val split by clusters
        if self.val_size and self.val_size > 0:
            frac = derive_val_fraction(self.test_size, self.val_size)
            tv_clusters = trainval[_INTERNAL_CLUSTER_COL].cast(pl.String)
            train, val = _split_groups(trainval, tv_clusters, test_size=frac, seed=self.seed)

        # leakage checks using the internal cluster column (always present pre-drop)
        train_c = set(train[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list())
        test_c = set(test[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list())
        val_c = (
            set(val[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list()) if val is not None else set()
        )

        leak_tt = len(train_c & test_c)
        leak_tv = len(train_c & val_c) if val is not None else 0
        leak_vt = len(val_c & test_c) if val is not None else 0

        # cleanup internal column before returning
        train = train.drop([_INTERNAL_CLUSTER_COL])
        test = test.drop([_INTERNAL_CLUSTER_COL])
        if val is not None:
            val = val.drop([_INTERNAL_CLUSTER_COL])

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
            "cluster_col": self.cluster_col,
            "n_clusters_total": int(n_clusters),
            "cluster_source": used_source,
            "n_missing_cluster_assignments": int(missing),
            "leak_clusters_train_test": int(leak_tt),
            "leak_clusters_train_val": int(leak_tv),
            "leak_clusters_val_test": int(leak_vt),
            "note": (
                "cluster-aware split uses group-based splitting to prevent cluster leakage across splits."
            ),
        }

        params = {
            "test_size": self.test_size,
            "val_size": self.val_size,
            "seed": self.seed,
            "cluster_col": self.cluster_col,
            "cluster_map_path": self.cluster_map_path,
            "map_id_col": self.map_id_col,
            "map_cluster_col": self.map_cluster_col,
            "assign_singletons_for_missing": self.assign_singletons_for_missing,
        }

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params=params,
            stats=stats,
        )
