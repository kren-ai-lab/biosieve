from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.splitting.group import _split_groups, _validate_sizes
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

log = get_logger(__name__)

_INTERNAL_CLUSTER_COL = "_biosieve_cluster_id__"


def _load_cluster_map_csv(path: str, id_col: str, cluster_col: str) -> dict[str, str]:
    """Load a CSV mapping sample ids to cluster ids.

    Parameters
    ----------
    path:
        Path to the mapping CSV.
    id_col:
        Column name in mapping CSV containing sample ids.
    cluster_col:
        Column name in mapping CSV containing cluster ids.

    Returns
    -------
    dict[str, str]
        Dictionary mapping sample id -> cluster id.
        If the mapping CSV contains duplicates for the same id, the last occurrence wins.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If required columns are missing from the mapping CSV.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cluster_map_path not found: {p}")

    df = pd.read_csv(p)
    if id_col not in df.columns or cluster_col not in df.columns:
        raise ValueError(
            f"cluster map must contain columns '{id_col}' and '{cluster_col}'. Found: {df.columns.tolist()}"
        )

    return dict(zip(df[id_col].astype(str), df[cluster_col].astype(str)))


@dataclass(frozen=True)
class ClusterAwareSplitter:
    """Cluster-aware split (group-based) to prevent cluster leakage across splits.

    This strategy is a thin wrapper around group-based splitting: it ensures that a
    cluster identifier never appears in more than one of train/test/val.

    Cluster sources
    ---------------
    1) Dataset column: if `cluster_col` exists in the dataset, it is used directly.
    2) Mapping file: if `cluster_map_path` is provided, cluster ids are joined by `cols.id_col`
       using a mapping CSV (id -> cluster_id).

    Missing cluster assignments (mapping mode)
    -----------------------------------------
    If a sample id is not present in the mapping file:
    - If `assign_singletons_for_missing=True`, it is assigned to a singleton cluster `singleton:<id>`
      which is safe (does not create leakage).
    - Otherwise, an error is raised.

    Parameters
    ----------
    test_size:
        Fraction of samples assigned to the test split (cluster-disjoint from train/val).
    val_size:
        Fraction of samples assigned to validation (0 disables validation).
        Internally converted to a fraction of the remaining trainval split.
    seed:
        Random seed for deterministic group splitting.
    cluster_col:
        Column name in the dataset containing cluster ids (used if present).
    cluster_map_path:
        Optional path to a CSV mapping ids to cluster ids.
    map_id_col:
        Column in the mapping CSV with ids (typically "id").
    map_cluster_col:
        Column in the mapping CSV with cluster ids (typically "cluster_id").
    assign_singletons_for_missing:
        If True, assign singleton clusters for missing ids in the mapping file.

    Returns
    -------
    SplitResult
        Container with train/test/val splits plus:
        - params: effective parameters and mapping configuration
        - stats: counts, number of clusters, and leakage checks

        Leakage keys (must be zero):
        - leak_clusters_train_test
        - leak_clusters_val_test
        And if val exists:
        - leak_clusters_train_val

    Raises
    ------
    ImportError
        If scikit-learn is not installed (required by group splitting backend).
    ValueError
        If split sizes are invalid, if no cluster info is provided, if too few clusters
        exist to split, or if mapping coverage is incomplete and singletons are disabled.
    FileNotFoundError
        If `cluster_map_path` does not exist.

    Notes
    -----
    - This splitter enforces cluster-disjoint splits, but does not guarantee balanced class
      distributions. If you need both “no leakage” and balancing, a hybrid (future) strategy
      is recommended (e.g., cluster-level balancing with stratified_numeric).
    - The leakage checks are computed using an internal cluster column that is always present
      during splitting, ensuring correctness for both dataset-column and mapping modes.

    Examples
    --------
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
        return "cluster_aware"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)

        # Always create an internal cluster id column to support correct leakage checks
        if self.cluster_col in work.columns:
            work[_INTERNAL_CLUSTER_COL] = work[self.cluster_col].astype(str)
            used_source = "dataset_column"
            missing = 0
        elif self.cluster_map_path:
            cmap = _load_cluster_map_csv(self.cluster_map_path, self.map_id_col, self.map_cluster_col)
            ids = work[cols.id_col].astype(str)

            assigned = []
            missing = 0
            for sid in ids:
                cid = cmap.get(sid)
                if cid is None:
                    missing += 1
                    if self.assign_singletons_for_missing:
                        cid = f"singleton:{sid}"
                    else:
                        raise ValueError(
                            f"Missing cluster assignment for id={sid}. "
                            f"Either provide full mapping or set assign_singletons_for_missing=true."
                        )
                assigned.append(cid)

            work[_INTERNAL_CLUSTER_COL] = pd.Series(assigned, index=work.index, dtype="string").astype(str)
            used_source = "mapping_file"
        else:
            raise ValueError(
                "ClusterAwareSplitter requires either: "
                f"(i) a '{self.cluster_col}' column in the dataset OR "
                "(ii) cluster_map_path pointing to a CSV mapping id->cluster_id."
            )

        cluster_ids = work[_INTERNAL_CLUSTER_COL].astype(str)
        n_clusters = int(cluster_ids.nunique(dropna=False))
        if n_clusters < 2:
            raise ValueError(f"Need at least 2 clusters to split. Found {n_clusters} unique clusters.")

        # 1) split off test by clusters
        trainval, test = _split_groups(work, cluster_ids, test_size=self.test_size, seed=self.seed)

        val = None
        train = trainval

        # 2) optional val split by clusters
        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                raise ValueError("Derived val fraction invalid. Check test_size/val_size.")
            tv_clusters = trainval[_INTERNAL_CLUSTER_COL].astype(str)
            train, val = _split_groups(trainval, tv_clusters, test_size=frac, seed=self.seed)

        # leakage checks using the internal cluster column (always present pre-drop)
        train_c = set(train[_INTERNAL_CLUSTER_COL].astype(str).unique())
        test_c = set(test[_INTERNAL_CLUSTER_COL].astype(str).unique())
        val_c = set(val[_INTERNAL_CLUSTER_COL].astype(str).unique()) if val is not None else set()

        leak_tt = len(train_c & test_c)
        leak_tv = len(train_c & val_c) if val is not None else 0
        leak_vt = len(val_c & test_c) if val is not None else 0

        # Enforce invariants: cluster leakage must be zero
        if leak_tt != 0 or leak_vt != 0 or leak_tv != 0:
            raise ValueError(
                "Cluster leakage detected (this should never happen with group-based splitting). "
                f"leak_train_test={leak_tt}, leak_train_val={leak_tv}, leak_val_test={leak_vt}"
            )

        # cleanup internal column before returning
        train = train.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)
        test = test.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)
        if val is not None:
            val = val.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "cluster_col": self.cluster_col,
            "n_clusters_total": int(n_clusters),
            "cluster_source": used_source,
            "n_missing_cluster_assignments": int(missing),
            "leak_clusters_train_test": 0,
            "leak_clusters_train_val": 0,
            "leak_clusters_val_test": 0,
            "note": "cluster-aware split uses group-based splitting to prevent cluster leakage across splits.",
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
