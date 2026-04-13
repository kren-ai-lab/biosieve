"""Group-aware splitting strategy to prevent group leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from biosieve.types import Columns

log = get_logger(__name__)
MIN_GROUPS_FOR_SPLIT = 2


class _GroupShuffleSplit(Protocol):
    def split(self, X: object, *, groups: pd.Series) -> Iterator[tuple[list[int], list[int]]]: ...


class _GroupShuffleSplitFactory(Protocol):
    def __call__(self, *, n_splits: int, test_size: float, random_state: int) -> _GroupShuffleSplit: ...


def _try_import_gss() -> _GroupShuffleSplitFactory | None:
    try:
        from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

        return cast("_GroupShuffleSplitFactory", GroupShuffleSplit)
    except ImportError:
        return None


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if test_size + val_size >= 1.0:
        msg = "test_size + val_size must be < 1.0"
        raise ValueError(msg)


def _validate_inputs(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
) -> tuple[pd.Series, int]:
    _validate_sizes(test_size, val_size)
    if group_col not in df.columns:
        msg = f"Missing group column '{group_col}'. Columns: {df.columns.tolist()}"
        raise ValueError(msg)

    groups = df[group_col].astype(str)
    if groups.isna().any():
        msg = f"Found NaN group ids in '{group_col}'. Clean dataset before splitting."
        raise ValueError(msg)

    n_groups = int(groups.nunique(dropna=False))
    if n_groups < MIN_GROUPS_FOR_SPLIT:
        msg = f"Need at least 2 groups to split. Found {n_groups} unique groups in '{group_col}'."
        raise ValueError(msg)
    return groups, n_groups


def _split_groups(
    df: pd.DataFrame,
    groups: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test using GroupShuffleSplit.

    Args:
        df: Input DataFrame.
        groups: Group labels aligned with `df` rows.
        test_size: Fraction of samples assigned to the test split (group-aware).
        seed: Random seed.

    Returns:
        train, test:
        DataFrames with disjoint groups.

    Raises:
        ImportError: If scikit-learn is not installed.

    """
    GroupShuffleSplit = _try_import_gss()
    if GroupShuffleSplit is None:
        msg = "GroupSplitter requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
        raise ImportError(msg)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = df.index.to_numpy()

    train_idx, test_idx = next(gss.split(idx, groups=groups))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    return train, test


@dataclass(frozen=True)
class GroupSplitter:
    r"""Group-aware train/test(/val) split (leakage-aware by groups).

    This splitter ensures that a group identifier never appears in more than one
    of train/test/val. This is critical for biological datasets where samples
    are not i.i.d. (e.g., multiple proteins from the same organism, homologous
    clusters, subjects, families).

    Procedure:
    1) Split off the test set by groups (GroupShuffleSplit).
    2) Optionally split a validation set from the remaining trainval by groups.

    Args:
        group_col: Column defining group IDs (e.g., taxid, subject_id, cluster_id).
        test_size: Fraction of samples assigned to the test set (group-disjoint from train/val).
        val_size: Fraction of samples assigned to validation (0 disables validation).
            Internally converted to a fraction of the remaining trainval split.
        seed: Random seed for deterministic splitting.

    Returns:
        Container with train/test/val DataFrames plus:
        - params: effective parameters
        - stats: counts, unique groups per split, and leakage checks

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If group column is missing, if there are too few groups to split, or if
        `test_size`/`val_size` are invalid.

    Notes:
        - Leakage contract:
        `leak_groups_train_test == 0` and `leak_groups_val_test == 0` must hold.
        Validation is split by groups too, so `leak_groups_train_val == 0` also holds.
        - If your "group" is actually a homology cluster id, this is effectively
        homology-aware splitting.

    Examples:
        >>> biosieve split \\
        ...   --in dataset.csv \\
        ...   --outdir runs/split_group \\
        ...   --strategy group \\
        ...   --params params.yaml

    """

    group_col: str = "group"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "group"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        """Split data into group-disjoint train/test/(val) sets."""
        log.info(
            "group:start | group_col=%s | test_size=%.3f | val_size=%.3f",
            cols.group_col,
            self.test_size,
            self.val_size,
        )
        log.debug("group:params | %s", self.__dict__)

        work = df.copy().reset_index(drop=True)
        groups, n_groups = _validate_inputs(
            work,
            self.group_col,
            self.test_size,
            self.val_size,
        )

        # 1) test split
        trainval, test = _split_groups(work, groups, test_size=self.test_size, seed=self.seed)

        val = None
        train = trainval

        # 2) optional val split from trainval
        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                msg = "Derived val fraction invalid. Check test_size/val_size."
                raise ValueError(msg)

            tv_groups = trainval[self.group_col].astype(str)
            tv_n_groups = int(tv_groups.nunique(dropna=False))
            if tv_n_groups < MIN_GROUPS_FOR_SPLIT:
                msg = (
                    f"Not enough groups left after test split to create validation. "
                    f"Groups in trainval: {tv_n_groups}. Reduce test_size/val_size."
                )
                raise ValueError(msg)

            train, val = _split_groups(trainval, tv_groups, test_size=frac, seed=self.seed)

        # reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if val is not None:
            val = val.reset_index(drop=True)

        # leakage checks
        def _group_set(x: pd.DataFrame) -> set[str]:
            return set(x[self.group_col].astype(str).tolist())

        train_g = _group_set(train)
        test_g = _group_set(test)
        val_g = _group_set(val) if val is not None else set()

        leak_tt = len(train_g & test_g)
        leak_tv = len(train_g & val_g)
        leak_vt = len(val_g & test_g)

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "group_col": self.group_col,
            "n_groups_total": int(n_groups),
            "n_groups_train": len(train_g),
            "n_groups_test": len(test_g),
            "n_groups_val": len(val_g) if val is not None else 0,
            "leak_groups_train_test": int(leak_tt),
            "leak_groups_train_val": int(leak_tv),
            "leak_groups_val_test": int(leak_vt),
        }

        log.info(
            "group:stats | groups=%d | train=%d | val=%d | test=%d",
            n_groups,
            stats["n_train"],
            stats["n_val"],
            stats["n_test"],
        )

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "group_col": self.group_col,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "seed": self.seed,
            },
            stats=stats,
        )
