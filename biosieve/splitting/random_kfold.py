from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from biosieve.types import Columns

log = get_logger(__name__)


class _KFold(Protocol):
    def split(self, X: object) -> Iterator[tuple[list[int], list[int]]]: ...


class _KFoldFactory(Protocol):
    def __call__(self, *, n_splits: int, shuffle: bool, random_state: int) -> _KFold: ...


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def _try_import_kfold() -> _KFoldFactory | None:
    try:
        from sklearn.model_selection import KFold

        return cast("_KFoldFactory", KFold)
    except Exception:
        return None


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split

        return cast("_TrainTestSplitFn", train_test_split)
    except Exception:
        return None


@dataclass(frozen=True)
class RandomKFoldSplitter:
    """Random K-Fold splitting.

    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold
      - val:   optional validation subset sampled from train (val_size > 0)

    Notes
    -----
    - This is *random* KFold, not stratified.
    - For reproducibility, each fold uses seed + fold_index for the optional val split.

    """

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13

    # optional val inside train for each fold
    val_size: float = 0.0

    @property
    def strategy(self) -> str:
        return "random_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> list[SplitResult]:
        KFold = _try_import_kfold()
        if KFold is None:
            raise ImportError(
                "RandomKFoldSplitter requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
            )

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.val_size < 0 or self.val_size >= 1:
            raise ValueError("val_size must be in [0, 1)")

        work = df.copy().reset_index(drop=True)
        n = len(work)
        if n < self.n_splits:
            raise ValueError(f"Not enough samples (n={n}) for n_splits={self.n_splits}")

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                raise ImportError(
                    "val_size > 0 requires scikit-learn train_test_split. Install scikit-learn."
                )

        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(work)):
            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            val_df: pd.DataFrame | None = None

            if self.val_size and self.val_size > 0:
                # deterministic per-fold val split
                seed_fold = int(self.seed + fold_idx)
                assert tts is not None
                train_df, val_df = tts(
                    train_df,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)

            folds.append(
                SplitResult(
                    train=train_df,
                    test=test_df,
                    val=val_df,
                    strategy=self.strategy,
                    params={
                        "n_splits": self.n_splits,
                        "shuffle": self.shuffle,
                        "seed": self.seed,
                        "val_size": self.val_size,
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": int(n),
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "n_val": len(val_df) if val_df is not None else 0,
                    },
                )
            )

        return folds
