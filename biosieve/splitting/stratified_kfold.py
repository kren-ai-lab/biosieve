from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

log = get_logger(__name__)


def _try_import_stratified_kfold():
    try:
        from sklearn.model_selection import StratifiedKFold  # type: ignore

        return StratifiedKFold
    except Exception:
        return None


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split  # type: ignore

        return train_test_split
    except Exception:
        return None


@dataclass(frozen=True)
class StratifiedKFoldSplitter:
    """
    Stratified K-Fold splitting for classification labels.

    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold
      - val:   optional validation subset sampled from train (val_size > 0)

    Notes
    -----
    - Stratification is performed using `label_col`.
    - If val_size > 0, validation is sampled randomly from the fold's train
      (non-stratified by default to keep it simple and robust).
      If you want stratified val too, we can add `val_stratify=true`.
    """

    label_col: str = "label"

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13

    # optional validation inside train for each fold
    val_size: float = 0.0

    # behavior
    dropna: bool = True  # drop rows with NaN label
    cast_to_str: bool = False  # optionally cast labels to str (useful if labels are mixed types)

    @property
    def strategy(self) -> str:
        return "stratified_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> List[SplitResult]:
        StratifiedKFold = _try_import_stratified_kfold()
        if StratifiedKFold is None:
            raise ImportError(
                "StratifiedKFoldSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.val_size < 0 or self.val_size >= 1:
            raise ValueError("val_size must be in [0, 1)")
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column '{self.label_col}'. Columns: {df.columns.tolist()}")

        work = df.copy().reset_index(drop=True)

        y = work[self.label_col]

        # handle missing labels
        if self.dropna:
            keep = ~pd.to_numeric(y).isna() if y.dtype.kind in "fiu" else ~y.isna()
            # above is defensive; for object labels ~isna() is enough
            keep = ~y.isna()
            dropped = int((~keep).sum())
            work = work.loc[keep].reset_index(drop=True)
            y = work[self.label_col].reset_index(drop=True)
        else:
            if y.isna().any():
                raise ValueError(f"Found NaN labels in '{self.label_col}'. Set dropna=true or clean dataset.")
            dropped = 0

        if self.cast_to_str:
            y = y.astype(str)

        n = len(work)
        if n < self.n_splits:
            raise ValueError(f"Not enough samples (n={n}) for n_splits={self.n_splits}")

        # sanity: each class must have at least n_splits members for StratifiedKFold
        vc = pd.Series(y).value_counts(dropna=False)
        too_small = vc[vc < self.n_splits]
        if len(too_small) > 0:
            raise ValueError(
                "Some classes have fewer samples than n_splits, cannot stratify k-fold. "
                f"n_splits={self.n_splits}. Problem classes: {too_small.to_dict()}"
            )

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                raise ImportError("val_size > 0 requires scikit-learn train_test_split.")

        folds: List[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(work, y)):
            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            val_df: Optional[pd.DataFrame] = None

            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)

                # default: random val from train (robust)
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
                        "label_col": self.label_col,
                        "n_splits": self.n_splits,
                        "shuffle": self.shuffle,
                        "seed": self.seed,
                        "val_size": self.val_size,
                        "dropna": self.dropna,
                        "cast_to_str": self.cast_to_str,
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": int(len(df)),
                        "n_used": int(n),
                        "n_dropped_nan": int(dropped),
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "n_val": int(len(val_df)) if val_df is not None else 0,
                        "train_label_counts": train_df[self.label_col].astype(str).value_counts().to_dict(),
                        "test_label_counts": test_df[self.label_col].astype(str).value_counts().to_dict(),
                    },
                )
            )

        return folds
