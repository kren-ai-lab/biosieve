from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        return train_test_split
    except Exception:
        return None


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


@dataclass(frozen=True)
class StratifiedSplitter:
    """
    Stratified train/test(/val) split for classification.
    Requires a label column (e.g., y).
    """
    label_col: str = "label"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    @property
    def strategy(self) -> str:
        return "stratified"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        tts = _try_import_train_test_split()
        if tts is None:
            raise ImportError("StratifiedSplitter requires scikit-learn. Install: conda install -c conda-forge scikit-learn")

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        if self.label_col not in work.columns:
            raise ValueError(f"Missing label column '{self.label_col}'. Columns: {work.columns.tolist()}")

        y = work[self.label_col]

        # 1) split off test
        trainval, test = tts(
            work,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=y,
        )

        val = None
        train = trainval

        # 2) optional split train/val (stratified within trainval)
        if self.val_size and self.val_size > 0:
            # val_size is fraction of TOTAL; convert to fraction of trainval
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                raise ValueError("Derived val fraction invalid. Check test_size/val_size.")
            y_tv = trainval[self.label_col]
            train, val = tts(
                trainval,
                test_size=frac,
                random_state=self.seed,
                shuffle=True,
                stratify=y_tv,
            )

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if val is not None:
            val = val.reset_index(drop=True)

        stats: Dict[str, Any] = {
            "n_total": int(len(work)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "label_col": self.label_col,
            "train_label_counts": train[self.label_col].value_counts(dropna=False).to_dict(),
            "test_label_counts": test[self.label_col].value_counts(dropna=False).to_dict(),
        }
        if val is not None:
            stats["val_label_counts"] = val[self.label_col].value_counts(dropna=False).to_dict()

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "label_col": self.label_col,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "seed": self.seed,
            },
            stats=stats,
        )
