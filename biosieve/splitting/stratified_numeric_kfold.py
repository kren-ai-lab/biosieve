# ruff: noqa: ANN202, ANN401, D102, EM102, TRY003, TRY300

"""Stratified k-fold splitter for numeric targets via binning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import sklearn_required_message, try_import_train_test_split, validate_kfold
from biosieve.splitting.stratified_numeric import _bin_counts, _label_stats, _make_bins


def _try_import_stratified_kfold():
    try:
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        return StratifiedKFold
    except ImportError:
        return None


@dataclass(frozen=True)
class StratifiedNumericKFoldSplitter:
    """Stratified K-fold splitting for numeric labels via binning."""

    label_col: str = "y"
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13
    n_bins: int = 10
    binning: Literal["quantile", "uniform"] = "quantile"
    duplicates: Literal["drop", "raise"] = "drop"
    auto_reduce_bins: bool = True
    min_bin_count: int = 2
    val_size: float = 0.0
    dropna: bool = True
    report_bin_edges: bool = False

    @property
    def strategy(self) -> str:
        return "stratified_numeric_kfold"

    def run_folds(self, df: pl.DataFrame, _cols: Any) -> list[SplitResult]:
        skf_cls = _try_import_stratified_kfold()
        if skf_cls is None:
            raise ImportError(sklearn_required_message("StratifiedNumericKFoldSplitter"))
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column '{self.label_col}'. Columns: {df.columns}")

        work = df.clone()
        y_series = work[self.label_col].cast(pl.Float64, strict=False)
        if self.dropna:
            keep = y_series.is_not_null()
            dropped = int((~keep).sum())
            work = work.filter(keep)
            y_series = work[self.label_col].cast(pl.Float64, strict=False)
        else:
            dropped = int(y_series.is_null().sum())
            if dropped:
                raise ValueError(f"Found {dropped} NaN labels in '{self.label_col}'.")

        y = y_series.to_numpy()
        validate_kfold(self.n_splits, self.val_size, n_samples=work.height)
        bins, n_eff = _make_bins(
            y,
            n_bins=self.n_bins,
            binning=self.binning,
            min_bin_count=max(self.min_bin_count, self.n_splits),
            auto_reduce_bins=self.auto_reduce_bins,
        )

        skf = skf_cls(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        tts = try_import_train_test_split() if self.val_size > 0 else None
        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(work.height), bins)):
            train = work[train_idx]
            test = work[test_idx]
            val = None
            train_global_idx = np.asarray(train_idx, dtype=int)
            val_global_idx = np.asarray([], dtype=int)
            if self.val_size > 0:
                if tts is None:
                    raise ImportError(
                        sklearn_required_message("StratifiedNumericKFoldSplitter with val_size > 0")
                    )
                inner_idx = np.arange(train.height)
                train_keep_idx, val_idx = tts(
                    inner_idx,
                    test_size=self.val_size,
                    random_state=self.seed + fold_idx,
                    shuffle=True,
                    stratify=None,
                )
                val = train[val_idx]
                train = train[train_keep_idx]
                train_global_idx = np.asarray(train_idx, dtype=int)[np.asarray(train_keep_idx, dtype=int)]
                val_global_idx = np.asarray(train_idx, dtype=int)[np.asarray(val_idx, dtype=int)]

            folds.append(
                SplitResult(
                    train=train,
                    test=test,
                    val=val,
                    strategy=self.strategy,
                    params={
                        "label_col": self.label_col,
                        "n_splits": self.n_splits,
                        "seed": self.seed,
                        "n_bins": self.n_bins,
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": fold_idx,
                        "n_total": df.height,
                        "n_used": work.height,
                        "n_dropped_nan": dropped,
                        "n_train": train.height,
                        "n_test": test.height,
                        "n_val": val.height if val is not None else 0,
                        "n_bins_effective": n_eff,
                        "train_bin_counts": _bin_counts(bins[train_global_idx])
                        if train_global_idx.size
                        else {},
                        "test_bin_counts": _bin_counts(bins[np.asarray(test_idx, dtype=int)]),
                        "train_label_stats": _label_stats(
                            train[self.label_col].cast(pl.Float64, strict=False).to_numpy()
                        ),
                        "test_label_stats": _label_stats(
                            test[self.label_col].cast(pl.Float64, strict=False).to_numpy()
                        ),
                    },
                )
            )
            if val is not None:
                folds[-1].stats["val_bin_counts"] = (
                    _bin_counts(bins[val_global_idx]) if val_global_idx.size else {}
                )
        return folds
