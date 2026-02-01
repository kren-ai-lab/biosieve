from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
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


def _make_bins(
    y: pd.Series,
    *,
    n_bins: int,
    binning: str,
    duplicates: str,
) -> pd.Series:
    """
    Convert numeric labels to categorical bins for stratification.

    binning:
      - "quantile": pd.qcut
      - "uniform": pd.cut

    duplicates:
      - passed to qcut (drop/raise) to handle repeated quantiles
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if binning == "quantile":
        # qcut returns categories; labels=False returns integer bins
        try:
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates=duplicates)
        except ValueError as e:
            # common case: too many duplicates -> fewer unique bin edges
            raise ValueError(
                f"qcut failed for n_bins={n_bins}. "
                f"Try fewer bins or duplicates='drop'. Original error: {e}"
            )
        return bins.astype("Int64")

    if binning == "uniform":
        bins = pd.cut(y, bins=n_bins, labels=False, include_lowest=True)
        return bins.astype("Int64")

    raise ValueError("binning must be 'quantile' or 'uniform'")


@dataclass(frozen=True)
class StratifiedNumericSplitter:
    """
    Stratified split for numeric labels via binning.

    Steps:
      1) bin numeric label y into categorical bins
      2) stratify train/test(/val) using bins

    Parameters
    ----------
    label_col:
        Column containing numeric target values.
    n_bins:
        Number of bins used to stratify.
    binning:
        "quantile" (recommended) or "uniform".
    duplicates:
        For quantile binning only: "drop" or "raise".
        - "drop" reduces number of bins if quantiles are not unique.
    """

    label_col: str = "y"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    n_bins: int = 10
    binning: str = "quantile"     # "quantile" | "uniform"
    duplicates: str = "drop"      # "drop" | "raise"

    dropna: bool = True           # if True, rows with NaN label are removed
    report_bin_edges: bool = False

    @property
    def strategy(self) -> str:
        return "stratified_numeric"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        tts = _try_import_train_test_split()
        if tts is None:
            raise ImportError(
                "StratifiedNumericSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)

        if self.label_col not in work.columns:
            raise ValueError(f"Missing numeric label column '{self.label_col}'. Columns: {work.columns.tolist()}")

        y_raw = pd.to_numeric(work[self.label_col], errors="coerce")

        if self.dropna:
            keep = ~y_raw.isna()
            dropped = int((~keep).sum())
            work = work.loc[keep].reset_index(drop=True)
            y_raw = y_raw.loc[keep].reset_index(drop=True)
        else:
            dropped = int(y_raw.isna().sum())
            if dropped > 0:
                raise ValueError(
                    f"Found {dropped} NaN labels in '{self.label_col}'. "
                    "Set dropna=true or clean the dataset."
                )

        if len(work) < 3:
            raise ValueError("Not enough samples after dropping NaNs to split.")

        # bins for stratification
        bins = _make_bins(y_raw, n_bins=self.n_bins, binning=self.binning, duplicates=self.duplicates)

        # If duplicates='drop', the actual number of bins can be smaller
        n_bins_effective = int(pd.Series(bins).nunique(dropna=True))
        if n_bins_effective < 2:
            raise ValueError(
                f"Effective number of bins is {n_bins_effective}. "
                f"Cannot stratify. Try fewer bins or different binning."
            )

        # Some bins may have very low counts; sklearn will error if any bin has < 2 when splitting.
        bin_counts = bins.value_counts(dropna=False)
        min_count = int(bin_counts.min())
        if min_count < 2:
            raise ValueError(
                "Some bins have <2 samples, which makes stratified splitting impossible. "
                f"min bin count={min_count}. Try fewer bins (n_bins) or different binning."
            )

        # split off test
        trainval, test = tts(
            work,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=bins,
        )

        val = None
        train = trainval

        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                raise ValueError("Derived val fraction invalid. Check test_size/val_size.")

            y_tv = pd.to_numeric(trainval[self.label_col], errors="coerce")
            bins_tv = _make_bins(y_tv, n_bins=self.n_bins, binning=self.binning, duplicates=self.duplicates)

            # same min-count check
            bc_tv = bins_tv.value_counts(dropna=False)
            if int(bc_tv.min()) < 2:
                raise ValueError(
                    "After test split, some bins have <2 samples in trainval, cannot create validation. "
                    "Reduce val_size/test_size or use fewer bins."
                )

            train, val = tts(
                trainval,
                test_size=frac,
                random_state=self.seed,
                shuffle=True,
                stratify=bins_tv,
            )

        # reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if val is not None:
            val = val.reset_index(drop=True)

        # report bin distributions
        def _bin_counts_for(split_df: pd.DataFrame) -> Dict[str, int]:
            yy = pd.to_numeric(split_df[self.label_col], errors="coerce")
            bb = _make_bins(yy, n_bins=self.n_bins, binning=self.binning, duplicates=self.duplicates)
            vc = bb.value_counts(dropna=False).sort_index()
            return {str(k): int(v) for k, v in vc.to_dict().items()}

        stats: Dict[str, Any] = {
            "n_total": int(len(df)),
            "n_used": int(len(work)),
            "n_dropped_nan": int(dropped),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "label_col": self.label_col,
            "binning": self.binning,
            "n_bins_requested": int(self.n_bins),
            "n_bins_effective": int(n_bins_effective),
            "train_bin_counts": _bin_counts_for(train),
            "test_bin_counts": _bin_counts_for(test),
        }
        if val is not None:
            stats["val_bin_counts"] = _bin_counts_for(val)

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
                "n_bins": self.n_bins,
                "binning": self.binning,
                "duplicates": self.duplicates,
                "dropna": self.dropna,
            },
            stats=stats,
        )
