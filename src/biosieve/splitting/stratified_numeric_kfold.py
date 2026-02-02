from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult


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


def _label_stats(y: pd.Series) -> Dict[str, Any]:
    yy = pd.to_numeric(y, errors="coerce").dropna()
    if len(yy) == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(yy)),
        "min": float(yy.min()),
        "max": float(yy.max()),
        "mean": float(yy.mean()),
        "std": float(yy.std(ddof=1)) if len(yy) > 1 else 0.0,
        "median": float(yy.median()),
        "q25": float(yy.quantile(0.25)),
        "q75": float(yy.quantile(0.75)),
    }


def _bin_counts(bins: pd.Series) -> Dict[str, int]:
    vc = bins.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _make_bins_once(
    y: pd.Series,
    *,
    n_bins: int,
    binning: str,
    duplicates: str,
) -> Tuple[pd.Series, int, Optional[List[float]]]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    yy = pd.to_numeric(y, errors="coerce")
    if yy.isna().any():
        raise ValueError("NaN values found in label series while binning.")

    edges: Optional[List[float]] = None

    if binning == "quantile":
        bins, bin_edges = pd.qcut(
            yy, q=n_bins, labels=False, duplicates=duplicates, retbins=True
        )
        bins = pd.Series(bins, index=y.index).astype("Int64")
        n_eff = int(pd.Series(bins).nunique(dropna=True))
        edges = [float(x) for x in np.asarray(bin_edges).tolist()]
        return bins, n_eff, edges

    if binning == "uniform":
        bins, bin_edges = pd.cut(
            yy, bins=n_bins, labels=False, include_lowest=True, retbins=True
        )
        bins = pd.Series(bins, index=y.index).astype("Int64")
        n_eff = int(pd.Series(bins).nunique(dropna=True))
        edges = [float(x) for x in np.asarray(bin_edges).tolist()]
        return bins, n_eff, edges

    raise ValueError("binning must be 'quantile' or 'uniform'")


def _make_bins_safe(
    y: pd.Series,
    *,
    n_bins: int,
    binning: str,
    duplicates: str,
    min_bin_count: int,
    auto_reduce_bins: bool,
) -> Tuple[pd.Series, int, Optional[List[float]], List[int], bool]:
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")

    attempted: List[int] = []
    auto_reduced = False
    candidates = list(range(n_bins, 1, -1)) if auto_reduce_bins else [n_bins]

    last_error: Optional[Exception] = None

    for b in candidates:
        attempted.append(int(b))
        try:
            bins, n_eff, edges = _make_bins_once(y, n_bins=b, binning=binning, duplicates=duplicates)

            if n_eff < 2:
                raise ValueError(f"Effective bins={n_eff} (requested={b}). Cannot stratify.")

            counts = bins.value_counts(dropna=False)
            if int(counts.min()) < min_bin_count:
                raise ValueError(f"Some bins have <{min_bin_count} samples (min={int(counts.min())}) for n_bins={b}.")

            if b != n_bins:
                auto_reduced = True

            return bins, n_eff, edges, attempted, auto_reduced

        except Exception as e:
            last_error = e
            continue

    raise ValueError(
        "Could not create valid stratification bins for numeric kfold. "
        f"Attempted bins={attempted}. Last error: {last_error}"
    )


@dataclass(frozen=True)
class StratifiedNumericKFoldSplitter:
    """
    Stratified K-Fold splitting for numeric labels via binning.

    Approach
    --------
    1) Convert numeric label y into categorical bins (quantile/uniform).
    2) Run StratifiedKFold using the bins as y_strat.
    3) Optionally create a validation subset from the fold's train split.

    This is useful for regression datasets where you want fold-level balance across
    the label range.

    Parameters
    ----------
    label_col:
        Numeric target column.
    n_splits:
        Number of folds.
    shuffle, seed:
        Controls deterministic fold generation.
    n_bins, binning, duplicates:
        Binning configuration.
    auto_reduce_bins, min_bin_count:
        Robustness controls for real-world datasets with repeated values.
    val_size:
        Optional validation fraction sampled from fold train (random).
    dropna:
        If True, drop rows with NaN labels; else raise.
    report_bin_edges:
        If True, include bin edges in each fold report.

    Notes
    -----
    - StratifiedKFold requires each bin to have at least `n_splits` samples.
      This splitter checks this and will auto-reduce bins if enabled.
    """

    label_col: str = "y"

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13

    n_bins: int = 10
    binning: str = "quantile"
    duplicates: str = "drop"

    auto_reduce_bins: bool = True
    min_bin_count: int = 2

    val_size: float = 0.0
    dropna: bool = True
    report_bin_edges: bool = False

    @property
    def strategy(self) -> str:
        return "stratified_numeric_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> List[SplitResult]:
        StratifiedKFold = _try_import_stratified_kfold()
        if StratifiedKFold is None:
            raise ImportError(
                "StratifiedNumericKFoldSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0.0 <= self.val_size < 1.0):
            raise ValueError("val_size must be in [0, 1)")
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column '{self.label_col}'. Columns: {df.columns.tolist()}")

        work = df.copy().reset_index(drop=True)
        y_raw = pd.to_numeric(work[self.label_col], errors="coerce")

        dropped = 0
        if self.dropna:
            keep = ~y_raw.isna()
            dropped = int((~keep).sum())
            work = work.loc[keep].reset_index(drop=True)
            y_raw = y_raw.loc[keep].reset_index(drop=True)
        else:
            if y_raw.isna().any():
                raise ValueError(
                    f"Found NaN labels in '{self.label_col}'. Set dropna=true or clean dataset."
                )

        if len(work) < self.n_splits:
            raise ValueError(f"Not enough samples (n={len(work)}) for n_splits={self.n_splits}")

        # Build bins robustly (may auto-reduce)
        bins, n_eff, edges, attempted_bins, auto_reduced = _make_bins_safe(
            y_raw,
            n_bins=self.n_bins,
            binning=self.binning,
            duplicates=self.duplicates,
            min_bin_count=self.min_bin_count,
            auto_reduce_bins=self.auto_reduce_bins,
        )

        # StratifiedKFold constraint: each stratum must have >= n_splits
        vc = bins.value_counts(dropna=False)
        too_small = vc[vc < self.n_splits]
        if len(too_small) > 0:
            raise ValueError(
                "Some bins have fewer samples than n_splits; cannot stratify k-fold. "
                f"n_splits={self.n_splits}, problematic bins: {too_small.to_dict()}. "
                "Try fewer bins (n_bins) or auto_reduce_bins."
            )

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                raise ImportError("val_size > 0 requires scikit-learn train_test_split.")

        folds: List[SplitResult] = []
        X_dummy = work.index.values

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, bins)):
            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            val_df: Optional[pd.DataFrame] = None
            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)
                train_df, val_df = tts(
                    train_df,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)

            # For reporting, compute bins per split using the same chosen binning setup
            # but allow min_bin_count=1 (reporting only).
            def _bins_for_split(split_df: pd.DataFrame) -> pd.Series:
                yy = pd.to_numeric(split_df[self.label_col], errors="coerce").dropna()
                bb, _, _, _, _ = _make_bins_safe(
                    yy,
                    n_bins=self.n_bins,
                    binning=self.binning,
                    duplicates=self.duplicates,
                    min_bin_count=1,
                    auto_reduce_bins=True,
                )
                return bb

            train_bins = _bins_for_split(train_df)
            test_bins = _bins_for_split(test_df)

            stats: Dict[str, Any] = {
                "fold_index": int(fold_idx),
                "n_total": int(len(df)),
                "n_used": int(len(work)),
                "n_dropped_nan": int(dropped),

                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_val": int(len(val_df)) if val_df is not None else 0,

                "label_col": self.label_col,
                "binning": self.binning,
                "duplicates": self.duplicates,
                "n_bins_requested": int(self.n_bins),
                "n_bins_effective": int(n_eff),
                "min_bin_count": int(self.min_bin_count),
                "auto_reduce_bins": bool(self.auto_reduce_bins),
                "attempted_bins": attempted_bins,
                "auto_reduced": bool(auto_reduced),

                "train_bin_counts": _bin_counts(train_bins),
                "test_bin_counts": _bin_counts(test_bins),

                "train_label_stats": _label_stats(train_df[self.label_col]),
                "test_label_stats": _label_stats(test_df[self.label_col]),
            }

            if val_df is not None:
                val_bins = _bins_for_split(val_df)
                stats["val_bin_counts"] = _bin_counts(val_bins)
                stats["val_label_stats"] = _label_stats(val_df[self.label_col])

            if self.report_bin_edges:
                stats["bin_edges"] = edges

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
                        "n_bins": self.n_bins,
                        "binning": self.binning,
                        "duplicates": self.duplicates,
                        "auto_reduce_bins": self.auto_reduce_bins,
                        "min_bin_count": self.min_bin_count,
                        "val_size": self.val_size,
                        "dropna": self.dropna,
                        "report_bin_edges": self.report_bin_edges,
                        "fold_index": fold_idx,
                    },
                    stats=stats,
                )
            )

        return folds
