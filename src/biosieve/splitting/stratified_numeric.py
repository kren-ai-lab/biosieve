from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

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


def _label_stats(y: pd.Series) -> Dict[str, Any]:
    yy = pd.to_numeric(y, errors="coerce")
    yy = yy.dropna()
    if len(yy) == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }

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
    """
    Return:
      bins: Int64 categorical bin ids (0..K-1)
      n_bins_effective: number of unique bins actually produced
      edges: list of bin edges if available (optional)
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    yy = pd.to_numeric(y, errors="coerce")
    if yy.isna().any():
        # caller decides dropna/raise earlier; here we assume no NaN
        raise ValueError("NaN values found in label series while binning.")

    edges: Optional[List[float]] = None

    if binning == "quantile":
        # qcut can return edges with retbins=True
        try:
            bins, bin_edges = pd.qcut(
                yy, q=n_bins, labels=False, duplicates=duplicates, retbins=True
            )
        except ValueError as e:
            raise ValueError(
                f"qcut failed for n_bins={n_bins}. "
                f"Try fewer bins or duplicates='drop'. Original error: {e}"
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
    """
    Try to create stratification bins that satisfy:
      - at least 2 effective bins
      - each bin count >= min_bin_count

    If auto_reduce_bins=True, will decrement bins until success or reach 2.
    Returns:
      bins, n_bins_effective, edges, attempted_bins, auto_reduced
    """
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
                raise ValueError(
                    f"Effective number of bins is {n_eff} (requested={b}). Cannot stratify."
                )

            counts = bins.value_counts(dropna=False)
            if int(counts.min()) < min_bin_count:
                raise ValueError(
                    f"Some bins have <{min_bin_count} samples (min={int(counts.min())}) for n_bins={b}."
                )

            if b != n_bins:
                auto_reduced = True

            return bins, n_eff, edges, attempted, auto_reduced

        except Exception as e:
            last_error = e
            continue

    # failed all attempts
    if last_error is not None:
        raise ValueError(
            "Could not create valid stratification bins. "
            f"Attempted bins={attempted}. Last error: {last_error}"
        )
    raise ValueError("Could not create valid stratification bins.")


@dataclass(frozen=True)
class StratifiedNumericSplitter:
    """
    Stratified split for numeric labels via binning.

    Steps
    -----
    1) Convert numeric labels into categorical bins (quantile or uniform).
    2) Use stratified train/test split over bins.
    3) Optionally split validation from trainval (also stratified over bins).

    Parameters
    ----------
    label_col:
        Column containing numeric target values.
    test_size, val_size:
        Fractions for test and validation. Must satisfy test_size + val_size < 1.
    seed:
        Random seed for deterministic splits.
    n_bins:
        Requested number of bins for stratification.
    binning:
        "quantile" (recommended) or "uniform".
    duplicates:
        For quantile binning only: "drop" or "raise".
        - "drop" reduces number of bins if quantiles are not unique.
    dropna:
        If True, rows with NaN label are dropped. If False, NaNs raise an error.
    auto_reduce_bins:
        If True, automatically decreases n_bins until stratification is feasible.
    min_bin_count:
        Minimum required count per bin to allow stratified splitting.
    report_bin_edges:
        If True, store bin edges used in the report.

    Raises
    ------
    ValueError
        If label column is missing, label cannot be parsed to numeric,
        split sizes are invalid, or bins cannot be created for stratification.
    ImportError
        If scikit-learn is not installed.
    """

    label_col: str = "y"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    n_bins: int = 10
    binning: str = "quantile"      # "quantile" | "uniform"
    duplicates: str = "drop"       # "drop" | "raise" (qcut only)

    dropna: bool = True

    auto_reduce_bins: bool = True
    min_bin_count: int = 2

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

        dropped = 0
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

        # Build bins (safe, may auto-reduce bins)
        bins, n_eff, edges, attempted_bins, auto_reduced = _make_bins_safe(
            y_raw,
            n_bins=self.n_bins,
            binning=self.binning,
            duplicates=self.duplicates,
            min_bin_count=self.min_bin_count,
            auto_reduce_bins=self.auto_reduce_bins,
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

        # Optional val split from trainval
        val_attempted_bins: Optional[List[int]] = None
        val_auto_reduced: Optional[bool] = None
        val_edges: Optional[List[float]] = None
        val_n_eff: Optional[int] = None

        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                raise ValueError("Derived val fraction invalid. Check test_size/val_size.")

            y_tv = pd.to_numeric(trainval[self.label_col], errors="coerce")

            bins_tv, n_eff_tv, edges_tv, attempted_tv, auto_red_tv = _make_bins_safe(
                y_tv,
                n_bins=self.n_bins,
                binning=self.binning,
                duplicates=self.duplicates,
                min_bin_count=self.min_bin_count,
                auto_reduce_bins=self.auto_reduce_bins,
            )

            train, val = tts(
                trainval,
                test_size=frac,
                random_state=self.seed,
                shuffle=True,
                stratify=bins_tv,
            )

            val_attempted_bins = attempted_tv
            val_auto_reduced = auto_red_tv
            val_edges = edges_tv
            val_n_eff = n_eff_tv

        # reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if val is not None:
            val = val.reset_index(drop=True)

        # helper to compute split bin counts using the same safe binning logic
        def _split_bin_counts(split_df: pd.DataFrame) -> Dict[str, int]:
            yy = pd.to_numeric(split_df[self.label_col], errors="coerce")
            # split_df should not have NaNs if dropna=True, but keep safe anyway:
            yy = yy.dropna()
            bb, _, _, _, _ = _make_bins_safe(
                yy,
                n_bins=self.n_bins,
                binning=self.binning,
                duplicates=self.duplicates,
                min_bin_count=1,               # just for reporting, allow small bins here
                auto_reduce_bins=True,
            )
            return _bin_counts(bb)

        stats: Dict[str, Any] = {
            "n_total": int(len(df)),
            "n_used": int(len(work)),
            "n_dropped_nan": int(dropped),

            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,

            "label_col": self.label_col,
            "binning": self.binning,
            "duplicates": self.duplicates,

            "n_bins_requested": int(self.n_bins),
            "n_bins_effective": int(n_eff),
            "min_bin_count": int(self.min_bin_count),
            "auto_reduce_bins": bool(self.auto_reduce_bins),
            "attempted_bins": attempted_bins,
            "auto_reduced": bool(auto_reduced),

            "train_bin_counts": _split_bin_counts(train),
            "test_bin_counts": _split_bin_counts(test),

            "train_label_stats": _label_stats(train[self.label_col]),
            "test_label_stats": _label_stats(test[self.label_col]),
        }

        if val is not None:
            stats["val_bin_counts"] = _split_bin_counts(val)
            stats["val_label_stats"] = _label_stats(val[self.label_col])

            # binning info for the val split stage
            stats["val_stage"] = {
                "n_bins_effective": int(val_n_eff) if val_n_eff is not None else None,
                "attempted_bins": val_attempted_bins,
                "auto_reduced": val_auto_reduced,
            }

        if self.report_bin_edges:
            stats["bin_edges"] = edges
            if val is not None:
                stats["val_stage"]["bin_edges"] = val_edges

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
                "auto_reduce_bins": self.auto_reduce_bins,
                "min_bin_count": self.min_bin_count,
                "report_bin_edges": self.report_bin_edges,
            },
            stats=stats,
        )
