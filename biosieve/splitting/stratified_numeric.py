"""Stratified splitting strategy for numeric targets via binning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

log = get_logger(__name__)
MIN_BINS = 2


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return train_test_split
    except ImportError:
        return None


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _label_stats(y: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "median": None}
    return {
        "n": int(y.size),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        "median": float(np.median(y)),
    }


def _bin_counts(bins: np.ndarray) -> dict[str, int]:
    uniq, counts = np.unique(bins, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq, counts, strict=False)}


def _make_bins(
    values: np.ndarray,
    *,
    n_bins: int,
    binning: Literal["quantile", "uniform"],
    min_bin_count: int,
    auto_reduce_bins: bool,
) -> tuple[np.ndarray, int]:
    if n_bins < MIN_BINS:
        raise ValueError("n_bins must be >= 2")
    candidates = range(n_bins, 1, -1) if auto_reduce_bins else [n_bins]
    last_error = "unknown"
    for candidate in candidates:
        if binning == "quantile":
            edges = np.quantile(values, np.linspace(0, 1, candidate + 1))
        else:
            edges = np.linspace(float(values.min()), float(values.max()), candidate + 1)
        edges = np.unique(edges)
        if edges.size < 3:
            last_error = "not enough unique bin edges"
            continue
        bins = np.digitize(values, edges[1:-1], right=True)
        _, counts = np.unique(bins, return_counts=True)
        if counts.min() < min_bin_count:
            last_error = f"minimum bin count {counts.min()} < {min_bin_count}"
            continue
        return bins.astype(int), int(np.unique(bins).size)
    raise ValueError(f"Could not create valid stratification bins: {last_error}")


@dataclass(frozen=True)
class StratifiedNumericSplitter:
    """Stratified split for numeric labels via binning."""

    label_col: str = "y"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13
    n_bins: int = 10
    binning: Literal["quantile", "uniform"] = "quantile"
    duplicates: Literal["drop", "raise"] = "drop"
    dropna: bool = True
    auto_reduce_bins: bool = True
    min_bin_count: int = 2
    report_bin_edges: bool = False

    @property
    def strategy(self) -> str:
        return "stratified_numeric"

    def run(self, df: pl.DataFrame, _cols: Any) -> SplitResult:
        tts = _try_import_train_test_split()
        if tts is None:
            raise ImportError("StratifiedNumericSplitter requires scikit-learn.")
        _validate_sizes(self.test_size, self.val_size)
        if self.label_col not in df.columns:
            raise ValueError(f"Missing numeric label column '{self.label_col}'. Columns: {df.columns}")

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
        bins, n_eff = _make_bins(
            y,
            n_bins=self.n_bins,
            binning=self.binning,
            min_bin_count=self.min_bin_count,
            auto_reduce_bins=self.auto_reduce_bins,
        )

        all_idx = np.arange(work.height)
        trainval_idx, test_idx = tts(
            all_idx, test_size=self.test_size, random_state=self.seed, shuffle=True, stratify=bins
        )
        trainval = work[trainval_idx]
        test = work[test_idx]
        train = trainval
        val = None

        if self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            y_tv = trainval[self.label_col].cast(pl.Float64, strict=False).to_numpy()
            bins_tv, _ = _make_bins(
                y_tv,
                n_bins=self.n_bins,
                binning=self.binning,
                min_bin_count=self.min_bin_count,
                auto_reduce_bins=self.auto_reduce_bins,
            )
            inner_idx = np.arange(trainval.height)
            train_idx, val_idx = tts(
                inner_idx, test_size=frac, random_state=self.seed, shuffle=True, stratify=bins_tv
            )
            train = trainval[train_idx]
            val = trainval[val_idx]

        stats: dict[str, Any] = {
            "n_total": df.height,
            "n_used": work.height,
            "n_dropped_nan": dropped,
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
            "label_col": self.label_col,
            "n_bins_requested": self.n_bins,
            "n_bins_effective": n_eff,
            "train_label_stats": _label_stats(train[self.label_col].cast(pl.Float64, strict=False).to_numpy()),
            "test_label_stats": _label_stats(test[self.label_col].cast(pl.Float64, strict=False).to_numpy()),
            "train_bin_counts": _bin_counts(bins[np.asarray(trainval_idx, dtype=int)[: train.height]])
            if train.height <= len(trainval_idx)
            else {},
        }
        if val is not None:
            stats["val_label_stats"] = _label_stats(val[self.label_col].cast(pl.Float64, strict=False).to_numpy())

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
            },
            stats=stats,
        )
