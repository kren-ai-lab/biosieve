"""Shared helpers for splitting strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

MIN_KFOLD_SPLITS = 2
SKLEARN_REQUIRED_MESSAGE = (
    "scikit-learn could not be imported, but it is a required biosieve dependency. "
    "Check that biosieve was installed correctly in this environment."
)


def validate_sizes(test_size: float, val_size: float) -> None:
    """Validate test/validation fractions shared by single-split strategies."""
    if not (0.0 < test_size < 1.0):
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if test_size + val_size >= 1.0:
        msg = "test_size + val_size must be < 1.0"
        raise ValueError(msg)


def validate_kfold(n_splits: int, val_size: float, *, n_samples: int | None = None) -> None:
    """Validate shared k-fold parameters."""
    if n_splits < MIN_KFOLD_SPLITS:
        msg = "n_splits must be >= 2"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if n_samples is not None and n_samples < n_splits:
        msg = f"Not enough samples (n={n_samples}) for n_splits={n_splits}"
        raise ValueError(msg)


def derive_val_fraction(test_size: float, val_size: float) -> float:
    """Convert global val_size into a fraction of the train+val remainder."""
    frac = val_size / (1.0 - test_size)
    if not (0.0 < frac < 1.0):
        msg = "Derived val fraction invalid. Check test_size/val_size."
        raise ValueError(msg)
    return frac


def require_train_test_split(feature: str) -> Any:  # noqa: ANN401
    """Return sklearn.model_selection.train_test_split with a consistent ImportError."""
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
    except ImportError as e:
        msg = sklearn_required_message(feature)
        raise ImportError(msg) from e
    return train_test_split


def sklearn_required_message(feature: str) -> str:
    """Build a consistent error message for required scikit-learn-backed features."""
    return f"{feature} requires scikit-learn. {SKLEARN_REQUIRED_MESSAGE}"


def split_train_val(
    train_df: pl.DataFrame,
    *,
    val_size: float,
    seed: int,
    feature: str,
    stratify: object = None,
    train_test_split: Any | None = None,  # noqa: ANN401
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a training frame into train/val using sklearn when val is enabled."""
    if val_size <= 0:
        msg = "split_train_val requires val_size > 0"
        raise ValueError(msg)

    tts = train_test_split or require_train_test_split(feature)

    inner_idx = np.arange(train_df.height)
    train_keep_idx, val_idx = tts(
        inner_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df[train_keep_idx], train_df[val_idx]


def value_counts_dict(series: pl.Series) -> dict[str, int]:
    """Return value counts in the JSON-friendly shape used in split stats."""
    return {str(row[0]): int(row[1]) for row in series.cast(pl.String).value_counts().iter_rows()}
