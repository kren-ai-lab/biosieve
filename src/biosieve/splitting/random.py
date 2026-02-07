from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult

from biosieve.utils.logging import get_logger
log = get_logger(__name__)

def _validate_sizes(test_size: float, val_size: float) -> None:
    """
    Validate split fractions.

    Raises
    ------
    ValueError
        If sizes are out of range or leave no samples for training.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _index_split(
    n: int, test_size: float, val_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Create index splits for train/test(/val) using a seeded RNG.

    Parameters
    ----------
    n:
        Number of rows.
    test_size:
        Fraction assigned to test.
    val_size:
        Fraction assigned to val (0 disables validation).
    seed:
        RNG seed.

    Returns
    -------
    train_idx, test_idx, val_idx
        Numpy arrays of indices. `val_idx` may be None.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size)) if val_size > 0 else 0
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Split sizes leave no training samples. Reduce test_size/val_size.")

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val] if n_val > 0 else None
    test_idx = idx[n_train + n_val:]

    return train_idx, test_idx, val_idx


@dataclass(frozen=True)
class RandomSplitter:
    """
    Random train/test(/val) split (deterministic via seed).

    Parameters
    ----------
    test_size:
        Fraction of samples assigned to the test set.
    val_size:
        Fraction of samples assigned to the validation set (0 disables validation).
        This fraction is taken from the *whole dataset*, not only from train.
    seed:
        Random seed used to shuffle indices.

    Returns
    -------
    SplitResult
        A container with:
        - train/test/val DataFrames
        - params: {"test_size","val_size","seed"}
        - stats: counts and effective sizes

    Raises
    ------
    ValueError
        If `test_size` or `val_size` are invalid or leave no samples for training.

    Notes
    -----
    - This strategy does not consider labels, groups, homology, time, or distances.
      It is appropriate for quick baselines but can lead to leakage in biological datasets
      where redundancy or relatedness exists.

    Examples
    --------
    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_random \\
    ...   --strategy random \\
    ...   --params params.yaml
    """
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    @property
    def strategy(self) -> str:
        return "random"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:

        log.info(
            "random:start | test_size=%.3f | val_size=%.3f | seed=%s",
            self.test_size, self.val_size, self.seed
        )
        log.debug("random:params | %s", self.__dict__)

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        n = len(work)

        train_idx, test_idx, val_idx = _index_split(n, self.test_size, self.val_size, self.seed)

        train = work.iloc[train_idx].reset_index(drop=True)
        test = work.iloc[test_idx].reset_index(drop=True)
        val = work.iloc[val_idx].reset_index(drop=True) if val_idx is not None else None

        stats: Dict[str, Any] = {
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "test_size": float(self.test_size),
            "val_size": float(self.val_size),
            "seed": int(self.seed),
        }

        log.info(
            "random:stats | train=%d | val=%d | test=%d",
            int(len(train)), int(len(val)), int(len(test))
        )
        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={"test_size": self.test_size, "val_size": self.val_size, "seed": self.seed},
            stats=stats,
        )
