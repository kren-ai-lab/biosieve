"""Random splitting baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_sizes(test_size: float, val_size: float) -> None:
    """Validate split fractions.

    Raises:
        ValueError: If sizes are out of range or leave no samples for training.

    """
    if not (0.0 < test_size < 1.0):
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if test_size + val_size >= 1.0:
        msg = "test_size + val_size must be < 1.0"
        raise ValueError(msg)


def _validate_inputs(test_size: float, val_size: float) -> None:
    _validate_sizes(test_size, val_size)


def _index_split(
    n: int, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Create index splits for train/test(/val) using a seeded RNG.

    Args:
        n: Number of rows.
        test_size: Fraction assigned to test.
        val_size: Fraction assigned to val (0 disables validation).
        seed: RNG seed.

    Returns:
        train_idx, test_idx, val_idx: Numpy arrays of indices. `val_idx` may be None.

    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = round(n * test_size)
    n_val = round(n * val_size) if val_size > 0 else 0
    n_train = n - n_test - n_val
    if n_train <= 0:
        msg = "Split sizes leave no training samples. Reduce test_size/val_size."
        raise ValueError(msg)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val] if n_val > 0 else None
    test_idx = idx[n_train + n_val :]

    return train_idx, test_idx, val_idx


@dataclass(frozen=True)
class RandomSplitter:
    r"""Random train/test(/val) split (deterministic via seed).

    Args:
        test_size: Fraction of samples assigned to the test set.
        val_size: Fraction of samples assigned to the validation set (0 disables validation).
        This fraction is taken from the *whole dataset*, not only from train.
        seed: Random seed used to shuffle indices.

    Returns:
        A container with:
        - train/test/val DataFrames: - params: {"test_size","val_size","seed"}
        - stats: counts and effective sizes

    Raises:
        ValueError: If `test_size` or `val_size` are invalid or leave no samples for training.

    Notes:
        - This strategy does not consider labels, groups, homology, time, or distances.
        It is appropriate for quick baselines but can lead to leakage in biological datasets
        where redundancy or relatedness exists.

    Examples:
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
        """Return the strategy identifier."""
        return "random"

    def run(self, df: pl.DataFrame, _cols: Columns) -> SplitResult:
        """Generate deterministic random train/test/(val) splits."""
        log.info(
            "random:start | test_size=%.3f | val_size=%.3f | seed=%s",
            self.test_size,
            self.val_size,
            self.seed,
        )
        log.debug("random:params | %s", self.__dict__)

        _validate_inputs(self.test_size, self.val_size)

        work = df.clone()
        n = work.height

        train_idx, test_idx, val_idx = _index_split(n, self.test_size, self.val_size, self.seed)

        train = work[train_idx]
        test = work[test_idx]
        val = work[val_idx] if val_idx is not None else None

        stats: dict[str, Any] = {
            "n_total": int(n),
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
            "test_size": float(self.test_size),
            "val_size": float(self.val_size),
            "seed": int(self.seed),
        }

        log.info(
            "random:stats | train=%d | val=%d | test=%d",
            train.height,
            val.height if val is not None else 0,
            test.height,
        )
        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={"test_size": self.test_size, "val_size": self.val_size, "seed": self.seed},
            stats=stats,
        )
