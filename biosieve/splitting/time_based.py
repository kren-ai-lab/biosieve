"""Time-based splitting strategy for chronological model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if test_size + val_size >= 1.0:
        msg = "test_size + val_size must be < 1.0"
        raise ValueError(msg)


def _to_datetime(s: pd.Series, fmt: str | None) -> pd.Series:
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="raise")
    return pd.to_datetime(s, errors="raise")


@dataclass(frozen=True)
class TimeSplitter:
    r"""Time-based split (chronological): train is earlier, test is later.

    This strategy sorts the dataset by a time column and splits chronologically:
      train | (optional val) | test

    Args:
        time_col: Column containing timestamps (string datetime) or numeric time values.
        test_size: Fraction assigned to the test split (latest samples if ascending=True).
        val_size: Fraction assigned to a validation split between train and test (0 disables validation).
        parse_datetime: If True, parse `time_col` using `pandas.to_datetime`. If False, parse as numeric.
        time_format: Optional datetime format string (e.g., "%Y-%m-%d"). Only used when parse_datetime=True.
        ascending: If True, sorts from older to newer. If False, newer to older (reverses split direction).

    Returns:
        Container with train/test/val DataFrames plus:
        - params: effective parameters
        - stats: counts and time ranges per split

    Raises:
        ValueError: If time column missing, sizes invalid, parsing fails, or rounding produces empty splits.

    Notes:
        - No shuffling is performed.
        - This does not enforce homology/group/structure leakage constraints by itself.
        For time-first with leakage guardrails, consider a hybrid (future feature).

    Examples:
        >>> biosieve split \\
        ...   --in dataset.csv \\
        ...   --outdir runs/split_time \\
        ...   --strategy time \\
        ...   --params params.yaml

    """

    time_col: str = "time"
    test_size: float = 0.2
    val_size: float = 0.0

    parse_datetime: bool = True
    time_format: str | None = None
    ascending: bool = True

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "time"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        """Create chronological train/test/(val) partitions."""
        log.info("time:start | date_col=%s", cols.date_col)

        log.debug("time:params | %s", self.__dict__)

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        if self.time_col not in work.columns:
            msg = f"Missing time column '{self.time_col}'. Columns: {work.columns.tolist()}"
            raise ValueError(msg)

        t_raw = work[self.time_col]

        if t_raw.isna().any():
            msg = f"Found NaN timestamps in '{self.time_col}'. Clean dataset before splitting."
            raise ValueError(msg)

        if self.parse_datetime:
            t = _to_datetime(t_raw, self.time_format)
        else:
            t = pd.to_numeric(t_raw, errors="raise")

        work["_biosieve_time__"] = t
        work = work.sort_values("_biosieve_time__", ascending=self.ascending, kind="mergesort").reset_index(
            drop=True
        )

        n = len(work)
        n_test = round(n * self.test_size)
        n_val = round(n * self.val_size) if self.val_size > 0 else 0
        n_train = n - n_test - n_val

        if n_train <= 0:
            msg = "Split sizes leave no training samples. Reduce test_size/val_size."
            raise ValueError(msg)
        if n_test <= 0:
            msg = "test_size too small -> no test samples after rounding. Increase test_size."
            raise ValueError(msg)
        if self.val_size > 0 and n_val <= 0:
            msg = "val_size too small -> no validation samples after rounding. Increase val_size."
            raise ValueError(msg)

        train = work.iloc[:n_train].drop(columns=["_biosieve_time__"]).reset_index(drop=True)
        val = (
            work.iloc[n_train : n_train + n_val].drop(columns=["_biosieve_time__"]).reset_index(drop=True)
            if n_val > 0
            else None
        )
        test = work.iloc[n_train + n_val :].drop(columns=["_biosieve_time__"]).reset_index(drop=True)

        def _range(x: pd.DataFrame) -> dict[str, Any]:
            if len(x) == 0:
                return {"min": None, "max": None}
            tt = x[self.time_col]
            if self.parse_datetime:
                tt = pd.to_datetime(tt, format=self.time_format, errors="coerce")
            return {"min": str(tt.min()), "max": str(tt.max())}

        stats: dict[str, Any] = {
            "n_total": int(n),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "time_col": self.time_col,
            "ascending": bool(self.ascending),
            "parse_datetime": bool(self.parse_datetime),
            "time_format": self.time_format,
            "train_time_range": _range(train),
            "test_time_range": _range(test),
        }

        log.info(
            "time:stats | train=%d | val=%d | test=%d", stats["n_train"], stats["n_val"], stats["n_test"]
        )

        if val is not None:
            stats["val_time_range"] = _range(val)

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "time_col": self.time_col,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "parse_datetime": self.parse_datetime,
                "time_format": self.time_format,
                "ascending": self.ascending,
            },
            stats=stats,
        )
