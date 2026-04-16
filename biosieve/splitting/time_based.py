"""Time-based splitting strategy for chronological model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

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


def _to_datetime(s: pl.Series, fmt: str | None) -> pl.Series:
    if fmt:
        return s.str.to_datetime(format=fmt, strict=True)
    return s.str.to_datetime(strict=True)


def _validate_inputs(df: pl.DataFrame, time_col: str, test_size: float, val_size: float) -> pl.Series:
    _validate_sizes(test_size, val_size)
    if time_col not in df.columns:
        msg = f"Missing time column '{time_col}'. Columns: {df.columns}"
        raise ValueError(msg)
    t_raw = df[time_col]
    if t_raw.is_null().any():
        msg = f"Found NaN timestamps in '{time_col}'. Clean dataset before splitting."
        raise ValueError(msg)
    return t_raw


@dataclass(frozen=True)
class TimeSplitter:
    r"""Time-based split (chronological): train is earlier, test is later.

    This strategy sorts the dataset by a time column and splits chronologically:
      train | (optional val) | test

    Args:
        time_col: Column containing timestamps (string datetime) or numeric time values.
        test_size: Fraction assigned to the test split (latest samples if ascending=True).
        val_size: Fraction assigned to a validation split between train and test (0 disables validation).
        parse_datetime: If True, parse `time_col` as datetimes. If False, parse as numeric.
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

    def run(self, df: pl.DataFrame, cols: Columns) -> SplitResult:
        """Create chronological train/test/(val) partitions."""
        log.info("time:start | date_col=%s", cols.date_col)

        log.debug("time:params | %s", self.__dict__)

        work = df.clone()
        t_raw = _validate_inputs(work, self.time_col, self.test_size, self.val_size)

        if self.parse_datetime:
            t = _to_datetime(t_raw, self.time_format)
        else:
            t = t_raw.cast(pl.Float64, strict=True)

        work = work.with_columns(t.alias("_biosieve_time__")).sort(
            "_biosieve_time__", descending=not self.ascending, maintain_order=True
        )

        n = work.height
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

        train = work[:n_train].drop(["_biosieve_time__"])
        val = work[n_train : n_train + n_val].drop(["_biosieve_time__"]) if n_val > 0 else None
        test = work[n_train + n_val :].drop(["_biosieve_time__"])

        def _range(x: pl.DataFrame) -> dict[str, Any]:
            if x.height == 0:
                return {"min": None, "max": None}
            tt = x[self.time_col]
            if self.parse_datetime:
                tt = _to_datetime(tt, self.time_format)
            return {"min": str(tt.min()), "max": str(tt.max())}

        stats: dict[str, Any] = {
            "n_total": int(n),
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
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
