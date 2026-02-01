from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _to_datetime(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="raise")
    return pd.to_datetime(s, errors="raise")


@dataclass(frozen=True)
class TimeSplitter:
    """
    Temporal split: train is earlier, test is later (and optional val in-between).

    Behavior:
      - parse time_col as datetime (or numeric if you set parse_datetime=False)
      - sort by time
      - split by fractions: train | val | test (chronological)
      - no shuffling

    Params:
      - time_col: column with timestamps (string or numeric)
      - parse_datetime: if True (default), parse with pandas.to_datetime
      - time_format: optional strptime-like format string (e.g., "%Y-%m-%d")
      - ascending: True means older->newer
      - test_size, val_size
    """

    time_col: str = "time"
    test_size: float = 0.2
    val_size: float = 0.0

    parse_datetime: bool = True
    time_format: Optional[str] = None
    ascending: bool = True

    @property
    def strategy(self) -> str:
        return "time"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        if self.time_col not in work.columns:
            raise ValueError(f"Missing time column '{self.time_col}'. Columns: {work.columns.tolist()}")

        t_raw = work[self.time_col]

        if self.parse_datetime:
            t = _to_datetime(t_raw, self.time_format)
        else:
            # numeric time is allowed
            t = pd.to_numeric(t_raw, errors="raise")

        work["_biosieve_time__"] = t
        work = work.sort_values("_biosieve_time__", ascending=self.ascending, kind="mergesort").reset_index(drop=True)

        n = len(work)
        n_test = int(round(n * self.test_size))
        n_val = int(round(n * self.val_size)) if self.val_size > 0 else 0
        n_train = n - n_test - n_val

        if n_train <= 0:
            raise ValueError("Split sizes leave no training samples. Reduce test_size/val_size.")
        if n_test <= 0:
            raise ValueError("test_size too small -> no test samples after rounding. Increase test_size.")
        if self.val_size > 0 and n_val <= 0:
            raise ValueError("val_size too small -> no validation samples after rounding. Increase val_size.")

        train = work.iloc[:n_train].drop(columns=["_biosieve_time__"]).reset_index(drop=True)
        val = (
            work.iloc[n_train : n_train + n_val].drop(columns=["_biosieve_time__"]).reset_index(drop=True)
            if n_val > 0
            else None
        )
        test = work.iloc[n_train + n_val :].drop(columns=["_biosieve_time__"]).reset_index(drop=True)

        # stats: time ranges
        def _range(x: pd.DataFrame) -> Dict[str, Any]:
            if len(x) == 0:
                return {"min": None, "max": None}
            tt = x[self.time_col]
            if self.parse_datetime:
                tt = pd.to_datetime(tt, format=self.time_format, errors="coerce")
            return {"min": str(tt.min()), "max": str(tt.max())}

        stats: Dict[str, Any] = {
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "time_col": self.time_col,
            "ascending": self.ascending,
            "train_time_range": _range(train),
            "test_time_range": _range(test),
        }
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
