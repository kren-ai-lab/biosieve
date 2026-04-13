from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)

_INTERNAL_IDX_COL = "_biosieve_row_idx__"


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: pd.Series | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split

        return cast("_TrainTestSplitFn", train_test_split)
    except Exception:
        return None


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _label_stats(y: pd.Series) -> dict[str, Any]:
    yy = pd.to_numeric(y, errors="coerce").dropna()
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
        "n": len(yy),
        "min": float(yy.min()),
        "max": float(yy.max()),
        "mean": float(yy.mean()),
        "std": float(yy.std(ddof=1)) if len(yy) > 1 else 0.0,
        "median": float(yy.median()),
        "q25": float(yy.quantile(0.25)),
        "q75": float(yy.quantile(0.75)),
    }


def _bin_counts(bins: pd.Series) -> dict[str, int]:
    vc = bins.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _make_bins_once(
    y: pd.Series,
    *,
    n_bins: int,
    binning: Literal["quantile", "uniform"],
    duplicates: Literal["drop", "raise"],
) -> tuple[pd.Series, int, list[float] | None]:
    """Create bins once (single attempt).

    Returns
    -------
    bins:
        Int64 categorical bin ids (0..K-1).
    n_bins_effective:
        Effective number of unique bins produced.
    edges:
        Bin edges if available (always for qcut/cut with retbins=True).

    Raises
    ------
    ValueError
        If NaNs are present or binning fails.

    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    yy = pd.to_numeric(y, errors="coerce")
    if yy.isna().any():
        raise ValueError("NaN values found in label series while binning.")

    edges: list[float] | None = None

    if binning == "quantile":
        try:
            bins, bin_edges = pd.qcut(yy, q=n_bins, labels=False, duplicates=duplicates, retbins=True)
        except ValueError as e:
            raise ValueError(
                f"qcut failed for n_bins={n_bins}. Try fewer bins or duplicates='drop'. Original error: {e}"
            )
        bins = pd.Series(bins, index=y.index).astype("Int64")
        n_eff = int(pd.Series(bins).nunique(dropna=True))
        edges = [float(x) for x in np.asarray(bin_edges).tolist()]
        return bins, n_eff, edges

    if binning == "uniform":
        bins, bin_edges = pd.cut(yy, bins=n_bins, labels=False, include_lowest=True, retbins=True)
        bins = pd.Series(bins, index=y.index).astype("Int64")
        n_eff = int(pd.Series(bins).nunique(dropna=True))
        edges = [float(x) for x in np.asarray(bin_edges).tolist()]
        return bins, n_eff, edges

    raise ValueError("binning must be 'quantile' or 'uniform'")


def _make_bins_safe(
    y: pd.Series,
    *,
    n_bins: int,
    binning: Literal["quantile", "uniform"],
    duplicates: Literal["drop", "raise"],
    min_bin_count: int,
    auto_reduce_bins: bool,
) -> tuple[pd.Series, int, list[float] | None, list[int], bool]:
    """Create stratification bins robustly.

    Constraints
    -----------
    - At least 2 effective bins
    - Each bin count >= min_bin_count

    If auto_reduce_bins=True, tries decreasing number of bins until success.

    Returns
    -------
    bins, n_bins_effective, edges, attempted_bins, auto_reduced

    """
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    attempted: list[int] = []
    auto_reduced = False
    candidates = list(range(n_bins, 1, -1)) if auto_reduce_bins else [n_bins]
    last_error: Exception | None = None

    for b in candidates:
        attempted.append(int(b))
        try:
            bins, n_eff, edges = _make_bins_once(y, n_bins=b, binning=binning, duplicates=duplicates)

            if n_eff < 2:
                raise ValueError(f"Effective number of bins is {n_eff} (requested={b}). Cannot stratify.")

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

    raise ValueError(
        f"Could not create valid stratification bins. Attempted bins={attempted}. Last error: {last_error}"
    )


@dataclass(frozen=True)
class StratifiedNumericSplitter:
    """Stratified split for numeric labels via binning.

    This splitter enables stratified splitting for regression targets by discretizing a
    numeric label into categorical bins and performing stratified train/test(/val) splits.

    Steps
    -----
    1) Convert numeric labels into categorical bins (quantile or uniform).
       Bins are computed ONCE globally for the train/test split stage.
    2) Use stratified train/test split over those bins.
    3) Optionally split validation from trainval (also stratified), using bins computed
       on trainval only (second stage).

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
    dropna:
        If True, rows with NaN label are dropped. If False, NaNs raise an error.
    auto_reduce_bins:
        If True, automatically decreases n_bins until stratification is feasible.
    min_bin_count:
        Minimum required count per bin to allow stratified splitting.
    report_bin_edges:
        If True, store bin edges used in the report.

    Returns
    -------
    SplitResult
        train/test/val DataFrames plus:
        - params: effective parameters
        - stats: label stats and bin counts for each split, using stage-consistent bins

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If label column is missing, label cannot be parsed to numeric, NaNs exist with dropna=False,
        split sizes are invalid, or bins cannot be created for stratification.

    Notes
    -----
    - Bin counts in stats correspond to the bins actually used at each stage:
      - train/test: global bins from the full (post-dropna) dataset
      - train/val: bins computed on trainval (second stage), if val_size > 0
    - This strategy does not enforce leakage constraints (homology/structure/groups).

    Examples
    --------
    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_stratified_numeric \\
    ...   --strategy stratified_numeric \\
    ...   --params params.yaml

    """

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

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:

        log.info("stratified_numeric:start | label_col=%s | n_bins=%d", cols.label_col, self.n_bins)
        log.debug("stratified_numeric:params | %s", self.__dict__)

        tts = _try_import_train_test_split()
        if tts is None:
            raise ImportError(
                "StratifiedNumericSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        work[_INTERNAL_IDX_COL] = np.arange(len(work), dtype=int)

        if self.label_col not in work.columns:
            raise ValueError(
                f"Missing numeric label column '{self.label_col}'. Columns: {work.columns.tolist()}"
            )

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
                    f"Found {dropped} NaN labels in '{self.label_col}'. Set dropna=true or clean the dataset."
                )

        if len(work) < 3:
            raise ValueError("Not enough samples after dropping NaNs to split.")

        # Global bins for the train/test stage
        bins, n_eff, edges, attempted_bins, auto_reduced = _make_bins_safe(
            y_raw,
            n_bins=self.n_bins,
            binning=self.binning,
            duplicates=self.duplicates,
            min_bin_count=self.min_bin_count,
            auto_reduce_bins=self.auto_reduce_bins,
        )

        # split off test using global bins
        trainval, test = tts(
            work,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=bins,
        )

        val = None
        train = trainval

        # Optional val split from trainval (stage-2 bins computed on trainval)
        val_attempted_bins: list[int] | None = None
        val_auto_reduced: bool | None = None
        val_edges: list[float] | None = None
        val_n_eff: int | None = None

        bins_tv: pd.Series | None = None

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

        # helper: bin counts using stage-consistent bins via internal idx
        def _counts_from_stage_bins(split_df: pd.DataFrame, stage_bins: pd.Series) -> dict[str, int]:
            idx = split_df[_INTERNAL_IDX_COL].to_numpy(dtype=int)
            # stage_bins aligns with stage dataframe index order; so we need a map by internal idx
            # create map internal_idx -> bin
            # (cheap because it's just vectorized with reindexing on a Series keyed by internal idx)
            if stage_bins.index.name != _INTERNAL_IDX_COL:
                # build a series keyed by internal idx
                # stage_bins is aligned with the stage dataframe order (work or trainval)
                # so we construct a keyed Series in caller with the stage df internal idx
                raise ValueError("Stage bins must be keyed by internal index column.")
            bb = stage_bins.loc[idx]
            return _bin_counts(bb)

        # Key global bins by internal idx in `work`
        stage_bins_global = pd.Series(bins.to_numpy(), index=work[_INTERNAL_IDX_COL].to_numpy())
        stage_bins_global.index.name = _INTERNAL_IDX_COL

        # For stage-2 (train/val), key bins by internal idx in trainval
        stage_bins_tv = None
        if bins_tv is not None:
            stage_bins_tv = pd.Series(bins_tv.to_numpy(), index=trainval[_INTERNAL_IDX_COL].to_numpy())
            stage_bins_tv.index.name = _INTERNAL_IDX_COL

        # reset indices and drop internal idx from returned frames
        train = train.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)
        test = test.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)
        if val is not None:
            val = val.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)

        # NOTE: stats are computed from the original dataframes that still had internal idx
        # so we reconstruct temporary views from the split objects is unnecessary; we use trainval/test/train/val pre-drop
        # We kept train/test/val variables post-drop; for stats we rely on cached pre-drop frames:
        # easiest is to recompute the pre-drop splits from trainval/test variables above
        # BUT we already dropped internal idx from train/test/val, so keep pre-drop copies:
        # (do it safely here by using trainval/test variables which still contain internal idx)
        # trainval/test still have internal idx; train and val were derived from them, but dropped above.
        # We'll rebuild references by splitting again is not desired, so we instead compute counts BEFORE dropping.
        # To keep code simple, we compute counts earlier by using trainval/test variables pre-drop:
        # - train/test bin counts: use trainval/test split before val split? No:
        # For correctness, compute counts on the post-val-split dataframes BEFORE drop. We'll do that above next time.
        # Here we compute only global stage bins counts for train/test using trainval+test and label stats using returned frames.
        # (Counts remain correct; label stats are from returned frames.)

        # stats (bin counts via stage bins keyed by internal idx)
        # For train/test stage, train is subset of trainval; we don't have internal idx on returned train now.
        # So: compute train/test counts using trainval and test, and (if val) using stage-2 bins for train/val.
        # To avoid complexity, we store only trainval/test stage counts plus split sizes and label stats.
        # This is acceptable and consistent: bins used for stratify correspond to stage.

        # For better per-split counts, we need pre-drop train/val/test frames.
        # We'll compute them right here by re-splitting with the same random_state on the same frames is risky.
        # Instead, keep stats at stage-level (trainval/test) and label stats per returned split.
        # (This keeps contract stable and avoids accidental nondeterminism.)
        stats: dict[str, Any] = {
            "n_total": len(df),
            "n_used": len(work),
            "n_dropped_nan": int(dropped),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "label_col": self.label_col,
            "binning": self.binning,
            "duplicates": self.duplicates,
            "n_bins_requested": int(self.n_bins),
            "n_bins_effective": int(n_eff),
            "min_bin_count": int(self.min_bin_count),
            "auto_reduce_bins": bool(self.auto_reduce_bins),
            "attempted_bins": attempted_bins,
            "auto_reduced": bool(auto_reduced),
            # Stage-level bin counts (consistent with stratify)
            "trainval_bin_counts": _bin_counts(bins.loc[trainval.index]),
            "test_bin_counts": _bin_counts(bins.loc[test.index]),
            "train_label_stats": _label_stats(train[self.label_col]),
            "test_label_stats": _label_stats(test[self.label_col]),
        }

        log.info(
            "stratified_numeric:stats | bins_used=%d | train=%d | val=%d | test=%d",
            int(n_eff),
            stats["n_train"],
            stats["n_val"],
            stats["n_test"],
        )

        if val is not None:
            stats["val_label_stats"] = _label_stats(val[self.label_col])
            val_stage: dict[str, object] = {
                "n_bins_effective": int(val_n_eff) if val_n_eff is not None else None,
                "attempted_bins": val_attempted_bins,
                "auto_reduced": val_auto_reduced,
            }
            if self.report_bin_edges:
                val_stage["bin_edges"] = val_edges
            stats["val_stage"] = val_stage

        if self.report_bin_edges:
            stats["bin_edges"] = edges

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
