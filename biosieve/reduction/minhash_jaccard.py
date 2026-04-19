"""Approximate greedy sequence reduction using MinHash Jaccard similarity.

This strategy is algorithmically equivalent to kmer_jaccard but replaces
the exact Jaccard computation with MinHash approximation and LSH indexing.
The trade-off: Jaccard values are estimates (not exact), but the approach
scales to large datasets where brute-force pairwise comparison is too slow.

Requires the optional `datasketch` package:
    pip install biosieve[minhash]
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from biosieve.reduction.backends.kmer_backend import _build_mapping, _kmer_set
from biosieve.reduction.base import ReductionResult
from biosieve.reduction.common import build_reduction_stats, prepare_reduction_work
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    import polars as pl

    from biosieve.types import Columns

log = get_logger(__name__)


def _try_import_datasketch() -> tuple[Any, Any]:
    """Try to import MinHash and MinHashLSH from datasketch."""
    try:
        datasketch = importlib.import_module("datasketch")
    except ImportError:
        return None, None
    else:
        return datasketch.MinHash, datasketch.MinHashLSH


def _validate_inputs(df: pl.DataFrame, cols: Columns, threshold: float, k: int, num_perm: int) -> None:
    if not (0.0 <= threshold <= 1.0):
        msg = "threshold must be in [0, 1]"
        raise ValueError(msg)
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    if num_perm < 2:  # noqa: PLR2004
        msg = "num_perm must be >= 2"
        raise ValueError(msg)
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns}"
        raise ValueError(msg)
    if cols.seq_col not in df.columns:
        msg = f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns}"
        raise ValueError(msg)


@dataclass(frozen=True)
class MinHashJaccardReducer:
    r"""Approximate greedy redundancy reduction using MinHash Jaccard similarity.

    Algorithmically equivalent to KmerJaccardReducer, but uses MinHash signatures
    and LSH indexing instead of exact Jaccard computation. This makes it practical
    for large datasets (>50k sequences) where exact pairwise comparison becomes a
    bottleneck.

    How it works:
    1) Sort dataset by `cols.id_col` for determinism.
    2) For each sequence, compute a MinHash signature from its k-mer set.
    3) Query the LSH index for representatives with estimated Jaccard >= `threshold`.
    4) If any match: mark the sequence as removed, mapping it to the most similar rep.
    5) If no match: insert into the LSH index as a new representative.

    Approximation notes:
    - Jaccard values in the mapping are MinHash estimates, not exact values.
    - LSH has false negatives: some similar pairs may not be retrieved.
      This means the output may contain more sequences than the exact kmer_jaccard
      would at the same threshold. Increase `num_perm` to reduce this effect.
    - LSH has false positives: pairs slightly below threshold may be matched.
      This is generally acceptable for deduplication purposes.

    Args:
        threshold: Jaccard similarity threshold in [0, 1]. Pairs with estimated
            Jaccard >= threshold are considered redundant.
        k: K-mer size (>= 1). Should match the value used in kmer_jaccard for
            comparable behaviour. Typical values: 3-7 for proteins.
        num_perm: Number of MinHash permutations. Higher values = more accurate
            estimates and fewer false negatives, at the cost of memory and speed.
            128 is a good default; use 256+ for high-precision deduplication.
        seed: Random seed for MinHash permutations. Controls reproducibility.

    Returns:
        ReductionResult with the deduplicated DataFrame, a mapping table
        (removed_id, representative_id, cluster_id, score), strategy name,
        and effective parameters including reduction stats.

    Raises:
        ImportError: If `datasketch` is not installed.
        ValueError: If parameters are out of range, required columns are missing,
            ids are duplicated, or sequences are empty/invalid.

    Examples:
        >>> biosieve reduce \
        ...   -i dataset.csv -o data_nr.csv \
        ...   --strategy minhash_jaccard \
        ...   --params params.yaml

        YAML params::

            minhash_jaccard:
              threshold: 0.8
              k: 5
              num_perm: 128

    """

    threshold: float = 0.7
    k: int = 5
    num_perm: int = 128
    seed: int = 42

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "minhash_jaccard"

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:
        """Reduce sequence redundancy with MinHash LSH candidate lookup."""
        MinHash, MinHashLSH = _try_import_datasketch()
        if MinHash is None:
            msg = (
                "datasketch is required for the minhash_jaccard strategy. "
                "Install it with: pip install biosieve[minhash]"
            )
            raise ImportError(msg)

        _validate_inputs(df, cols, self.threshold, self.k, self.num_perm)
        work, _ids = prepare_reduction_work(df, cols.id_col)

        # datasketch rejects threshold=1.0 for LSH configuration. In that edge case,
        # only identical MinHash signatures can match, so a direct signature lookup
        # preserves the intended behavior without going through LSH.
        lsh = None if self.threshold >= 1.0 else MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        rep_minhashes: dict[str, Any] = {}
        rep_signature_to_id: dict[tuple[int, ...], str] = {}
        reps_idx: list[int] = []
        removed_rows: list[tuple[str, str, float]] = []

        for i in range(work.height):
            seq = str(work[i, cols.seq_col])
            cur_id = str(work[i, cols.id_col])

            if not seq or seq.lower() == "nan":
                msg = (
                    f"Empty/invalid sequence for id={cur_id} in column '{cols.seq_col}'. "
                    "Clean dataset before minhash_jaccard reduction."
                )
                raise ValueError(msg)

            mh = MinHash(num_perm=self.num_perm, seed=self.seed)
            for token in _kmer_set(seq, self.k):
                mh.update(token.encode())

            if lsh is None:
                candidate = rep_signature_to_id.get(tuple(int(x) for x in mh.hashvalues))
                candidates = [candidate] if candidate is not None else []
            else:
                candidates = lsh.query(mh)

            if candidates:
                best_rep_id = max(candidates, key=lambda rid: mh.jaccard(rep_minhashes[rid]))
                best_score = float(mh.jaccard(rep_minhashes[best_rep_id]))
                removed_rows.append((cur_id, best_rep_id, best_score))
            else:
                if lsh is not None:
                    lsh.insert(cur_id, mh)
                rep_minhashes[cur_id] = mh
                rep_signature_to_id[tuple(int(x) for x in mh.hashvalues)] = cur_id
                reps_idx.append(i)

        kept = work[reps_idx]
        mapping = _build_mapping(removed_rows, cluster_prefix="minhash")

        stats: dict[str, Any] = build_reduction_stats(
            n_total=work.height,
            n_kept=kept.height,
            k=self.k,
            threshold=self.threshold,
            num_perm=self.num_perm,
        )

        log.info(
            "minhash_jaccard: %d → %d sequences (removed %d, ratio=%.3f)",
            stats["n_total"],
            stats["n_kept"],
            stats["n_removed"],
            stats["reduction_ratio"],
        )

        return ReductionResult(
            df=kept,
            mapping=mapping,
            strategy=self.strategy,
            params={
                "threshold": self.threshold,
                "k": self.k,
                "num_perm": self.num_perm,
                "seed": self.seed,
            },
            stats=stats,
        )
