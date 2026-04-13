"""Greedy sequence reduction strategy using k-mer Jaccard similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _kmer_set(seq: str, k: int) -> set[str]:
    """Convert a sequence into a set of k-mers.

    Args:
        seq: Input sequence string.
        k: K-mer size (>= 1).

    Returns:
        Set of unique k-mer tokens. If len(seq) < k, returns {seq}.

    Raises:
        ValueError: If k < 1.

    """
    if k <= 0:
        msg = "k must be >= 1"
        raise ValueError(msg)
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets.

    Args:
        a: First input set.
        b: Second input set.

    Returns:
        Jaccard similarity in [0, 1]. If both empty, returns 1.0.

    """
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _validate_inputs(df: pd.DataFrame, cols: Columns, threshold: float, k: int, max_candidates: int) -> None:
    if not (0.0 <= threshold <= 1.0):
        msg = "threshold must be in [0, 1]"
        raise ValueError(msg)
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    if max_candidates < 1:
        msg = "max_candidates must be >= 1"
        raise ValueError(msg)
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}"
        raise ValueError(msg)
    if cols.seq_col not in df.columns:
        msg = f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns.tolist()}"
        raise ValueError(msg)


def _prepare_work(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy().sort_values(id_col, kind="mergesort").reset_index(drop=True)
    ids = work[id_col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


def _build_mapping(removed_rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "score"])
    if len(mapping) == 0:
        return pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"])
    mapping["cluster_id"] = mapping["representative_id"].astype(str).apply(lambda x: f"kmer:{x}")
    return mapping[["removed_id", "representative_id", "cluster_id", "score"]]


@dataclass(frozen=True)
class KmerJaccardReducer:
    r"""Greedy redundancy reduction using Jaccard similarity of k-mer sets.

    This reducer approximates sequence redundancy without alignment by comparing
    k-mer token sets. A sequence is considered redundant if its Jaccard similarity
    with an existing representative is >= `threshold`.

    Greedy policy:
    1) Sort dataset rows by `cols.id_col` (stable, deterministic).
    2) Iterate sequences in that order.
    3) First unseen sequence becomes a representative.
    4) For each new sequence, retrieve candidate representatives using an inverted
       index (kmer -> reps containing that kmer).
    5) Compute Jaccard similarity against top candidates (capped by `max_candidates`).
       If best >= threshold, mark as removed and map to representative; otherwise
       accept as a new representative.

    Candidate pruning:
    The inverted index provides candidate reps ordered by the count of shared k-mers.
    Only the top `max_candidates` are evaluated to cap runtime.

    Args:
        threshold:
            Jaccard similarity threshold in [0, 1]. If score >= threshold, the sequence
            is removed as redundant.
        k: K-mer size (>= 1). Typical values: 3-7 for proteins (tradeoff speed/specificity).
        max_candidates:
            Maximum number of representative candidates to evaluate per sequence (>= 1).
            Higher values are more accurate but slower.

    Returns:
        ReductionResult:
            Result containing representative-only data, a removed-to-
            representative mapping with Jaccard score, strategy name, and
            effective parameters. The reduced dataframe includes
            `kmer_cluster_id` (`kmer:<rep_id>`) and params include reduction
            stats.

    Raises:
        ValueError: If threshold is out of range, k < 1, max_candidates < 1, required columns are
        missing, ids are duplicated, or sequences are empty/invalid.

    Notes:
        - This is a greedy algorithm: results depend on representative ordering
        (here: sorted by id for determinism).
        - Jaccard(k-mer) is an approximation. It does not guarantee homology clustering
        and may behave differently from alignment-based tools (e.g., MMseqs2).
        - Missing edges are not applicable here; similarity is computed on-the-fly from sequences.

    Examples:
        >>> biosieve reduce \\
        ...   --in dataset.csv \\
        ...   --out data_nr_kmer.csv \\
        ...   --strategy kmer_jaccard \\
        ...   --map map_kmer.csv \\
        ...   --report report_kmer.json \\
        ...   --params params.yaml

    """

    threshold: float = 0.7
    k: int = 5
    max_candidates: int = 80  # cap comparisons per sequence

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "kmer_jaccard"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:  # noqa: C901
        """Reduce sequence redundancy with k-mer candidate pruning."""
        _validate_inputs(df, cols, self.threshold, self.k, self.max_candidates)
        work, _ids = _prepare_work(df, cols.id_col)

        # Representatives are tracked by work index
        reps_idx: list[int] = []
        reps_kmers: list[set[str]] = []

        removed_rows: list[tuple[str, str, float]] = []

        # Inverted index for kmer -> rep positions
        kmer_to_rep: dict[str, list[int]] = {}

        def add_rep(work_idx: int, seq: str) -> None:
            reps_idx.append(work_idx)
            km = _kmer_set(seq, self.k)
            reps_kmers.append(km)
            rep_pos = len(reps_idx) - 1
            for token in km:
                kmer_to_rep.setdefault(token, []).append(rep_pos)

        for i in range(len(work)):
            seq = str(work.loc[i, cols.seq_col])
            cur_id = str(work.loc[i, cols.id_col])

            if not seq or seq.lower() == "nan":
                msg = (
                    f"Empty/invalid sequence for id={cur_id} in column '{cols.seq_col}'. "
                    "Clean dataset before kmer_jaccard reduction."
                )
                raise ValueError(
                    msg
                )

            if not reps_idx:
                add_rep(i, seq)
                continue

            km_cur = _kmer_set(seq, self.k)

            # Candidate rep scoring by shared k-mer counts
            cand_counts: dict[int, int] = {}
            for token in km_cur:
                for rep_pos in kmer_to_rep.get(token, []):
                    cand_counts[rep_pos] = cand_counts.get(rep_pos, 0) + 1

            candidates = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)
            if not candidates:
                add_rep(i, seq)
                continue

            best_rep_pos = None
            best_score = -1.0

            for rep_pos, _cnt in candidates[: self.max_candidates]:
                score = _jaccard(km_cur, reps_kmers[rep_pos])
                if score > best_score:
                    best_score = score
                    best_rep_pos = rep_pos
                if best_score >= self.threshold:
                    break

            if best_rep_pos is not None and best_score >= self.threshold:
                rep_work_idx = reps_idx[best_rep_pos]
                rep_id = str(work.loc[rep_work_idx, cols.id_col])
                removed_rows.append((cur_id, rep_id, float(best_score)))
            else:
                add_rep(i, seq)

        kept = work.iloc[reps_idx].reset_index(drop=True)

        mapping = _build_mapping(removed_rows)

        kept["kmer_cluster_id"] = kept[cols.id_col].astype(str).apply(lambda x: f"kmer:{x}")

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_kept": len(kept),
            "n_removed": len(mapping),
            "reduction_ratio": float(len(kept) / len(work)) if len(work) else 0.0,
            "k": int(self.k),
            "threshold": float(self.threshold),
            "max_candidates": int(self.max_candidates),
        }

        return ReductionResult(
            df=kept,
            mapping=mapping,
            strategy=self.strategy,
            params={
                "threshold": self.threshold,
                "k": self.k,
                "max_candidates": self.max_candidates,
                "stats": stats,
            },
        )
