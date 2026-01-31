from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns


def _kmer_set(seq: str, k: int) -> set[str]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class KmerJaccardReducer:
    """
    Greedy redundancy reduction by Jaccard similarity of k-mer sets.

    Deterministic:
    - stable sort by id
    - first-accepted representative policy
    """

    threshold: float = 0.7
    k: int = 5
    max_candidates: int = 80  # cap comparisons per sequence

    @property
    def strategy(self) -> str:
        return "kmer_jaccard"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        reps_idx: List[int] = []
        reps_kmers: List[set[str]] = []
        removed_rows: List[Tuple[str, str, float]] = []

        # Inverted index for kmer -> rep indices
        kmer_to_rep: Dict[str, List[int]] = {}

        def add_rep(work_idx: int, seq: str) -> None:
            reps_idx.append(work_idx)
            km = _kmer_set(seq, self.k)
            reps_kmers.append(km)
            rep_pos = len(reps_idx) - 1
            for token in km:
                kmer_to_rep.setdefault(token, []).append(rep_pos)

        for i in range(len(work)):
            seq = str(work.at[i, cols.seq_col])
            cur_id = str(work.at[i, cols.id_col])

            if not reps_idx:
                add_rep(i, seq)
                continue

            km_cur = _kmer_set(seq, self.k)
            cand_counts: Dict[int, int] = {}
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
                rep_id = str(work.at[rep_work_idx, cols.id_col])
                removed_rows.append((cur_id, rep_id, float(best_score)))
            else:
                add_rep(i, seq)

        kept = work.iloc[reps_idx].reset_index(drop=True)
        mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "score"])

        return ReductionResult(
            df=kept,
            mapping=mapping,
            strategy=self.strategy,
            params={"threshold": self.threshold, "k": self.k, "max_candidates": self.max_candidates},
        )
