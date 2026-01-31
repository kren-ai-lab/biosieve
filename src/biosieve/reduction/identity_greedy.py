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
        return {seq}  # degenerate sketch
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _approx_identity(a: str, b: str) -> float:
    """
    Approx identity without alignment: compare positions up to min length.
    Normalize by max length to penalize length differences.
    """
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    m = min(la, lb)
    matches = sum(1 for i in range(m) if a[i] == b[i])
    return matches / max(la, lb)


@dataclass(frozen=True)
class IdentityGreedyReducer:
    """
    Greedy redundancy reduction using:
    - k-mer Jaccard prefilter to shortlist candidates
    - approximate identity (position-wise) for thresholding

    Deterministic by:
    - stable sort by id
    - first-accepted representative policy
    """

    threshold: float = 0.9
    k: int = 5
    jaccard_prefilter: float = 0.2
    length_tolerance: float = 0.15  # relative tolerance on length (±15%)
    max_candidates: int = 50        # cap comparisons per sequence (speed bound)

    @property
    def strategy(self) -> str:
        return "identity_greedy"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if not (0.0 <= self.jaccard_prefilter <= 1.0):
            raise ValueError("jaccard_prefilter must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        reps_idx: List[int] = []
        reps_kmers: List[set[str]] = []
        reps_len: List[int] = []

        removed_rows: List[Tuple[str, str, float]] = []  # (removed_id, rep_id, score)

        # Simple inverted index for kmers -> rep indices (to avoid full scan)
        kmer_to_rep: Dict[str, List[int]] = {}

        def add_rep(rep_index_in_work: int, rep_seq: str) -> None:
            reps_idx.append(rep_index_in_work)
            km = _kmer_set(rep_seq, self.k)
            reps_kmers.append(km)
            reps_len.append(len(rep_seq))
            rep_pos = len(reps_idx) - 1
            for token in km:
                kmer_to_rep.setdefault(token, []).append(rep_pos)

        for i in range(len(work)):
            seq = str(work.at[i, cols.seq_col])
            cur_id = str(work.at[i, cols.id_col])
            cur_len = len(seq)

            if not reps_idx:
                add_rep(i, seq)
                continue

            # Candidate reps via shared kmers
            km_cur = _kmer_set(seq, self.k)
            cand_counts: Dict[int, int] = {}
            for token in km_cur:
                for rep_pos in kmer_to_rep.get(token, []):
                    cand_counts[rep_pos] = cand_counts.get(rep_pos, 0) + 1

            # Rank candidates by shared-token count (desc)
            candidates = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)
            if not candidates:
                add_rep(i, seq)
                continue

            # Evaluate shortlist
            best_rep_pos = None
            best_score = -1.0

            # Compare at most max_candidates reps
            for rep_pos, _cnt in candidates[: self.max_candidates]:
                rep_len = reps_len[rep_pos]
                # length filter
                if rep_len == 0 or cur_len == 0:
                    continue
                rel_diff = abs(rep_len - cur_len) / max(rep_len, cur_len)
                if rel_diff > self.length_tolerance:
                    continue

                # jaccard prefilter
                jac = _jaccard(km_cur, reps_kmers[rep_pos])
                if jac < self.jaccard_prefilter:
                    continue

                rep_seq = str(work.at[reps_idx[rep_pos], cols.seq_col])
                ident = _approx_identity(seq, rep_seq)
                if ident > best_score:
                    best_score = ident
                    best_rep_pos = rep_pos

                # Early stop if we already meet threshold with a high candidate
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
            params={
                "threshold": self.threshold,
                "k": self.k,
                "jaccard_prefilter": self.jaccard_prefilter,
                "length_tolerance": self.length_tolerance,
                "max_candidates": self.max_candidates,
            },
        )
