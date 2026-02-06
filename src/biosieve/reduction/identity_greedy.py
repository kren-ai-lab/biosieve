from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns


def _kmer_set(seq: str, k: int) -> set[str]:
    """Return set of k-mers for a sequence."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _approx_identity(a: str, b: str) -> float:
    """
    Approximate identity without alignment.

    Compares positions up to min length and normalizes by max length
    to penalize length differences.
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
    Greedy redundancy reduction using an approximate identity score.

    This reducer approximates sequence identity without alignment by combining:
    1) k-mer Jaccard prefilter to shortlist candidate representatives
    2) approximate identity (position-wise match rate) to apply a final threshold

    A sequence is removed if there exists an earlier representative such that:
    - relative length difference <= length_tolerance
    - k-mer Jaccard >= jaccard_prefilter
    - approx_identity >= threshold

    Determinism
    ----------
    - Rows are sorted by `cols.id_col` (stable).
    - "First accepted representative" policy.

    Parameters
    ----------
    threshold:
        Identity threshold in [0, 1]. If approx_identity >= threshold, the sequence
        is considered redundant.
    k:
        K-mer size for the Jaccard prefilter (>= 1).
    jaccard_prefilter:
        Jaccard cutoff in [0, 1] used to shortlist candidates.
    length_tolerance:
        Relative length tolerance in [0, 1]. Candidate is skipped if:
        abs(len_a - len_b) / max(len_a, len_b) > length_tolerance
    max_candidates:
        Maximum number of candidate representatives evaluated per sequence (>= 1).

    Returns
    -------
    ReductionResult
        - df:
            Reduced DataFrame containing representatives only.
            Adds `identity_cluster_id` as `ident:<rep_id>` for convenience.
        - mapping:
            DataFrame with columns:
              * removed_id
              * representative_id
              * cluster_id (`ident:<rep_id>`)
              * score (approx identity; higher means more similar)
        - strategy:
            "identity_greedy"
        - params:
            Effective parameters plus `stats` summary.

    Raises
    ------
    ValueError
        If required columns are missing, ids are duplicated, sequences are empty/invalid,
        or parameters are out of range.

    Notes
    -----
    - This is NOT a true alignment-based identity; it is a fast approximation.
    - It is best used as a lightweight heuristic reducer, not as a substitute for
      MMseqs2/CD-HIT when strict homology reduction is required.
    - Greedy selection means ordering matters; we sort by id for reproducibility.

    Examples
    --------
    >>> biosieve reduce \\
    ...   --in dataset.csv \\
    ...   --out data_nr_ident.csv \\
    ...   --strategy identity_greedy \\
    ...   --map map_ident.csv \\
    ...   --report report_ident.json \\
    ...   --params params.yaml
    """

    threshold: float = 0.9
    k: int = 5
    jaccard_prefilter: float = 0.2
    length_tolerance: float = 0.15
    max_candidates: int = 50

    @property
    def strategy(self) -> str:
        return "identity_greedy"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        # --- parameter validation ---
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not (0.0 <= self.jaccard_prefilter <= 1.0):
            raise ValueError("jaccard_prefilter must be in [0, 1]")
        if not (0.0 <= self.length_tolerance <= 1.0):
            raise ValueError("length_tolerance must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")

        # --- column validation ---
        if cols.id_col not in df.columns:
            raise ValueError(f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}")
        if cols.seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns.tolist()}")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        ids = work[cols.id_col].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids detected. IDs must be unique for deterministic reduction mapping.")

        reps_idx: List[int] = []
        reps_kmers: List[set[str]] = []
        reps_len: List[int] = []

        removed_rows: List[Tuple[str, str, float]] = []

        # inverted index: kmer -> rep positions
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

            if not seq or seq.lower() == "nan":
                raise ValueError(
                    f"Empty/invalid sequence for id={cur_id} in column '{cols.seq_col}'. "
                    "Clean dataset before identity_greedy reduction."
                )

            cur_len = len(seq)

            if not reps_idx:
                add_rep(i, seq)
                continue

            km_cur = _kmer_set(seq, self.k)

            # candidates via shared kmers (counts)
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
                rep_len = reps_len[rep_pos]
                if rep_len == 0 or cur_len == 0:
                    continue

                rel_diff = abs(rep_len - cur_len) / max(rep_len, cur_len)
                if rel_diff > self.length_tolerance:
                    continue

                jac = _jaccard(km_cur, reps_kmers[rep_pos])
                if jac < self.jaccard_prefilter:
                    continue

                rep_seq = str(work.at[reps_idx[rep_pos], cols.seq_col])
                ident = _approx_identity(seq, rep_seq)

                if ident > best_score:
                    best_score = ident
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
        if len(mapping) > 0:
            mapping["cluster_id"] = mapping["representative_id"].astype(str).apply(lambda x: f"ident:{x}")
            mapping = mapping[["removed_id", "representative_id", "cluster_id", "score"]]
        else:
            mapping = pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"])

        kept["identity_cluster_id"] = kept[cols.id_col].astype(str).apply(lambda x: f"ident:{x}")

        stats: Dict[str, Any] = {
            "n_total": int(len(work)),
            "n_kept": int(len(kept)),
            "n_removed": int(len(mapping)),
            "reduction_ratio": float(len(kept) / len(work)) if len(work) else 0.0,
            "threshold": float(self.threshold),
            "k": int(self.k),
            "jaccard_prefilter": float(self.jaccard_prefilter),
            "length_tolerance": float(self.length_tolerance),
            "max_candidates": int(self.max_candidates),
            "note": "Approx identity without alignment; k-mer Jaccard used as prefilter.",
        }

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
                "stats": stats,
            },
        )
