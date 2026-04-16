"""Greedy sequence reduction strategy using approximate identity heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _kmer_set(seq: str, k: int) -> set[str]:
    """Return set of k-mers for a sequence."""
    if k <= 0:
        msg = "k must be >= 1"
        raise ValueError(msg)
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
    """Approximate identity without alignment.

    Compares positions up to min length and normalizes by max length
    to penalize length differences.
    """
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    m = min(la, lb)
    matches = sum(1 for i in range(m) if a[i] == b[i])
    return matches / max(la, lb)


def _validate_inputs(
    *,
    df: pl.DataFrame,
    cols: Columns,
    threshold: float,
    k: int,
    jaccard_prefilter: float,
    length_tolerance: float,
    max_candidates: int,
) -> None:
    if not (0.0 <= threshold <= 1.0):
        msg = "threshold must be in [0, 1]"
        raise ValueError(msg)
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    if not (0.0 <= jaccard_prefilter <= 1.0):
        msg = "jaccard_prefilter must be in [0, 1]"
        raise ValueError(msg)
    if not (0.0 <= length_tolerance <= 1.0):
        msg = "length_tolerance must be in [0, 1]"
        raise ValueError(msg)
    if max_candidates < 1:
        msg = "max_candidates must be >= 1"
        raise ValueError(msg)
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns}"
        raise ValueError(msg)
    if cols.seq_col not in df.columns:
        msg = f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns}"
        raise ValueError(msg)


def _prepare_work(df: pl.DataFrame, id_col: str) -> tuple[pl.DataFrame, list[str]]:
    work = df.clone().sort(id_col, maintain_order=True)
    ids = work[id_col].cast(pl.String).to_list()
    if len(ids) != len(set(ids)):
        msg = "Duplicate ids detected. IDs must be unique for deterministic reduction mapping."
        raise ValueError(msg)
    return work, ids


@dataclass(frozen=True)
class IdentityGreedyReducer:
    r"""Greedy redundancy reduction using an approximate identity score.

    This reducer approximates sequence identity without alignment by combining:
    1) k-mer Jaccard prefilter to shortlist candidate representatives
    2) approximate identity (position-wise match rate) to apply a final threshold

    A sequence is removed if there exists an earlier representative such that:
    - relative length difference <= length_tolerance
    - k-mer Jaccard >= jaccard_prefilter
    - approx_identity >= threshold

    Determinism:
    - Rows are sorted by `cols.id_col` (stable).
    - "First accepted representative" policy.

    Args:
        threshold: Identity threshold in [0, 1]. If approx_identity >= threshold, the sequence
            is considered redundant.
        k: K-mer size for the Jaccard prefilter (>= 1).
        jaccard_prefilter: Jaccard cutoff in [0, 1] used to shortlist candidates.
        length_tolerance: Relative length tolerance in [0, 1]. Candidate is skipped if:
            abs(len_a - len_b) / max(len_a, len_b) > length_tolerance
        max_candidates: Maximum number of candidate representatives evaluated per sequence (>= 1).

    Returns:
        ReductionResult:
            Result containing representative-only data, a removed-to-
            representative mapping with approximate identity score, strategy name,
            and effective parameters with summary stats. The reduced dataframe
            includes `identity_cluster_id` (`ident:<rep_id>`).

    Raises:
        ValueError: If required columns are missing, ids are duplicated, sequences are empty/invalid,
        or parameters are out of range.

    Notes:
        - This is NOT a true alignment-based identity; it is a fast approximation.
        - It is best used as a lightweight heuristic reducer, not as a substitute for
        MMseqs2/CD-HIT when strict homology reduction is required.
        - Greedy selection means ordering matters; we sort by id for reproducibility.

    Examples:
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
        """Return the strategy identifier."""
        return "identity_greedy"

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:  # noqa: C901,PLR0912,PLR0915
        """Reduce sequence redundancy with k-mer prefilter and identity scoring."""
        _validate_inputs(
            df=df,
            cols=cols,
            threshold=self.threshold,
            k=self.k,
            jaccard_prefilter=self.jaccard_prefilter,
            length_tolerance=self.length_tolerance,
            max_candidates=self.max_candidates,
        )
        work, _ = _prepare_work(df, cols.id_col)

        reps_idx: list[int] = []
        reps_kmers: list[set[str]] = []
        reps_len: list[int] = []

        removed_rows: list[tuple[str, str, float]] = []

        # inverted index: kmer -> rep positions
        kmer_to_rep: dict[str, list[int]] = {}

        def add_rep(rep_index_in_work: int, rep_seq: str) -> None:
            reps_idx.append(rep_index_in_work)
            km = _kmer_set(rep_seq, self.k)
            reps_kmers.append(km)
            reps_len.append(len(rep_seq))
            rep_pos = len(reps_idx) - 1
            for token in km:
                kmer_to_rep.setdefault(token, []).append(rep_pos)

        for i in range(work.height):
            seq = str(work[i, cols.seq_col])
            cur_id = str(work[i, cols.id_col])

            if not seq or seq.lower() == "nan":
                msg = (
                    f"Empty/invalid sequence for id={cur_id} in column '{cols.seq_col}'. "
                    "Clean dataset before identity_greedy reduction."
                )
                raise ValueError(msg)

            cur_len = len(seq)

            if not reps_idx:
                add_rep(i, seq)
                continue

            km_cur = _kmer_set(seq, self.k)

            # candidates via shared kmers (counts)
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
                rep_len = reps_len[rep_pos]
                if rep_len == 0 or cur_len == 0:
                    continue

                rel_diff = abs(rep_len - cur_len) / max(rep_len, cur_len)
                if rel_diff > self.length_tolerance:
                    continue

                jac = _jaccard(km_cur, reps_kmers[rep_pos])
                if jac < self.jaccard_prefilter:
                    continue

                rep_seq = str(work[reps_idx[rep_pos], cols.seq_col])
                ident = _approx_identity(seq, rep_seq)

                if ident > best_score:
                    best_score = ident
                    best_rep_pos = rep_pos

                if best_score >= self.threshold:
                    break

            if best_rep_pos is not None and best_score >= self.threshold:
                rep_work_idx = reps_idx[best_rep_pos]
                rep_id = str(work[rep_work_idx, cols.id_col])
                removed_rows.append((cur_id, rep_id, float(best_score)))
            else:
                add_rep(i, seq)

        kept = work[reps_idx]

        if removed_rows:
            mapping = pl.DataFrame(
                removed_rows,
                schema=["removed_id", "representative_id", "score"],
                orient="row",
            ).with_columns(
                (pl.lit("ident:") + pl.col("representative_id").cast(pl.String)).alias("cluster_id")
            ).select(["removed_id", "representative_id", "cluster_id", "score"])
        else:
            mapping = pl.DataFrame(
                schema={
                    "removed_id": pl.String,
                    "representative_id": pl.String,
                    "cluster_id": pl.String,
                    "score": pl.Float64,
                }
            )

        kept = kept.with_columns(
            (pl.lit("ident:") + pl.col(cols.id_col).cast(pl.String)).alias("identity_cluster_id")
        )

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_kept": kept.height,
            "n_removed": mapping.height,
            "reduction_ratio": float(kept.height / work.height) if work.height else 0.0,
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
