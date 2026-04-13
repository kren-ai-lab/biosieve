"""Homology-aware reduction strategy backed by MMseqs2 clustering."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from biosieve.reduction.backends.mmseqs2_backend import (
    build_cluster_ids,
    parse_cluster_tsv,
    run_mmseqs_easy_cluster,
    write_fasta,
)
from biosieve.reduction.base import ReductionResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_inputs(
    df: pd.DataFrame,
    cols: Columns,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> None:
    if not (0.0 <= min_seq_id <= 1.0):
        msg = "min_seq_id must be in [0, 1]"
        raise ValueError(msg)
    if not (0.0 <= coverage <= 1.0):
        msg = "coverage must be in [0, 1]"
        raise ValueError(msg)
    if threads < 1:
        msg = "threads must be >= 1"
        raise ValueError(msg)
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}"
        raise ValueError(msg)
    if cols.seq_col not in df.columns:
        msg = f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns.tolist()}"
        raise ValueError(msg)


def _build_sequence_map(work: pd.DataFrame, cols: Columns) -> dict[str, str]:
    seqs: dict[str, str] = {}
    empty_seq = 0
    for _, row in work.iterrows():
        sid = str(row[cols.id_col])
        seq = str(row[cols.seq_col])
        if sid in seqs:
            msg = f"Duplicate ids detected: {sid}. IDs must be unique for MMseqs2 FASTA."
            raise ValueError(msg)
        if not seq or seq.lower() == "nan":
            empty_seq += 1
        seqs[sid] = seq
    if empty_seq > 0:
        msg = (
            f"Found {empty_seq} empty/invalid sequences in column '{cols.seq_col}'. "
            "Clean dataset before mmseqs2 reduction."
        )
        raise ValueError(msg)
    return seqs


def _build_outputs(
    *,
    work: pd.DataFrame,
    cols: Columns,
    member_to_rep: dict[str, str],
    member_to_cluster: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reps = {m for m, rep in member_to_rep.items() if m == rep}
    kept_df = work[work[cols.id_col].astype(str).isin(reps)].copy()
    kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
    removed_rows = [
        {
            "removed_id": member,
            "representative_id": rep,
            "cluster_id": member_to_cluster[member],
            "score": None,
        }
        for member, rep in member_to_rep.items()
        if member != rep
    ]
    mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "cluster_id", "score"])
    kept_df["mmseqs2_cluster_id"] = kept_df[cols.id_col].astype(str).map(member_to_cluster)
    stats: dict[str, Any] = {
        "n_total": len(work),
        "n_kept": len(kept_df),
        "n_removed": len(mapping),
        "n_clusters": len(reps),
        "reduction_ratio": float(len(kept_df) / len(work)) if len(work) else 0.0,
        "note": "Representative selection delegated to MMseqs2 easy-cluster.",
    }
    return kept_df, mapping, stats


@dataclass(frozen=True)
class MMseqs2Reducer:
    r"""Homology-based redundancy reduction using MMseqs2 easy-cluster.

    This reducer clusters sequences by homology using MMseqs2 (`easy-cluster`) and keeps
    one representative per cluster (as determined by MMseqs2). All non-representatives
    are removed and recorded in the returned mapping.

    Args:
        min_seq_id: Minimum sequence identity threshold in [0, 1] (MMseqs2 `--min-seq-id`).
        coverage: Alignment coverage threshold in [0, 1] (MMseqs2 `-c`).
        cov_mode: Coverage mode (MMseqs2 `--cov-mode`).
        cluster_mode: Cluster mode (MMseqs2 `--cluster-mode`).
        threads: Number of threads for MMseqs2.
        tmp_root:
            Optional parent directory for the temporary work directory. If None,
            uses system temp.
        keep_tmp:
            If True, copies the temporary folder to
            `./biosieve_mmseqs2_debug/` for debugging.

    Returns:
        ReductionResult:
            Result containing MMseqs2 representatives, removed-to-representative
            mapping, strategy name, and effective parameters. The reduced
            dataframe includes `mmseqs2_cluster_id`, and mapping rows contain
            removed id, representative id, cluster id, and score (`None` in
            easy-cluster mode).

    Raises:
        ValueError: If required columns are missing, ids are duplicated, sequences are empty,
        or parameter ranges are invalid.
        FileNotFoundError: If the mmseqs2 binary is missing (raised by backend).
        RuntimeError: If MMseqs2 returns a non-zero exit code (raised by backend).

    Notes:
        - MMseqs2 decides the representative per cluster. BioSieve treats that decision
        as authoritative for this strategy.
        - This strategy does not provide per-pair scores by default in `easy-cluster`.
        Hence `score` is always None.
        - IDs must be unique to produce a valid FASTA mapping.

    Examples:
        >>> biosieve reduce \\
        ...   --in dataset.csv \\
        ...   --out out_nr_mmseqs2.csv \\
        ...   --strategy mmseqs2 \\
        ...   --map map_mmseqs2.csv \\
        ...   --report report_mmseqs2.json \\
        ...   --params params.yaml

    """

    min_seq_id: float = 0.9
    coverage: float = 0.8
    cov_mode: int = 0
    cluster_mode: int = 0
    threads: int = 4

    # housekeeping
    tmp_root: str | None = None  # if None, uses system temp
    keep_tmp: bool = False

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "mmseqs2"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        """Run MMseqs2 clustering and return reduced data plus mapping."""
        _validate_inputs(df, cols, self.min_seq_id, self.coverage, self.threads)

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        seqs = _build_sequence_map(work, cols)

        tmp_base = self.tmp_root if self.tmp_root is not None else None
        if tmp_base is not None:
            Path(tmp_base).mkdir(parents=True, exist_ok=True)
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="biosieve_mmseqs2_", dir=tmp_base)
        tmpdir = Path(tmpdir_obj.name)

        try:
            fasta_path = tmpdir / "input.fasta"
            out_prefix = tmpdir / "mmseqs2_out"
            mmseqs_tmp = tmpdir / "mmseqs_tmp"

            write_fasta(seqs, fasta_path)

            paths = run_mmseqs_easy_cluster(
                fasta_path=fasta_path,
                out_prefix=out_prefix,
                tmp_dir=mmseqs_tmp,
                min_seq_id=self.min_seq_id,
                coverage=self.coverage,
                cov_mode=self.cov_mode,
                cluster_mode=self.cluster_mode,
                threads=self.threads,
            )

            member_to_rep = parse_cluster_tsv(paths.cluster_tsv)
            member_to_cluster = build_cluster_ids(member_to_rep)
            kept_df, mapping, stats = _build_outputs(
                work=work,
                cols=cols,
                member_to_rep=member_to_rep,
                member_to_cluster=member_to_cluster,
            )

            # If ReductionResult supports stats, include it; if not, keep inside params.
            params = {
                "min_seq_id": self.min_seq_id,
                "coverage": self.coverage,
                "cov_mode": self.cov_mode,
                "cluster_mode": self.cluster_mode,
                "threads": self.threads,
                "keep_tmp": self.keep_tmp,
                "stats": stats,
            }

            return ReductionResult(
                df=kept_df,
                mapping=mapping,
                strategy=self.strategy,
                params=params,
            )

        finally:
            if self.keep_tmp:
                debug_dir = Path("biosieve_mmseqs2_debug")
                try:
                    if debug_dir.exists():
                        shutil.rmtree(debug_dir)
                    shutil.copytree(tmpdir, debug_dir)
                except OSError as e:
                    log.warning("Failed to copy debug artifacts to %s: %s", debug_dir, e)
            tmpdir_obj.cleanup()
