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


@dataclass(frozen=True)
class MMseqs2Reducer:
    """Homology-based redundancy reduction using MMseqs2 easy-cluster.

    This reducer clusters sequences by homology using MMseqs2 (`easy-cluster`) and keeps
    one representative per cluster (as determined by MMseqs2). All non-representatives
    are removed and recorded in the returned mapping.

    Parameters
    ----------
    min_seq_id:
        Minimum sequence identity threshold in [0, 1] (MMseqs2 `--min-seq-id`).
    coverage:
        Alignment coverage threshold in [0, 1] (MMseqs2 `-c`).
    cov_mode:
        Coverage mode (MMseqs2 `--cov-mode`).
    cluster_mode:
        Cluster mode (MMseqs2 `--cluster-mode`).
    threads:
        Number of threads for MMseqs2.

    tmp_root:
        Optional parent directory for the temporary work directory. If None, uses system temp.
    keep_tmp:
        If True, copies the temporary folder to `./biosieve_mmseqs2_debug/` for debugging.

    Returns
    -------
    ReductionResult
        - df:
            DataFrame containing only MMseqs2 representatives (one per cluster).
            A helper column `mmseqs2_cluster_id` is added to representatives.
        - mapping:
            DataFrame with columns:
            * removed_id: removed member id
            * representative_id: cluster representative id
            * cluster_id: deterministic cluster id (rep-based)
            * score: always None in easy-cluster mode
        - strategy:
            "mmseqs2"
        - params:
            Effective parameters used for clustering and temp handling.

    Raises
    ------
    ValueError
        If required columns are missing, ids are duplicated, sequences are empty,
        or parameter ranges are invalid.
    FileNotFoundError
        If the mmseqs2 binary is missing (raised by backend).
    RuntimeError
        If MMseqs2 returns a non-zero exit code (raised by backend).

    Notes
    -----
    - MMseqs2 decides the representative per cluster. BioSieve treats that decision
      as authoritative for this strategy.
    - This strategy does not provide per-pair scores by default in `easy-cluster`.
      Hence `score` is always None.
    - IDs must be unique to produce a valid FASTA mapping.

    Examples
    --------
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
        return "mmseqs2"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        # --- basic checks ---
        if not (0.0 <= self.min_seq_id <= 1.0):
            raise ValueError("min_seq_id must be in [0, 1]")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0, 1]")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")

        if cols.id_col not in df.columns:
            raise ValueError(f"Missing id column '{cols.id_col}'. Columns: {df.columns.tolist()}")
        if cols.seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{cols.seq_col}'. Columns: {df.columns.tolist()}")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        # Build sequence dict (id -> sequence)
        seqs: dict[str, str] = {}
        empty_seq = 0
        for _, row in work.iterrows():
            sid = str(row[cols.id_col])
            seq = str(row[cols.seq_col])

            if sid in seqs:
                raise ValueError(f"Duplicate ids detected: {sid}. IDs must be unique for MMseqs2 FASTA.")
            if not seq or seq.lower() == "nan":
                empty_seq += 1
            seqs[sid] = seq

        if empty_seq > 0:
            raise ValueError(
                f"Found {empty_seq} empty/invalid sequences in column '{cols.seq_col}'. "
                "Clean dataset before mmseqs2 reduction."
            )

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

            # Representatives are those where member_id == representative_id
            reps = {m for m, rep in member_to_rep.items() if m == rep}

            kept_df = work[work[cols.id_col].astype(str).isin(reps)].copy()
            kept_df = kept_df.sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

            # mapping: every non-representative maps to rep
            removed_rows = []
            for member, rep in member_to_rep.items():
                if member == rep:
                    continue
                removed_rows.append(
                    {
                        "removed_id": member,
                        "representative_id": rep,
                        "cluster_id": member_to_cluster[member],
                        "score": None,  # not provided by easy-cluster
                    }
                )

            mapping = pd.DataFrame(
                removed_rows,
                columns=["removed_id", "representative_id", "cluster_id", "score"],
            )

            # Optional: attach cluster_id to kept_df too (handy for later)
            kept_df["mmseqs2_cluster_id"] = kept_df[cols.id_col].astype(str).map(member_to_cluster)

            # extra stats (safe, does not change schema)
            stats: dict[str, Any] = {
                "n_total": len(work),
                "n_kept": len(kept_df),
                "n_removed": len(mapping),
                "n_clusters": len(reps),
                "reduction_ratio": float(len(kept_df) / len(work)) if len(work) else 0.0,
                "note": "Representative selection delegated to MMseqs2 easy-cluster.",
            }

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
                except Exception:
                    # best-effort debug copy
                    pass
            tmpdir_obj.cleanup()
