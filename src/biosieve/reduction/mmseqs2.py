from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns
from biosieve.reduction.backends.mmseqs2_backend import (
    write_fasta,
    run_mmseqs_easy_cluster,
    parse_cluster_tsv,
    build_cluster_ids,
)


@dataclass(frozen=True)
class MMseqs2Reducer:
    """
    Homology-based redundancy reduction using MMseqs2 easy-cluster.

    Output mapping contains:
      removed_id, representative_id, cluster_id

    Notes:
    - Representative selection is handled by MMseqs2.
    - No per-pair identity score is produced by default in this mode.
    """

    min_seq_id: float = 0.9
    coverage: float = 0.8
    cov_mode: int = 0
    cluster_mode: int = 0
    threads: int = 4

    # housekeeping
    tmp_root: Optional[str] = None   # if None, uses system temp
    keep_tmp: bool = False

    @property
    def strategy(self) -> str:
        return "mmseqs2"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        # Basic checks
        if not (0.0 <= self.min_seq_id <= 1.0):
            raise ValueError("min_seq_id must be in [0,1]")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0,1]")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        # Build sequence dict (id -> sequence)
        seqs: Dict[str, str] = {}
        for _, row in work.iterrows():
            sid = str(row[cols.id_col])
            seq = str(row[cols.seq_col])
            if sid in seqs:
                raise ValueError(f"Duplicate ids detected: {sid}. IDs must be unique for MMseqs2 FASTA.")
            seqs[sid] = seq

        tmp_base = self.tmp_root if self.tmp_root is not None else None
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
            mapping = pd.DataFrame(removed_rows, columns=["removed_id", "representative_id", "cluster_id", "score"])

            # Optional: attach cluster_id to kept_df too (handy for later)
            kept_df["mmseqs2_cluster_id"] = kept_df[cols.id_col].astype(str).map(member_to_cluster)

            return ReductionResult(
                df=kept_df,
                mapping=mapping,
                strategy=self.strategy,
                params={
                    "min_seq_id": self.min_seq_id,
                    "coverage": self.coverage,
                    "cov_mode": self.cov_mode,
                    "cluster_mode": self.cluster_mode,
                    "threads": self.threads,
                    "keep_tmp": self.keep_tmp,
                },
            )

        finally:
            if self.keep_tmp:
                # persist temp directory by not cleaning it
                # (workaround: copy it out then cleanup; simplest: do nothing and leak path? no.)
                # We'll copy to cwd/biosieve_mmseqs2_debug
                debug_dir = Path("biosieve_mmseqs2_debug")
                if debug_dir.exists():
                    shutil.rmtree(debug_dir)
                shutil.copytree(tmpdir, debug_dir)
            tmpdir_obj.cleanup()
