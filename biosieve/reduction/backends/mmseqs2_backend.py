from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MMseqs2ClusterPaths:
    tmp_dir: Path
    fasta_path: Path
    cluster_tsv: Path
    rep_fasta: Path


def _which_mmseqs() -> str:
    exe = shutil.which("mmseqs")
    if not exe:
        raise FileNotFoundError(
            "mmseqs executable not found in PATH. Install MMseqs2 and ensure `mmseqs` is available."
        )
    return exe


def write_fasta(seqs: dict[str, str], fasta_path: Path) -> None:
    """Write dict of {id: sequence} as FASTA.
    IDs are used as FASTA headers, must be unique.
    """
    with fasta_path.open("w", encoding="utf-8") as f:
        for sid, seq in seqs.items():
            f.write(f">{sid}\n{seq}\n")


def run_mmseqs_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    threads: int = 4,
    extra_args: list[str] | None = None,
) -> MMseqs2ClusterPaths:
    """Runs:
      mmseqs easy-cluster <input.fasta> <out_prefix> <tmp_dir> --min-seq-id ... -c ... --cov-mode ... --cluster-mode ...
    Produces (among others):
      <out_prefix>_cluster.tsv
      <out_prefix>_rep_seq.fasta
    """
    mmseqs = _which_mmseqs()

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs,
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--cluster-mode",
        str(cluster_mode),
        "--threads",
        str(threads),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 failed.\nCommand: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n"
        )

    # MMseqs2 naming convention for easy-cluster
    cluster_tsv = Path(str(out_prefix) + "_cluster.tsv")
    rep_fasta = Path(str(out_prefix) + "_rep_seq.fasta")

    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected cluster TSV not found: {cluster_tsv}")
    if not rep_fasta.exists():
        raise FileNotFoundError(f"Expected representative FASTA not found: {rep_fasta}")

    return MMseqs2ClusterPaths(
        tmp_dir=tmp_dir,
        fasta_path=fasta_path,
        cluster_tsv=cluster_tsv,
        rep_fasta=rep_fasta,
    )


def parse_cluster_tsv(cluster_tsv: Path) -> dict[str, str]:
    """MMseqs2 *_cluster.tsv format (2 columns):
      representative_id <tab> member_id

    Returns: member_id -> representative_id
    """
    mapping: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            mapping[member] = rep
    return mapping


def build_cluster_ids(member_to_rep: dict[str, str]) -> dict[str, str]:
    """Assigns a stable cluster_id for each member based on representative id.
    cluster_id = "mmseqs2:<rep_id>"
    """
    return {member: f"mmseqs2:{rep}" for member, rep in member_to_rep.items()}
