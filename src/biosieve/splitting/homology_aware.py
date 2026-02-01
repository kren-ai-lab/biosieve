from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult

# We reuse the same group split logic (sklearn GroupShuffleSplit)
from biosieve.splitting.group import _split_groups, _validate_sizes


def _write_fasta(df: pd.DataFrame, id_col: str, seq_col: str, out_fa: Path) -> None:
    lines = []
    for _, row in df.iterrows():
        sid = str(row[id_col])
        seq = str(row[seq_col])
        if not seq:
            raise ValueError(f"Empty sequence for id={sid}")
        lines.append(f">{sid}\n{seq}\n")
    out_fa.write_text("".join(lines), encoding="utf-8")


def _run_mmseqs_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    *,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    cov_mode: int = 0,
    threads: int = 8,
) -> Path:
    """
    Runs:
      mmseqs easy-cluster input.fasta out_prefix tmp_dir --min-seq-id ... -c ... --cov-mode ... --threads ...
    Output mapping file typically: out_prefix_cluster.tsv (rep \t member)
    """
    import subprocess

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs_bin,
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
        "--threads",
        str(threads),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"mmseqs2 binary not found ('{mmseqs_bin}'). Install mmseqs2 or provide precomputed clusters."
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"mmseqs2 easy-cluster failed. Command: {' '.join(cmd)}\n{msg}")

    tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv.exists():
        raise FileNotFoundError(
            f"Expected mmseqs2 output not found: {tsv}. "
            "Check mmseqs2 version/output naming."
        )
    return tsv


def _load_mmseqs_cluster_tsv(cluster_tsv: Path) -> pd.DataFrame:
    """
    mmseqs easy-cluster typically outputs: rep<TAB>member
    We'll return a DataFrame with columns: representative_id, member_id, cluster_id
    where cluster_id is representative_id by default (stable, deterministic).
    """
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative_id", "member_id"])
    df["representative_id"] = df["representative_id"].astype(str)
    df["member_id"] = df["member_id"].astype(str)
    df["cluster_id"] = df["representative_id"]
    return df


def _build_cluster_id_map(
    mapping_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    cluster_col: str = "cluster_id",
) -> Dict[str, str]:
    if member_col not in mapping_df.columns or cluster_col not in mapping_df.columns:
        raise ValueError(
            f"mapping_df must contain '{member_col}' and '{cluster_col}'. "
            f"Found: {mapping_df.columns.tolist()}"
        )
    return dict(zip(mapping_df[member_col].astype(str), mapping_df[cluster_col].astype(str)))


@dataclass(frozen=True)
class HomologyAwareSplitter:
    """
    Homology-aware split: clusters sequences by homology and uses cluster_id as group,
    ensuring no homologous cluster appears in multiple splits.

    Modes:
      - mode="precomputed": provide clusters_path with mapping member->cluster
      - mode="mmseqs2": run mmseqs2 easy-cluster on the input dataset sequences

    Outputs:
      - train/test(/val) CSVs via run_split
      - split_report includes cluster coverage and group leakage checks
    """

    # core sizes
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    # homology clustering mode
    mode: str = "mmseqs2"  # "mmseqs2" | "precomputed"

    # precomputed mapping
    clusters_path: Optional[str] = None
    clusters_format: str = "mmseqs_tsv"  # "mmseqs_tsv" | "csv"
    member_col: str = "member_id"
    cluster_col: str = "cluster_id"

    # mmseqs2 params
    mmseqs_bin: str = "mmseqs"
    min_seq_id: float = 0.9
    coverage: float = 0.8
    cov_mode: int = 0
    threads: int = 8

    # work dirs for mmseqs2 mode
    work_dir: str = "runs/mmseqs2_homology"
    keep_work: bool = False

    @property
    def strategy(self) -> str:
        return "homology_aware"

    def _get_cluster_map(self, df: pd.DataFrame, cols: Columns) -> Tuple[Dict[str, str], Dict[str, Any]]:
        work = Path(self.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        # PRECOMPUTED
        if self.mode == "precomputed":
            if not self.clusters_path:
                raise ValueError("mode='precomputed' requires clusters_path.")
            p = Path(self.clusters_path)
            if not p.exists():
                raise FileNotFoundError(f"clusters_path not found: {p}")

            if self.clusters_format == "mmseqs_tsv":
                mdf = _load_mmseqs_cluster_tsv(p)
            elif self.clusters_format == "csv":
                mdf = pd.read_csv(p)
            else:
                raise ValueError("clusters_format must be 'mmseqs_tsv' or 'csv'")

            cmap = _build_cluster_id_map(mdf, member_col=self.member_col, cluster_col=self.cluster_col)
            meta = {
                "mode": "precomputed",
                "clusters_path": str(p),
                "clusters_format": self.clusters_format,
                "member_col": self.member_col,
                "cluster_col": self.cluster_col,
                "n_mapped_members": int(len(cmap)),
            }
            return cmap, meta

        # MMSEQS2
        if self.mode == "mmseqs2":
            if cols.seq_col not in df.columns:
                raise ValueError(
                    f"Homology-aware split (mmseqs2 mode) requires sequence column '{cols.seq_col}'. "
                    f"Columns: {df.columns.tolist()}"
                )

            fasta = work / "input.fasta"
            _write_fasta(df, cols.id_col, cols.seq_col, fasta)

            out_prefix = work / "clusters"
            tmp_dir = work / "tmp"

            tsv = _run_mmseqs_easy_cluster(
                fasta,
                out_prefix,
                tmp_dir,
                mmseqs_bin=self.mmseqs_bin,
                min_seq_id=self.min_seq_id,
                coverage=self.coverage,
                cov_mode=self.cov_mode,
                threads=self.threads,
            )

            mdf = _load_mmseqs_cluster_tsv(tsv)
            cmap = _build_cluster_id_map(mdf, member_col="member_id", cluster_col="cluster_id")

            meta = {
                "mode": "mmseqs2",
                "mmseqs_bin": self.mmseqs_bin,
                "min_seq_id": self.min_seq_id,
                "coverage": self.coverage,
                "cov_mode": self.cov_mode,
                "threads": self.threads,
                "work_dir": str(work),
                "cluster_tsv": str(tsv),
                "n_mapped_members": int(len(cmap)),
            }
            return cmap, meta

        raise ValueError("mode must be 'mmseqs2' or 'precomputed'")

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)

        cmap, cmeta = self._get_cluster_map(work, cols)

        # attach cluster_id; for any missing, assign unique singleton cluster
        ids = work[cols.id_col].astype(str)
        cluster_ids = []
        missing = 0
        for sid in ids:
            cid = cmap.get(str(sid))
            if cid is None:
                missing += 1
                cid = f"singleton:{sid}"
            cluster_ids.append(cid)

        work["_biosieve_cluster_id__"] = cluster_ids

        # group split using cluster ids
        groups = work["_biosieve_cluster_id__"].astype(str)

        # 1) split off test
        trainval, test = _split_groups(work, groups, test_size=self.test_size, seed=self.seed)

        val = None
        train = trainval

        # 2) optional val split from trainval (group-aware)
        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            tv_groups = trainval["_biosieve_cluster_id__"].astype(str)
            train, val = _split_groups(trainval, tv_groups, test_size=frac, seed=self.seed)

        # cleanup
        train = train.drop(columns=["_biosieve_cluster_id__"]).reset_index(drop=True)
        test = test.drop(columns=["_biosieve_cluster_id__"]).reset_index(drop=True)
        if val is not None:
            val = val.drop(columns=["_biosieve_cluster_id__"]).reset_index(drop=True)

        # leakage checks on clusters
        def _cset(x: pd.DataFrame) -> set[str]:
            return set(x[cols.id_col].astype(str).map(lambda s: cmap.get(s, f"singleton:{s}")).tolist())

        train_c = _cset(train)
        test_c = _cset(test)
        val_c = _cset(val) if val is not None else set()

        stats: Dict[str, Any] = {
            "n_total": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "n_clusters_total": int(len(set(cluster_ids))),
            "n_clusters_train": int(len(train_c)),
            "n_clusters_test": int(len(test_c)),
            "n_clusters_val": int(len(val_c)) if val is not None else 0,
            "leak_clusters_train_test": int(len(train_c & test_c)),
            "leak_clusters_train_val": int(len(train_c & val_c)),
            "leak_clusters_val_test": int(len(val_c & test_c)),
            "n_missing_cluster_assignments": int(missing),
            "cluster_meta": cmeta,
        }

        # Respect keep_work flag (mmseqs2 mode)
        if self.mode == "mmseqs2" and not self.keep_work:
            # best-effort cleanup
            try:
                import shutil
                shutil.rmtree(Path(self.work_dir), ignore_errors=True)
            except Exception:
                pass

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "test_size": self.test_size,
                "val_size": self.val_size,
                "seed": self.seed,
                "mode": self.mode,
                "clusters_path": self.clusters_path,
                "clusters_format": self.clusters_format,
                "member_col": self.member_col,
                "cluster_col": self.cluster_col,
                "mmseqs_bin": self.mmseqs_bin,
                "min_seq_id": self.min_seq_id,
                "coverage": self.coverage,
                "cov_mode": self.cov_mode,
                "threads": self.threads,
                "work_dir": self.work_dir,
                "keep_work": self.keep_work,
            },
            stats=stats,
        )
