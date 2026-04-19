# ruff: noqa: ANN401, D102, EM101, EM102, TRY003

"""Homology-aware splitting strategy using sequence clustering as grouping."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import derive_val_fraction, validate_sizes
from biosieve.splitting.group import _split_groups

_INTERNAL_CLUSTER_COL = "_biosieve_cluster_id__"


def _write_fasta(df: pl.DataFrame, id_col: str, seq_col: str, out_fa: Path) -> None:
    lines: list[str] = []
    for row in df.select([id_col, seq_col]).iter_rows(named=True):
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
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> Path:
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as e:
        raise FileNotFoundError(f"mmseqs2 binary not found ('{mmseqs_bin}').") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError((e.stderr or e.stdout or "").strip()) from e
    tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv.exists():
        raise FileNotFoundError(f"Expected mmseqs2 output not found: {tsv}")
    return tsv


def _load_mmseqs_cluster_tsv(cluster_tsv: Path) -> pl.DataFrame:
    return pl.read_csv(
        cluster_tsv,
        separator="\t",
        has_header=False,
        new_columns=["representative_id", "member_id"],
    ).with_columns(cluster_id=pl.col("representative_id"))


def _build_cluster_id_map(mapping_df: pl.DataFrame, *, member_col: str, cluster_col: str) -> dict[str, str]:
    if member_col not in mapping_df.columns or cluster_col not in mapping_df.columns:
        raise ValueError(f"mapping_df must contain '{member_col}' and '{cluster_col}'.")
    return dict(
        zip(
            mapping_df[member_col].cast(pl.String).to_list(),
            mapping_df[cluster_col].cast(pl.String).to_list(),
            strict=False,
        )
    )


@dataclass(frozen=True)
class HomologyAwareSplitter:
    """Homology-aware split using sequence clusters as groups."""

    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13
    mode: str = "mmseqs2"
    clusters_path: str | None = None
    clusters_format: str = "mmseqs_tsv"
    member_col: str = "member_id"
    cluster_col: str = "cluster_id"
    mmseqs_bin: str = "mmseqs"
    min_seq_id: float = 0.9
    coverage: float = 0.8
    cov_mode: int = 0
    threads: int = 8
    work_dir: str = "runs/mmseqs2_homology"
    keep_work: bool = False

    @property
    def strategy(self) -> str:
        return "homology_aware"

    def _get_cluster_map(self, df: pl.DataFrame, cols: Any) -> tuple[dict[str, str], dict[str, Any]]:
        work = Path(self.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        if self.mode == "precomputed":
            if not self.clusters_path:
                raise ValueError("mode='precomputed' requires clusters_path.")
            p = Path(self.clusters_path)
            if not p.exists():
                raise FileNotFoundError(f"clusters_path not found: {p}")
            if self.clusters_format == "mmseqs_tsv":
                mdf = _load_mmseqs_cluster_tsv(p)
                cmap = _build_cluster_id_map(mdf, member_col="member_id", cluster_col="cluster_id")
            else:
                mdf = pl.read_csv(p)
                cmap = _build_cluster_id_map(mdf, member_col=self.member_col, cluster_col=self.cluster_col)
            return cmap, {"mode": "precomputed", "clusters_path": str(p)}

        if self.mode == "mmseqs2":
            if cols.seq_col not in df.columns:
                raise ValueError(f"Homology-aware split requires sequence column '{cols.seq_col}'.")
            fasta = work / "input.fasta"
            _write_fasta(df, cols.id_col, cols.seq_col, fasta)
            tsv = _run_mmseqs_easy_cluster(
                fasta,
                work / "clusters",
                work / "tmp",
                mmseqs_bin=self.mmseqs_bin,
                min_seq_id=self.min_seq_id,
                coverage=self.coverage,
                cov_mode=self.cov_mode,
                threads=self.threads,
            )
            mdf = _load_mmseqs_cluster_tsv(tsv)
            return _build_cluster_id_map(mdf, member_col="member_id", cluster_col="cluster_id"), {
                "mode": "mmseqs2"
            }

        raise ValueError("mode must be 'mmseqs2' or 'precomputed'")

    def run(self, df: pl.DataFrame, cols: Any) -> SplitResult:
        validate_sizes(self.test_size, self.val_size)
        work = df.clone()
        cmap, meta = self._get_cluster_map(work, cols)
        ids = work[cols.id_col].cast(pl.String).to_list()
        cluster_ids = [cmap.get(sid, f"singleton:{sid}") for sid in ids]
        work = work.with_columns(pl.Series(_INTERNAL_CLUSTER_COL, cluster_ids))

        groups = work[_INTERNAL_CLUSTER_COL].cast(pl.String)
        trainval, test = _split_groups(work, groups, test_size=self.test_size, seed=self.seed)
        train = trainval
        val = None
        if self.val_size > 0:
            frac = derive_val_fraction(self.test_size, self.val_size)
            train, val = _split_groups(
                trainval, trainval[_INTERNAL_CLUSTER_COL].cast(pl.String), test_size=frac, seed=self.seed
            )

        train_c = set(train[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list())
        test_c = set(test[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list())
        val_c = (
            set(val[_INTERNAL_CLUSTER_COL].cast(pl.String).unique().to_list()) if val is not None else set()
        )

        if self.keep_work is False and self.mode == "mmseqs2":
            shutil.rmtree(self.work_dir, ignore_errors=True)

        train = train.drop([_INTERNAL_CLUSTER_COL])
        test = test.drop([_INTERNAL_CLUSTER_COL])
        if val is not None:
            val = val.drop([_INTERNAL_CLUSTER_COL])

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={"mode": self.mode, **meta},
            stats={
                "n_total": df.height,
                "n_train": train.height,
                "n_test": test.height,
                "n_val": val.height if val is not None else 0,
                "n_clusters_total": len(set(cluster_ids)),
                "leak_clusters_train_test": len(train_c & test_c),
                "leak_clusters_train_val": len(train_c & val_c) if val is not None else 0,
                "leak_clusters_val_test": len(val_c & test_c) if val is not None else 0,
            },
        )
