from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from biosieve.splitting.base import SplitResult

# Reuse group split logic (sklearn GroupShuffleSplit)
from biosieve.splitting.group import _split_groups, _validate_sizes
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)

_INTERNAL_CLUSTER_COL = "_biosieve_cluster_id__"


def _write_fasta(df: pd.DataFrame, id_col: str, seq_col: str, out_fa: Path) -> None:
    """Write a FASTA file from a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    id_col:
        Column containing sequence/sample ids.
    seq_col:
        Column containing sequences.
    out_fa:
        Output FASTA path.

    Raises
    ------
    ValueError
        If any sequence is empty.

    """
    lines = []
    for _, row in df.iterrows():
        sid = str(row[id_col])
        seq = str(row[seq_col])
        if not seq:
            msg = f"Empty sequence for id={sid}"
            raise ValueError(msg)
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
    """Run mmseqs2 easy-cluster and return the produced cluster TSV path.

    Command
    -------
    mmseqs easy-cluster input.fasta out_prefix tmp_dir
        --min-seq-id <min_seq_id> -c <coverage> --cov-mode <cov_mode> --threads <threads>

    Parameters
    ----------
    fasta_path:
        Input FASTA file.
    out_prefix:
        Output prefix for mmseqs2.
    tmp_dir:
        Temporary directory for mmseqs2.
    mmseqs_bin:
        Path to mmseqs binary ("mmseqs" by default).
    min_seq_id:
        Minimum sequence identity threshold.
    coverage:
        Coverage threshold (-c).
    cov_mode:
        Coverage mode (--cov-mode).
    threads:
        Number of threads.

    Returns
    -------
    Path
        Path to the expected TSV file `{out_prefix}_cluster.tsv` containing `rep<TAB>member`.

    Raises
    ------
    FileNotFoundError
        If the mmseqs binary is not found or the expected output TSV is missing.
    RuntimeError
        If mmseqs2 returns a non-zero code.

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
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as e:
        msg = f"mmseqs2 binary not found ('{mmseqs_bin}'). Install mmseqs2 or use mode='precomputed'."
        raise FileNotFoundError(
            msg
        ) from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        msg_0 = f"mmseqs2 easy-cluster failed. Command: {' '.join(cmd)}\n{msg}"
        raise RuntimeError(msg_0) from e

    tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv.exists():
        msg_0 = f"Expected mmseqs2 output not found: {tsv}. Check mmseqs2 version/output naming."
        raise FileNotFoundError(
            msg_0
        )
    return tsv


def _load_mmseqs_cluster_tsv(cluster_tsv: Path) -> pd.DataFrame:
    """Load mmseqs2 easy-cluster mapping TSV.

    Expected format: rep<TAB>member

    Returns
    -------
    pd.DataFrame
        Columns:
        - representative_id
        - member_id
        - cluster_id (defaults to representative_id for stable deterministic naming)

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
) -> dict[str, str]:
    """Build a member->cluster_id dict from a mapping DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.

    """
    if member_col not in mapping_df.columns or cluster_col not in mapping_df.columns:
        msg = (
            f"mapping_df must contain '{member_col}' and '{cluster_col}'. "
            f"Found: {mapping_df.columns.tolist()}"
        )
        raise ValueError(
            msg
        )
    return dict(
        zip(
            mapping_df[member_col].astype(str),
            mapping_df[cluster_col].astype(str),
            strict=False,
        )
    )


@dataclass(frozen=True)
class HomologyAwareSplitter:
    """Homology-aware split using sequence clusters as groups (no homology leakage).

    This strategy clusters sequences by homology and then performs a group-based split
    over cluster ids, ensuring that no homology cluster appears in multiple splits.

    Modes
    -----
    1) mode="mmseqs2":
       Runs `mmseqs easy-cluster` on the input dataset and derives cluster ids from
       the output mapping (rep -> member).
    2) mode="precomputed":
       Uses a precomputed mapping file with `member_id -> cluster_id`.

    Missing mapping coverage
    ------------------------
    Any sample id not present in the mapping is assigned a safe singleton cluster:
    `singleton:<id>`, preventing accidental leakage.

    Parameters
    ----------
    test_size, val_size, seed:
        Split fractions and seed (group-based split is deterministic given seed).
    mode:
        "mmseqs2" or "precomputed".
    clusters_path:
        Path to precomputed clusters mapping (required if mode="precomputed").
    clusters_format:
        "mmseqs_tsv" (rep<TAB>member) or "csv".
    member_col, cluster_col:
        Column names in the precomputed mapping (csv mode).
    mmseqs_bin, min_seq_id, coverage, cov_mode, threads:
        mmseqs2 parameters (used only in mode="mmseqs2").
    work_dir:
        Directory for intermediate mmseqs2 artefacts (FASTA, tmp, TSV).
    keep_work:
        If False, removes `work_dir` after success (best-effort).

    Returns
    -------
    SplitResult
        train/test/val splits plus:
        - params: effective configuration (including mmseqs2 options)
        - stats: cluster counts, leakage checks, and mapping metadata

    Raises
    ------
    ValueError
        If split sizes are invalid, mode is unknown, required inputs are missing,
        or sequence column is missing in mmseqs2 mode.
    FileNotFoundError
        If clusters_path is missing (precomputed) or mmseqs2 binary/output is missing.
    RuntimeError
        If mmseqs2 fails.

    Notes
    -----
    - Leakage contract (must be zero):
      leak_clusters_train_test == 0 and leak_clusters_val_test == 0.
      If val exists, leak_clusters_train_val == 0 as well (val is group-split from trainval).
    - This prevents homology leakage, but does not enforce label balancing.
      For balanced yet leakage-aware splits, a hybrid cluster-level balancing strategy
      can be added later.

    Examples
    --------
    mmseqs2 mode:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_homology_mmseqs2 \\
    ...   --strategy homology_aware \\
    ...   --params params.yaml

    precomputed mode:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_homology_precomputed \\
    ...   --strategy homology_aware \\
    ...   --params params.yaml

    """

    # core sizes
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    # clustering mode
    mode: str = "mmseqs2"  # "mmseqs2" | "precomputed"

    # precomputed mapping
    clusters_path: str | None = None
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

    def _get_cluster_map(self, df: pd.DataFrame, cols: Columns) -> tuple[dict[str, str], dict[str, Any]]:
        work = Path(self.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        # PRECOMPUTED
        if self.mode == "precomputed":
            if not self.clusters_path:
                msg = "mode='precomputed' requires clusters_path."
                raise ValueError(msg)
            p = Path(self.clusters_path)
            if not p.exists():
                msg = f"clusters_path not found: {p}"
                raise FileNotFoundError(msg)

            if self.clusters_format == "mmseqs_tsv":
                mdf = _load_mmseqs_cluster_tsv(p)
                cmap = _build_cluster_id_map(mdf, member_col="member_id", cluster_col="cluster_id")
            elif self.clusters_format == "csv":
                mdf = pd.read_csv(p)
                cmap = _build_cluster_id_map(mdf, member_col=self.member_col, cluster_col=self.cluster_col)
            else:
                msg = "clusters_format must be 'mmseqs_tsv' or 'csv'"
                raise ValueError(msg)

            meta = {
                "mode": "precomputed",
                "clusters_path": str(p),
                "clusters_format": self.clusters_format,
                "member_col": self.member_col,
                "cluster_col": self.cluster_col,
                "n_mapped_members": len(cmap),
            }
            return cmap, meta

        # MMSEQS2
        if self.mode == "mmseqs2":
            if cols.seq_col not in df.columns:
                msg = (
                    f"Homology-aware split (mmseqs2 mode) requires sequence column '{cols.seq_col}'. "
                    f"Columns: {df.columns.tolist()}"
                )
                raise ValueError(
                    msg
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
                "n_mapped_members": len(cmap),
            }
            return cmap, meta

        msg = "mode must be 'mmseqs2' or 'precomputed'"
        raise ValueError(msg)

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)

        cmap, cmeta = self._get_cluster_map(work, cols)

        log.info("homology_aware:start | source=%s", "precomputed" if cmeta["mode"] else "mmseqs2")
        log.debug("homology_aware:params | %s", self.__dict__)

        # attach cluster_id; missing -> singleton
        ids = work[cols.id_col].astype(str)
        cluster_ids = []
        missing = 0
        for sid in ids:
            cid = cmap.get(str(sid))
            if cid is None:
                missing += 1
                cid = f"singleton:{sid}"
            cluster_ids.append(cid)

        work[_INTERNAL_CLUSTER_COL] = pd.Series(cluster_ids, index=work.index, dtype="string").astype(str)

        # group split using cluster ids
        groups = work[_INTERNAL_CLUSTER_COL].astype(str)

        # 1) split off test
        trainval, test = _split_groups(work, groups, test_size=self.test_size, seed=self.seed)

        val = None
        train = trainval

        # 2) optional val split from trainval (group-aware)
        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                msg = "Derived val fraction invalid. Check test_size/val_size."
                raise ValueError(msg)
            tv_groups = trainval[_INTERNAL_CLUSTER_COL].astype(str)
            train, val = _split_groups(trainval, tv_groups, test_size=frac, seed=self.seed)

        # leakage checks using internal cluster id column (correct, deterministic)
        train_c = set(train[_INTERNAL_CLUSTER_COL].astype(str).unique())
        test_c = set(test[_INTERNAL_CLUSTER_COL].astype(str).unique())
        val_c = set(val[_INTERNAL_CLUSTER_COL].astype(str).unique()) if val is not None else set()

        leak_tt = len(train_c & test_c)
        leak_tv = len(train_c & val_c) if val is not None else 0
        leak_vt = len(val_c & test_c) if val is not None else 0

        if leak_tt != 0 or leak_tv != 0 or leak_vt != 0:
            msg = (
                "Homology leakage detected (should not happen with group-based splitting). "
                f"leak_train_test={leak_tt}, leak_train_val={leak_tv}, leak_val_test={leak_vt}"
            )
            raise ValueError(
                msg
            )

        # cleanup
        train = train.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)
        test = test.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)
        if val is not None:
            val = val.drop(columns=[_INTERNAL_CLUSTER_COL]).reset_index(drop=True)

        stats: dict[str, Any] = {
            "n_total": len(df),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "n_clusters_total": len(set(cluster_ids)),
            "n_clusters_train": len(train_c),
            "n_clusters_test": len(test_c),
            "n_clusters_val": len(val_c) if val is not None else 0,
            "leak_clusters_train_test": 0,
            "leak_clusters_train_val": 0,
            "leak_clusters_val_test": 0,
            "n_missing_cluster_assignments": int(missing),
            "cluster_meta": cmeta,
        }

        log.info(
            "homology_aware:stats | train=%d | val=%d | test=%d",
            stats["n_train"],
            stats["n_val"],
            stats["n_test"],
        )

        # best-effort cleanup
        if self.mode == "mmseqs2" and not self.keep_work:
            import shutil

            shutil.rmtree(Path(self.work_dir), ignore_errors=True)

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
