"""CLI end-to-end tests via subprocess."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import biosieve

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run(*args, cwd=None):
    """Run a biosieve CLI command via uv run; return CompletedProcess."""
    return subprocess.run(
        ["uv", "run", "biosieve", *args],
        capture_output=True,
        cwd=str(cwd or PROJECT_ROOT),
    )


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory):
    """Self-contained CSV fixture (module-scoped, no dependency on function-scoped df_base)."""
    import numpy as np

    rng = np.random.default_rng(42)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    ids = [f"pep_{i:03d}" for i in range(50)]
    seqs = ["".join(rng.choice(aa, size=int(rng.integers(20, 41)))) for _ in range(50)]
    df = pd.DataFrame({"id": ids, "sequence": seqs})
    p = tmp_path_factory.mktemp("cli") / "dataset.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Meta commands
# ---------------------------------------------------------------------------


def test_cli_version():
    r = _run("--version")
    assert r.returncode == 0
    assert biosieve.__version__.encode() in r.stdout


def test_cli_help():
    r = _run("--help")
    assert r.returncode == 0
    assert b"biosieve" in r.stdout.lower()


def test_cli_info_all():
    r = _run("info")
    assert r.returncode == 0
    assert b"exact" in r.stdout
    assert b"random" in r.stdout


def test_cli_info_reduce_only():
    r = _run("info", "--kind", "reduce")
    assert r.returncode == 0
    assert b"exact" in r.stdout


def test_cli_info_split_only():
    r = _run("info", "--kind", "split")
    assert r.returncode == 0
    assert b"random" in r.stdout


# ---------------------------------------------------------------------------
# biosieve reduce
# ---------------------------------------------------------------------------


def test_cli_reduce_exact(csv_file, tmp_path):
    out = tmp_path / "out.csv"
    r = _run("reduce", "--input-data", str(csv_file), "--output", str(out), "--strategy", "exact")
    assert r.returncode == 0, r.stderr.decode()
    assert out.exists()
    df_out = pd.read_csv(out)
    assert "id" in df_out.columns


def test_cli_reduce_with_map_and_report(csv_file, tmp_path):
    out = tmp_path / "out.csv"
    map_out = tmp_path / "map.csv"
    report_out = tmp_path / "report.json"
    r = _run(
        "reduce",
        "--input-data",
        str(csv_file),
        "--output",
        str(out),
        "--strategy",
        "exact",
        "--mapping-output",
        str(map_out),
        "--report-output",
        str(report_out),
    )
    assert r.returncode == 0, r.stderr.decode()
    assert map_out.exists()
    assert report_out.exists()
    report = json.loads(report_out.read_text())
    assert "stats" in report


def test_cli_reduce_unknown_strategy_exits_nonzero(csv_file, tmp_path):
    r = _run(
        "reduce",
        "--input-data",
        str(csv_file),
        "--output",
        str(tmp_path / "out.csv"),
        "--strategy",
        "NOPE",
    )
    assert r.returncode != 0


def test_cli_reduce_missing_input_exits_nonzero(tmp_path):
    r = _run(
        "reduce",
        "--input-data",
        str(tmp_path / "nonexistent.csv"),
        "--output",
        str(tmp_path / "out.csv"),
        "--strategy",
        "exact",
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# biosieve split
# ---------------------------------------------------------------------------


def test_cli_split_random(csv_file, tmp_path):
    outdir = tmp_path / "splits"
    r = _run("split", "--input-data", str(csv_file), "--output-dir", str(outdir), "--strategy", "random")
    assert r.returncode == 0, r.stderr.decode()
    assert (outdir / "train.csv").exists()
    assert (outdir / "test.csv").exists()


def test_cli_split_kfold(csv_file, tmp_path):
    outdir = tmp_path / "splits"
    r = _run(
        "split",
        "--input-data",
        str(csv_file),
        "--output-dir",
        str(outdir),
        "--strategy",
        "random_kfold",
        "--set",
        "random_kfold.n_splits=3",
    )
    assert r.returncode == 0, r.stderr.decode()
    assert (outdir / "fold_00" / "train.csv").exists()
    assert (outdir / "fold_01" / "train.csv").exists()


def test_cli_split_unknown_strategy_exits_nonzero(csv_file, tmp_path):
    r = _run(
        "split",
        "--input-data",
        str(csv_file),
        "--output-dir",
        str(tmp_path / "out"),
        "--strategy",
        "NOPE",
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# biosieve validate
# ---------------------------------------------------------------------------


def test_cli_validate_basic(csv_file):
    r = _run("validate", "--input-data", str(csv_file))
    # validate exits 0 if no hard errors found
    # (mmseqs2 check may be informational only)
    assert r.returncode in (0, 1)  # 1 only if mmseqs2 not found and it's a required check
    assert b"OK" in r.stdout or b"SKIP" in r.stdout


def test_cli_validate_missing_input_exits_nonzero(tmp_path):
    r = _run("validate", "--input-data", str(tmp_path / "nonexistent.csv"))
    assert r.returncode != 0
