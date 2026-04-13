"""Integration tests for run_split() — real file I/O, full pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest



import json

import pandas as pd
import pytest

from biosieve.core.split_runner import run_split
from biosieve.core.strategies import build_registry

REGISTRY = build_registry()


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_split_random_writes_train_test(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    outdir = tmp_path / "splits"

    run_split(str(csv_in), str(outdir), "random", REGISTRY)

    assert (outdir / "train.csv").exists()
    assert (outdir / "test.csv").exists()
    train = pd.read_csv(outdir / "train.csv")
    test = pd.read_csv(outdir / "test.csv")
    assert len(train) + len(test) == len(df_base)
    assert set(train["id"]) & set(test["id"]) == set()


def test_split_random_no_val_by_default(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    outdir = tmp_path / "splits"
    run_split(str(csv_in), str(outdir), "random", REGISTRY)
    assert not (outdir / "val.csv").exists()


def test_split_writes_report_json(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    outdir = tmp_path / "splits"
    run_split(str(csv_in), str(outdir), "random", REGISTRY)

    report_path = outdir / "split_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "0.1"
    assert report["strategy"] == "random"


def test_split_kfold_writes_fold_dirs(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    outdir = tmp_path / "splits"

    run_split(str(csv_in), str(outdir), "random_kfold", REGISTRY, strategy_params={"n_splits": 3})

    for i in range(3):
        fold_dir = outdir / f"fold_{i:02d}"
        assert fold_dir.exists(), f"Missing {fold_dir}"
        assert (fold_dir / "train.csv").exists()
        assert (fold_dir / "test.csv").exists()


def test_split_kfold_writes_kfold_report(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    outdir = tmp_path / "splits"
    run_split(str(csv_in), str(outdir), "random_kfold", REGISTRY, strategy_params={"n_splits": 3})
    report = json.loads((outdir / "kfold_report.json").read_text())
    assert report["kfold"] is True
    assert report["n_folds"] == 3


def test_split_unknown_strategy_raises(df_base: pd.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    with pytest.raises(ValueError, match="Unknown split strategy"):
        run_split(str(csv_in), str(tmp_path / "out"), "nonexistent", REGISTRY)


def test_split_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        run_split(str(tmp_path / "nonexistent.csv"), str(tmp_path / "out"), "random", REGISTRY)
