"""Integration tests for run_reduce() — real file I/O, full pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    import pytest


import json

import polars as pl
import pytest

from biosieve.core.runner import run_reduce
from biosieve.core.strategies import build_registry

REGISTRY = build_registry()


def _write_csv(df: pl.DataFrame, path: Path) -> Path:
    df.write_csv(path)
    return path


def test_reduce_exact_writes_output(df_base: pl.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    csv_out = tmp_path / "out.csv"

    run_reduce(str(csv_in), str(csv_out), "exact", REGISTRY)

    assert csv_out.exists()
    df_out = pl.read_csv(csv_out)
    assert "id" in df_out.columns
    assert df_out.height <= df_base.height


def test_reduce_writes_map_csv(df_base: pl.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    map_path = tmp_path / "map.csv"

    run_reduce(str(csv_in), str(tmp_path / "out.csv"), "exact", REGISTRY, map_path=str(map_path))

    assert map_path.exists()
    map_df = pl.read_csv(map_path)
    # Even if nothing was removed, the file must exist with correct schema
    assert "removed_id" in map_df.columns
    assert "representative_id" in map_df.columns


def test_reduce_writes_json_report(df_base: pl.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    report_path = tmp_path / "report.json"

    run_reduce(str(csv_in), str(tmp_path / "out.csv"), "exact", REGISTRY, report_path=str(report_path))

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "0.1"
    assert "summary" in report
    assert report["summary"]["n_in"] == df_base.height


def test_reduce_unknown_strategy_raises(df_base: pl.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    with pytest.raises(ValueError, match="Unknown reducer"):
        run_reduce(str(csv_in), str(tmp_path / "out.csv"), "nonexistent", REGISTRY)


def test_reduce_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_reduce(str(tmp_path / "nonexistent.csv"), str(tmp_path / "out.csv"), "exact", REGISTRY)


def test_reduce_creates_parent_dirs(df_base: pl.DataFrame, tmp_path: Path) -> None:
    csv_in = _write_csv(df_base, tmp_path / "in.csv")
    nested_out = tmp_path / "nested" / "subdir" / "out.csv"

    run_reduce(str(csv_in), str(nested_out), "exact", REGISTRY)

    assert nested_out.exists()
