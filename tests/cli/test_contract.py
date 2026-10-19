"""Shared ecosystem CLI contract: config checks, exit codes, record identity, formats and provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from jsonschema import Draft202012Validator

from biosieve.cli.main import main

if TYPE_CHECKING:
    import pytest

SCHEMAS = Path(__file__).parent.parent / "contract"


def _schema(name: str) -> Any:  # noqa: ANN401
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


def test_reduce_embeddings_flags_write_valid_provenance(
    df_base: pl.DataFrame, embeddings_files: tuple[Path, Path], tmp_path: Path
) -> None:
    emb, ids = embeddings_files
    data = tmp_path / "in.csv"
    df_base.select("sequence", "id").write_csv(data)
    out = tmp_path / "out.csv"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"embedding_cosine": {"threshold": 0.99, "use_faiss": False}}))

    argv = ["reduce", "-i", str(data), "-o", str(out), "-s", "embedding_cosine", "--config", str(cfg)]
    code = main([*argv, "--embeddings", str(emb), "--embedding-ids", str(ids), "--quiet"])

    assert code == 0
    assert pl.read_csv(out).columns[0] == "id"
    prov = json.loads((tmp_path / "out.csv.provenance.json").read_text())
    _schema("provenance.schema.json").validate(prov)
    assert prov["parameters"]["params"]["threshold"] == 0.99
    assert "embeddings_path" not in prov["parameters"]["params"]
    assert len(prov["inputs"]) == 3

    # without --embedding-ids the sibling <stem>.ids.txt is read, so provenance records it too
    emb.with_suffix(".ids.txt").write_text("\n".join(pl.read_csv(ids)["id"].cast(pl.String)) + "\n")
    assert main([*argv, "--embeddings", str(emb), "--quiet"]) == 0
    prov = json.loads((tmp_path / "out.csv.provenance.json").read_text())
    assert prov["inputs"][-1]["path"].endswith("embeddings.ids.txt")


def test_split_parquet_in_parquet_out_with_sibling_provenance(df_base: pl.DataFrame, tmp_path: Path) -> None:
    data = tmp_path / "in.parquet"
    df_base.write_parquet(data)
    outdir = tmp_path / "splits"

    assert main(["split", "-i", str(data), "-o", str(outdir), "-s", "random", "--quiet"]) == 0

    assert pl.read_parquet(outdir / "train.parquet").columns[0] == "id"
    prov = json.loads((tmp_path / "splits.provenance.json").read_text())
    _schema("provenance.schema.json").validate(prov)
    assert prov["command"] == "split"


def test_validate_config_reports_all_problems_as_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "embedding_cosine:\n  threshold: high\n  embeddings_path: e.npy\nrandom:\n  seed: 1\n",
        encoding="utf-8",
    )

    code = main(["validate-config", str(cfg), "-s", "embedding_cosine", "--json", "--quiet"])

    assert code == 2
    problems = json.loads(capsys.readouterr().out)
    _schema("validate-errors.schema.json").validate(problems)
    assert {p["path"] for p in problems} == {
        "embedding_cosine.threshold",
        "embedding_cosine.embeddings_path",
        "random",
    }


def test_validate_config_valid_exits_0(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("random:\n  seed: 1\n", encoding="utf-8")

    assert main(["validate-config", str(cfg), "-s", "random", "--kind", "split", "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_exit_codes_config_runtime_and_missing_id(df_base: pl.DataFrame, tmp_path: Path) -> None:
    data = tmp_path / "in.csv"
    df_base.write_csv(data)
    out = tmp_path / "out.csv"
    base = ["reduce", "-i", str(data), "-o", str(out), "--quiet"]

    assert main([*base, "-s", "no_such_strategy"]) == 2
    assert main([*base, "-s", "embedding_cosine"]) == 2  # --embeddings is required
    assert main([*base, "-s", "exact", "--id-column", "nope"]) == 3
    assert (
        main(["reduce", "-i", str(tmp_path / "missing.csv"), "-o", str(out), "-s", "exact", "--quiet"]) == 3
    )


def test_npz_embeddings_carry_ids_and_duplicates_fail(
    df_base: pl.DataFrame, embeddings_fixture: tuple[np.ndarray, list[str]], tmp_path: Path
) -> None:
    X, ids = embeddings_fixture
    data = tmp_path / "in.csv"
    df_base.write_csv(data)
    argv = ["reduce", "-i", str(data), "-o", str(tmp_path / "o.csv"), "-s", "embedding_cosine", "--quiet"]

    good = tmp_path / "good.npz"
    np.savez(good, values=X, ids=np.array(ids))
    assert main([*argv, "--embeddings", str(good), "--set", "embedding_cosine.use_faiss=false"]) == 0

    dup = tmp_path / "dup.npz"
    np.savez(dup, values=X, ids=np.array([ids[0]] * len(ids)))
    assert main([*argv, "--embeddings", str(dup)]) == 3


def test_data_validate_failure_exits_2(df_base: pl.DataFrame, tmp_path: Path) -> None:
    data = tmp_path / "in.csv"
    df_base.write_csv(data)

    assert main(["validate-inputs", "-i", str(data), "--columns", '{"id_col": "nope"}', "--quiet"]) == 2


def test_public_validate_config_mapping_and_path(tmp_path: Path) -> None:
    from types import MappingProxyType

    import biosieve

    assert biosieve.validate_config({"random": {"seed": 1}}, strategy="random", kind="split") == []
    bad = MappingProxyType({"embedding_cosine": {"threshold": "high"}})
    issues = biosieve.validate_config(bad, strategy="embedding_cosine")
    assert [i.path for i in issues] == ["embedding_cosine.threshold"]
    assert isinstance(issues[0], biosieve.ConfigIssue)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("random:\n  seed: 1\n", encoding="utf-8")
    assert biosieve.validate_config(cfg, strategy="random", kind="split") == []
    assert biosieve.validate_config(str(cfg), strategy="random", kind="split") == []

    missing = biosieve.validate_config(tmp_path / "nope.yaml", strategy="random", kind="split")
    assert [i.path for i in missing] == [""]
    cfg.write_text("random: [unclosed\n", encoding="utf-8")
    assert [i.path for i in biosieve.validate_config(cfg, strategy="random", kind="split")] == [""]
    assert [i.path for i in biosieve.validate_config({}, strategy="nope")] == [""]
    assert [i.path for i in biosieve.validate_config({}, strategy="random", kind="nope")] == [""]


def test_missing_data_inputs_exit_3(
    df_base: pl.DataFrame, embeddings_files: tuple[Path, Path], tmp_path: Path
) -> None:
    emb, _ = embeddings_files
    data = tmp_path / "in.csv"
    df_base.write_csv(data)
    gone = str(tmp_path / "gone")
    out = str(tmp_path / "out.csv")
    reduce = ["reduce", "-i", str(data), "-o", out, "--quiet"]
    assert main([*reduce, "-s", "embedding_cosine", "--embeddings", f"{gone}.npy"]) == 3
    assert (
        main([*reduce, "-s", "embedding_cosine", "--embeddings", str(emb), "--embedding-ids", f"{gone}.csv"])
        == 3
    )
    assert main([*reduce, "-s", "structural_distance", "--edges", f"{gone}.csv"]) == 3
    split = ["split", "-i", str(data), "-o", str(tmp_path / "o"), "--quiet"]
    assert (
        main(
            [
                *split,
                "-s",
                "homology_aware",
                "--clusters",
                f"{gone}.csv",
                "--set",
                'homology_aware.mode="precomputed"',
            ]
        )
        == 3
    )
