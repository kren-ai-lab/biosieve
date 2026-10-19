"""CLI command for dataset splitting runs."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer

from biosieve.cli.common import (
    CONFIG_OPTION,
    EMBEDDING_IDS_OPTION,
    EMBEDDINGS_OPTION,
    LOG_FILE_OPTION,
    LOG_LEVEL_OPTION,
    QUIET_OPTION,
    build_run_inputs,
    collect_paths,
    setup_runtime,
)
from biosieve.core.split_runner import run_split

INPUT_DATA_OPTION = typer.Option(
    ...,
    "--input-data",
    "-i",
    help="Input CSV or Parquet path.",
)
OUTPUT_DIR_OPTION = typer.Option(
    ...,
    "--output-dir",
    "-o",
    help="Output directory for the splits (same format as the input).",
)
STRATEGY_OPTION = typer.Option(
    ...,
    "--strategy",
    "-s",
    help="Split strategy name (e.g., random, stratified, cluster_aware).",
)
REPORT_OUTPUT_OPTION = typer.Option(
    None,
    "--report-output",
    help="Optional JSON report path.",
)
ID_COLUMN_OPTION = typer.Option(
    "id",
    "--id-column",
    help="Column name for unique sample ids.",
    show_default=True,
)
SEQUENCE_COLUMN_OPTION = typer.Option(
    "sequence",
    "--sequence-column",
    help="Column name for sequences.",
    show_default=True,
)
CLUSTERS_OPTION = typer.Option(
    None,
    "--clusters",
    help="Precomputed clusters file (homology_aware, mode=precomputed).",
)
SET_VALUES_OPTION = typer.Option(
    None,
    "--set",
    help="Override params. Example: --set random.seed=13",
)
CSV_SEPARATOR_OPTION = typer.Option(
    ",",
    "--csv-separator",
    help="CSV delimiter used to read input.",
    show_default=True,
)
ENCODING_OPTION = typer.Option(
    "utf-8",
    "--encoding",
    help="CSV encoding used to read input.",
    show_default=True,
)


def split(
    input_data: Path = INPUT_DATA_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    strategy: str = STRATEGY_OPTION,
    report_output: Path | None = REPORT_OUTPUT_OPTION,
    id_column: str = ID_COLUMN_OPTION,
    sequence_column: str = SEQUENCE_COLUMN_OPTION,
    config: Path | None = CONFIG_OPTION,
    embeddings: Path | None = EMBEDDINGS_OPTION,
    embedding_ids: Path | None = EMBEDDING_IDS_OPTION,
    clusters: Path | None = CLUSTERS_OPTION,
    set_values: list[str] | None = SET_VALUES_OPTION,
    csv_separator: str = CSV_SEPARATOR_OPTION,
    encoding: str = ENCODING_OPTION,
    log_level: str = LOG_LEVEL_OPTION,
    quiet: bool = QUIET_OPTION,  # noqa: FBT001
    log_file: Path | None = LOG_FILE_OPTION,
) -> None:
    """Split a dataset into train/test(/val) using a selected strategy."""
    registry = setup_runtime(log_level, quiet=quiet, log_file=log_file)
    cols, strat_params, read_csv_kwargs = build_run_inputs(
        registry=registry,
        kind="split",
        strategy=strategy,
        id_column=id_column,
        sequence_column=sequence_column,
        params_path=config,
        set_values=set_values,
        csv_separator=csv_separator,
        encoding=encoding,
    )
    run_split(
        in_path=str(input_data),
        outdir=str(output_dir),
        strategy=strategy,
        registry=registry,
        cols=cols,
        report_path=str(report_output) if report_output is not None else None,
        strategy_params=strat_params,
        paths=collect_paths(embeddings=embeddings, ids=embedding_ids, clusters=clusters),
        read_csv_kwargs=read_csv_kwargs,
    )
