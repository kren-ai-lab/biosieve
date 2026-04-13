from __future__ import annotations

from pathlib import Path  # noqa: TC003
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import typer

from biosieve.cli.common import LOG_FILE_OPTION, LOG_LEVEL_OPTION, QUIET_OPTION, setup_runtime
from biosieve.core.split_runner import run_split
from biosieve.io.params import load_params, params_for_strategy
from biosieve.types import Columns

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry

INPUT_DATA_OPTION = typer.Option(
    ...,
    "--input-data",
    "-i",
    help="Input CSV path.",
)
OUTPUT_DIR_OPTION = typer.Option(
    ...,
    "--output-dir",
    "-o",
    help="Output directory for split CSVs.",
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
PARAMS_OPTION = typer.Option(
    None,
    "--params",
    help="YAML/JSON file with strategy parameters.",
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
    params: Path | None = PARAMS_OPTION,
    set_values: list[str] | None = SET_VALUES_OPTION,
    csv_separator: str = CSV_SEPARATOR_OPTION,
    encoding: str = ENCODING_OPTION,
    log_level: str = LOG_LEVEL_OPTION,
    quiet: bool = QUIET_OPTION,  # noqa: FBT001
    log_file: Path | None = LOG_FILE_OPTION,
) -> None:
    """Split a dataset into train/test(/val) using a selected strategy."""
    registry = setup_runtime(log_level, quiet=quiet, log_file=log_file)
    args = SimpleNamespace(
        in_path=str(input_data),
        outdir=str(output_dir),
        strategy=strategy,
        report_path=str(report_output) if report_output is not None else None,
        id_col=id_column,
        seq_col=sequence_column,
        params_path=str(params) if params is not None else None,
        overrides=list(set_values or []),
        sep=csv_separator,
        encoding=encoding,
    )
    _run_split(args, registry)


def _run_split(args: SimpleNamespace, registry: StrategyRegistry) -> None:
    cols = Columns(id_col=args.id_col, seq_col=args.seq_col)

    all_params = load_params(args.params_path, overrides=args.overrides)
    strat_params: dict[str, Any] = params_for_strategy(all_params, args.strategy)

    read_csv_kwargs = {"sep": args.sep, "encoding": args.encoding}

    run_split(
        in_path=args.in_path,
        outdir=args.outdir,
        strategy=args.strategy,
        registry=registry,
        cols=cols,
        report_path=args.report_path,
        strategy_params=strat_params,
        read_csv_kwargs=read_csv_kwargs,
    )
