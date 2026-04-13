from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import typer

from biosieve.cli.common import LOG_FILE_OPTION, LOG_LEVEL_OPTION, QUIET_OPTION, setup_runtime
from biosieve.core.runner import run_reduce
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
OUTPUT_OPTION = typer.Option(
    ...,
    "--output",
    "-o",
    help="Output CSV path (non-redundant).",
)
STRATEGY_OPTION = typer.Option(
    ...,
    "--strategy",
    "-s",
    help="Reducer strategy name (e.g., exact, mmseqs2, embedding_cosine).",
)
MAPPING_OUTPUT_OPTION = typer.Option(
    None,
    "--mapping-output",
    help="CSV mapping path (removed_id -> representative_id).",
)
REPORT_OUTPUT_OPTION = typer.Option(
    None,
    "--report-output",
    help="JSON report path.",
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
    help="YAML/JSON file with strategy parameters. Format: {strategy_name: {param: value}}.",
)
SET_VALUES_OPTION = typer.Option(
    None,
    "--set",
    help="Override params. Example: --set embedding_cosine.threshold=0.97",
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


def reduce(
    input_data: Path = INPUT_DATA_OPTION,
    output: Path = OUTPUT_OPTION,
    strategy: str = STRATEGY_OPTION,
    mapping_output: Path | None = MAPPING_OUTPUT_OPTION,
    report_output: Path | None = REPORT_OUTPUT_OPTION,
    id_column: str = ID_COLUMN_OPTION,
    sequence_column: str = SEQUENCE_COLUMN_OPTION,
    params: Path | None = PARAMS_OPTION,
    set_values: list[str] | None = SET_VALUES_OPTION,
    csv_separator: str = CSV_SEPARATOR_OPTION,
    encoding: str = ENCODING_OPTION,
    log_level: str = LOG_LEVEL_OPTION,
    quiet: bool = QUIET_OPTION,
    log_file: Path | None = LOG_FILE_OPTION,
) -> None:
    """Reduce redundancy in a dataset using a selected strategy."""
    registry = setup_runtime(log_level, quiet=quiet, log_file=log_file)
    args = SimpleNamespace(
        in_path=str(input_data),
        out_path=str(output),
        strategy=strategy,
        map_path=str(mapping_output) if mapping_output is not None else None,
        report_path=str(report_output) if report_output is not None else None,
        id_col=id_column,
        seq_col=sequence_column,
        params_path=str(params) if params is not None else None,
        overrides=list(set_values or []),
        sep=csv_separator,
        encoding=encoding,
    )
    _run_reduce(args, registry)


def _run_reduce(args: SimpleNamespace, registry: StrategyRegistry) -> None:
    """Handler executed by main CLI.
    """
    cols = Columns(id_col=args.id_col, seq_col=args.seq_col)

    # Load and resolve params
    all_params = load_params(args.params_path, overrides=args.overrides)
    strat_params: dict[str, Any] = params_for_strategy(all_params, args.strategy)

    # Read CSV kwargs
    read_csv_kwargs = {
        "sep": args.sep,
        "encoding": args.encoding,
    }

    run_reduce(
        in_path=args.in_path,
        out_path=args.out_path,
        strategy=args.strategy,
        registry=registry,
        cols=cols,
        map_path=args.map_path,
        report_path=args.report_path,
        strategy_params=strat_params,
        read_csv_kwargs=read_csv_kwargs,
    )
