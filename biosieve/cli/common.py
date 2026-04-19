"""Shared CLI options and runtime setup helpers."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import typer

import biosieve
from biosieve.core.strategies import build_registry
from biosieve.io.params import load_params, params_for_strategy
from biosieve.types import Columns
from biosieve.utils.logging import configure_logging

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

LOG_LEVEL_OPTION = typer.Option(
    "INFO",
    "--log-level",
    help=f"Log level. One of: {', '.join(LOG_LEVELS)}.",
    show_default=True,
)
QUIET_OPTION = typer.Option(
    False,  # noqa: FBT003
    "--quiet/--no-quiet",
    help="Suppress console logs.",
    show_default=True,
)
LOG_FILE_OPTION = typer.Option(
    None,
    "--log-file",
    help="Optional file path to append logs.",
)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Show version and exit."""
    if value:
        typer.echo(f"biosieve {biosieve.__version__}")
        raise typer.Exit


def setup_runtime(log_level: str, *, quiet: bool, log_file: Path | None) -> StrategyRegistry:
    """Configure logging and build the strategy registry lazily per command."""
    configure_logging(
        level=log_level,
        quiet=quiet,
        log_file=str(log_file) if log_file is not None else None,
    )
    return build_registry()


def build_run_inputs(
    *,
    strategy: str,
    id_column: str,
    sequence_column: str,
    params_path: Path | None,
    set_values: list[str] | None,
    csv_separator: str,
    encoding: str,
) -> tuple[Columns, dict[str, object], dict[str, object]]:
    """Build the shared runner inputs used by the split and reduce CLI commands."""
    cols = Columns(id_col=id_column, seq_col=sequence_column)
    all_params = load_params(
        str(params_path) if params_path is not None else None, overrides=list(set_values or [])
    )
    strat_params = params_for_strategy(all_params, strategy)
    read_csv_kwargs: dict[str, object] = {"sep": csv_separator, "encoding": encoding}
    return cols, strat_params, read_csv_kwargs
