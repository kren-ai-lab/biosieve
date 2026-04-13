from __future__ import annotations

from pathlib import Path

import typer

import biosieve
from biosieve.core.registry import StrategyRegistry
from biosieve.core.strategies import build_registry
from biosieve.utils.logging import configure_logging

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

LOG_LEVEL_OPTION = typer.Option(
    "INFO",
    "--log-level",
    help=f"Log level. One of: {', '.join(LOG_LEVELS)}.",
    show_default=True,
)
QUIET_OPTION = typer.Option(
    False,
    "--quiet/--no-quiet",
    help="Suppress console logs.",
    show_default=True,
)
LOG_FILE_OPTION = typer.Option(
    None,
    "--log-file",
    help="Optional file path to append logs.",
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo(f"biosieve {biosieve.__version__}")
        raise typer.Exit()


def setup_runtime(log_level: str, quiet: bool, log_file: Path | None) -> StrategyRegistry:
    """Configure logging and build the strategy registry lazily per command."""
    configure_logging(
        level=log_level,
        quiet=quiet,
        log_file=str(log_file) if log_file is not None else None,
    )
    return build_registry()
