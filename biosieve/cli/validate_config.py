"""CLI command that validates a params file without reading any data."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import typer

from biosieve.cli.common import (
    LOG_FILE_OPTION,
    LOG_LEVEL_OPTION,
    QUIET_OPTION,
    setup_runtime,
)
from biosieve.cli.reduce import SET_VALUES_OPTION, STRATEGY_OPTION
from biosieve.config import validate_config as validate_api

KIND_OPTION = typer.Option(
    "reduce",
    "--kind",
    help="Whether --strategy refers to a reducer or a splitter.",
    show_default=True,
)
JSON_OPTION = typer.Option(False, "--json", help="Print problems as a JSON array on stdout.")  # noqa: FBT003


def validate_config(
    config: Path = typer.Argument(..., help="YAML/JSON params file."),  # noqa: B008
    strategy: str = STRATEGY_OPTION,
    kind: str = KIND_OPTION,
    set_values: list[str] | None = SET_VALUES_OPTION,
    as_json: bool = JSON_OPTION,  # noqa: FBT001
    log_level: str = LOG_LEVEL_OPTION,
    quiet: bool = QUIET_OPTION,  # noqa: FBT001
    log_file: Path | None = LOG_FILE_OPTION,
) -> None:
    """Validate CONFIG as accepted by `biosieve reduce|split --config CONFIG --strategy NAME`."""
    if kind not in {"reduce", "split"}:
        msg = "kind must be one of: reduce, split"
        raise typer.BadParameter(msg)
    setup_runtime(log_level, quiet=quiet, log_file=log_file)
    issues = validate_api(config, strategy=strategy, kind=kind, set_values=set_values)
    if as_json:
        typer.echo(json.dumps([{"path": i.path, "message": i.message} for i in issues]))
    else:
        for i in issues:
            typer.echo(f"{i.path}: {i.message}", err=True)
    raise typer.Exit(2 if issues else 0)
