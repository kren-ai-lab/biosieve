"""Shared CLI options and runtime setup helpers."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import typer

import biosieve
from biosieve.core.strategies import build_registry
from biosieve.errors import ConfigError
from biosieve.io.params import Issue, check_params, read_config
from biosieve.types import Columns
from biosieve.utils.logging import configure_logging

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

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


EMBEDDINGS_OPTION = typer.Option(
    None,
    "--embeddings",
    help="Embeddings array (.npy or .npz with 'values' and 'ids').",
)
EMBEDDING_IDS_OPTION = typer.Option(
    None,
    "--embedding-ids",
    help="Embedding ids: <stem>.ids.txt (one id per line), .npz with 'ids', or CSV with an 'id' column. "
    "Defaults to the .npz's own ids or the sibling <stem>.ids.txt.",
)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "--params",
    help="YAML/JSON file with strategy parameters. Format: {strategy_name: {param: value}}.",
)


def collect_paths(**flags: Path | None) -> dict[str, str]:
    """Map the given input-path flags to the strategy field names that receive them."""
    return {f"{k}_path": str(v) for k, v in flags.items() if v is not None}


def load_strategy_params(
    registry: StrategyRegistry,
    kind: str,
    strategy: str,
    config: str | os.PathLike[str] | Mapping[str, object] | None,
    set_values: list[str] | None,
) -> tuple[dict[str, object], list[Issue]]:
    """Read and check the params for one strategy, collecting every problem."""
    raw, issues = read_config(config, list(set_values or []))
    try:
        cls = (
            registry.get_reducer_class(strategy)
            if kind == "reduce"
            else registry.get_splitter_class(strategy)
        )
    except KeyError as exc:
        return {}, [*issues, ("", str(exc.args[0]))]
    if not issues:
        issues = check_params(raw, strategy, cls)
    block = raw.get(strategy)
    return (cast("dict[str, object]", block) if isinstance(block, dict) else {}), issues


def build_run_inputs(
    *,
    registry: StrategyRegistry,
    kind: str,
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
    strat_params, issues = load_strategy_params(
        registry, kind, strategy, str(params_path) if params_path is not None else None, set_values
    )
    if issues:
        raise ConfigError("\n".join(f"{path or 'config'}: {msg}" for path, msg in issues))
    read_csv_kwargs: dict[str, object] = {"sep": csv_separator, "encoding": encoding}
    return cols, strat_params, read_csv_kwargs
