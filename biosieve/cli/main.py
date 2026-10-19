"""Top-level BioSieve CLI application wiring."""

from __future__ import annotations

import traceback

import polars as pl
import typer

from biosieve.cli.common import version_callback
from biosieve.cli.doctor import doctor
from biosieve.cli.info import info
from biosieve.cli.reduce import reduce
from biosieve.cli.split import split
from biosieve.cli.validate import validate
from biosieve.cli.validate_config import validate_config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="biosieve",
    add_completion=False,
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="BioSieve: dataset splitting and redundancy reduction toolkit.",
)

# Exit codes (shared ecosystem CLI contract): 2 usage/config errors (Click handles them, ConfigError is a
# BadParameter), 3 runtime failures (listed here), 4 anything else (a bug, with traceback).
RUNTIME_ERRORS = (OSError, ValueError, RuntimeError, ImportError, pl.exceptions.PolarsError)


@app.callback()
def root(
    _version: bool | None = typer.Option(  # noqa: FBT001
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """BioSieve CLI root."""


app.command("reduce", context_settings=CONTEXT_SETTINGS)(reduce)
app.command("split", context_settings=CONTEXT_SETTINGS)(split)
app.command("info", context_settings=CONTEXT_SETTINGS)(info)
app.command("validate", context_settings=CONTEXT_SETTINGS)(validate)
app.command("validate-config", context_settings=CONTEXT_SETTINGS)(validate_config)
app.command("doctor", context_settings=CONTEXT_SETTINGS)(doctor)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI application and return a process exit code."""
    try:
        # Typer handles usage errors, aborts and interrupts itself, then calls
        # sys.exit(); we translate that into a return code for the entry point.
        typer.main.get_command(app).main(args=argv, prog_name="biosieve")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except RUNTIME_ERRORS as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        return 3
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
