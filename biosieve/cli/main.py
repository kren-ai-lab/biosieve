from __future__ import annotations

import click
import typer

from biosieve.cli.common import version_callback
from biosieve.cli.info import info
from biosieve.cli.reduce import reduce
from biosieve.cli.split import split
from biosieve.cli.validate import validate

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="biosieve",
    add_completion=False,
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="BioSieve: dataset splitting and redundancy reduction toolkit.",
)


@app.callback()
def root(
    _version: bool | None = typer.Option(
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


def main(argv: list[str] | None = None) -> int:
    try:
        typer.main.get_command(app).main(
            args=argv,
            prog_name="biosieve",
            standalone_mode=False,
        )
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except (click.Abort, KeyboardInterrupt):
        typer.echo("Interrupted.", err=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"ERROR: {exc}", err=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
