"""CLI command for environment diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys

import typer

import biosieve

MMSEQS2_BINARY_OPTION = typer.Option(
    "mmseqs",
    "--mmseqs2-binary",
    help="Name/path to mmseqs binary to check.",
    show_default=True,
)


def _check_binary(name: str, binary: str) -> tuple[bool, str]:
    path = shutil.which(binary)
    if path is None:
        return False, f"FAIL {name}: '{binary}' not found in PATH"
    try:
        result = subprocess.run(  # noqa: S603
            [binary, "version"], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception:  # noqa: BLE001
        return True, f"OK   {name}: {path}"
    else:
        version = (result.stdout or result.stderr).strip().split("\n")[0]
        return True, f"OK   {name}: {path} ({version})"


def _check_python_dep(import_name: str, dist_name: str | None = None) -> tuple[bool, str]:
    try:
        importlib.import_module(import_name)
    except ImportError:
        return False, f"MISS {import_name}: not installed"
    else:
        pkg = dist_name or import_name
        try:
            version = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return True, f"OK   {import_name}: installed (version unknown)"
        else:
            return True, f"OK   {import_name}: {version}"


def doctor(
    mmseqs2_binary: str = MMSEQS2_BINARY_OPTION,
) -> None:
    """Print environment info: versions, binary paths, optional dep availability."""
    typer.echo(f"biosieve {biosieve.__version__}")
    typer.echo(f"python   {sys.version.split()[0]}  ({sys.executable})")
    typer.echo("")

    errors = 0
    missing = 0

    def show(ok: bool, msg: str, *, optional: bool = False) -> None:  # noqa: FBT001
        nonlocal errors, missing
        typer.echo(msg)
        if not ok:
            if optional:
                missing += 1
            else:
                errors += 1

    typer.echo("── Core dependencies ────────────────────────────────")
    for import_name, dist_name in [
        ("polars", "polars"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("typer", "typer"),
        ("click", "click"),
    ]:
        ok, msg = _check_python_dep(import_name, dist_name)
        show(ok, msg)

    typer.echo("")
    typer.echo("── Optional dependencies ────────────────────────────")
    for import_name, dist_name in [
        ("faiss", "faiss-cpu"),
        ("datasketch", "datasketch"),
    ]:
        ok, msg = _check_python_dep(import_name, dist_name)
        show(ok, msg, optional=True)

    typer.echo("")
    typer.echo("── External binaries ────────────────────────────────")
    ok, msg = _check_binary("mmseqs2", mmseqs2_binary)
    show(ok, msg, optional=True)

    typer.echo("")
    if errors > 0:
        typer.echo(f"RESULT: {errors} required dep(s) missing — some strategies will fail.", err=True)
        raise SystemExit(1)
    if missing > 0:
        typer.echo(f"RESULT: OK (core). {missing} optional dep(s) missing — limited strategies available.")
    else:
        typer.echo("RESULT: OK — all deps present.")
