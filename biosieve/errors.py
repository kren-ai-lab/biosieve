"""Exceptions shared by the CLI and the strategies."""

from __future__ import annotations

import typer


class ConfigError(typer.BadParameter):
    """Invalid configuration or usage; the CLI exits with code 2."""
