from __future__ import annotations

import logging
import sys
from pathlib import Path

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_logging(
    level: str = "INFO",
    quiet: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure global logging for BioSieve.

    Parameters
    ----------
    level:
        Logging level name. One of {"DEBUG","INFO","WARNING","ERROR"}.
    quiet:
        If True, suppress console logs (still logs to file if log_file is set).
    log_file:
        Optional file path to append logs.

    Notes
    -----
    - Safe to call multiple times; it replaces handlers on the root logger.
    - We configure the root logger so module loggers propagate by default.

    """
    lvl = _LEVELS.get(level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Remove existing handlers to avoid duplicates in notebooks/CLI reruns
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not quiet:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, mode="a", encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Get a module logger (propagates to root configured by configure_logging).
    """
    return logging.getLogger(name)
