from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class StrategySpec:
    """Lazy strategy specification.

    Attributes
    ----------
    name:
        Strategy name as used in CLI (e.g., "embedding_cosine").
    kind:
        "reducer" or "splitter".
    import_path:
        Import path in the form "module.submodule:ClassName".
    summary:
        Optional short description.

    """

    name: str
    kind: str  # "reducer" | "splitter"
    import_path: str
    summary: str | None = None


def lazy_import_class(import_path: str) -> type[Any]:
    """Import a class from an import path "pkg.mod:ClassName".

    Raises
    ------
    ValueError
        If import_path is malformed.
    ImportError
        If module or attribute cannot be imported.

    """
    if ":" not in import_path:
        msg = f"Invalid import_path '{import_path}'. Expected format 'module:ClassName'."
        raise ValueError(msg)
    mod_name, cls_name = import_path.split(":", 1)
    mod = import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls
