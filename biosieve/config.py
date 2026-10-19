"""Public config validation: check a params mapping or file without reading any data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping


@dataclass(frozen=True)
class ConfigIssue:
    """One problem found in a config: dotted ``path`` ("" is the file root) and ``message``."""

    path: str
    message: str


def validate_config(
    config: Mapping[str, object] | str | os.PathLike[str],
    *,
    strategy: str,
    kind: str = "reduce",
    set_values: list[str] | None = None,
) -> list[ConfigIssue]:
    """Validate a ``{strategy: {param: value}}`` mapping or YAML/JSON file; return every problem found."""
    # Imported here so `import biosieve` stays light.
    from biosieve.cli.common import load_strategy_params  # noqa: PLC0415
    from biosieve.core.strategies import build_registry  # noqa: PLC0415

    if kind not in {"reduce", "split"}:
        return [ConfigIssue("", f"unknown kind {kind!r}; expected one of: reduce, split")]
    _, issues = load_strategy_params(build_registry(), kind, strategy, config, set_values)
    return [ConfigIssue(p, m) for p, m in issues]
