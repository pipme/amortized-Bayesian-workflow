from __future__ import annotations

import logging
from typing import Iterable


def resolve_log_level(level: str | int) -> int:
    """Resolve a string or numeric logging level to an integer."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        text = level.strip()
        if text.isdigit():
            return int(text)
        resolved = logging.getLevelName(text.upper())
        if isinstance(resolved, int):
            return resolved
        raise ValueError(
            "log level must be a valid logging name or integer. "
            "Examples: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
        )
    raise TypeError("log level must be a string or integer")


def configure_logging(
    level: str | int = logging.WARNING,
    *,
    include_external_loggers: bool = True,
    external_loggers: Iterable[str] = ("pymc.sampling.forward",),
) -> int:
    """Configure ABW package logging and optionally external noisy loggers.

    Returns the resolved integer level for convenience.
    """
    resolved = resolve_log_level(level)
    logging.getLogger("amortized_bayesian_workflow").setLevel(resolved)
    if include_external_loggers:
        for name in external_loggers:
            logging.getLogger(name).setLevel(resolved)
    return resolved
