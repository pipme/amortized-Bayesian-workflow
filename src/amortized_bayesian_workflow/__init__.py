"""Public package API for the refactored Amortized Bayesian Workflow."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING

from .logging_utils import configure_logging

_default_level = os.getenv("ABW_LOG_LEVEL", "WARNING")
_include_external = os.getenv(
    "ABW_LOG_INCLUDE_EXTERNAL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
try:
    configure_logging(
        _default_level,
        include_external_loggers=_include_external,
    )
except (TypeError, ValueError):
    # Fall back to WARNING if an invalid env var value is provided.
    configure_logging(logging.WARNING)

if TYPE_CHECKING:
    from .api import retry_failed_datasets, run_workflow
    from .config import ArtifactLayout, WorkflowConfig
    from .report import DatasetResult, WorkflowReport
    from .workflow import WorkflowRunner

__all__ = [
    "ArtifactLayout",
    "DatasetResult",
    "WorkflowConfig",
    "WorkflowReport",
    "WorkflowRunner",
    "configure_logging",
    "run_workflow",
    "retry_failed_datasets",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"ArtifactLayout", "WorkflowConfig"}:
        mod = import_module(".config", __name__)
        return getattr(mod, name)
    if name == "configure_logging":
        mod = import_module(".logging_utils", __name__)
        return getattr(mod, name)
    if name in {"DatasetResult", "WorkflowReport"}:
        mod = import_module(".report", __name__)
        return getattr(mod, name)
    if name == "WorkflowRunner":
        mod = import_module(".workflow", __name__)
        return getattr(mod, name)
    if name in {"run_workflow", "retry_failed_datasets"}:
        mod = import_module(".api", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
