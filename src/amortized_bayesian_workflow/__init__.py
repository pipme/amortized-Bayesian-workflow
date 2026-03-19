"""Public package API for the refactored Amortized Bayesian Workflow."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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
    "run_workflow",
    "retry_failed_datasets",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"ArtifactLayout", "WorkflowConfig"}:
        mod = import_module(".config", __name__)
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
