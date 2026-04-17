from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TaskMetadata, WorkflowTask
    from .examples import (
        PsychometricTask,
    )
    from .jax_task import JAXTask
    from .pymc_task import PyMCTask

__all__ = [
    "JAXTask",
    "GeneralizedExtremeValue",
    "PsychometricTask",
    "PyMCTask",
    "WorkflowTask",
]


def __getattr__(name: str):
    if name in {"TaskMetadata", "WorkflowTask"}:
        mod = import_module(".base", __name__)
        return getattr(mod, name)
    if name == "JAXTask":
        mod = import_module(".jax_task", __name__)
        return getattr(mod, name)
    if name == "PyMCTask":
        mod = import_module(".pymc_task", __name__)
        return getattr(mod, name)
    if name in {
        "GeneralizedExtremeValue",
        "PsychometricTask",
    }:
        mod = import_module(".examples", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
