from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TaskMetadata, WorkflowTask
    from .examples import (
        BernoulliGLMTask,
        GEVObservationData,
        PsychometricTask,
        make_bernoulli_glm_task,
        make_gev_pymc_task,
        make_psychometric_task,
    )
    from .jax_task import JAXTask
    from .pymc_task import PyMCTask

__all__ = [
    "BernoulliGLMTask",
    "GEVObservationData",
    "JAXTask",
    "PsychometricTask",
    "PyMCTask",
    "TaskMetadata",
    "WorkflowTask",
    "make_bernoulli_glm_task",
    "make_gev_pymc_task",
    "make_psychometric_task",
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
        "BernoulliGLMTask",
        "GEVObservationData",
        "PsychometricTask",
        "make_bernoulli_glm_task",
        "make_gev_pymc_task",
        "make_psychometric_task",
    }:
        mod = import_module(".examples", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
