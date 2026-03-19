from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import ArtifactLayout, WorkflowConfig
from .report import WorkflowReport
from .workflow import WorkflowRunner


def run_workflow(
    *,
    task,
    approximator,
    observations: Sequence[np.ndarray],
    config: WorkflowConfig | None = None,
    fit: bool = False,
    layout: ArtifactLayout | None = None,
) -> WorkflowReport:
    """High-level notebook-friendly entry point.

    Users can provide a `JAXTask` or `PyMCTask`, a compatible amortizer
    (e.g. `BayesFlowAmortizedPosterior`), and a batch of observed datasets.
    """

    runner = WorkflowRunner(
        task=task,
        approximator=approximator,
        config=config or WorkflowConfig(),
        layout=layout,
    )
    if fit:
        runner.amortized_training()
    return runner.run(observations)


def retry_failed_datasets(
    *,
    runner: WorkflowRunner,
    report: WorkflowReport,
    observations: Sequence[np.ndarray],
    config_override: dict[str, Any] | None = None,
) -> WorkflowReport:
    """Convenience wrapper for notebook retry flows."""

    return runner.retry_failed(
        report,
        observations,
        config_override=config_override,
    )
