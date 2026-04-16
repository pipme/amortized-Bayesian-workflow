from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class TaskMetadata:
    name: str
    parameter_names: tuple[str, ...] = ()
    parameter_dims: dict[str, int] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def total_parameter_dim(self) -> int:
        if self.parameter_dims:
            return int(sum(int(v) for v in self.parameter_dims.values()))
        return 0


class WorkflowTask(Protocol):
    metadata: TaskMetadata

    def simulate_prior_predictive(
        self,
        num_samples: int,
        *,
        seed: int,
    ) -> dict[str, np.ndarray]: ...

    def vectorized_log_posterior_fn(
        self,
        observation: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]: ...

    def single_log_posterior_fn(
        self,
        observation: np.ndarray,
    ) -> Callable[[Any], Any]: ...
