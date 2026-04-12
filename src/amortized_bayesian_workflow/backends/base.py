from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

LogProbFn = Callable[[np.ndarray], np.ndarray]
SingleLogProbFn = Callable[[Any], Any]


@dataclass(frozen=True)
class SamplerRequest:
    """Backend-agnostic sampling request."""

    log_prob_fn: LogProbFn
    num_warmup: int
    num_samples: int
    seed: int = 0
    initial_positions: np.ndarray | None = None
    single_log_prob_fn: SingleLogProbFn | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerResult:
    """Standardized MCMC backend output."""

    backend: str
    draws: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def convergence_value(self, metric: str = "rhat") -> float:
        value = self.diagnostics.get(metric)
        if value is None:
            return np.inf
        if isinstance(value, np.ndarray):
            return float(np.nanmax(value))
        return float(value)

    def is_converged(
        self,
        *,
        metric: str = "rhat",
        threshold: float = 1.1,
    ) -> bool:
        value = self.convergence_value(metric=metric)
        return bool(np.isfinite(value) and value <= threshold)


class SamplerBackend(Protocol):
    name: str

    def run(self, request: SamplerRequest) -> SamplerResult: ...
