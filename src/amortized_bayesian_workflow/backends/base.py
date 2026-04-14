from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

LogProbFn = Callable[[Any], Any]


@dataclass(frozen=True)
class SamplerRequest:
    """Backend-agnostic sampling request."""

    log_prob_fn: LogProbFn
    iter_warmup: int
    iter_sampling: int
    seed: int = 0
    initial_positions: np.ndarray | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerResult:
    """Standardized MCMC backend output."""

    backend: str
    draws: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_converged(
        self,
        *,
        threshold: float = 1.01,
    ) -> bool:
        if "num_superchains" in self.metadata:
            metric = "nested_rhat"
        else:
            metric = "rhat"
        value = self.diagnostics.get(metric)
        value = float(np.max(value))
        return bool(np.isfinite(value) and value <= threshold)


class SamplerBackend(Protocol):
    name: str

    def run(self, request: SamplerRequest) -> SamplerResult: ...
