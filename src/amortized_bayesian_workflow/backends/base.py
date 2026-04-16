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

    @property
    def rhat_name(self) -> str:
        if "num_superchains" in self.metadata:
            return "nested_rhat"
        else:
            return "rhat"

    def is_converged(
        self,
        *,
        threshold: float = 1.01,
    ) -> bool:
        if self.rhat_name not in self.diagnostics:
            raise ValueError(f"R-hat diagnostic '{self.rhat_name}' not found.")
        value = self.diagnostics[self.rhat_name]
        value = float(np.max(value))
        return bool(np.isfinite(value) and value <= threshold)


class SamplerBackend(Protocol):
    name: str

    def run(self, request: SamplerRequest) -> SamplerResult: ...
