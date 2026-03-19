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


class SamplerBackend(Protocol):
    name: str

    def run(self, request: SamplerRequest) -> SamplerResult: ...
