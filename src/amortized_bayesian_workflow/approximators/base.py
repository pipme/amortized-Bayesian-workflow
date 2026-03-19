from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class AmortizedDraws:
    samples: np.ndarray
    log_prob: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AmortizedPosterior(Protocol):
    def sample_and_log_prob(
        self,
        observation: np.ndarray,
        *,
        num_samples: int,
        seed: int,
    ) -> AmortizedDraws: ...

    def summary_statistics(self, observations: np.ndarray) -> np.ndarray: ...
