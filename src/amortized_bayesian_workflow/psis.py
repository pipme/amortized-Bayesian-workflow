from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PSISResult:
    pareto_k: float
    ess: float
    smoothed_normalized_weights: np.ndarray
    smoothed_log_weights: np.ndarray
    log_weights: np.ndarray
    num_proposal_samples: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def needs_mcmc(self) -> bool:
        if self.num_proposal_samples >= 2000:
            threshold = 0.7
        else:
            threshold = min(1 - 1 / np.log10(self.num_proposal_samples), 0.7)
        return (not np.isfinite(self.pareto_k)) or (self.pareto_k > threshold)


def _normalize_log_weights(logw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shift = np.max(logw)
    stable = logw - shift
    w = np.exp(stable)
    total = np.sum(w)
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid importance weights.")
    return stable - np.log(total), w / total


def compute_psis(
    *,
    log_target: np.ndarray,
    log_proposal: np.ndarray,
) -> PSISResult:
    log_target = np.asarray(log_target, dtype=float)
    log_proposal = np.asarray(log_proposal, dtype=float)
    if log_target.shape != log_proposal.shape:
        raise ValueError(
            "log_target and log_proposal must have the same shape."
        )
    if log_target.ndim != 1:
        raise ValueError("PSIS expects 1D arrays for a single dataset.")

    raw_logw = log_target - log_proposal
    try:
        arviz_stats = import_module("arviz.stats")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PSIS requires ArviZ. Install the `pymc` extra or add `arviz`."
        ) from exc

    smoothed_logw, pareto_k = arviz_stats.psislw(raw_logw)
    smoothed_logw = np.asarray(smoothed_logw, dtype=float)
    _, smoothed_normalized_weights = _normalize_log_weights(smoothed_logw)
    ess = float(1.0 / np.sum(smoothed_normalized_weights**2))
    return PSISResult(
        pareto_k=float(np.asarray(pareto_k)),
        ess=ess,
        smoothed_normalized_weights=smoothed_normalized_weights,
        smoothed_log_weights=smoothed_logw,
        log_weights=np.asarray(raw_logw, dtype=float),
        num_proposal_samples=int(log_target.shape[0]),
        metadata={"num_samples": int(log_target.shape[0])},
    )


def resample_with_weights(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    num_draws: int,
    seed: int | None = None,
    replace=True,
) -> np.ndarray:
    samples = np.asarray(samples)
    weights = np.asarray(weights, dtype=float)
    if samples.shape[0] != weights.shape[0]:
        raise ValueError("weights length must match number of samples.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(
        samples.shape[0],
        size=int(num_draws),
        replace=replace,
        p=weights,
    )
    return samples[idx]
