from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class InitialPositionFilterResult:
    positions: np.ndarray
    log_prob: np.ndarray


def filter_initial_positions(
    initial_positions: np.ndarray,
    log_prob_fn: Callable[[np.ndarray], Any],
    *,
    unique: bool = True,
) -> InitialPositionFilterResult:
    """
    Filter initial positions by checking for finite log probabilities and optionally keeping only unique positions.
    """
    positions = np.asarray(initial_positions)
    if positions.ndim != 2:
        raise ValueError(
            "initial_positions must have shape (num_positions, dim)."
        )

    if unique:
        positions = np.unique(positions, axis=0)

    batch_exc: Exception | None = None
    try:
        evaluated = log_prob_fn(positions)
        log_prob = np.asarray(evaluated)
    except Exception as exc:
        batch_exc = exc
        log_prob = np.asarray([])

    # If log_prob_fn is scalar-style, vectorize only here where batch
    # evaluation is required for filtering.
    if (
        batch_exc is not None
        or log_prob.ndim == 0
        or (log_prob.ndim == 1 and log_prob.shape[0] != positions.shape[0])
    ):
        try:
            import jax

            log_prob = np.asarray(jax.vmap(log_prob_fn)(positions))
        except Exception as exc:  # pragma: no cover - fallback error path
            raise ValueError(
                "log_prob_fn must support batched input or be vmappable over single positions."
            ) from exc if batch_exc is None else batch_exc

    if log_prob.ndim != 1:
        raise ValueError("log_prob_fn must return a one-dimensional array.")
    if log_prob.shape[0] != positions.shape[0]:
        raise ValueError(
            "log_prob_fn must return one value per input position."
        )

    valid_mask = np.isfinite(log_prob)
    positions = positions[valid_mask]
    log_prob = log_prob[valid_mask]
    return InitialPositionFilterResult(
        positions=positions,
        log_prob=log_prob,
    )
