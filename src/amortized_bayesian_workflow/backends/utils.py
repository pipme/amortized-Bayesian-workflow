from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class InitialPositionFilterResult:
    positions: np.ndarray
    log_prob: np.ndarray
    valid_mask: np.ndarray


def filter_initial_positions(
    initial_positions: np.ndarray,
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    *,
    unique: bool = True,
    sort_descending: bool = False,
) -> InitialPositionFilterResult:
    positions = np.asarray(initial_positions)
    if positions.ndim != 2:
        raise ValueError("initial_positions must have shape (num_positions, dim).")

    if unique:
        positions = np.unique(positions, axis=0)

    log_prob = np.asarray(log_prob_fn(positions))
    if log_prob.shape[0] != positions.shape[0]:
        raise ValueError("log_prob_fn must return one value per input position.")

    valid_mask = np.isfinite(log_prob)
    positions = positions[valid_mask]
    log_prob = log_prob[valid_mask]

    if sort_descending and len(log_prob) > 0:
        order = np.argsort(-log_prob)
        positions = positions[order]
        log_prob = log_prob[order]

    return InitialPositionFilterResult(
        positions=positions,
        log_prob=log_prob,
        valid_mask=valid_mask,
    )

