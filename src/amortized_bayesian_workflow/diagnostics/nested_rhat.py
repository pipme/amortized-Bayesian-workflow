from __future__ import annotations

import numpy as np


def nested_rhat(
    ary: np.ndarray,
    *,
    num_superchains: int | None = None,
    superchains_id: np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute nested R-hat for 2D or 3D chain arrays.

    `ary` shape is `(chains, draws)` or `(chains, draws, dims)`.
    """

    if ary.ndim == 2:
        return _nested_rhat_2d(
            ary, num_superchains=num_superchains, superchains_id=superchains_id
        )
    if ary.ndim == 3:
        return np.asarray(
            [
                _nested_rhat_2d(
                    ary[:, :, i],
                    num_superchains=num_superchains,
                    superchains_id=superchains_id,
                )
                for i in range(ary.shape[2])
            ]
        )
    raise ValueError("Expected a 2D or 3D array.")


def _nested_rhat_2d(
    ary: np.ndarray,
    *,
    num_superchains: int | None,
    superchains_id: np.ndarray | None,
) -> float:
    if (num_superchains is None) == (superchains_id is None):
        raise ValueError("Provide exactly one of num_superchains or superchains_id.")
    if num_superchains is not None:
        if ary.shape[0] % num_superchains != 0:
            raise ValueError("Number of chains must be divisible by num_superchains.")
        superchains_id = np.repeat(
            np.arange(num_superchains), ary.shape[0] // num_superchains
        )
    assert superchains_id is not None

    unique_ids, counts = np.unique(superchains_id, return_counts=True)
    if counts.min() != counts.max():
        raise ValueError("Each superchain must contain the same number of subchains.")

    k = len(unique_ids)
    m = ary.shape[0] // k
    n = ary.shape[1]
    if n <= 0 or m <= 0:
        raise ValueError("Invalid chain shape.")

    reorder = np.argsort(superchains_id)
    ordered = ary[reorder].T
    ordered = np.reshape(ordered, (n, m, k), order="F")

    mean_super = ordered.mean(axis=(0, 1))
    mean_all = mean_super.mean()
    mean_sub = ordered.mean(axis=0)

    if k <= 1:
        raise ValueError("At least two superchains are required.")
    between_super = np.sum((mean_super - mean_all) ** 2) / (k - 1)

    if m > 1:
        between_chain = np.sum((mean_sub - mean_super) ** 2, axis=0) / (m - 1)
    else:
        between_chain = np.zeros(k)

    if n > 1:
        within_chain = np.sum((ordered - mean_sub) ** 2, axis=0) / (n - 1)
        within_chain = within_chain.mean(axis=0)
    else:
        within_chain = np.zeros(k)

    within_super = np.mean(between_chain + within_chain)
    if within_super == 0:
        return float("inf")
    return float(np.sqrt(1.0 + between_super / within_super))

