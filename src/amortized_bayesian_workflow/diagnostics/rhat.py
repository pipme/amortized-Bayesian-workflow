from __future__ import annotations

from typing import Any

import arviz_stats as azs
import numpy as np

from .nested_rhat import nested_rhat as _nested_rhat_fallback


def _to_numpy_stat(x: Any) -> np.ndarray | float:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return arr


def rhat(chains: np.ndarray) -> np.ndarray | float:
    """Compute standard R-hat on chain arrays of shape (chains, draws[, dims]).

    Prefer `arviz_stats.rhat` when available and fall back to a NumPy
    implementation for compatibility with older/newer `arviz_stats` APIs.
    """
    ary = np.asarray(chains)
    if ary.ndim not in {2, 3}:
        raise ValueError(
            "Expected chain array with shape (chains, draws[, dims])."
        )
    return _to_numpy_stat(azs.rhat(ary, chain_axis=0, draw_axis=1))


def rhat_nested(
    chains: np.ndarray,
    *,
    num_superchains: int | None = None,
    superchain_ids: np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute nested R-hat, preferring `arviz_stats.rhat_nested` if available."""
    ary = np.asarray(chains)
    if ary.ndim not in {2, 3}:
        raise ValueError(
            "Expected chain array with shape (chains, draws[, dims])."
        )

    if (num_superchains is None) == (superchain_ids is None):
        raise ValueError(
            "Provide exactly one of num_superchains or superchain_ids."
        )
    if superchain_ids is None:
        if ary.shape[0] % int(num_superchains) != 0:
            raise ValueError(
                "Number of chains must be divisible by num_superchains."
            )
        superchain_ids = np.repeat(
            np.arange(int(num_superchains)),
            ary.shape[0] // int(num_superchains),
        )
    else:
        superchain_ids = np.asarray(superchain_ids)
    return _nested_rhat_fallback(ary, superchains_id=superchain_ids)
    # We can use the following when arviz_stats.rhat_nested works as expected, but it now restricts the number of draws to be at least 4, which is not always the case in our experiments. https://github.com/arviz-devs/arviz-stats/issues/354
    # return _to_numpy_stat(
    #     azs.rhat_nested(
    #         ary,
    #         superchain_ids=superchain_ids.tolist(),
    #         chain_axis=0,
    #         draw_axis=1,
    #     )
    # )


def summarize_chain_convergence(
    chains: np.ndarray,
    *,
    num_superchains: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Return (nested) R-hat diagnostics."""
    ary = np.asarray(chains)
    out: dict[str, np.ndarray | float] = {}
    if num_superchains is not None and 1 < num_superchains < ary.shape[0]:
        out["nested_rhat"] = rhat_nested(ary, num_superchains=num_superchains)
    else:
        out["rhat"] = rhat(ary)

    return out
