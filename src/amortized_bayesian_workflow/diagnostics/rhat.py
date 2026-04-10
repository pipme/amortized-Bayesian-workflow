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


def _rhat_fallback(chains: np.ndarray) -> np.ndarray | float:
    """Compute classic split-free Gelman-Rubin R-hat for (chains, draws[, dims])."""
    ary = np.asarray(chains, dtype=float)
    if ary.ndim == 2:
        ary = ary[:, :, None]

    n_chains, n_draws, _ = ary.shape
    if n_chains < 2 or n_draws < 2:
        out = np.full(ary.shape[-1], np.nan, dtype=float)
        return float(out[0]) if out.size == 1 else out

    chain_means = np.mean(ary, axis=1)
    # Between-chain variance B and within-chain variance W.
    b = n_draws * np.var(chain_means, axis=0, ddof=1)
    w = np.mean(np.var(ary, axis=1, ddof=1), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        var_hat = ((n_draws - 1.0) / n_draws) * w + (1.0 / n_draws) * b
        r_hat = np.sqrt(var_hat / w)
    r_hat = np.where(np.isfinite(r_hat), r_hat, np.nan)

    if r_hat.size == 1:
        return float(r_hat[0])
    return r_hat


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
    if hasattr(azs, "rhat"):
        return _to_numpy_stat(azs.rhat(ary, chain_axis=0, draw_axis=1))
    return _rhat_fallback(ary)


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

    try:
        return _to_numpy_stat(
            azs.rhat_nested(
                ary,
                superchain_ids=superchain_ids.tolist(),
                chain_axis=0,
                draw_axis=1,
            )
        )
    except Exception:
        # Keep a compatibility bridge for nested R-hat if arviz_stats API changes.
        return _nested_rhat_fallback(ary, superchains_id=superchain_ids)


def summarize_chain_convergence(
    chains: np.ndarray,
    *,
    num_superchains: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Return standard R-hat and (when meaningful) nested R-hat diagnostics."""
    ary = np.asarray(chains)
    out: dict[str, np.ndarray | float] = {"rhat": rhat(ary)}
    if num_superchains is not None and 1 < num_superchains < ary.shape[0]:
        try:
            out["nested_rhat"] = rhat_nested(
                ary, num_superchains=num_superchains
            )
        except Exception:
            # Keep standard rhat even if nested rhat is not available.
            pass
    return out
