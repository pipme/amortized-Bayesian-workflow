from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from ..psis import resample_with_weights
from .base import SamplerBackend, SamplerRequest, SamplerResult
from .resolve import get_backend

if TYPE_CHECKING:
    from .blackjax_chees_hmc import BlackJAXCheesHMCBackend
    from .blackjax_nuts import BlackJAXNUTSBackend
    from .tfp_chees_hmc import TFPCheesHMCBackend

__all__ = [
    "BlackJAXCheesHMCBackend",
    "BlackJAXNUTSBackend",
    "SamplerBackend",
    "SamplerRequest",
    "SamplerResult",
    "TFPCheesHMCBackend",
    "get_backend",
    "prepare_sampler_request",
    "run_mcmc",
]


def __getattr__(name: str):
    if name == "BlackJAXCheesHMCBackend":
        mod = import_module(".blackjax_chees_hmc", __name__)
        return getattr(mod, name)
    if name == "BlackJAXNUTSBackend":
        mod = import_module(".blackjax_nuts", __name__)
        return getattr(mod, name)
    if name == "TFPCheesHMCBackend":
        mod = import_module(".tfp_chees_hmc", __name__)
        return getattr(mod, name)
    raise AttributeError(name)


def _resolve_log_prob_functions(
    *,
    log_prob_fn: Callable[[np.ndarray], np.ndarray] | None,
    single_log_prob_fn: Callable[[Any], Any] | None,
) -> Callable[[Any], Any]:
    if log_prob_fn is None and single_log_prob_fn is None:
        raise ValueError(
            "Provide at least one of log_prob_fn or single_log_prob_fn."
        )

    if single_log_prob_fn is not None:
        return single_log_prob_fn
    assert log_prob_fn is not None

    def _single(theta: Any) -> Any:
        ary = theta[None, ...]
        return log_prob_fn(ary)[0]

    return _single


def prepare_sampler_request(
    *,
    backend_name: str,
    q_samples: np.ndarray,
    log_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int,
    overrides: Mapping[str, Any] | None = None,
    psis_log_weights: np.ndarray | None = None,
    psis_weights: np.ndarray | None = None,
) -> SamplerRequest:
    """Build a `SamplerRequest` using sampler-owned defaults.

    Users can pass minimal input (sampler name + core functions) and optionally
    override defaults through `overrides`.
    """

    fallback_defaults: dict[str, dict[str, Any]] = {
        "blackjax_nuts": {
            "iter_warmup": 1_000,
            "iter_sampling": 1_000,
            "initial_position_source": "amortized",
            "options": {
                "num_chains": 4,
                "target_accept": 0.8,
            },
        },
        "blackjax_chees_hmc": {
            "iter_warmup": 200,
            "iter_sampling": 1,
            "initial_position_source": "amortized",
            "options": {
                "num_superchains": 16,
                "num_subchains_per_superchain": 128,
                "initial_step_size": 0.1,
            },
        },
        "tfp_chees_hmc": {
            "iter_warmup": 200,
            "iter_sampling": 1,
            "initial_position_source": "amortized",
            "options": {
                "num_superchains": 16,
                "num_subchains_per_superchain": 128,
                "init_step_size": 0.1,
            },
        },
    }
    backend_fallback = fallback_defaults.get(
        backend_name,
        {
            "iter_warmup": 1_000,
            "iter_sampling": 1_000,
            "initial_position_source": "amortized",
            "options": {"num_chains": 8},
        },
    )

    override_dict = dict(overrides or {})
    options_override = dict(override_dict.pop("options", {}))

    iter_warmup = int(
        override_dict.pop("iter_warmup", backend_fallback["iter_warmup"])
    )
    iter_sampling = int(
        override_dict.pop("iter_sampling", backend_fallback["iter_sampling"])
    )
    init_source = (
        str(
            override_dict.pop(
                "initial_position_source",
                backend_fallback["initial_position_source"],
            )
        )
        .strip()
        .lower()
    )

    options = {
        **dict(backend_fallback["options"]),
        **options_override,
        **override_dict,
    }

    if backend_name in {"blackjax_chees_hmc", "tfp_chees_hmc"}:
        chain_count = int(options.pop("num_superchains"))
    else:
        chain_count = int(options.pop("num_chains"))

    assert chain_count <= q_samples.shape[0], (
        "Not enough q_samples to initialize the requested number of (super)chains."
    )
    if init_source in {"amortized"}:
        initial_positions = np.asarray(q_samples[:chain_count], dtype=float)
    elif init_source in {"psis_resampled"}:
        if psis_weights is None:
            raise ValueError(
                "psis_weights is required when initial_position_source='psis_resampled'."
            )
        initial_positions = np.asarray(
            resample_with_weights(
                samples=q_samples,
                weights=psis_weights,
                num_draws=chain_count,
                seed=int(seed),
                replace=False,
            ),
            dtype=float,
        )
    else:
        raise ValueError(
            "Unsupported initial_position_source. "
            "Use one of: amortized, psis_resampled."
        )

    if "num_subchains_per_superchain" in options:
        options["subchains_per_superchain"] = int(
            options.pop("num_subchains_per_superchain")
        )

    return SamplerRequest(
        initial_positions=initial_positions,
        log_prob_fn=log_prob_fn,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=int(seed),
        options=options,
    )


def run_mcmc(
    *,
    backend_name: str,
    initial_positions: np.ndarray,
    iter_warmup: int,
    iter_sampling: int,
    log_prob_fn: Callable[[Any], Any],
    seed: int = 0,
    options: Mapping[str, Any] | None = None,
) -> SamplerResult:
    """Run MCMC with a backend-agnostic, notebook-friendly call.

    ``log_prob_fn`` is the scalar log density for a single parameter vector.
    A batched form is derived internally for initialization filtering.
    """

    backend = get_backend(backend_name)
    request = SamplerRequest(
        initial_positions=np.asarray(initial_positions),
        log_prob_fn=log_prob_fn,
        iter_warmup=int(iter_warmup),
        iter_sampling=int(iter_sampling),
        seed=int(seed),
        options=dict(options or {}),
    )
    return backend.run(request)
