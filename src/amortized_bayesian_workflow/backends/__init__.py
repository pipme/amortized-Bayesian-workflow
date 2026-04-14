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
    single_log_prob_fn: Callable[[Any], Any] | None = None,
    seed: int,
    overrides: Mapping[str, Any] | None = None,
    psis_log_weights: np.ndarray | None = None,
    psis_weights: np.ndarray | None = None,
) -> SamplerRequest:
    """Build a `SamplerRequest` using sampler-owned defaults.

    Users can pass minimal input (sampler name + core functions) and optionally
    override defaults through `overrides`.
    """

    get_backend(backend_name)
    single_log_prob_fn = _resolve_log_prob_functions(
        log_prob_fn=log_prob_fn,
        single_log_prob_fn=single_log_prob_fn,
    )
    fallback_defaults: dict[str, dict[str, Any]] = {
        "blackjax_nuts": {
            "num_warmup": 1_000,
            "num_samples": 1_000,
            "initial_position_source": "amortized_random",
            "chain_count_option": "num_chains",
            "chain_count": 8,
            "options": {"target_accept": 0.8},
        },
        "blackjax_chees_hmc": {
            "num_warmup": 200,
            "num_samples": 1,
            "initial_position_source": "amortized_random",
            "chain_count_option": "num_superchains",
            "chain_count": 16,
            "options": {
                "num_subchains_per_superchain": 128,
                "initial_step_size": 0.1,
            },
        },
        "tfp_chees_hmc": {
            "num_warmup": 200,
            "num_samples": 1_000,
            "initial_position_source": "amortized_random",
            "chain_count_option": "num_superchains",
            "chain_count": 16,
            "options": {
                "num_subchains_per_superchain": 128,
                "init_step_size": 0.1,
            },
        },
    }
    backend_fallback = fallback_defaults.get(
        backend_name,
        {
            "num_warmup": 1_000,
            "num_samples": 1_000,
            "initial_position_source": "amortized_random",
            "chain_count_option": "num_chains",
            "chain_count": 8,
            "options": {},
        },
    )

    override_dict = dict(overrides or {})
    options_override = dict(override_dict.pop("options", {}))

    num_warmup = int(
        override_dict.pop("num_warmup", backend_fallback["num_warmup"])
    )
    num_samples = int(
        override_dict.pop("num_samples", backend_fallback["num_samples"])
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

    chain_key = str(backend_fallback["chain_count_option"])
    default_chain_count = int(backend_fallback["chain_count"])
    if chain_key == "num_chains":
        chain_count = int(
            options.pop(
                "num_chains",
                options.pop("num_superchains", default_chain_count),
            )
        )
    else:
        chain_count = int(
            options.pop(
                "num_superchains",
                options.pop("num_chains", default_chain_count),
            )
        )
    chain_count = max(1, min(chain_count, q_samples.shape[0]))

    rng = np.random.default_rng(int(seed))
    if init_source in {"amortized", "amortized_random", "random"}:
        init_idx = rng.choice(
            q_samples.shape[0], size=chain_count, replace=False
        )
        initial_positions = np.asarray(q_samples[init_idx], dtype=float)
    elif init_source in {"psis_resampled", "psis_weighted"}:
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
    # elif init_source in {"psis_top", "top_weighted"}:
    #     if psis_log_weights is None:
    #         raise ValueError(
    #             "psis_log_weights is required when initial_position_source='psis_top'."
    #         )
    #     init_idx = np.argsort(-np.asarray(psis_log_weights))[:chain_count]
    #     initial_positions = np.asarray(q_samples[init_idx], dtype=float)
    else:
        raise ValueError(
            "Unsupported initial_position_source. "
            "Use one of: amortized_random, psis_resampled, psis_top."
        )

    if chain_key == "num_chains":
        options["num_chains"] = int(chain_count)
    else:
        options["num_superchains"] = int(chain_count)

    if "num_subchains_per_superchain" in options:
        options["subchains_per_superchain"] = int(
            options.pop("num_subchains_per_superchain")
        )

    return SamplerRequest(
        initial_positions=initial_positions,
        log_prob_fn=single_log_prob_fn,
        iter_warmup=num_warmup,
        iter_sampling=num_samples,
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
