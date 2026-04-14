from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from ..diagnostics import summarize_chain_convergence
from .base import SamplerRequest, SamplerResult
from .utils import filter_initial_positions


class BlackJAXNUTSBackend:
    """BlackJAX NUTS backend with per-chain warmup and sampling."""

    name = "blackjax_nuts"

    def run(self, request: SamplerRequest) -> SamplerResult:
        if request.initial_positions.ndim != 2:
            raise ValueError(
                "initial_positions must have shape (num_chains, dim)."
            )

        opt = dict(request.options)
        num_chains = int(opt.pop("num_chains", 1))
        store_warmup_state = bool(opt.pop("store_warmup_state", False))
        target_accept = float(opt.pop("target_accept", 0.8))

        filtered = filter_initial_positions(
            request.initial_positions,
            request.log_prob_fn,
        )
        chains = filtered.positions[:num_chains]

        if filtered.positions.shape[0] < num_chains:
            raise ValueError("Not enough valid unique initial positions.")

        draws = []
        accept_rates = []
        warmup_states = []

        for chain_id, init in enumerate(chains):
            key = jax.random.PRNGKey(request.seed + chain_id)
            chain_draws, accept_rate, warm_state = self._run_single_chain(
                logdensity_fn=request.log_prob_fn,
                init_position=jnp.asarray(init),
                key=key,
                iter_warmup=request.iter_warmup,
                iter_sampling=request.iter_sampling,
                target_accept=target_accept,
            )
            draws.append(np.asarray(chain_draws))
            accept_rates.append(float(accept_rate))
            if store_warmup_state:
                warmup_states.append(warm_state)

        stacked = np.stack(draws, axis=0)
        diagnostics = {
            **summarize_chain_convergence(
                stacked,
                num_superchains=num_chains
                if 1 < num_chains < stacked.shape[0]
                else None,
            ),
            "accept_rate_mean": float(np.mean(accept_rates)),
            "accept_rate_per_chain": np.asarray(accept_rates),
        }
        metadata = {"num_chains": int(chains.shape[0]), "sampler": "nuts"}
        if store_warmup_state:
            metadata["warmup_states"] = warmup_states
        return SamplerResult(
            backend=self.name,
            draws=stacked,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _run_single_chain(
        self,
        *,
        logdensity_fn,
        init_position,
        key,
        iter_warmup: int,
        iter_sampling: int,
        target_accept: float,
    ):
        adapt = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=target_accept,
        )

        warmup_key, sample_key = jax.random.split(key)
        state, kernel = self._run_warmup(
            adapt,
            logdensity_fn=logdensity_fn,
            key=warmup_key,
            init_position=init_position,
            iter_warmup=iter_warmup,
        )

        def one_step(carry, rng_key):
            state = carry
            state, info = kernel.step(rng_key, state)
            if hasattr(state, "position"):
                pos = state.position
            else:
                pos = state[0]
            is_acc = getattr(info, "is_accepted", None)
            if is_acc is None:
                acc_prob = getattr(info, "acceptance_probability", 1.0)
            else:
                acc_prob = is_acc
            return state, (pos, acc_prob)

        keys = jax.random.split(sample_key, int(iter_sampling))
        _, (positions, acc) = jax.lax.scan(one_step, state, keys)
        accept_rate = jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
        return positions, accept_rate, state

    def _run_warmup(
        self,
        adapt,
        *,
        logdensity_fn,
        key,
        init_position,
        iter_warmup: int,
    ):
        out = adapt.run(key, init_position, num_steps=int(iter_warmup))

        # Handle common BlackJAX API variants.
        if isinstance(out, tuple) and len(out) == 2:
            first, _ = out
            # Variant A: ((state, parameters), adaptation_info)
            if isinstance(first, tuple) and len(first) == 2:
                state, params = first
                kernel = self._build_nuts_kernel(logdensity_fn, params)
                return state, kernel
            # Variant B: (AdaptationResults(state, parameters, kernel), info)
            state = getattr(first, "state", None)
            kernel = getattr(first, "kernel", None)
            params = getattr(first, "parameters", None)
            if state is not None and kernel is not None:
                return state, kernel
            if state is not None and params is not None:
                kernel = self._build_nuts_kernel(logdensity_fn, params)
                return state, kernel

        raise RuntimeError("Unsupported BlackJAX warmup output format.")

    def _build_nuts_kernel(self, logdensity_fn, params):
        try:
            if isinstance(params, dict):
                return blackjax.nuts(logdensity_fn, **params)
            if hasattr(params, "_asdict"):
                return blackjax.nuts(logdensity_fn, **params._asdict())
            if hasattr(params, "__dict__"):
                return blackjax.nuts(logdensity_fn, **vars(params))
        except TypeError:
            # Fallback for step_size/inverse_mass_matrix style params object.
            if hasattr(params, "step_size") and hasattr(
                params, "inverse_mass_matrix"
            ):
                return blackjax.nuts(
                    logdensity_fn,
                    step_size=params.step_size,
                    inverse_mass_matrix=params.inverse_mass_matrix,
                )
            raise
        raise RuntimeError(
            "Unsupported BlackJAX NUTS parameter format after warmup."
        )
