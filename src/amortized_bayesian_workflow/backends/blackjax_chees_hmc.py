from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..diagnostics import summarize_chain_convergence
from .base import SamplerRequest, SamplerResult
from .utils import filter_initial_positions


class BlackJAXCheesHMCBackend:
    """BlackJAX ChEES-adapted dynamic HMC backend with vmapped chains."""

    name = "blackjax_chees_hmc"

    def run(self, request: SamplerRequest) -> SamplerResult:
        chains = np.asarray(request.initial_positions, dtype=float)
        if chains.ndim != 2:
            raise ValueError(
                "initial_positions must have shape (num_initial_positions, dim)."
            )

        opt = dict(request.options)
        num_superchains = int(opt.pop("num_superchains", 16))
        subchains_per_superchain = int(
            opt.pop("subchains_per_superchain", 128)
        )
        store_warmup_state = bool(opt.pop("store_warmup_state", False))

        initial_step_size = float(
            opt.pop("initial_step_size", opt.pop("init_step_size", 0.1))
        )
        learning_rate = float(opt.pop("learning_rate", 5e-2))
        target_accept = float(opt.pop("target_accept", 0.651))
        decay_rate = float(opt.pop("decay_rate", 0.5))
        jitter_amount = float(opt.pop("jitter_amount", 1.0))
        max_leapfrog_steps = int(opt.pop("max_leapfrog_steps", 1000))

        filtered = filter_initial_positions(
            chains,
            request.log_prob_fn,
        )
        if filtered.positions.shape[0] < num_superchains:
            raise ValueError("Not enough valid unique initial positions.")

        base_positions = filtered.positions[:num_superchains].astype(
            np.float64
        )
        tiled_positions = np.repeat(
            base_positions, subchains_per_superchain, axis=0
        )
        num_chains, dim = tiled_positions.shape

        key = jax.random.PRNGKey(request.seed)
        warmup_key, sample_key = jax.random.split(key)

        warmup = blackjax.chees_adaptation(
            request.log_prob_fn,
            num_chains=num_chains,
            target_acceptance_rate=target_accept,
            decay_rate=decay_rate,
            jitter_amount=jitter_amount,
            max_leapfrog_steps=max_leapfrog_steps,
        )
        optim = optax.adam(learning_rate)
        (warmup_results, _warmup_info) = warmup.run(
            warmup_key,
            jnp.asarray(tiled_positions),
            step_size=initial_step_size,
            optim=optim,
            num_steps=int(request.iter_warmup),
        )
        states = warmup_results.state
        params = warmup_results.parameters

        kernel = blackjax.dynamic_hmc(request.log_prob_fn, **params).step

        def one_step(carry, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            new_states, info = jax.vmap(kernel)(keys, carry)
            positions = new_states.position
            accept = getattr(info, "acceptance_rate", None)
            if accept is None:
                accept = getattr(info, "acceptance_probability", 1.0)
            return new_states, (positions, jnp.asarray(accept))

        # JIT the sampling loop to avoid Python overhead across many iterations.
        run_sampling = jax.jit(
            lambda init_states, keys: jax.lax.scan(one_step, init_states, keys)
        )

        sample_keys = jax.random.split(sample_key, int(request.iter_sampling))
        final_states, (positions_t, accept_t) = run_sampling(states, sample_keys)

        draws = np.asarray(jnp.swapaxes(positions_t, 0, 1))
        accept_t = jnp.asarray(accept_t)
        accept_rate_per_chain = np.asarray(jnp.mean(accept_t, axis=0))

        diagnostics = {
            **summarize_chain_convergence(
                draws,
                num_superchains=num_superchains
                if 1 < num_superchains < draws.shape[0]
                else None,
            ),
            "accept_rate_mean": float(np.mean(accept_rate_per_chain)),
            "accept_rate_per_chain": accept_rate_per_chain,
        }

        metadata = {
            "num_superchains": num_superchains,
            "subchains_per_superchain": subchains_per_superchain,
            "num_chains": int(num_chains),
            "sampler": "blackjax_dynamic_hmc_chees",
        }
        if store_warmup_state:
            metadata["warmup_state"] = final_states

        return SamplerResult(
            backend=self.name,
            draws=draws,
            diagnostics=diagnostics,
            metadata=metadata,
        )
