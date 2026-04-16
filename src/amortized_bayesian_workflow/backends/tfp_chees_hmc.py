from __future__ import annotations

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from ..diagnostics import summarize_chain_convergence
from .base import SamplerRequest, SamplerResult
from .utils import filter_initial_positions


class TFPCheesHMCBackend:
    """Optional TFP/JAX ChEES-HMC backend.

    Intended as a compatibility backend which may be more robust than BlackJAX's implementation.
    """

    name = "tfp_chees_hmc"

    def run(self, request: SamplerRequest) -> SamplerResult:
        opt = dict(request.options)
        num_superchains = int(
            opt.pop("num_superchains", request.initial_positions.shape[0])
        )
        subchains_per_superchain = int(opt.pop("subchains_per_superchain", 1))
        init_step_size = float(
            opt.pop("init_step_size", opt.pop("initial_step_size", 0.1))
        )

        filtered = filter_initial_positions(
            request.initial_positions,
            request.log_prob_fn,
        )
        if filtered.positions.shape[0] < num_superchains:
            raise ValueError("Not enough valid unique initial positions.")

        seeds = jax.random.PRNGKey(request.seed)
        base_positions = filtered.positions[:num_superchains].astype(
            np.float64
        )
        tiled_positions = np.repeat(
            base_positions, subchains_per_superchain, axis=0
        )

        leapfrog_steps = int(opt.pop("num_leapfrog_steps", 1))
        total_steps = request.iter_warmup + request.iter_sampling

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=request.log_prob_fn,
            step_size=init_step_size,
            num_leapfrog_steps=leapfrog_steps,
        )
        kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=request.iter_warmup,
        )
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=request.iter_warmup,
            reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
        )

        draws_t, _ = tfp.mcmc.sample_chain(
            num_results=total_steps,
            current_state=tiled_positions,
            kernel=kernel,
            seed=seeds,
            trace_fn=lambda _, kernel_results: kernel_results,
        )
        draws = np.asarray(draws_t[request.iter_warmup :].swapaxes(0, 1))
        diagnostics = {
            **summarize_chain_convergence(
                draws, num_superchains=num_superchains
            ),
            "filtered_initial_log_prob": filtered.log_prob[:num_superchains],
            "num_valid_initial_positions": int(filtered.positions.shape[0]),
        }
        return SamplerResult(
            backend=self.name,
            draws=draws,
            diagnostics=diagnostics,
            metadata={
                "num_superchains": num_superchains,
                "subchains_per_superchain": subchains_per_superchain,
                "sampler": "chees_hmc",
            },
        )
