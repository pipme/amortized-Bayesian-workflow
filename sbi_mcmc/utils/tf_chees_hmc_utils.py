import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from sbi_mcmc.diagnostics import nested_rhat


def filter_invalid_init_positions(initial_positions, lp_fn, sort=False):
    initial_positions = np.unique(initial_positions, axis=0)
    logp_values = lp_fn(initial_positions)
    # Some parameters can induce -inf logp (e.g., since the likelihood for GEV has finite support in data space)
    inds_valid = np.isfinite(logp_values)
    print(f"Valid initial positions: {np.sum(inds_valid)}/{len(inds_valid)}")
    initial_positions = initial_positions[inds_valid]
    logp_values_valid = logp_values[inds_valid]
    if sort:
        # sort according to logp values, from high to low
        sort_inds = np.argsort(-logp_values_valid)
        initial_positions = initial_positions[sort_inds]
        logp_values_valid = logp_values_valid[sort_inds]
    print(
        f"Unique initial positions: {len(initial_positions)}/{np.sum(inds_valid)}"
    )
    return initial_positions, logp_values_valid


def run_chees_hmc(
    initial_positions,
    K,
    M,
    lp_fn,
    num_warmup,
    num_sampling,
    D,
    init_step_size=0.1,
    seed=None,
    sort=False,
    target_log_prob_fn=None,
):
    if seed is None:
        seed = jax.random.PRNGKey(0)
    num_chains = K * M
    result_record = {}
    initial_positions, logp_values_valid = filter_invalid_init_positions(
        initial_positions, lp_fn, sort=sort
    )

    if len(initial_positions) < K:
        raise ValueError("Not enough valid and unique initial positions")

    initial_positions = initial_positions[:K]
    logp_values_valid = logp_values_valid[:K]
    result_record["initial_positions"] = initial_positions
    result_record["logp_values_initial_positions"] = logp_values_valid

    assert D == initial_positions.shape[1], (
        f"{D} != {initial_positions.shape[1]}, D must be the same as the dimension of initial positions"
    )
    # Repeat the initial positions M times for each superchain's subchains
    initial_positions_numpy = np.repeat(initial_positions, M, axis=0)
    initial_positions_numpy = initial_positions_numpy.astype(np.float64)

    if target_log_prob_fn is None:
        target_log_prob_fn = lp_fn

    kernel_mcmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn, init_step_size, 1
    )
    kernel_mcmc = (
        tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
            kernel_mcmc, num_warmup
        )
    )
    kernel_mcmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel_mcmc,
        num_warmup,
        # target_accept_prob=0.75,
        reduce_fn=tfp.math.reduce_log_harmonic_mean_exp,
    )
    total_num_iterations = num_warmup + num_sampling

    def trace_everything(states, previous_kernel_results):
        return previous_kernel_results

    print("Running ChEES-HMC...")
    result_mcmc, states_and_trace_tfp = tfp.mcmc.sample_chain(
        total_num_iterations,
        initial_positions_numpy,
        kernel=kernel_mcmc,
        seed=seed,
        trace_fn=trace_everything,
    )
    positions_tfp = result_mcmc[num_warmup:]
    positions_tfp = np.array(positions_tfp.swapaxes(0, 1))
    assert positions_tfp.shape == (num_chains, num_sampling, D), (
        f"{positions_tfp.shape} != {(num_chains, num_sampling, D)}"
    )
    print(positions_tfp.shape)
    # samples_tfp = positions_tfp.reshape(-1, D)
    # samples_tfp = np.array(samples_tfp)

    n_rhat_value = nested_rhat(positions_tfp, num_superchains=K)
    result_record["n_rhat"] = n_rhat_value
    result_record["chees_draws_tfp"] = positions_tfp
    # result_record["states_and_trace_tfp"] = states_and_trace_tfp
    print(n_rhat_value)
    return result_record
