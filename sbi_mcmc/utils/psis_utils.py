import arviz as az
import numpy as np


def sampling_importance_resampling(
    log_pdfs_target,
    log_pdfs_proposal,
    samples_proposal,
    reff=1,
    num_samples=10000,
    return_weights=False,
    with_replacement=True,
):
    pareto_log_weights, k_stat = psis_weights(
        log_pdfs_target, log_pdfs_proposal, reff
    )
    print(f"K_stat: {k_stat}")

    # Sampling Importance Resampling
    resamples = _sir(
        samples_proposal, pareto_log_weights, num_samples, with_replacement
    )
    if return_weights:
        return resamples, k_stat, pareto_log_weights
    return resamples, k_stat


def _sir(
    samples_proposal, log_weights, num_samples=10000, with_replacement=True
):
    if not with_replacement:
        assert num_samples <= samples_proposal.shape[0], (
            "Not enough proposal samples to draw from without replacement"
        )
    inds = np.random.choice(
        samples_proposal.shape[0],
        num_samples,
        replace=with_replacement,
        p=np.exp(log_weights),
    )
    resamples = np.array(samples_proposal)[inds]
    return resamples


def psis_weights(log_pdfs_target, log_pdfs_proposal, reff=1):
    log_weights = np.array(log_pdfs_target - log_pdfs_proposal)
    pareto_log_weights, k_stat = az.psislw(log_weights, reff=reff)
    return pareto_log_weights, k_stat.item()
