from __future__ import annotations

import numpy as np
import pytest

from amortized_bayesian_workflow.tasks.examples import make_bernoulli_glm_task


@pytest.mark.parametrize(
    ("summary", "n", "expected_obs_dim"),
    [("sufficient", 8, 10), ("raw", 6, 100)],
)
def test_bernoulli_glm_sampling_shapes(
    summary: str, n: int, expected_obs_dim: int
):
    task = make_bernoulli_glm_task(summary=summary)
    sims = task.sample_prior_predictive(n, seed=123)
    assert sims["parameters"].shape == (n, 10)
    assert sims["observables"].shape == (n, expected_obs_dim)
    assert sims["observables_raw"].shape == (n, 100)
    assert sims["observables_summary"].shape == (n, 10)


@pytest.mark.parametrize(("summary", "seed"), [("sufficient", 7), ("raw", 9)])
def test_bernoulli_glm_vectorized_matches_single(summary: str, seed: int):
    task = make_bernoulli_glm_task(summary=summary)
    sims = task.sample_prior_predictive(5, seed=seed)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)
