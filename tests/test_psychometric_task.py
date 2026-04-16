from __future__ import annotations

import numpy as np

from amortized_bayesian_workflow.tasks.examples import PsychometricTask


def test_psychometric_overdispersion_shapes_and_vectorized_matches_single():
    task = PsychometricTask(overdispersion=True)
    sims = task.simulate_prior_predictive(6, seed=22)
    assert sims["parameters"].shape == (6, 5)
    assert sims["observables"].shape == (6, 9, 3)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)
