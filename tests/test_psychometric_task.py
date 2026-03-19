from __future__ import annotations

import numpy as np

from amortized_bayesian_workflow.tasks.examples import make_psychometric_task


def test_psychometric_sample_shapes():
    task = make_psychometric_task(overdispersion=False)
    sims = task.sample_prior_predictive(7, seed=10)
    assert sims["parameters"].shape == (7, 4)
    assert sims["observables"].shape == (7, 9, 3)


def test_psychometric_overdispersion_sample_shapes():
    task = make_psychometric_task(overdispersion=True)
    sims = task.sample_prior_predictive(6, seed=11)
    assert sims["parameters"].shape == (6, 5)
    assert sims["observables"].shape == (6, 9, 3)


def test_psychometric_vectorized_matches_single():
    task = make_psychometric_task(overdispersion=False)
    sims = task.sample_prior_predictive(5, seed=21)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)


def test_psychometric_overdispersion_vectorized_matches_single():
    task = make_psychometric_task(overdispersion=True)
    sims = task.sample_prior_predictive(5, seed=22)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)


def test_psychometric_can_load_csv_observation():
    task = make_psychometric_task()
    obs = task.load_first_csv_observation()
    assert obs.shape == (9, 3)
