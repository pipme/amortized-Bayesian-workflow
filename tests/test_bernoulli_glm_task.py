from __future__ import annotations

import numpy as np

from amortized_bayesian_workflow.tasks.examples import make_bernoulli_glm_task


def test_bernoulli_glm_sampling_shapes_sufficient():
    task = make_bernoulli_glm_task(summary="sufficient")
    sims = task.sample_prior_predictive(8, seed=123)
    assert sims["parameters"].shape == (8, 10)
    assert sims["observables"].shape == (8, 10)
    assert sims["observables_raw"].shape == (8, 100)
    assert sims["observables_summary"].shape == (8, 10)


def test_bernoulli_glm_sampling_shapes_raw():
    task = make_bernoulli_glm_task(summary="raw")
    sims = task.sample_prior_predictive(6, seed=123)
    assert sims["parameters"].shape == (6, 10)
    assert sims["observables"].shape == (6, 100)
    assert sims["observables_raw"].shape == (6, 100)
    assert sims["observables_summary"].shape == (6, 10)


def test_bernoulli_glm_vectorized_matches_single_sufficient():
    task = make_bernoulli_glm_task(summary="sufficient")
    sims = task.sample_prior_predictive(5, seed=7)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)


def test_bernoulli_glm_vectorized_matches_single_raw():
    task = make_bernoulli_glm_task(summary="raw")
    sims = task.sample_prior_predictive(5, seed=9)
    obs = sims["observables"][0]
    theta = sims["parameters"]

    vec = task.vectorized_log_posterior_fn(obs)(theta)
    single = task.single_log_posterior_fn(obs)
    one_by_one = np.asarray([single(row) for row in theta], dtype=float)
    assert np.allclose(vec, one_by_one, atol=1e-6)
