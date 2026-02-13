"""Tests for `sbi_mcmc` package."""

import arviz as az
import numpy as np
import pymc as pm
import pytest
import scipy.stats as sps
from pymc import Normal, Uniform
from pymc.stats.log_density import compute_log_likelihood, compute_log_prior
from sbi_mcmc.tasks import GeneralizedExtremeValue
from sbi_mcmc.tasks.tasks import (
    PyMCPriorWrapper,
    PyMCTaskVariableInfo,
    ndarray_values_as_dict,
)
from sbi_mcmc.tasks.tasks_utils import get_task_logp_func


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return "fixture"


def test_prior_computation():
    with pm.Model() as pymc_model:
        sigma = Uniform("sigma", 0, 2)
        intercept = Normal("intercept", 1, sigma=2)
        slope = Normal("slope", 2, sigma=1)

    var_names = ["slope", "intercept", "sigma"]
    var_dims = {k: 1 for k in var_names}
    var_info = PyMCTaskVariableInfo(
        var_names,
        None,
        var_dims,
        None,
        None,
        None,
    )
    prior = PyMCPriorWrapper(pymc_model, var_info)
    pt = np.array([0.9, 3, 2])
    v1 = prior.log_prob(np.atleast_2d(pt)).item()

    v2 = (
        sps.norm.logpdf(pt[0], loc=2, scale=1)
        + sps.norm.logpdf(pt[1], loc=1, scale=2)
        + np.log(1 / 2)
    )

    assert np.isclose(v1, v2, atol=1e-5)

    pt = np.array([2, 1, 3])
    v1 = prior.log_prob(np.atleast_2d(pt)).item()
    assert np.isneginf(v1)


def test_transform_to_unconstrained_space():
    task = GeneralizedExtremeValue()
    prior_samples, _ = task.prior._sample_prior_predictive(10)

    prior_samples_transformed = task.transform_to_unconstrained_space(
        prior_samples, in_place=False
    )
    assert np.allclose(
        task.transform_to_constrained_space(prior_samples_transformed),
        prior_samples,
    ), "Transformed samples not transformed back correctly"

    values = ndarray_values_as_dict(prior_samples, task.var_dims)
    transformed_values = ndarray_values_as_dict(
        prior_samples_transformed, task.var_dims
    )
    for var_name in values.keys():
        if var_name == "tau":
            assert np.allclose(
                np.log(values[var_name]), transformed_values[var_name]
            ), f"{var_name} not transformed correctly"
        else:
            assert np.allclose(
                values[var_name], transformed_values[var_name]
            ), f"{var_name} not transformed incorrectly"

    task.transform_to_unconstrained_space(prior_samples, in_place=True)
    assert np.allclose(prior_samples, prior_samples_transformed), (
        "Transformed in place not working correctly"
    )


def test_gev_inf_likelihood():
    """A test to check if the log likelihood values are inf while log densities are not. Also serve as a reference for the usage of some functions."""
    task = GeneralizedExtremeValue()
    # Generate some simulated data observations
    N_simulated = 100
    parameter_samples_constrained = task.sample(
        N_simulated, unconstrained=False
    )["parameters"]
    parameter_samples_constrained_dict = ndarray_values_as_dict(
        parameter_samples_constrained, task.var_dims
    )
    dataset = az.convert_to_inference_data(parameter_samples_constrained_dict)
    simulated_obs = pm.sample_posterior_predictive(dataset, task.pymc_model)
    test_datasets = np.array(simulated_obs.posterior_predictive.gev)[0]

    # Get the logp function for the task, parsing a specific observation
    test_data_obs = test_datasets[0]
    lp_fn = get_task_logp_func(task, observation=test_data_obs)

    # Evaluate the logp function on some new parameter samples
    p_constrained_new = task.sample(N_simulated, unconstrained=False)[
        "parameters"
    ]
    p_unconstrained_new = task.transform_to_unconstrained_space(
        p_constrained_new
    )
    values = lp_fn(p_unconstrained_new)
    ind = np.nanargmin(np.where(np.isinf(values), np.nan, values))
    print(f"Number of inf log densities: {np.isinf(values).sum()}")
    print(f"Min log density (except inf): {values[ind]}")

    p_constrained_new_dict = ndarray_values_as_dict(
        p_constrained_new, task.var_dims
    )
    dataset = az.convert_to_inference_data(p_constrained_new_dict)
    pymc_model = task.setup_pymc_model(observation=test_data_obs)
    compute_log_likelihood(dataset, model=pymc_model)
    compute_log_prior(dataset, model=pymc_model)
    log_likelihood_elem = dataset.log_likelihood.gev[0].values
    log_likelihood = log_likelihood_elem.sum(1)
    wrong_inds = np.logical_and(np.isinf(log_likelihood), (~np.isinf(values)))
    if wrong_inds.sum() > 0:
        raise ValueError(
            "For some parameters, the log likelihood values are inf while log densities are not! Something must be wrong. Check the implementation."
        )
