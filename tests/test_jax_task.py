from __future__ import annotations

import jax
import numpy as np

from amortized_bayesian_workflow.tasks import JAXTask


def _simulator(key, num_samples: int):
    key_theta, key_noise = jax.random.split(key)
    theta = jax.random.normal(key_theta, shape=(int(num_samples), 1))
    y = theta + jax.random.normal(key_noise, shape=(int(num_samples), 3))
    return {
        "parameters": np.asarray(theta),
        "observables": np.asarray(y),
    }


def _log_prior(theta):
    return -0.5 * np.asarray(theta).sum() * 0.0


def _log_likelihood(theta, observation):
    return -0.5 * np.asarray(observation).sum() * 0.0


def test_jax_task_init_infers_metadata():
    task = JAXTask(
        simulator=_simulator,
        log_likelihood=_log_likelihood,
        log_prior=_log_prior,
    )
    assert task.metadata.name == "JAXTask"
    assert task.metadata.parameter_dims == {"theta": 1}
    assert task.metadata.total_parameter_dim == 1
