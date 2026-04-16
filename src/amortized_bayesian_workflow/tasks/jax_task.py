from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .base import TaskMetadata

SimulatorFn = Callable[[Any, int], dict[str, Any]]
LogDensityFn = Callable[[Any, Any], Any]


class JAXTask:
    """Task wrapper for JAX-native simulators and log density functions.

    Users provide:
    - `simulator(key, num_samples) -> {"parameters": ..., "observables": ...}`
    - `log_likelihood(theta, x) -> scalar` (JAX-compatible)
    - `log_prior(theta) -> scalar` (JAX-compatible)
    """

    _SHAPE_INFERENCE_SEED = 0

    def __init__(
        self,
        *,
        simulator: SimulatorFn,
        log_likelihood: LogDensityFn,
        log_prior: LogDensityFn,
        metadata: TaskMetadata | None = None,
    ) -> None:
        self.simulator = simulator
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.metadata = metadata or self._infer_metadata(
            simulator=simulator,
        )

    @staticmethod
    def _infer_metadata(
        *,
        simulator: SimulatorFn,
        name: str = "JAXTask",
        parameter_names: tuple[str, ...] | None = None,
        parameter_dims: dict[str, int] | None = None,
    ) -> TaskMetadata:
        dims = dict(parameter_dims) if parameter_dims is not None else None
        names = tuple(parameter_names) if parameter_names is not None else None
        if dims is None:
            out = simulator(
                jax.random.PRNGKey(JAXTask._SHAPE_INFERENCE_SEED), 1
            )
            theta = np.asarray(out["parameters"])
            if theta.ndim < 2:
                raise ValueError(
                    "Simulator must return 'parameters' with shape (num_samples, ...)."
                )
            inferred = int(theta.reshape(theta.shape[0], -1).shape[1])
            dims = {"theta": inferred}
            names = ("theta",) if names is None else names
        elif names is None:
            names = tuple(dims.keys())
        return TaskMetadata(
            name=name,
            parameter_names=names or tuple(dims.keys()),
            parameter_dims=dims,
        )

    def simulate_prior_predictive(
        self,
        num_samples: int,
        *,
        seed: int,
    ) -> dict[str, np.ndarray]:
        out = self.simulator(jax.random.PRNGKey(seed), int(num_samples))
        if not isinstance(out, dict):
            raise TypeError("simulator must return a dict with array values.")
        result = {key: np.asarray(value) for key, value in out.items()}
        if "parameters" not in result or "observables" not in result:
            raise ValueError(
                "simulator output must contain 'parameters' and 'observables'."
            )
        return result

    def vectorized_log_posterior_fn(self, observation: np.ndarray):
        obs = jnp.asarray(observation)

        def single(theta):
            return self.log_prior(theta) + self.log_likelihood(theta, obs)

        vmapped = jax.jit(jax.vmap(single))

        def wrapped(theta_batch: np.ndarray) -> np.ndarray:
            return np.asarray(vmapped(jnp.asarray(theta_batch)))

        return wrapped

    def single_log_posterior_fn(self, observation: np.ndarray):
        obs = jnp.asarray(observation)

        @jax.jit
        def single(theta):
            return self.log_prior(theta) + self.log_likelihood(theta, obs)

        return single
