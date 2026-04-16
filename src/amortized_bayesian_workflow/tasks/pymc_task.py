from __future__ import annotations

import logging
import numbers
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc import Normal, Uniform
from pymc.stats import compute_log_prior
from pymc.util import RandomState
from scipy.special import erf

from amortized_bayesian_workflow.utils import read_from_file, save_to_file

from .base import TaskMetadata
from .pymc_utils import (
    _infer_value_var_dims_from_initial_point,
    get_pymc_free_RVs_names,
    get_task_func,
    ndarray_values_as_dict,
)
from .pymc_utils import (
    _transform_to_constrained_space_vmap as _transform_to_constrained_space,
)
from .pymc_utils import (
    _transform_to_unconstrained_space_vmap as _transform_to_unconstrained_space,
)

logger = logging.getLogger(__name__)


class PyMCTask(ABC):
    def __init__(
        self,
        task_name: str = None,
    ) -> None:
        self.pymc_model = self.setup_pymc_model()

        # Store names of the parameters that needed to be inferred
        self.var_names, self.value_var_names, self.var_names_to_transforms = (
            get_pymc_free_RVs_names(self.pymc_model)
        )
        self.var_names_to_value_var_names = dict(
            zip(self.var_names, self.value_var_names, strict=True)
        )
        self.value_var_names_to_var_names = dict(
            zip(self.value_var_names, self.var_names, strict=True)
        )
        self.value_var_dims = _infer_value_var_dims_from_initial_point(
            self.pymc_model, self.value_var_names
        )
        self.var_dims = OrderedDict(
            (self.value_var_names_to_var_names[vv], self.value_var_dims[vv])
            for vv in self.value_var_names
        )
        # All parameters should have a dimension specified
        assert set(self.var_names).issubset(set(self.var_dims.keys())), (
            f"var_dims should contain dimensions for all parameters. "
            f"var_dims: {self.var_dims.keys()}, var_names: {self.var_names}"
        )
        self.D = sum(self.var_dims.values())
        # Create a variable info object for convenience
        self.var_info = PyMCTaskVariableInfo(
            self.var_names,
            self.value_var_names,
            self.var_dims,
            self.value_var_dims,
            self.var_names_to_value_var_names,
            self.var_names_to_transforms,
        )

        self.task_name = task_name or self.__class__.__name__

    @property
    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            name=self.task_name,
            parameter_names=tuple(self.var_names),
            parameter_dims=dict(self.var_dims),
        )

    @property
    def prior(self):
        return PyMCPriorWrapper(self.pymc_model, self.var_info)

    @abstractmethod
    def setup_pymc_model(self) -> pm.Model: ...

    def run_pymc_sampler(
        self, draws=1000, load_from_file=True, save_result=True, pm_kwargs=None
    ):
        if pm_kwargs is None:
            pm_kwargs = {}
        file_path = (
            Path(__file__).parent.parent
            / f"task_results/{self.name}_pymc_samples.pkl"
        )
        if load_from_file:
            try:
                idata_post = read_from_file(file_path)
                logger.info(f"Loaded PyMC samples from file {file_path}.")
                return idata_post
            except FileNotFoundError:
                logger.info(f"File {file_path} not found.")

        logger.info("Sampling from PyMC model.")
        idata_post = pm.sample(draws, model=self.pymc_model, **pm_kwargs)
        if save_result:
            save_to_file(idata_post, file_path)
        return idata_post

    def simulate_observation(
        self, param_values: np.ndarray, unconstrained=True, **kwargs
    ):
        """Simulate the observation given the parameter values. The parameter values can be in either constrained or unconstrained space. The function will transform the parameter values to the constrained space if needed."""
        if unconstrained:
            # Transform the parameter values to the constrained space
            param_values = self.transform_to_constrained_space(
                param_values, in_place=False
            )
        param_dict = ndarray_values_as_dict(param_values, self.var_dims)
        for p, v in param_dict.items():
            param_dict[p] = v[None, ...]  # add a fake "chain" dimension
            # print(f"{p}: {param_dict[p].shape}")
        # Create InferenceData object from the parameter values dict
        idata_post = az.convert_to_inference_data(param_dict)
        idata_post = pm.sample_posterior_predictive(
            idata_post,
            model=self.pymc_model,
            progressbar=kwargs.get("progressbar", False),
        )

        _, observation_sims = az.sel_utils.xarray_to_ndarray(
            idata_post.posterior_predictive
        )
        observation_sims = observation_sims.T
        return {"observables": observation_sims}

    def simulate_prior_predictive(
        self,
        batch_size: int,
        unconstrained=True,
        seed: RandomState = None,
        **kwargs,
    ):
        logger.info("Simulating from PyMC model...")
        prior_samples, observation_sims = self.prior._sample_prior_predictive(
            batch_size, seed=seed
        )

        if unconstrained:
            # for now we have to manually transform the prior samples to unconstrained space, see discussions in https://github.com/pymc-devs/pymc/pull/6309#
            logger.info("Transforming prior samples to unconstrained space.")
            self.transform_to_unconstrained_space(prior_samples, in_place=True)
        assert np.isfinite(prior_samples).all(), (
            "Invalid prior samples. Should be finite."
        )
        assert np.isfinite(observation_sims).all(), (
            "Invalid simulation. Should be finite."
        )

        return {
            "parameters": prior_samples,
            "observables": observation_sims,
        }

    def transform_to_constrained_space(
        self, value_var_values: np.ndarray, in_place=False
    ) -> np.ndarray:
        """Transform the unconstrained value variable values to the constrained space."""
        return _transform_to_constrained_space(
            value_var_values,
            self.value_var_dims,
            self.var_dims,
            self.pymc_model,
            in_place=in_place,
        )

    def transform_to_unconstrained_space(
        self, var_values: np.ndarray, in_place=False
    ) -> np.ndarray:
        """Transform the constrained parameter values to the unconstrained space (PyMC's value variables)."""
        return _transform_to_unconstrained_space(
            var_values, self.var_dims, self.pymc_model, in_place=in_place
        )

    def vectorized_log_posterior_fn(self, observation: np.ndarray):
        """Return a batched log-posterior function for the given observation. Batched here means that the function can take in a batch of parameter values and return a batch of log-posterior values."""
        return get_task_func(
            task=self,
            func_type="posterior",
            observation=observation,
            static=True,
            vmap=True,
        )

    def single_log_posterior_fn(self, observation: np.ndarray):
        """Return a single-sample log-posterior function for the given observation."""
        return get_task_func(
            task=self,
            func_type="posterior",
            observation=observation,
            static=True,
            vmap=False,
        )

    def axis_names_for_corner_plot(self, value_var=True):
        if value_var:
            var_dims = self.value_var_dims
        else:
            var_dims = self.var_dims
        axis_names = []
        for var_name, var_dim in var_dims.items():
            if var_dim == 1:
                axis_names.append(var_name)
            else:
                axis_names.extend([f"{var_name}_{i}" for i in range(var_dim)])
        return axis_names


@dataclass(frozen=True)
class PyMCTaskVariableInfo:
    var_names: tuple[str, ...]
    value_var_names: tuple[str, ...]
    var_dims: OrderedDict[str, int]
    value_var_dims: OrderedDict[str, int]
    var_names_to_value_var_names: dict[str, str]
    var_names_to_transforms: dict[str, pm.logprob.Transform]

    @cached_property
    def OrderedTransform_var_names(self):
        """Get the variable names that are associated with pymc.distributions.transforms.Ordered transform."""
        var_names = []
        for var_name, transform in self.var_names_to_transforms.items():
            if isinstance(transform, pm.distributions.transforms.Ordered):
                var_names.append(var_name)
        return var_names

    @cached_property
    def var_start_end_indices(self):
        return self._get_start_end_indices(self.var_dims)

    @cached_property
    def value_var_start_end_indices(self):
        return self._get_start_end_indices(self.value_var_dims)

    @staticmethod
    def _get_start_end_indices(var_dims: OrderedDict[str, int]):
        indices = {}
        start = 0
        for var_name, var_dim in var_dims.items():
            end = start + var_dim
            indices[var_name] = (start, end)
            start = end
        return indices

    @cached_property
    def var_names_flatten(self):
        _var_names_flatten = []
        for var_name, dim in self.var_dims.items():
            _var_names_flatten.extend(
                [f"$\\text{{{var_name}}}$"]
                if dim == 1
                else [
                    f"$\\text{{{var_name}}}_{{{i + 1}}}$" for i in range(dim)
                ]
            )
        return tuple(_var_names_flatten)


class PyMCPriorWrapper:
    def __init__(self, pymc_model: pm.Model, var_info: PyMCTaskVariableInfo):
        self.pymc_model = pymc_model

        self.var_info = var_info
        self.var_names = var_info.var_names
        self.var_dims = var_info.var_dims

        self.unconstrained = False  # Prior is assumed to be defined on the constrained space by default.

    def sample(self, sample_shape=None):
        prior_samples, observation_sims = self._sample_prior_predictive(
            sample_shape
        )
        if self.unconstrained:
            # Transform to unconstrained space
            _transform_to_unconstrained_space(
                prior_samples, self.var_dims, self.pymc_model, in_place=True
            )
        return prior_samples

    def _sample_prior_predictive(
        self,
        sample_shape: tuple | int | None = None,
        seed: RandomState = None,
    ):
        """Sample from the prior predictive distribution. This is used to
        generate prior samples and the corresponding synthetic simulations.
        In PyMC, `Transforms` are not applied during forward sampling
        and are only applied when sampling unobserved random variables with
        `pymc.sample()` (see https://www.pymc.io/projects/docs/en/latest/api/distributions/transforms.html). Therefore, the prior parameter samples
        and the simulated observations are all in the constrained space when
        using `pm.sample_prior_predictive`.

        Parameters
        ----------
        sample_shape : tuple or int, optional
            The shape of the samples. If None, the shape is (1,).
        """

        if sample_shape is None:
            sample_shape = (1,)
        if isinstance(sample_shape, numbers.Number):
            sample_shape = (sample_shape,)
        if len(sample_shape) == 0:
            sample_shape = (1,)
        assert len(sample_shape) == 1
        N_sims = sample_shape[0]
        idata_prior = pm.sample_prior_predictive(
            samples=N_sims, model=self.pymc_model, random_seed=seed
        )
        _, observation_sims = az.sel_utils.xarray_to_ndarray(
            idata_prior.prior_predictive
        )
        _, prior_samples = az.sel_utils.xarray_to_ndarray(
            idata_prior.prior, var_names=self.var_names
        )
        prior_samples = prior_samples.T
        observation_sims = observation_sims.T
        # We need to be careful here with the prior parameter samples from `sample_prior_predictive`. For example, if a parameter are specified to be transformed by `pm.distributions.transforms.Ordered`, we need to ensure that the its values are sorted in correct order. Otherwise, when transforming the prior samples to the unconstrained space, the transformed values might be invalid (nan). TODO: Possbily other transforms might need to be considered. We will see if this is a problem in practice.
        for var_name in self.var_info.OrderedTransform_var_names:
            start, end = self.var_info.var_start_end_indices[var_name]
            prior_samples[:, start:end] = np.sort(
                prior_samples[:, start:end], axis=1
            )
        return prior_samples, observation_sims
