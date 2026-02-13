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
import numpy as np
import pymc as pm
import xarray as xr
from pymc.stats import compute_log_prior
from pymc.util import RandomState

from sbi_mcmc.utils.logging import get_logger
from sbi_mcmc.utils.utils import read_from_file, save_to_file

logger = get_logger(__name__)
logger.propagate = False
logging.getLogger("pymc.sampling.forward").setLevel(logging.WARNING)


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


class PriorSupportConstrained:
    """A dummy class for SBI toolkit."""


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
        random_seed: RandomState = None,
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
            samples=N_sims, model=self.pymc_model, random_seed=random_seed
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

    def log_prob(self, params_values):
        """Return the log probability of the parameter values under the prior distribution.

        Parameters
        ----------
        params_values : np.ndarray
            The parameter values for which the log probability should be computed. The shape of the array should be (N, D), where N is the number of parameter values and D is the number of parameters.

        Returns
        -------
        np.ndarray
            The log probability of the parameter values under the prior distribution. For comptabiltiy with the SBI toolkit, the shape of the returned array should be (N, 1), where N is the number of parameter values.
        """
        if self.unconstrained:
            raise NotImplementedError(
                "Computing prior probabilities in the unconstrained space is not implemented."
            )
        if params_values.ndim > 2:
            params_values = params_values.squeeze()
        assert params_values.ndim == 2, (
            "params_values should be 2D (n_samples, n_dims)."
        )

        values_dict = {}
        start = 0
        for _, var_name in enumerate(self.var_names):
            end = start + self.var_dims[var_name]
            values = params_values[:, start:end][
                None, ...
            ]  # (1, n_samples, n_var_dims)
            if values.shape[-1] == 1:
                values = values.squeeze(-1)  # (1, n_samples)
            values_dict[var_name] = values
            start = end
        idata = az.convert_to_inference_data(values_dict)
        compute_log_prior(idata, model=self.pymc_model, progressbar=False)
        _, values = az.sel_utils.xarray_to_ndarray(
            idata.log_prior, var_names=self.var_names
        )
        values = values.T  # (n_samples, n_dims)
        values = np.sum(values, axis=1, keepdims=True)
        return values

    @property
    def support(self):
        if self.unconstrained:
            from torch.distributions import constraints

            return constraints._Real()
        else:
            return PriorSupportConstrained()


def scalar_params_as_xr_array(params_dict):
    params_values = []
    names = []
    for k, v in params_dict.items():
        names.append(k)
        params_values.append(v)
    params_values = np.array(params_values)
    params_xr = xr.DataArray(
        params_values,
        dims=("x"),
        coords={"x": names},
    )
    return params_xr


def ndarray_values_as_dict(params_values: np.ndarray, var_dims: OrderedDict):
    """Convert a 2D numpy array of parameter values to a dictionary of
    parameter values. The shape of the array should be (n_samples, n_dims)."""
    if params_values.ndim == 1:
        params_values = params_values[None, ...]
    assert params_values.ndim == 2
    assert sum(var_dims.values()) == params_values.shape[1]

    params_dict = OrderedDict()
    start = 0
    for var_name, var_dim in var_dims.items():
        end = start + var_dim
        params_dict[var_name] = params_values[:, start:end].squeeze()
        start = end

    return params_dict


def dict_values_as_ndarray(params_dict: dict, var_dims: OrderedDict):
    """Convert a dictionary of parameter values to a 2D numpy array of parameter values."""
    params_values = []
    for var_name, _ in var_dims.items():
        if params_dict[var_name].ndim == 0:
            values = np.atleast_2d(params_dict[var_name])
        elif params_dict[var_name].ndim == 1:
            values = params_dict[var_name][None, :]
        elif params_dict[var_name].ndim == 2:
            values = params_dict[var_name]
        else:
            raise ValueError(
                f"Parameter values should be 0/1/2D. Not {params_dict[var_name].ndim}."
            )
        params_values.append(values)
    params_values = np.concatenate(params_values, axis=1)
    return params_values


def var_name_to_variable(var_name, pymc_model):
    all_vars = (
        pymc_model.unobserved_RVs
        + pymc_model.observed_RVs
        + pymc_model.unobserved_value_vars
    )

    for var in all_vars:
        if var.name == var_name:
            return var


def split_to_list_of_arrays(value, var_dims, squeeze=True):
    value = jnp.atleast_2d(value)
    lv = []
    start = 0
    for _, var_dim in var_dims.items():
        end = start + var_dim
        lv.append(
            value[..., start:end].squeeze()
            if squeeze
            else value[..., start:end]
        )
        start = end
    return lv


def _transform_to_unconstrained_space(
    params_values: np.ndarray,
    var_dims: OrderedDict,
    pymc_model: pm.Model,
    in_place: bool | None = True,
):
    """Transform the constrained parameter values to the unconstrained space."""
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)
    assert params_values.ndim == 2, (
        "params_values should be 2D. (n_samples, n_dims)."
    )
    model_var_names = get_pymc_free_RVs_names(pymc_model)[0]
    assert set(model_var_names) == set(var_dims.keys())
    assert len(model_var_names) == len(var_dims)

    m: pm.Model = pymc_model
    transformed_rvs = []
    ind_dict = {}  # Store the indices of the relevant variables
    for free_rv in m.free_RVs:
        transform = m.rvs_to_transforms.get(free_rv)
        if transform is None:
            transformed_rvs.append(free_rv)
        else:
            transformed_rv = transform.forward(free_rv, *free_rv.owner.inputs)
            transformed_rvs.append(transformed_rv)
        ind_dict[free_rv.name] = len(transformed_rvs) - 1

    fn = m.compile_fn(inputs=m.free_RVs, outs=transformed_rvs)
    N_sims = params_values.shape[0]
    params_values_transformed = []
    for i in range(N_sims):
        # For now, it's a bit tedious to convert the ndarray to a dictionary
        # and then back to an ndarray. We might consider using something like
        # `pymc.blocking.DictToArrayBijection` in the future. Or wait until
        # https://github.com/pymc-devs/pymc/issues/6721 is resolved.
        value_dict = ndarray_values_as_dict(params_values[i], var_dims)
        value_unconstrained_list = fn(value_dict)
        value_unconstrained_dict = {}
        for var_name in var_dims.keys():
            value_unconstrained_dict[var_name] = value_unconstrained_list[
                ind_dict[var_name]
            ]
        value_unconstrained = dict_values_as_ndarray(
            value_unconstrained_dict, var_dims=var_dims
        )
        if in_place:
            params_values[i] = value_unconstrained
        else:
            params_values_transformed.append(value_unconstrained)
    if not in_place:
        params_values_transformed = np.concatenate(
            params_values_transformed, axis=0
        )
        return params_values_transformed
    return params_values


def _transform_to_constrained_space(
    params_values: np.ndarray,
    value_var_dims: OrderedDict,  # The dimensions of the value variables (unconstrainted space)
    var_dims: OrderedDict,  # The dimensions of the constrained variables
    pymc_model: pm.Model,
    in_place: bool | None = True,
):
    """Transform the unconstrained parameter values to the constrained space."""
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)
    assert params_values.ndim == 2, (
        "params_values should be 2D. (n_samples, n_dims)."
    )
    model_value_var_names = get_pymc_free_RVs_names(pymc_model)[1]
    assert (  # noqa: PT018
        set(model_value_var_names) == set(value_var_dims.keys())
        and len(model_value_var_names) == len(value_var_dims)
    ), (
        f"Model's value variable names {model_value_var_names} should be the same as the value_var_dims keys {value_var_dims.keys()}."
    )

    m = pymc_model
    inputs = []
    for name in value_var_dims.keys():
        var = var_name_to_variable(name, m)
        assert var is not None, (
            f"Variable with name {name} not found in the model."
        )
        inputs.append(var)

    # TODO: would it be more efficient to return only the relevant variables?
    # outputs = m.free_RVs  # why is it not working?

    outputs = m.unobserved_value_vars
    fn_inv = m.compile_fn(outs=outputs)
    # Get the indices of the relevant variables
    ind_dict = {}
    for j, value_var in enumerate(outputs):
        if value_var.name in var_dims.keys():
            ind_dict[value_var.name] = j

    assert len(ind_dict) == len(value_var_dims)
    N_sims = params_values.shape[0]
    params_values_transformed = []
    for i in range(N_sims):
        value_dict = ndarray_values_as_dict(params_values[i], value_var_dims)
        value_constrained_list = fn_inv(value_dict)
        value_constrained_dict = {}
        for name in var_dims.keys():
            value_constrained_dict[name] = value_constrained_list[
                ind_dict[name]
            ]
        value_constrained = dict_values_as_ndarray(
            value_constrained_dict, var_dims=var_dims
        )
        if in_place:
            params_values[i] = value_constrained
        else:
            params_values_transformed.append(value_constrained)
    if not in_place:
        params_values_transformed = np.concatenate(
            params_values_transformed, axis=0
        )
        return params_values_transformed
    return params_values


def sort_sequence(list_to_sort: tuple | list) -> tuple | list:
    """Sort a list or tuple and return the sorted list and the indices of the sorted list. Similar to `numpy.argsort`."""
    sort_index = [
        i for i, x in sorted(enumerate(list_to_sort), key=lambda x: x[1])
    ]
    if isinstance(list_to_sort, tuple):
        return tuple([list_to_sort[i] for i in sort_index]), sort_index
    return [list_to_sort[i] for i in sort_index], sort_index


def get_pymc_free_RVs_names(
    pymc_model,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    free_rvs_names = []
    value_rvs_names = []
    transforms = []
    for rv in pymc_model.free_RVs:
        free_rvs_names.append(rv.name)
        value_rvs_names.append(pymc_model.rvs_to_values[rv].name)
        transform = pymc_model.rvs_to_transforms.get(rv)
        transforms.append(transform)
    # Sort the names
    free_rvs_names, sort_index = sort_sequence(free_rvs_names)
    value_rvs_names = [value_rvs_names[i] for i in sort_index]
    transforms = [transforms[i] for i in sort_index]
    # To tuple
    free_rvs_names = tuple(free_rvs_names)
    value_rvs_names = tuple(value_rvs_names)
    var_names_to_transforms = dict(
        zip(free_rvs_names, transforms, strict=True)
    )
    return free_rvs_names, value_rvs_names, var_names_to_transforms


class PyMCTask(ABC):
    def __init__(
        self,
        true_params_dict: dict | None = None,
        var_names=None,
        var_dims=None,
        task_name=None,
    ) -> None:
        self.pymc_model = self.setup_pymc_model()

        # TODO: Ideally, we would like to infer the names and dimensions of the parameters from the model itself. For now, we assume that the names are model.free_RVs.name and dimensions are provided.
        self.var_dims = (
            OrderedDict()
        )  # The dimensions of the parameters, orders are important
        if var_dims is None:
            var_dims = {}

        # Store names of the parameters that needed to be inferred
        self.var_names, self.value_var_names, self.var_names_to_transforms = (
            get_pymc_free_RVs_names(self.pymc_model)
        )
        self.var_names_to_value_var_names = dict(
            zip(self.var_names, self.value_var_names, strict=True)
        )
        if var_names is not None:
            # Ensure that the provided var_names are the same as the model's free_RVs, possibly in a different order
            assert set(self.var_names) == set(var_names), (
                f"var_names should be the same as the model's free_RVs names, but it's {set(self.var_names)} and {set(var_names)}."
            )
            assert len(self.var_names) == len(var_names)

            # Caution: the var_names and value_var_names orders may be different to the default order. Therefore, calling e.g., get_jaxified_graph(inputs=task.pymc_model.value_vars, outputs=task.pymc_model.logp()) will not work. You need to change the order of the inputs to match the order of the value_var_names defined here.
            self.var_names = tuple(var_names)
            self.value_var_names = tuple(
                self.var_names_to_value_var_names[n] for n in self.var_names
            )

        self.value_var_dims = {}
        for i, var_name in enumerate(self.var_names):
            # By default, the parameters are assumed to be scalar
            self.var_dims[var_name] = var_dims.get(var_name, 1)
            # The value parameters are assumed to have the same dimension as the corresponding parameters
            self.value_var_dims[self.value_var_names[i]] = self.var_dims[
                var_name
            ]

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

        self.name = task_name
        if true_params_dict is not None:
            self.true_params_xr = scalar_params_as_xr_array(true_params_dict)
            if self.var_names is not None:
                self.true_params_xr = self.true_params_xr.loc[
                    list(self.var_names)
                ]

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

    def simulate(
        self,
        num_simulations=None,
        return_torch=False,
        unconstrained=True,
        random_seed: RandomState = None,
        **kwargs,
    ):
        logger.info("Simulating from PyMC model...")
        prior_samples, observation_sims = self.prior._sample_prior_predictive(
            num_simulations, random_seed=random_seed
        )

        if unconstrained:
            # for now we have to manually transform the prior samples to unconstrained space, see discussions in https://github.com/pymc-devs/pymc/pull/6309#
            logger.info("Transforming prior samples to unconstrained space.")
            self.transform_to_unconstrained_space(prior_samples, in_place=True)
            self.prior.unconstrained = True  # For SBI toolkit
        assert np.isfinite(prior_samples).all(), (
            "Invalid prior samples. Should be finite."
        )
        assert np.isfinite(observation_sims).all(), (
            "Invalid simulation. Should be finite."
        )
        if return_torch:
            raise NotImplementedError
        return prior_samples, observation_sims

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

    def sample(self, batch_size: int, numpy: bool = False, **kwargs):
        prior_samples, observation_samples = self.simulate(
            batch_size, **kwargs
        )

        return {
            "parameters": prior_samples,
            "observables": observation_samples,
        }

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
