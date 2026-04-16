from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
from numpy.typing import ArrayLike
from pymc import Model
from pymc.sampling.jax import get_jaxified_graph  # Assuming this can be used
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import (
    graph_inputs,  # Import graph_inputs utility
)
from pytensor.tensor.type import TensorType  # Import TensorType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pymc_task import PyMCTask


def get_jaxified_func(
    model: Model,
    func_type: str = "posterior",
    negative: bool = False,
    value_var_names=None,
) -> Callable[[ArrayLike], jax.Array]:
    """Get a JAX-compiled function for computing log-densities from a PyMC model.

    This function creates a JAX-compiled version of the specified density function
    (posterior, likelihood, or prior) from a PyMC model. The returned function
    operates on parameters in the unconstrained transformation space.

    Args:
        model: The PyMC model instance to compile.
        func_type: Type of density function to create. One of:
            - "posterior": Full log-probability including priors and likelihood
            - "likelihood": Log-likelihood only (observed variables)
            - "prior": Log-prior only (free variables)
        negative: If True, returns negative log-density values (useful for optimization).
            If False, returns standard log-density values. Defaults to False.
        value_var_names: Optional list specifying the order of value variables in the
            input arrays. If None, uses the default order from model.value_vars.
            Must contain the same variable names as model.value_vars.

    Returns:
        A JAX-compiled function with signature f(x: ArrayLike) -> jax.Array where:
        - x: Array of parameters in unconstrained space, shape (num_parameters,)
        - Returns: Scalar log-density value corresponding to func_type

    Note:
        Parameters should be provided in the unconstrained transformation space
        as used internally by PyMC. Potential terms in the model may affect
        the correctness of the computed densities and should not be included in the PyMC model.

    Example:
        >>> model = pm.Model()
        >>> # ... define model ...
        >>> posterior_fn = get_jaxified_func(model, "posterior", negative=False)
        >>> log_prob = posterior_fn(param_array)
    """
    func_type = func_type.lower()
    if func_type not in ["posterior", "likelihood", "prior"]:
        raise ValueError(
            f"func_type must be one of 'posterior', 'likelihood', 'prior', got '{func_type}'"
        )
    if func_type == "posterior":
        graph = model.logp()
    elif func_type == "likelihood":
        graph = model.logp(vars=model.observed_RVs)
    elif func_type == "prior":
        graph = model.logp(vars=model.free_RVs)

    if negative:
        graph = -graph

    inputs = model.value_vars
    value_var_names_default = [v.name for v in model.value_vars]
    if value_var_names is not None:
        assert set(value_var_names) == set(value_var_names_default), (
            "value_var_names must be the same as the default value variable names"
        )
        # Create mapping from name to index in default order
        name_to_idx = {
            name: i for i, name in enumerate(value_var_names_default)
        }
        # Reorder inputs based on value_var_names order
        inputs = [
            model.value_vars[name_to_idx[name]] for name in value_var_names
        ]
    logp_fn = get_jaxified_graph(inputs=inputs, outputs=[graph])

    def logp_fn_wrap(x: ArrayLike) -> jax.Array:
        return logp_fn(*x)[0]

    return logp_fn_wrap


def get_dynamic_jaxified_func(
    model: Model,
    func_type: str = "posterior",
    negative: bool = False,
    value_var_names=None,
) -> Callable[[ArrayLike, ArrayLike], jax.Array]:
    """Get a JAX-compiled function for computing log-densities with dynamic observations.

    This function creates a JAX-compiled version of the specified density function
    (posterior, likelihood, or prior) from a PyMC model, allowing for dynamic
    observation inputs. The returned function operates on parameters in the
    unconstrained transformation space.

    Args:
        model: The PyMC model instance to compile.
        func_type: Type of density function to create. One of:
            - "posterior": Full log-probability including priors and likelihood
            - "likelihood": Log-likelihood only (observed variables)
            - "prior": Log-prior only (free variables)
        negative: If True, returns negative log-density values (useful for optimization).
            If False, returns standard log-density values. Defaults to False.
        value_var_names: Optional list specifying the order of value variables in the
            input arrays. If None, uses the default order from model.value_vars.
            Must contain the same variable names as model.value_vars.

    Returns:
        A JAX-compiled function with signature f(x: ArrayLike, obs_values: ArrayLike) -> jax.Array where:
        - x: Array of parameters in unconstrained space, shape (num_parameters,)
        - obs_values: Array of observation values corresponding to model data variables
        - Returns: Scalar log-density value corresponding to func_type

    Note:
        Parameters should be provided in the unconstrained transformation space
        as used internally by PyMC. Potential terms in the model may affect
        the correctness of the computed densities and should not be included in the PyMC model.
    Example:
        >>> model = pm.Model()
        >>> # ... define model ...
        >>> posterior_fn = get_dynamic_jaxified_func(model, "posterior", negative=False)
        >>> log_prob = posterior_fn(param_array, obs_array)
    """
    func_type = func_type.lower()
    if func_type not in ["posterior", "likelihood", "prior"]:
        raise ValueError(
            f"func_type must be one of 'posterior', 'likelihood', 'prior', got '{func_type}'"
        )

    if func_type == "prior":
        # For prior, obs_values are not needed. We use the static function instead.
        return get_jaxified_func(
            model,
            func_type=func_type,
            negative=negative,
            value_var_names=value_var_names,
        )
    elif func_type == "posterior":
        graph = model.logp()
    elif func_type == "likelihood":
        graph = model.logp(vars=model.observed_RVs)

    if negative:
        graph = -graph

    # Handle parameter variable order
    param_vars_default = model.value_vars
    value_var_names_default = [v.name for v in model.value_vars]
    if value_var_names is not None:
        if set(value_var_names) != set(value_var_names_default):
            raise ValueError(
                "Provided value_var_names must contain the same names as "
                f"model.value_vars: {value_var_names_default}"
            )
        # Create mapping from name to index in default order
        name_to_idx = {
            name: i for i, name in enumerate(value_var_names_default)
        }
        # Reorder parameter variables based on value_var_names order
        param_vars = [
            param_vars_default[name_to_idx[name]] for name in value_var_names
        ]
        param_order_for_doc = value_var_names
    else:
        param_vars = param_vars_default
        param_order_for_doc = value_var_names_default
    # --- This is the tricky part: getting original data nodes reliably ---
    try:
        # Get the ordered list of SharedVariables associated with pm.Data containers
        target_data_order = list(model.data_vars)
        data_vars_in_model_set = set(target_data_order)

        if not data_vars_in_model_set:
            raise ValueError("Model does not contain any pm.Data variables.")

        # Find all inputs to the graph expression
        logp_graph_ins = graph_inputs([graph])

        # Identify which of these inputs are our pm.Data variables
        found_data_nodes = [
            node for node in logp_graph_ins if node in data_vars_in_model_set
        ]

        # Sort the found nodes based on their original index in model.data_vars
        # to ensure a predictable order for the user.
        original_data_nodes = sorted(
            found_data_nodes, key=lambda node: target_data_order.index(node)
        )
        data_order_for_doc = [node.name for node in original_data_nodes]

        if not original_data_nodes:
            raise ValueError(
                "Could not identify any pm.Data variables as inputs to the "
                "model's logp graph. Ensure pm.Data variables are used "
                "in the 'observed' argument of your distributions."
            )
        # Optional: Print the order for the user
        # print("Detected data input order:", data_order_for_doc)

    except Exception as e:
        # Catch specific ValueErrors above or other unexpected errors
        if isinstance(e, (ValueError, RuntimeError)):
            raise e  # Re-raise specific errors
        # Wrap other exceptions
        raise RuntimeError(
            "Failed to automatically identify observation data nodes. "
            "This process is complex and model-dependent."
        ) from e
    # --- End tricky part ---

    # Create placeholders using the TensorType instance as a factory
    placeholder_data_vars = []
    for i, node in enumerate(original_data_nodes):
        # Create TensorType first using dtype and shape from the original node's type
        tensor_type = TensorType(
            dtype=node.dtype, shape=node.type.broadcastable
        )
        # Use the type instance itself to create the variable
        placeholder = tensor_type(name=f"data_in_{i}")
        placeholder_data_vars.append(placeholder)

    # Create replacement map
    replacements = {
        orig: ph
        for orig, ph in zip(
            original_data_nodes, placeholder_data_vars, strict=False
        )
    }

    # Clone the logp graph expression
    cloned_logp = clone_replace(
        graph, replace=replacements, rebuild_strict=False
    )

    # Define inputs for JAXification: parameters (in specified order) + data placeholders
    all_inputs = list(param_vars) + placeholder_data_vars

    # JAXify the graph with parameters and data placeholders as inputs
    raw_jax_fn = get_jaxified_graph(inputs=all_inputs, outputs=[cloned_logp])

    # Create the final wrapper
    num_params = len(param_vars)

    def dynamic_logp_fn(parameters, observations):
        """
        JAX-compiled log-probability function.

        Args:
            parameters: A sequence/tuple of JAX arrays for model parameters,
                        matching the order: {param_order_for_doc}.
            observations: A sequence/tuple of JAX arrays for observed data,
                          matching the order identified internally: {data_order_for_doc}.
        Returns:
            Log-probability as a JAX scalar array.
        """
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(
                f"Parameters must be a list or tuple, got {type(parameters)}"
            )
        if len(parameters) != num_params:
            raise ValueError(
                f"Expected {num_params} parameter arrays (matching the order {param_order_for_doc}), got {len(parameters)}"
            )
        parameters_jax = tuple(jax.numpy.asarray(p) for p in parameters)

        if not isinstance(observations, (list, tuple)):
            observations = (observations,)  # Allow single observation input
        if len(observations) != len(placeholder_data_vars):
            raise ValueError(
                f"Expected {len(placeholder_data_vars)} observation arrays (matching the order {data_order_for_doc}), got {len(observations)}"
            )
        # Ensure observations are JAX arrays
        observations_jax = tuple(
            jax.numpy.asarray(obs) for obs in observations
        )
        # Ensure parameters are also JAX arrays if they aren't already
        combined_inputs = parameters_jax + observations_jax

        return raw_jax_fn(*combined_inputs)[0]

    return dynamic_logp_fn


def get_task_func(
    task: PyMCTask,
    func_type: str = "posterior",
    observation=None,
    static=True,
    pymc_model=None,
    vmap=True,
):
    """Get a density function (log-likelihood, log-prior, or log-posterior) for a PyMC task.

    Args:
        task: The PyMC task instance
        func_type: One of "posterior", "likelihood", "prior"
        observation: Observation data (used when static=True or func_type="posterior")
        static: Whether observation is baked into the function or provided dynamically
        pymc_model: Pre-built PyMC model (required when static=False)
        vmap: Whether to vectorize the function for batch inputs

    Returns:
        Compiled JAX function with signature:
        - static=True: f(theta) -> density
        - static=False: f(theta, obs_values) -> density (except prior ignores obs_values)
    """
    func_type = func_type.lower()
    if func_type not in ["posterior", "likelihood", "prior"]:
        raise ValueError(
            f"func_type must be one of 'posterior', 'likelihood', 'prior', got '{func_type}'"
        )

    if static:
        if observation is None:
            pymc_model = task.setup_pymc_model()
        else:
            pymc_model = task.setup_pymc_model(observation=observation)
        logdensity_fn = get_jaxified_func(
            pymc_model,
            func_type=func_type,
            value_var_names=task.value_var_names,
        )

        def lp_fn(x: jnp.ndarray, value_var_dims=task.value_var_dims):
            return logdensity_fn(split_to_list_of_arrays(x, value_var_dims))

        if vmap:
            lp_fn = jax.vmap(lp_fn)
    else:
        if pymc_model is None:
            raise ValueError(
                "pymc_model must be provided when static is False."
            )
        logdensity_fn = get_dynamic_jaxified_func(
            pymc_model,
            func_type=func_type,
            value_var_names=task.value_var_names,
        )

        def lp_fn(
            x: jnp.ndarray,
            obs_values: jnp.ndarray = None,
            value_var_dims=task.value_var_dims,
        ):
            param_values = split_to_list_of_arrays(x, value_var_dims)

            if func_type == "prior":
                # For prior, obs_values is ignored
                return logdensity_fn(param_values)
            # Call the logdensity function with parameters and observations
            return logdensity_fn(param_values, obs_values)

        if func_type == "prior":
            in_axes = (0,)
        else:
            in_axes = (0, None)
        if vmap:
            lp_fn = jax.vmap(lp_fn, in_axes=in_axes)
    return lp_fn


def scalar_params_as_xr_array(params_dict):
    import xarray as xr

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


def _transform_to_unconstrained_space_vmap(
    params_values: np.ndarray,
    var_dims: OrderedDict,
    pymc_model: pm.Model,
    in_place: bool | None = True,
) -> np.ndarray:
    """Transform constrained parameter values to the unconstrained space using JAX vmap.

    This is a vectorized (batched) alternative to :func:`_transform_to_unconstrained_space`
    that avoids the Python-level loop over samples.  Instead of compiling a
    Pytensor function and calling it once per sample, it

    1. Builds the forward-transform Pytensor graph with fresh ``TensorType``
       placeholders (one per free RV) so that special random-variable nodes are
       never used directly as graph inputs.
    2. JAXifies that graph with :func:`get_jaxified_graph`.
    3. Applies :func:`jax.vmap` over the leading (sample) axis.

    Caution
    -------
    A new JAX-compiled function is built on every call.  If you need to call
    this repeatedly for the *same* model/var_dims, consider caching the
    compiled vmapped function externally.

    Parameters
    ----------
    params_values:
        2-D array of shape ``(n_samples, n_dims)`` in the **constrained** space.
    var_dims:
        Ordered mapping from constrained variable name to its dimension size.
    pymc_model:
        The PyMC model that defines the transforms.
    in_place:
        If ``True`` (default) the result overwrites *params_values* in place.
        If ``False`` a new array is returned.
    """
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)
    assert params_values.ndim == 2, (
        "params_values should be 2D (n_samples, n_dims)."
    )

    model_var_names = get_pymc_free_RVs_names(pymc_model)[0]
    assert set(model_var_names) == set(var_dims.keys())
    assert len(model_var_names) == len(var_dims)

    m = pymc_model

    # Build the forward-transform graph using fresh TensorVariable placeholders
    # instead of the special RV nodes, so get_jaxified_graph can treat them as
    # free inputs without confusion.
    placeholders = []
    transformed_list = []
    ind_dict: dict[str, int] = {}
    for free_rv in m.free_RVs:
        tensor_type = TensorType(
            dtype=free_rv.dtype, shape=free_rv.type.broadcastable
        )
        placeholder = tensor_type(name=free_rv.name + "_ph")
        placeholders.append(placeholder)

        transform = m.rvs_to_transforms.get(free_rv)
        if transform is None:
            transformed_list.append(placeholder)
        else:
            transformed_list.append(
                transform.forward(placeholder, *free_rv.owner.inputs)
            )
        ind_dict[free_rv.name] = len(transformed_list) - 1

    fn_jax = get_jaxified_graph(inputs=placeholders, outputs=transformed_list)

    # Order of m.free_RVs decides the position of each placeholder.
    free_rv_order = [rv.name for rv in m.free_RVs]

    # Split (N_sims, total_dim) → per-variable batched arrays in free_rv order.
    # Scalar vars (var_dim == 1): shape (N_sims, 1) → (N_sims,) so vmap gets ()
    # per sample, matching the scalar pytensor type.
    # Vector vars (var_dim > 1): shape (N_sims, var_dim) already correct.
    per_var_batched: dict[str, np.ndarray] = {}
    start = 0
    for var_name, var_dim in var_dims.items():
        arr = params_values[:, start : start + var_dim]
        if var_dim == 1:
            arr = arr[:, 0]  # (N_sims,)
        per_var_batched[var_name] = arr
        start += var_dim

    batched_inputs = [per_var_batched[name] for name in free_rv_order]

    vmapped_fn = jax.vmap(fn_jax)
    results = vmapped_fn(
        *batched_inputs
    )  # tuple: each element (N_sims, ...) or (N_sims,)

    # Reconstruct output ordered by var_dims.
    out_parts = []
    for var_name in var_dims.keys():
        arr = np.asarray(results[ind_dict[var_name]])
        if arr.ndim == 1:
            arr = arr[:, None]  # (N_sims, 1) for scalars
        out_parts.append(arr)
    result = np.concatenate(out_parts, axis=1)  # (N_sims, total_dim)

    if in_place:
        params_values[:] = result
        return params_values
    return result


def _transform_to_constrained_space_vmap(
    params_values: np.ndarray,
    value_var_dims: OrderedDict,  # dimensions in the unconstrained (value-var) space
    var_dims: OrderedDict,  # dimensions in the constrained (free-RV) space
    pymc_model: pm.Model,
    in_place: bool | None = True,
) -> np.ndarray:
    """Transform unconstrained (value-variable) parameters to the constrained space using JAX vmap.

    This is the vectorized counterpart of :func:`_transform_to_constrained_space`.
    Instead of a Python loop it

    1. Builds the backward-transform Pytensor graph using ``m.rvs_to_values``
       TensorVariables as symbolic inputs (these are regular ``TensorVariable``
       nodes, safe to use with :func:`get_jaxified_graph`).
    2. JAXifies that graph.
    3. Applies :func:`jax.vmap` over the leading sample axis.

    Parameters
    ----------
    params_values:
        2-D array of shape ``(n_samples, n_dims)`` in the **unconstrained**
        (value-variable) space.
    value_var_dims:
        Ordered mapping from value-variable name to its dimension size
        (unconstrained space).
    var_dims:
        Ordered mapping from constrained variable name to its dimension size.
    pymc_model:
        The PyMC model that defines the transforms.
    in_place:
        If ``True`` (default) the result overwrites *params_values* in place.
        If ``False`` a new array is returned.
    """
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)
    assert params_values.ndim == 2, (
        "params_values should be 2D (n_samples, n_dims)."
    )

    model_value_var_names = get_pymc_free_RVs_names(pymc_model)[1]
    assert (  # noqa: PT018
        set(model_value_var_names) == set(value_var_dims.keys())
        and len(model_value_var_names) == len(value_var_dims)
    ), (
        f"Model's value variable names {model_value_var_names} should be the same as "
        f"the value_var_dims keys {value_var_dims.keys()}."
    )

    m = pymc_model

    # Build helper look-ups indexed by name.
    name_to_value_var = {
        m.rvs_to_values[rv].name: m.rvs_to_values[rv] for rv in m.free_RVs
    }
    name_to_free_rv = {rv.name: rv for rv in m.free_RVs}

    # Ordered inputs for get_jaxified_graph: value vars in value_var_dims order.
    ordered_value_vars = [
        name_to_value_var[name] for name in value_var_dims.keys()
    ]

    # Build backward-transform graph in var_dims order (constrained names).
    constrained_list = []
    for free_rv_name in var_dims.keys():
        free_rv = name_to_free_rv[free_rv_name]
        value_var = m.rvs_to_values[free_rv]
        transform = m.rvs_to_transforms.get(free_rv)
        if transform is None:
            constrained_list.append(value_var)
        else:
            constrained_list.append(
                transform.backward(value_var, *free_rv.owner.inputs)
            )

    fn_jax_inv = get_jaxified_graph(
        inputs=ordered_value_vars, outputs=constrained_list
    )

    # Split (N_sims, total_dim) → per-value-var batched arrays in value_var_dims order.
    batched_inputs = []
    start = 0
    for var_dim in value_var_dims.values():
        arr = params_values[:, start : start + var_dim]
        if var_dim == 1:
            arr = arr[:, 0]  # (N_sims,) for scalar value vars
        batched_inputs.append(arr)
        start += var_dim

    vmapped_fn = jax.vmap(fn_jax_inv)
    results = vmapped_fn(
        *batched_inputs
    )  # tuple: each (N_sims, ...) or (N_sims,)

    # Reconstruct output in var_dims order.
    out_parts = []
    for idx in range(len(var_dims)):
        arr = np.asarray(results[idx])
        if arr.ndim == 1:
            arr = arr[:, None]  # (N_sims, 1)
        out_parts.append(arr)
    result = np.concatenate(out_parts, axis=1)

    if in_place:
        params_values[:] = result
        return params_values
    return result


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


def _infer_value_var_dims_from_initial_point(
    model, value_var_names: tuple[str, ...]
) -> OrderedDict[str, int]:
    init_point = model.initial_point()
    dims: OrderedDict[str, int] = OrderedDict()
    for name in value_var_names:
        if name not in init_point:
            raise ValueError(
                f"Value variable {name!r} missing from model.initial_point()."
            )
        value = np.asarray(init_point[name])
        dims[name] = int(value.size if value.size > 0 else 1)
    return dims
