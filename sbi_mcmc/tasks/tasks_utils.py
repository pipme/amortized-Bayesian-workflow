from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from pymc import Model
from pymc.sampling.jax import get_jaxified_graph  # Assuming this can be used
from pytensor.graph.basic import graph_inputs  # Import graph_inputs utility
from pytensor.graph.replace import clone_replace
from pytensor.tensor.type import TensorType  # Import TensorType

from .tasks import PyMCTask, split_to_list_of_arrays


def get_task_logp_func(
    task: PyMCTask, observation=None, static=True, pymc_model=None, vmap=True
):
    """Get a logp function for a PyMC task. The input to the logp function is an array of shape (num_samples, num_parameters). The input array represents parameter values in unconstrained space."""
    if static:
        if observation is None:
            pymc_model = task.setup_pymc_model()
        else:
            pymc_model = task.setup_pymc_model(observation=observation)
        logdensity_fn = get_jaxified_logp(
            pymc_model,
            negative_logp=True,
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
        logdensity_fn = get_dynamic_jax_logp(
            pymc_model,
            negative_logp=True,
            value_var_names=task.value_var_names,
        )

        def lp_fn(
            x: jnp.ndarray,
            obs_values: jnp.ndarray,
            value_var_dims=task.value_var_dims,
        ):
            param_values = split_to_list_of_arrays(x, value_var_dims)
            # Call the logdensity function with parameters and observations
            return logdensity_fn(param_values, obs_values)

        if vmap:
            lp_fn = jax.vmap(lp_fn, in_axes=(0, None))
    return lp_fn


def get_jaxified_logp(
    model: Model, negative_logp: bool = True, value_var_names=None
) -> Callable[[ArrayLike], jax.Array]:
    model_logp = model.logp()
    if not negative_logp:
        model_logp = -model_logp
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
    logp_fn = get_jaxified_graph(inputs=inputs, outputs=[model_logp])

    def logp_fn_wrap(x: ArrayLike) -> jax.Array:
        return logp_fn(*x)[0]

    return logp_fn_wrap


def get_dynamic_jax_logp(model, negative_logp=True, value_var_names=None):
    """
    Attempts to create a JAX logp function where observations are dynamic inputs.

    NOTE: This requires identifying pm.Data nodes that are inputs to the model's
          logp graph. The order of the 'observations' input tuple must match
          the order in which these nodes appear in model.data_vars.
    Args:
        model: The PyMC model.
        negative_logp: Whether to return the negative log-probability.
        value_var_names: Optional list of value variable names to specify the
                         order of parameters in the input tuple. If None, the
                         default order from model.value_vars is used.
    """
    model_logp_graph = model.logp()
    if not negative_logp:
        model_logp_graph = -model_logp_graph

    # Handle parameter variable order
    param_vars_default = model.value_vars
    value_var_names_default = [v.name for v in param_vars_default]

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

        # Find all inputs to the logp graph expression
        logp_graph_ins = graph_inputs([model_logp_graph])

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
        model_logp_graph, replace=replacements, rebuild_strict=False
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
        parameters_jax = tuple(jax.numpy.asarray(p) for p in parameters)
        combined_inputs = parameters_jax + observations_jax
        return raw_jax_fn(*combined_inputs)[0]

    return dynamic_logp_fn
