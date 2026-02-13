import bayesflow as bf
import keras
import numpy as np
from bayesflow.utils import (
    filter_kwargs,
)


def bf_transform_to_base(parameters, observables, approximator, **kwargs):
    """Function to transform the parameters to the flow base space, conditioned on the observables."""
    assert parameters.shape[0] == observables.shape[0], (
        "The batch size of the parameters and observables must be the same."
    )
    # Reshape the observables to the original dimensions if provided
    if original_dims := kwargs.pop("observable_original_dims"):
        print(
            f"Reshaping the observables to the original dimensions: {original_dims}"
        )
        observables = observables.reshape(
            (observables.shape[0], *original_dims)
        )
    parameters = keras.ops.convert_to_numpy(parameters)
    observables = keras.ops.convert_to_numpy(observables)
    conditions = {
        "parameters": parameters,
        "observables": observables,
    }
    num_samples = 1

    inference_conditions = compute_summary_statistics(
        approximator,
        conditions,
        **kwargs,
    )

    # conditions must always have shape (batch_size, dims)
    batch_size = keras.ops.shape(inference_conditions)[0]
    inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
    inference_conditions = keras.ops.broadcast_to(
        inference_conditions,
        (
            batch_size,
            num_samples,
            *keras.ops.shape(inference_conditions)[2:],
        ),
    )
    batch_shape = (batch_size, num_samples)

    parameters_base = approximator.inference_network(
        parameters[:, None, :],
        conditions=inference_conditions,
        inverse=False,
        density=False,
    )
    if num_samples == 1:
        parameters_base = keras.ops.squeeze(parameters_base, axis=1)
    return parameters_base


def compute_summary_statistics(
    approximator: bf.approximators.ContinuousApproximator,
    conditions: dict,
    return_inference_variables=False,
    **kwargs,
):
    if not isinstance(conditions, dict):
        # If conditions is not a dict, we assume it is a numpy array representing the observables
        conditions = {"observables": conditions}
    conditions = approximator.adapter(
        conditions, strict=False, stage="inference", **kwargs
    )
    # Get the adapter parameters
    inference_variables = conditions.pop("inference_variables", None)
    conditions = keras.tree.map_structure(
        keras.ops.convert_to_tensor, conditions
    )
    summary_variables = conditions.pop("summary_variables", None)
    inference_conditions = conditions.pop("inference_conditions", None)
    if approximator.summary_network is None:
        if summary_variables is not None:
            raise ValueError(
                "Cannot use summary variables without a summary network."
            )
    else:
        if summary_variables is None:
            raise ValueError(
                "Summary variables are required when a summary network is present."
            )

        summary_outputs = approximator.summary_network(
            summary_variables,
            **filter_kwargs(kwargs, approximator.summary_network.call),
        )

        if inference_conditions is None:
            inference_conditions = summary_outputs
        else:
            inference_conditions = keras.ops.concatenate(
                [inference_conditions, summary_outputs], axis=1
            )

    assert inference_conditions is not None, (
        "Inference conditions should not be None."
    )
    assert inference_conditions.ndim == 2, (
        "Inference conditions should have shape (batch_size, dims)."
    )
    if return_inference_variables:
        return inference_conditions, inference_variables
    return inference_conditions


def bf_log_prob_posterior(approximator, parameters, observables, **kwargs):
    """
    Compute the log probability of the parameters given the observables.
    Args:
        parameters: The parameters to evaluate the log probability for. Shape (batch_size, num_samples, parameters_dims) or (num_samples, parameters_dims) if batch_size is one.
        observables: The observables to condition on. Shape (batch_size, observable_dims) or (observable_dims,) if batch_size is one.
    """

    if parameters.ndim == 3:
        batch_size = parameters.shape[0]
    else:
        assert parameters.ndim == 2
        batch_size = 1
        parameters = np.expand_dims(parameters, axis=0)
    n_samples = parameters.shape[1]
    data = {"observables": observables, "inference_variables": parameters}
    inference_conditions, inference_variables = compute_summary_statistics(
        approximator, data, return_inference_variables=True, **kwargs
    )
    assert np.allclose(inference_variables, parameters), (
        "Inference variables should be the same as parameters, i.e., the adapter should not change them."
    )
    assert inference_conditions.shape[0] == batch_size, "Batch size mismatch"
    assert inference_conditions.ndim == 2, (
        f"Expected 2D input, got {inference_conditions.ndim}D"
    )
    inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
    inference_conditions = keras.ops.repeat(
        inference_conditions, n_samples, axis=1
    )
    log_prob = approximator.inference_network.log_prob(
        inference_variables,
        conditions=inference_conditions,
        **filter_kwargs(kwargs, approximator.inference_network.log_prob),
    )
    log_prob = keras.tree.map_structure(keras.ops.convert_to_numpy, log_prob)
    return log_prob
