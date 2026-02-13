"""PyMC implementation of the Bernoulli GLM task based on sbibm."""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from sbi_mcmc.tasks.tasks import PyMCTask
from sbi_mcmc.utils.logging import get_logger

logger = get_logger(__name__)


class BernoulliGLMTask(PyMCTask):
    def __init__(self):
        """Bernoulli GLM Task"""
        summary = "sufficient"
        self.summary = summary
        dim_parameters = 10
        self.dim_data_raw = 100
        self.dim_data_summary = 10
        name = "BernoulliGLM"

        # Determine data dimension and task name based on summary type
        if self.summary == "sufficient":
            dim_data = self.dim_data_summary
            self.raw = False
        elif self.summary == "raw":
            dim_data = self.dim_data_raw
            self.raw = True
        else:
            raise ValueError(f"Unknown summary type: {self.summary}")

        self.dim_data = (
            dim_data  # This reflects the dimension of the *output* data
        )
        self.dim_parameters = dim_parameters

        # Define variable names and dimensions for PyMCTask
        var_names = ["theta"]
        var_dims = {"theta": self.dim_parameters}
        task_name = name

        # Load design matrix and stimulus (needed for summary stats)
        try:
            current_dir = Path(__file__).parent.absolute()
            # Adjust path relative to the current file location
            self.task_info_dir = (
                current_dir / f"info/{name}"
            )  # Adjust if needed
            self.design_matrix_path = self.task_info_dir / "design_matrix.npy"
            self.stimulus_I_path = self.task_info_dir / "stimulus_I.npy"

            self.design_matrix = np.load(self.design_matrix_path)
            self.design_matrix_pt = pt.as_tensor_variable(self.design_matrix)

            # Load stimulus_I needed for summary statistics calculation
            self.stimulus_I_np = np.load(self.stimulus_I_path)

        except FileNotFoundError as e:
            logger.error(
                f"Required file not found: {e}. Searched in {self.task_info_dir}. Please ensure sbibm files exist."
            )
            raise

        # Define prior parameters (converting precision to covariance)
        M = self.dim_parameters - 1
        D = np.diag(np.ones(M)) - np.diag(np.ones(M - 1), -1)
        F = np.matmul(D, D) + np.diag(1.0 * np.arange(M) / (M)) ** 0.5
        Binv = np.zeros((M + 1, M + 1))
        Binv[0, 0] = 0.5  # offset precision
        Binv[1:, 1:] = np.matmul(F.T, F)  # filter precision

        min_eig = np.min(np.linalg.eigvalsh(Binv))
        if min_eig <= 1e-12:
            Binv += np.eye(Binv.shape[0]) * 1e-10
            logger.warning(
                "Added jitter to prior precision matrix for numerical stability."
            )

        try:
            self.prior_cov = np.linalg.inv(Binv)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "Prior precision matrix inversion failed."
            ) from err

        self.prior_loc = np.zeros(self.dim_parameters)

        super().__init__(
            var_names=var_names,
            var_dims=var_dims,
            task_name=task_name,
        )

    def _calculate_summary_stats_torch(self, y_raw: np.ndarray) -> np.ndarray:
        """Calculates summary statistics (num_spikes, sta) from raw data."""
        import torch

        if y_raw.ndim == 1:
            y_raw = y_raw.reshape(1, -1)  # Ensure 2D for batch processing

        num_sims = y_raw.shape[0]
        summaries = np.zeros((num_sims, self.dim_data_summary))

        # Use torch for convolution as in original sbibm code for consistency
        y_torch = torch.from_numpy(y_raw).float().reshape(num_sims, 1, -1)
        stimulus_I_torch_r = torch.from_numpy(self.stimulus_I_np).reshape(
            1, 1, -1
        )

        for i in range(num_sims):
            num_spikes = torch.sum(y_torch[i])
            # Calculate Spike-Triggered Average (STA) using torch convolution
            sta = torch.nn.functional.conv1d(
                y_torch[i : i + 1], stimulus_I_torch_r, padding=8
            ).squeeze()[-9:]  # Get last 9 elements
            summaries[i, 0] = num_spikes.item()
            summaries[i, 1:] = sta.numpy()
        return summaries

    def _calculate_summary_stats(self, y_raw: np.ndarray) -> np.ndarray:
        """Calculates summary statistics (num_spikes, sta) from raw data using numpy."""
        if y_raw.ndim == 1:
            y_raw = y_raw.reshape(1, -1)
        num_sims = y_raw.shape[0]
        summaries = np.dot(y_raw, self.design_matrix)
        return summaries

    def observation_to_pymc_data(self, observation: np.ndarray = None) -> dict:
        """Prepares observation data for the PyMC model.
        NOTE: setup_pymc_model requires RAW data (100 dim). This function
        loads the specified data (raw or summary), but the model itself
        needs the raw version.
        """
        if observation is None:
            # Generate synthetic data if no observation is provided
            observation = np.random.binomial(
                n=1, p=0.5, size=(self.dim_data_raw,)
            )

        if observation.ndim > 1:
            observation = observation.flatten()

        # Explicitly check if the provided observation has the raw dimension
        if observation.shape[0] != self.dim_data_raw:
            raise ValueError(
                f"PyMC model setup requires RAW observation data (dim={self.dim_data_raw}). "
                f"Provided data has dimension {observation.shape[0]}."
            )
        return observation

    def setup_pymc_model(self, observation: np.ndarray = None) -> pm.Model:
        """Sets up the PyMC model for the Bernoulli GLM.
        IMPORTANT: This model definition requires RAW (100-dimensional)
        observation data, regardless of the `self.summary` setting.
        Ensure the provided `observation` is the raw data.
        """
        observation = self.observation_to_pymc_data(observation)

        with pm.Model() as model:
            # Define pm.Data container for RAW observations
            obs_data = pm.Data(
                "obs_data", observation, mutable=True, dims="obs_dim"
            )  # Add dims for clarity

            # Prior for parameters theta
            theta = pm.MvNormal(
                "theta",
                mu=self.prior_loc,
                cov=self.prior_cov,
                shape=self.dim_parameters,
                dims="param_dim",  # Add dims for clarity
            )

            # Linear predictor (psi = X * theta)
            # Ensure design_matrix_pt has compatible dimensions if needed
            psi = pm.math.dot(
                self.design_matrix_pt, theta
            )  # Output shape should match obs_data shape

            # Link function (sigmoid)
            p = pm.math.sigmoid(psi)

            # Likelihood (Bernoulli) - requires raw observations
            y = pm.Bernoulli(
                "y", p=p, observed=obs_data, dims="obs_dim"
            )  # Match dims
        return model

    def sample(self, batch_size: int, **kwargs):
        """Samples from the prior predictive distribution.
        Returns raw data if summary='raw', or summary statistics otherwise.
        """
        simulations = super().sample(batch_size=batch_size, **kwargs)
        simulations["observables_raw"] = simulations["observables"]
        # Calculate summary statistics from raw simulations
        observables = self._calculate_summary_stats(simulations["observables"])
        simulations["observables"] = observables
        return simulations

    def simulate_observation(
        self, param_values: np.ndarray, unconstrained=True, **kwargs
    ):
        simulations = super().simulate_observation(
            param_values=param_values, unconstrained=unconstrained, **kwargs
        )
        simulations["observables_raw"] = simulations["observables"]
        observables = self._calculate_summary_stats(simulations["observables"])
        simulations["observables"] = observables
        return simulations
