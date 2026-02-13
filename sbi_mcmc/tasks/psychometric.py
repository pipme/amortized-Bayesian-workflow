from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.special import erf

from sbi_mcmc.tasks.tasks import PyMCTask
from sbi_mcmc.utils.logging import get_logger

logger = get_logger(__name__)


class PsychometricTask(PyMCTask):
    """A psychometric curve model using a cumulative normal distribution, optionally with overdispersion.
    Paper reference: https://www.sciencedirect.com/science/article/pii/S0042698916000390
    """

    def __init__(self, overdispersion=False, observed_responses=None):
        stimulus_intensities = np.array(
            [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]
        )
        # Normalize to -1 to 1
        stimulus_intensities = stimulus_intensities / 100.0
        self.stimulus_intensities = stimulus_intensities
        self.n_stimuli = len(stimulus_intensities)
        var_dims = {"mu_": 1, "sigma": 1, "gamma": 1, "lambd": 1}
        task_name = "psychometric_curve"
        self.task_info_dir = (
            Path(__file__).parent.absolute() / f"info/{task_name}"
        )
        self.overdispersion = overdispersion
        if self.overdispersion:
            # Add overdispersion parameter
            var_dims["nu"] = 1
            task_name += "_overdispersion"
        var_names = list(var_dims.keys())

        if observed_responses is None:
            # Generate synthetic data using true parameters
            self.default_true_params = {
                "mu": 0.0,  # Response bias
                "sigma": 0.2,  # Contrast threshold
                "gamma": 0.1,  # Lapse rate for left stimuli
                "lambd": 0.1,  # Lapse rate for right stimuli
            }
            # if self.overdispersion:
            #     self.default_true_params["nu"] = 0.0
            self.n_trials = np.random.multinomial(
                90, np.ones(self.n_stimuli) / self.n_stimuli
            )
            self.observed_responses = self._compute_observation(
                self.default_true_params
            )
        else:
            self.observed_responses = observed_responses
            self.n_trials = len(observed_responses)

        super().__init__(
            var_names=var_names,
            var_dims=var_dims,
            task_name=task_name,
        )

    def observation_to_pymc_data(self, observation=None):
        if observation is None:
            stimulus_intensities = self.stimulus_intensities
            n_trials = self.n_trials
            observed_responses = self.observed_responses
        else:
            stimulus_intensities = observation[:, 0]
            n_trials = observation[:, 1]
            observed_responses = observation[:, 2]

        return stimulus_intensities, n_trials, observed_responses

    def setup_pymc_model(self, observation=None):
        stimulus_intensities, n_trials, observed_responses = (
            self.observation_to_pymc_data(observation)
        )

        with pm.Model() as pymc_model:
            # Define pm.Data containers
            stimulus_data = pm.Data(
                "stimulus_intensities", stimulus_intensities
            )
            n_trials_data = pm.Data(
                "n_trials", n_trials
            )  # Ensure integer type for counts
            observed_responses_data = pm.Data(
                "observed_responses",
                observed_responses,
            )  # Ensure integer type for counts

            # Priors
            gamma = pm.Beta(
                "gamma", alpha=1, beta=10
            )  # Lapse rate for left stimuli
            lambd = pm.Beta(
                "lambd", alpha=1, beta=10
            )  # Lapse rate for right stimuli
            mu_ = pm.Beta(
                "mu_",
                alpha=2,
                beta=2,
            )  # Using Beta distribution stretched and shifted to approximate cosine falloff near boundary
            mu = pm.Deterministic("mu", mu_ * 2 - 1)  # Response bias
            sigma = pm.HalfNormal("sigma", sigma=1)  # Contrast threshold

            # Psychometric function
            erf_term = pm.math.erf(
                (stimulus_data - mu) / (sigma * pm.math.sqrt(2))
            )
            p_bar = pm.Deterministic(
                "p_bar", gamma + (1 - gamma - lambd) * (1 + erf_term) / 2
            )
            if self.overdispersion:
                nu = pm.Beta("nu", alpha=1, beta=10)
                nu_prime = 1 / nu**2 - 1
                y = pm.BetaBinomial(
                    "y",
                    n=n_trials_data,
                    alpha=nu_prime * p_bar,
                    beta=nu_prime * (1 - p_bar),
                    observed=observed_responses_data,
                )
            else:
                p = p_bar
                # Likelihood: Varying number of trials per stimulus level
                y = pm.Binomial(
                    "y", n=n_trials_data, p=p, observed=observed_responses_data
                )  # Simulate responses

        return pymc_model

    def sample(self, batch_size: int, numpy: bool = False, **kwargs):
        n_samples_per_chunk = kwargs.get("n_samples_per_chunk", 100)
        n_samples_per_chunk = min(n_samples_per_chunk, batch_size)
        print(n_samples_per_chunk)
        # Vary the number of trials every n samples
        n_chunk = batch_size // n_samples_per_chunk
        simulations_list = []
        for i in range(n_chunk):
            if i == n_chunk - 1:
                n_samples_per_chunk = batch_size - i * n_samples_per_chunk
            simulations = self.sample_with_n_trials(
                n_samples_per_chunk, numpy, **kwargs
            )
            simulations_list.append(simulations)

        # Concatenate the simulations (list of dicts) into a single dict
        simulations = {
            k: np.concatenate([d[k] for d in simulations_list])
            for k in simulations_list[0].keys()
        }

        return simulations

    def sample_with_n_trials(
        self, batch_size: int, numpy: bool = False, **kwargs
    ):
        self.update_observation()
        prior_samples, observation_samples = self.simulate(
            batch_size, **kwargs
        )
        # Add the number of trials to the observables
        n_trials_repeated = np.tile(
            self.n_trials[None, :], (len(observation_samples), 1)
        )
        stimulus_repeated = np.tile(
            self.stimulus_intensities[None, :], (len(observation_samples), 1)
        )
        observation_samples = np.stack(
            [stimulus_repeated, n_trials_repeated, observation_samples],
            axis=-1,
        )
        return {
            "parameters": prior_samples,
            "observables": observation_samples,
        }

    def update_observation(self, observed_responses=None):
        stimulus_intensities = self.stimulus_intensities
        # Simulate responses from different number of trials, used for prior predictive

        if observed_responses is None:
            # Update number of trials for prior predictive

            # Randomly select which phase to use (0: equal, 1: left focus, 2: right focus)
            phase = np.random.randint(0, 3)
            if phase == 0:
                # Initial phase: Allocate trials uniformly across stimulus levels
                n_trials = np.random.multinomial(
                    90, np.ones(self.n_stimuli) / self.n_stimuli
                )
            else:
                # Main phase with biased probabilities
                n_trials_main = np.clip(
                    int(np.random.randn() * 144 + 332), 1, 1000
                )

                if phase == 1:
                    # Focus on left stimuli
                    probs = np.where(stimulus_intensities <= 0, 0.8, 0.2)
                else:  # phase == 2
                    # Focus on right stimuli
                    probs = np.where(stimulus_intensities >= 0, 0.8, 0.2)

                probs = probs / probs.sum()
                n_trials = np.random.multinomial(n_trials_main, probs)
            self.n_trials = n_trials
        else:
            self.observed_responses = observed_responses
            self.n_trials = len(observed_responses)

        self.pymc_model = self.setup_pymc_model()

    def _compute_observation(self, params):
        """Generate synthetic observations using true parameters."""
        # Compute the psychometric function
        true_p = self.psychometric_function(self.stimulus_intensities, params)
        # Generate binomial observations
        observed_responses = np.random.binomial(
            self.n_trials, true_p, size=len(self.stimulus_intensities)
        )
        return observed_responses

    def get_observation(self):
        """Return the observed responses."""
        return np.stack(
            [
                self.stimulus_intensities,
                self.n_trials,
                self.observed_responses,
            ],
            axis=-1,
        )

    def plot_psychometric_curve(self, params=None, ax=None):
        """Plot the psychometric curve with optional parameters."""
        if ax is None:
            _, ax = plt.subplots()

        # Plot observed data points
        ax.scatter(
            self.stimulus_intensities,
            self.observed_responses / self.n_trials,
            label="Observed Data",
            color="black",
            alpha=0.5,
        )

        # If parameters provided, plot the curve
        if params is not None:
            x = np.linspace(
                min(self.stimulus_intensities),
                max(self.stimulus_intensities),
                100,
            )
            p = self.psychometric_function(x, params)
            ax.plot(x, p, "r-", label="Model Fit")

        ax.set_xlabel("Stimulus Intensity")
        ax.set_ylabel("Response Probability")
        ax.legend()
        return ax

    def psychometric_function(self, x, params):
        erf_term = erf((x - params["mu"]) / (params["sigma"] * np.sqrt(2)))
        p = (
            params["gamma"]
            + (1 - params["gamma"] - params["lambd"]) * (1 + erf_term) / 2
        )
        return p
