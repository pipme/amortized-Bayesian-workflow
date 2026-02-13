import numpy as np
import pymc as pm
import pymc_extras.distributions as pmx

from sbi_mcmc.tasks.tasks import PyMCTask
from sbi_mcmc.utils.logging import get_logger

logger = get_logger(__name__)


class GeneralizedExtremeValue(PyMCTask):
    """From https://www.pymc.io/projects/examples/en/latest/case_studies/GEV.html"""

    def __init__(self):
        var_names = ["mu", "sigma", "xi"]
        task_name = "GEV"
        super().__init__(
            var_names=var_names,
            task_name=task_name,
        )

    def observation_to_pymc_data(self, observation=None):
        if observation is None:
            observation = self.get_observation()
        return observation

    def setup_pymc_model(self, observation=None) -> pm.Model:
        observation = self.observation_to_pymc_data(observation)
        p = 1 / 10
        with pm.Model() as pymc_model:
            # Priors
            observation = pm.Data("observation", observation)
            mu = pm.Normal("mu", mu=3.8, sigma=0.2)
            sigma = pm.HalfNormal("sigma", sigma=0.3)
            xi = pm.TruncatedNormal(
                "xi", mu=0, sigma=0.2, lower=-0.6, upper=0.6
            )

            # Estimation
            gev = pmx.GenExtreme(
                "gev",
                mu=mu,
                sigma=sigma,
                xi=xi,
                observed=observation,
            )
            # Return level
            z_p = pm.Deterministic(
                "z_p", mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi))
            )
        return pymc_model

    def get_observation(self):
        # fmt: off
        data = np.array([4.03, 3.83, 3.65, 3.88, 4.01, 4.08, 4.18, 3.80,
                        4.36, 3.96, 3.98, 4.69, 3.85, 3.96, 3.85, 3.93,
                        3.75, 3.63, 3.57, 4.25, 3.97, 4.05, 4.24, 4.22,
                        3.73, 4.37, 4.06, 3.71, 3.96, 4.06, 4.55, 3.79,
                        3.89, 4.11, 3.85, 3.86, 3.86, 4.21, 4.01, 4.11,
                        4.24, 3.96, 4.21, 3.74, 3.85, 3.88, 3.66, 4.11,
                        3.71, 4.18, 3.90, 3.78, 3.91, 3.72, 4.00, 3.66,
                        3.62, 4.33, 4.55, 3.75, 4.08, 3.90, 3.88, 3.94,
                        4.33])
        # fmt: on
        return data
