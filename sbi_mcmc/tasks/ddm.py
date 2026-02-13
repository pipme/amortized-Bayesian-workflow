"""Drift diffusion model"""

import time
from pathlib import Path

import hssm
import numpy as np
import pandas as pd
import pymc as pm
from hssm.likelihoods import DDM
from numba import njit

from sbi_mcmc.tasks.tasks import PyMCTask
from sbi_mcmc.utils.logging import get_logger

logger = get_logger(__name__)


class CustomDDM(PyMCTask):
    """A 6-parameter drift diffusion model used in the manuscript, from von Krausen et al. (2022)."""

    def __init__(self, uniform_prior: bool = False, dt: float = 0.001):
        var_names = ["v", "a", "ndt_plus", "ndt_minus"]
        task_name = "CustomDDM"
        self.uniform_prior = uniform_prior
        self.dt = dt
        self.task_info_dir = (
            Path(__file__).parent.absolute() / "info/CustomDDM"
        )
        if self.dt != 0.001:
            task_name += f"(dt-{self.dt})"
        super().__init__(
            var_names=var_names,
            var_dims={"v": 2, "a": 2, "ndt_plus": 1, "ndt_minus": 1},
            task_name=task_name,
        )

    def observation_to_pymc_data(self, observation: np.ndarray = None) -> dict:
        if observation is None:
            data = pd.read_csv(self.task_info_dir / "stan_data.csv").rename(
                columns={"id": "subject"}
            )
            ids = data["subject"].unique()
            subject_id = ids[0]

            data_1p = data[(data["subject"] == subject_id) & (data["rt"] > 0)]
            # Replace response 0 with -1
            data_1p.loc[:, "response"] = (
                data_1p["response"].replace(0, -1).astype(int)
            )
            data_c = data_1p[data_1p["block"] == 1]
            data_i = data_1p[data_1p["block"] == 0]

        else:
            # Negative rt means response is -1
            data = pd.DataFrame(
                observation,
                columns=["rt", "missing", "condition_type", "stimulus_type"],
            )
            data = data[data["rt"] != 0]
            # Remove missing trials
            data = data[data["missing"] == 0]
            data["response"] = np.sign(data["rt"])
            data["rt"] = np.abs(data["rt"])  # hssm DDM requires positive rt
            data_c = data[data["condition_type"] == 0]
            data_i = data[data["condition_type"] == 1]

        data_c = data_c[["rt", "response"]]
        data_i = data_i[["rt", "response"]]

        self.min_rts_c = data_c["rt"].min()
        self.min_rts_i = data_i["rt"].min()
        print(f"min rts: {self.min_rts_c}, {self.min_rts_i}")
        self.min_rts = min(self.min_rts_c, self.min_rts_i)
        mask_c_1 = data_c["response"] == 1
        mask_i_1 = data_i["response"] == 1
        N_c = len(data_c)
        N_i = len(data_i)
        print(f"N_c: {N_c}, N_i: {N_i}")
        c_1_flag = np.any(mask_c_1.values)
        i_1_flag = np.any(mask_i_1.values)
        c_0_flag = np.any(~mask_c_1.values)
        i_0_flag = np.any(~mask_i_1.values)
        print(
            f"c_1_flag: {c_1_flag}, i_1_flag: {i_1_flag}, c_0_flag: {c_0_flag}, i_0_flag: {i_0_flag}"
        )
        obs_c_1 = data_c[mask_c_1].values
        obs_c_0 = data_c[~mask_c_1].values
        obs_i_1 = data_i[mask_i_1].values
        obs_i_0 = data_i[~mask_i_1].values
        return obs_c_1, obs_c_0, obs_i_1, obs_i_0

    def setup_pymc_model(self, observation: np.ndarray = None) -> pm.Model:
        obs_c_1, obs_c_0, obs_i_1, obs_i_0 = self.observation_to_pymc_data(
            observation
        )
        C = 2

        # Define the model
        with pm.Model() as model:
            obs_c_1_data = pm.Data("obs_c_1", obs_c_1, mutable=True)
            obs_c_0_data = pm.Data("obs_c_0", obs_c_0, mutable=True)
            obs_i_1_data = pm.Data("obs_i_1", obs_i_1, mutable=True)
            obs_i_0_data = pm.Data("obs_i_0", obs_i_0, mutable=True)

            if self.uniform_prior:
                v = pm.Uniform(
                    "v", lower=[0.0, 0.0], upper=[7.0, 4.0], shape=C
                )
                a = pm.Uniform(
                    "a", lower=[0.0, 0.0], upper=[7.0, 4.0], shape=C
                )
                ndt_correct = pm.Uniform("ndt_plus", lower=0.1, upper=3.0)
                ndt_error = pm.Uniform("ndt_minus", lower=0.1, upper=7.0)
            else:
                v = pm.Gamma(
                    "v", alpha=2.0, beta=1.0, shape=C
                )  # drift rate per condition
                a = pm.Gamma(
                    "a", alpha=6.0, beta=1 / (0.3 / 2), shape=C
                )  # boundary per condition
                # non-decision time correct/error
                ndt_correct = pm.Gamma("ndt_plus", alpha=3.0, beta=1 / 0.15)
                ndt_error = pm.Gamma("ndt_minus", alpha=3.0, beta=1 / 0.5)
            z = 0.5

            # Use the pm.Data variables in the observed argument
            ddm_c_1 = DDM(
                "ddm_c_1",
                v=v[0],
                a=a[0],
                z=z,
                t=ndt_correct,
                observed=obs_c_1_data,  # Use pm.Data variable
            )
            ddm_c_0 = DDM(
                "ddm_c_0",
                v=v[0],
                a=a[0],
                z=z,
                t=ndt_error,
                observed=obs_c_0_data,  # Use pm.Data variable
            )  # flipping for z and v is done internally in the DDM likelihood
            ddm_i_1 = DDM(
                "ddm_i_1",
                v=v[1],
                a=a[1],
                z=z,
                t=ndt_correct,
                observed=obs_i_1_data,  # Use pm.Data variable
            )
            ddm_i_0 = DDM(
                "ddm_i_0",
                v=v[1],
                a=a[1],
                z=z,
                t=ndt_error,
                observed=obs_i_0_data,  # Use pm.Data variable
            )
        return model

    def sample(self, batch_size: int, **kwargs):
        # Measure simulation time
        sim_start = time.time()
        prior_samples = []
        observations_sims = []
        for _ in range(batch_size):
            theta = simple_ddm_prior_fun()
            sim_data = simple_ddm_simulator_fun(theta, dt=self.dt)
            prior_samples.append(theta)
            observations_sims.append(sim_data)
        prior_samples = np.array(prior_samples)
        observations_sims = np.array(observations_sims)
        sim_time = time.time() - sim_start

        # Measure transformation time
        transform_start = time.time()
        prior_samples = self.transform_to_unconstrained_space(
            prior_samples, in_place=True
        )
        transform_time = time.time() - transform_start

        logger.info(f"Simulation time: {sim_time:.2f}s")
        logger.info(f"Transform time: {transform_time:.2f}s")
        return {
            "parameters": prior_samples,
            "observables": observations_sims,
        }


class SimpleDDM(PyMCTask):
    """A 3-parameter standard drift diffusion model. Used for testing."""

    def __init__(
        self,
        rts_size: int = 120,
        simulator: str = "custom",
        fix_t=False,
    ):
        var_names = ["v", "a", "t"]
        var_dims = {"v": 1, "a": 1, "t": 1}
        task_name = "SimpleDDM"
        self.rts_size = rts_size
        self.simulator = simulator
        if simulator == "hssm":
            task_name += "(hssm_sims)"
        elif simulator == "hssm(fine)":
            task_name += "(hssm_sims_fine)"
        elif simulator == "custom":
            pass
        elif simulator == "pymc":
            task_name += "(pymc_sims)"
        else:
            raise ValueError(f"{simulator} unknow")

        self.fix_t = fix_t
        if self.fix_t:
            self.fixed_t = 0.5
            var_names = ["v", "a"]
            var_dims = {"v": 1, "a": 1}
            task_name += "_fixed_t"

        self.bounds = {
            "v": (1.0, 10.0),
            "a": (0.1, 1.0),
            "t": (0.2, 0.6),
        }
        super().__init__(
            var_names=var_names,
            var_dims=var_dims,
            task_name=task_name,
        )

    def setup_pymc_model(self, observation: np.ndarray = None) -> pm.Model:
        if observation is None:
            observation = np.ones((self.rts_size, 2))
        if observation.shape[1] == 2:
            dataset = pd.DataFrame(observation, columns=["rt", "response"])
            observation = dataset.values
        else:
            dataset = pd.DataFrame(
                observation,
                columns=["rt", "missing", "condition_type", "stimulus_type"],
            )
            dataset["response"] = np.sign(dataset["rt"])
            dataset["rt"] = np.abs(
                dataset["rt"]
            )  # hssm DDM requires positive rt
            dataset = dataset[dataset["rt"] != 0]
            dataset = dataset[["rt", "response"]]
            observation = dataset.values
        with pm.Model() as ddm_pymc:
            v = pm.Uniform(
                "v", lower=self.bounds["v"][0], upper=self.bounds["v"][1]
            )
            a = pm.Uniform(
                "a", lower=self.bounds["a"][0], upper=self.bounds["a"][1]
            )
            if self.fix_t:
                t = self.fixed_t
            else:
                t = pm.Uniform(
                    "t", lower=self.bounds["t"][0], upper=self.bounds["t"][1]
                )
            ddm = DDM("ddm", v=v, a=a, z=0.5, t=t, observed=observation)
        return ddm_pymc

    def sample(self, batch_size: int, **kwargs):
        if self.simulator == "pymc":
            simulations = super().sample(batch_size, **kwargs)
            # hack: due to usage of az.sel_utils.xarray_to_ndarray, the response times and response choices are concatenated, but should be separate
            simulations["observables"] = simulations["observables"].reshape(
                batch_size, self.rts_size, 2
            )
            return simulations

        # Measure simulation time
        sim_start = time.time()
        prior_samples = []
        observations_sims = []
        for _ in range(batch_size):
            v = np.random.uniform(*self.bounds["v"])
            a = np.random.uniform(*self.bounds["a"])
            if self.fix_t:
                t = self.fixed_t
            else:
                t = np.random.uniform(*self.bounds["t"])
            theta = np.array([v, a, t])
            if self.simulator == "hssm":
                dataset = hssm.simulate_data(
                    model="ddm",
                    theta=[v, a, 0.5, t],
                    size=self.rts_size,
                )
            elif self.simulator == "hssm(fine)":
                dataset = hssm.simulate_data(
                    model="ddm",
                    theta=[v, a, 0.5, t],
                    size=self.rts_size,
                    delta_t=0.00001,
                    smooth_unif=False,
                )
            else:
                ## test_simple_ddm_simulator is similar to hssm.simulate_data
                # dataset = test_simple_ddm_simulator(theta, self.rts_size)
                dataset = simple_ddm_simulator_fun(
                    np.repeat(theta, 2), self.rts_size
                )
                dataset = pd.DataFrame(
                    dataset,
                    columns=[
                        "rt",
                        "missing",
                        "condition_type",
                        "stimulus_type",
                    ],
                )

            if self.fix_t:
                theta = theta[:-1]
            prior_samples.append(theta)
            observations_sims.append(dataset.values)
        prior_samples = np.array(prior_samples)
        observations_sims = np.array(observations_sims)
        sim_time = time.time() - sim_start

        prior_samples = self.transform_to_unconstrained_space(
            prior_samples, in_place=True
        )
        return {
            "parameters": prior_samples,
            "observables": observations_sims,
        }


def test_simple_ddm_simulator(theta, num_obs=120):
    v_true = theta[0]
    a_true = theta[1]
    t_true = theta[2]
    rts = np.zeros(num_obs)
    for n in range(num_obs):
        rts[n] = simple_ddm_iat_diffusion_trial(v_true, a_true, t_true)

    dataset = pd.DataFrame({"rt": rts})
    dataset["response"] = np.sign(rts)
    dataset["rt"] = np.abs(dataset["rt"])  # hssm DDM requires positive rt
    return dataset


# Simple DDM simulator
# For a single trial
@njit
def simple_ddm_iat_diffusion_trial(v, a, ndtcorrect, ndterror=None, dt=0.001):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.0
    x = 0
    max_steps = int(20 / dt)

    # Simulate a single DM path
    while x > -a and x < a and n_steps < max_steps:
        # DDM equation
        x += v * dt + np.sqrt(dt) * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt
    if ndterror is None:
        ndterror = ndtcorrect
    return (
        rt + ndtcorrect if x > -a else -rt - ndterror
    )  # use different ndts for correct and error responses


# Simple DDM simulator
# For an entire subject
@njit
def simple_ddm_simulator_fun(theta, num_obs=120, dt=0.001):
    """Simulates data from a single subject in an IAT experiment."""
    assert num_obs % 4 == 0
    # assert num_obs == 120
    obs_per_condition = num_obs // 2

    condition_type = np.arange(2)
    condition_type = np.repeat(condition_type, obs_per_condition)

    # code stimulus types, picture == 1
    stimulus_type = np.concatenate(
        (
            np.zeros(obs_per_condition // 2),
            np.ones(obs_per_condition // 2),  # condition 1: congruent
            np.zeros(obs_per_condition // 2),
            np.ones(obs_per_condition // 2),
        )
    )  # condition 2: incongruent

    v = theta[0:2]
    a = theta[2:4]

    out = np.zeros(num_obs)

    for n in range(num_obs):
        out[n] = simple_ddm_iat_diffusion_trial(
            v[condition_type[n]], a[condition_type[n]], theta[4], theta[5], dt
        )
        # mark too slow trials with zero
        if abs(out[n]) >= 20.0:
            out[n] = 0

    missings = np.expand_dims(np.zeros(out.shape[0]), 1)
    missings[out == 0] = 1

    out = np.expand_dims(out, 1)
    condition_type = np.expand_dims(condition_type, 1)
    stimulus_type = np.expand_dims(stimulus_type, 1)
    out = np.concatenate(
        (out, missings, condition_type, stimulus_type), axis=1
    )

    return out


def simple_ddm_prior_fun(seed: int | None = None):
    """
    Samples from the prior (once).
    ----------

    # Prior ranges for the simulator
    # v1 & v2 ~ G(2.0, 1.0)
    # a1 & a2 ~ G(6.0, 0.3)
    # ndt_plus ~ G(3.0, 0.15)
    # ndt_minus ~ G(3.0, 0.5)

    """
    RNG = np.random.default_rng(seed)

    drifts = RNG.gamma(2.0, 1.0, size=2)

    thresholds = RNG.gamma(6.0, 0.3 / 2, size=2)

    ndt_plus = RNG.gamma(3.0, 0.15)
    ndt_minus = RNG.gamma(3.0, 0.5)

    return np.hstack((drifts, thresholds, ndt_plus, ndt_minus))
