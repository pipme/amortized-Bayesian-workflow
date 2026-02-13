import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from sbi_mcmc.tasks import (
    BernoulliGLMTask,
    CustomDDM,
    GeneralizedExtremeValue,
    PsychometricTask,
)
from sbi_mcmc.utils.utils import read_from_file


class PickleStatLogger:
    """
    A utility for logging experiment statistics to a pickle file using pathlib.

    Stores key-value dictionaries where each key represents a run or experiment,
    and values are statistics such as wall-clock time, accuracy, etc.

    Supports:
    - Reading and writing a persistent dictionary to disk
    - Updating or extending per-key statistics
    - Measuring and recording wall-clock time with a context manager

    Example usage:
    --------------
    from pathlib import Path

    logger = PickleStatLogger(Path("results/exp_stats.pkl"))

    # Time a code block and log wall-clock time under "run_005"
    with logger.timer("run_005"):
        time.sleep(0.8)

    # Add more stats to the same run
    logger.update("run_005", {"accuracy": 0.95})

    # Retrieve stats
    print(logger.get("run_005"))

    # Access all stats directly
    print(logger.data)
    """

    def __init__(self, filepath, overwrite=False, verbose=False):
        """
        If overwrite is True, the existing file will be deleted.
        """
        self.filepath = Path(filepath)
        if overwrite and self.filepath.exists():
            self.filepath.unlink()
        self.data = self._load()
        self.verbose = verbose

    def _load(self):
        if self.filepath.exists():
            with self.filepath.open("rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("wb") as f:
            pickle.dump(self.data, f)

    def update(self, key, value_dict):
        if key is None:
            self.data.update(value_dict)
        else:
            if key in self.data and isinstance(self.data[key], dict):
                self.data[key].update(value_dict)
            else:
                self.data[key] = value_dict
        self.save()

    def get(self, key):
        return self.data.get(key, None)

    @contextmanager
    def timer(self, label):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        elapsed = end - start
        self.update("wall_time", {label: elapsed})
        if self.verbose:
            print(f"[wall_time] {label}: {elapsed:.3f} sec")


def check_dataset_size_consistency(prior_simulations, dataset_size_dict):
    # Check if the number of prior simulations is the same as the specified dataset size
    for dataset_name, num_simulations in dataset_size_dict.items():
        assert len(prior_simulations[dataset_name]["observables"]) == len(
            prior_simulations[dataset_name]["parameters"]
        ), (
            f"Number of simulations for {dataset_name} "
            f"({len(prior_simulations[dataset_name])}) "
            f"does not match the number of parameters ({len(prior_simulations[dataset_name]['parameters'])})."
        )
        if (
            len(prior_simulations[dataset_name]["observables"])
            != num_simulations
        ):
            raise ValueError(
                f"Number of prior simulations for {dataset_name} "
                f"({len(prior_simulations[dataset_name]['observables'])}) "
                f"does not match the specified size ({num_simulations})."
            )


def check_prior_simulations(prior_simulations, task, check_duplicates=True):
    # Some sanity checks for prior simulations
    if check_duplicates:
        assert len(
            np.unique(prior_simulations["train"]["parameters"], axis=0)
        ) == len(prior_simulations["train"]["parameters"]), (
            "Prior simulations (parameters) contain duplicates. Please check the simulation process."
        )
    if "DDM" in task.name:
        # Check the number of missing trials. This should be small. Or else there may be a non-negligible discrepancy between DDM likelihood (not censored) and the prior simulation (censored).
        print(
            np.sort(prior_simulations["train"]["observables"][..., 1].sum(-1))
        )


def get_bf_configs(task, smoke_test=False):
    """Get configs for BayesFlow based on the task name."""
    import bayesflow as bf

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        # .standardize(exclude=["parameters"])
        .rename("parameters", "inference_variables")
        .rename("observables", "summary_variables")
    )
    # adapter = adapter.expand_dims("summary_variables", axis=-1)
    if task.name == "CustomDDM(dt-0.0001)":
        summary_network = bf.networks.SetTransformer()
    elif task.name == "BernoulliGLM":
        adapter = (
            bf.adapters.Adapter()
            .drop("observables_raw")
            .to_array()
            .convert_dtype("float64", "float32")
            .rename("parameters", "inference_variables")
            .rename("observables", "inference_conditions")
        )
        summary_network = None
    elif task.name == "GEV":
        adapter = adapter.expand_dims("summary_variables", axis=-1)
        summary_network = bf.networks.DeepSet(
            summary_dim=8,
            depth=1,
            activation="mish",
            inner_pooling="max",
            output_pooling="mean",
            dropout=0.05,
        )
    else:
        summary_network = bf.networks.DeepSet()
    # if smoke_test:
    #     print("Using DeepSet for smoke test.")
    #     summary_network = bf.networks.DeepSet()
    inference_network = bf.networks.CouplingFlow()

    if isinstance(inference_network, bf.networks.coupling_flow.CouplingFlow):
        flow_type = "coupling"
    elif isinstance(inference_network, bf.networks.flow_matching.FlowMatching):
        flow_type = "matching"
    else:
        raise NotImplementedError

    paths = get_paths(task, flow_type=flow_type, smoke_test=smoke_test)
    save_model_path = paths["save_model_path"]
    info = {
        "save_model_path": save_model_path,
        "flow_type": flow_type,
    }
    return {
        "simulator": None,
        "adapter": adapter,
        "summary_network": summary_network,
        "inference_network": inference_network,
        "checkpoint_filepath": save_model_path.parent,
        "checkpoint_name": save_model_path.stem,
        "save_best_only": True,
    }, info


def get_task_configs(task):
    epochs = 300
    train_dataset_size = 10_000
    val_dataset_size = 1000
    diagnostic_dataset_size = 200  # For SBC and parameter recovery
    lc2st_cal_dataset_size = 10_000  # For L-C2ST
    batch_size = min(256, train_dataset_size)

    if task.name == "CustomDDM(dt-0.0001)":
        train_dataset_size = 100_000
        epochs = 300
        batch_size = 512
    elif "psychometric_curve" in task.name:
        train_dataset_size = 50_000
        epochs = 300
        batch_size = 512
    elif "BernoulliGLM" in task.name:
        # train_dataset_size = 100_000
        epochs = 100
        batch_size = 512
    dataset_size_dict = {
        "train": train_dataset_size,
        "val": val_dataset_size,
        "diagnostic": diagnostic_dataset_size,
        "lc2st_cal": lc2st_cal_dataset_size,
    }
    return {
        "dataset_size_dict": dataset_size_dict,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def get_paths(task, **kwargs):
    paths = {"save_dir": Path(f"../experiments/results/{task.name}/")}
    paths["dataset_dir"] = paths["save_dir"] / "datasets"

    if kwargs.get("smoke_test", False):
        paths["save_dir"] /= "smoke_test/"
    paths["figure_dir"] = paths["save_dir"] / "figures"
    paths["training_result_dir"] = paths["save_dir"] / "training_results"

    flow_type = kwargs.get("flow_type", "coupling")
    paths["save_model_path"] = (
        paths["training_result_dir"]
        / f"neural_approximator_model({flow_type}).keras"
    )

    paths["inference_result_dir"] = (
        paths["save_dir"] / "inference_results" / paths["save_model_path"].stem
    )
    paths["abi_result_dir"] = paths["inference_result_dir"] / "abi"
    paths["inference_diagnostic_dir"] = (
        paths["inference_result_dir"] / "diagnostic"
    )
    paths["psis_result_dir"] = paths["inference_result_dir"] / "psis"
    paths["chees_hmc_result_dir"] = paths["inference_result_dir"] / "cheeshmc"
    paths["metrics_result_dir"] = paths["inference_result_dir"] / "metrics"
    for k, path in paths.items():
        if "dir" in k:
            path.mkdir(parents=True, exist_ok=True)

    test_dataset_name = kwargs.get("test_dataset_name", "")
    dataset_suffix = f"_{test_dataset_name}" if test_dataset_name else ""
    paths["abi_result"] = paths["abi_result_dir"] / f"abi{dataset_suffix}.pkl"
    paths["abi_stats"] = paths["abi_result_dir"] / f"stats{dataset_suffix}.pkl"

    paths["psis_result"] = lambda obs_id: (
        paths["psis_result_dir"] / f"psis{dataset_suffix}_id-{obs_id}.pkl"
    )
    paths["psis_stats"] = (
        paths["psis_result_dir"] / f"stats{dataset_suffix}.pkl"
    )
    paths["metrics_stats"] = paths["metrics_result_dir"] / f"stats{dataset_suffix}.pkl"
    paths["chees_hmc_result"] = lambda obs_id: (
        paths["chees_hmc_result_dir"]
        / f"chees_hmc{dataset_suffix}_id-{obs_id}.pkl"
    )
    paths["chees_hmc_stats"] = (
        paths["chees_hmc_result_dir"] / f"stats{dataset_suffix}.pkl"
    )
    paths["ood_stats"] = (
        paths["inference_diagnostic_dir"] / f"ood_stats{dataset_suffix}.pkl"
    )
    paths["lc2st_stats"] = (
        paths["inference_diagnostic_dir"] / f"lc2st_stats{dataset_suffix}.pkl"
    )
    return paths


def get_stuff(task=None, **kwargs):
    if task is None:
        config = OmegaConf.load("config/config.yaml")
        if kwargs.get("task_name") is not None:
            task_name = kwargs.get("task_name")
        else:
            task_name = config.task_name
        if task_name == "BernoulliGLM":
            task = BernoulliGLMTask()
        elif task_name in ["CustomDDM(dt=0.0001)", "CustomDDM(dt-0.0001)"]:
            task = CustomDDM(dt=0.0001)
        elif task_name == "CustomDDM":
            task = CustomDDM(dt=0.001)
        elif task_name == "psychometric_curve_overdispersion":
            task = PsychometricTask(overdispersion=True)
        elif task_name == "GEV":
            task = GeneralizedExtremeValue()
        else:
            raise NotImplementedError(
                f"Task {task_name} is not implemented. Please check the task name."
            )
    else:
        config = {}
    # update config with kwargs
    config.update(kwargs)
    smoke_test = config.get("smoke_test", False)
    paths = get_paths(task, **config)

    res = {
        "task": task,
        "paths": paths,
        "config": config,
    }
    job = config.get("job", None)
    if job == "training":
        return res
    if config.get("use_train_dataset", False):
        # Use the training dataset for testing
        print("Using the training dataset for testing. ")
        test_dataset_name = "train_dataset"
    else:
        assert "test_dataset_name" in config, (
            "Please provide a test dataset name."
        )
        test_dataset_name = config["test_dataset_name"]
    test_dataset = read_from_file(
        paths["dataset_dir"] / f"{test_dataset_name}.pkl"
    )
    if isinstance(test_dataset, dict):
        N_testdata = test_dataset["observables"].shape[0]
    elif isinstance(test_dataset, np.ndarray):
        N_testdata = test_dataset.shape[0]
        test_dataset = {"observables": test_dataset}
    else:
        raise ValueError(
            f"Unsupported test dataset type: {type(test_dataset)}. "
            "Expected a dictionary or numpy array."
        )
    if smoke_test:
        # Use only a small subset of the test dataset for smoke testing
        N_testdata = min(100, N_testdata)
        for k, v in test_dataset.items():
            test_dataset[k] = v[:N_testdata]
    print(f"Number of test data: {N_testdata}")

    res.update(
        {
            "N_testdata": N_testdata,
            "test_dataset": test_dataset,
            "test_dataset_name": test_dataset_name,
        }
    )
    if job == "lc2st":
        # Get the calibration dataset
        res["lc2st_cal_dataset"] = read_from_file(
            paths["dataset_dir"] / "lc2st_cal_dataset.pkl"
        )

    if job in ["lc2st", "ood", "psis", "abi", "chees_hmc"]:
        res["stats_logger"] = PickleStatLogger(
            paths[f"{job}_stats"], overwrite=config.get("overwrite_stats", False), verbose=True
        )

    return res


def to_torch(x, device="cpu", dtype=None):
    import torch

    # default to float32
    if dtype is None:
        dtype = torch.float32
    if isinstance(x, np.ndarray | list):
        x = torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, int | float):
        x = torch.tensor([x], device=device, dtype=dtype)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    return x


def pad_dataset(dataset, padded_value):
    for key in dataset.keys():
        if "observables" not in key:
            continue
        observables = dataset[key]
        if observables.ndim == 3:  # 3D case
            N, M, D = observables.shape
            padded_observables = np.full((N, 2 * M, D), padded_value)
            padded_observables[:, :M, :] = observables
        elif observables.ndim == 2:  # 2D case
            N, M = observables.shape
            padded_observables = np.full((N, 2 * M), padded_value)
            padded_observables[:, :M] = observables
        else:
            raise ValueError("Unexpected dimensionality of observables array.")

        dataset[key] = padded_observables
