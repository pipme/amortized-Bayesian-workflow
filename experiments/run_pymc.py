import argparse
import os
import time

from sbi_mcmc.utils.utils import read_from_file, save_to_file

os.environ["KERAS_BACKEND"] = "jax"
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from sbi_mcmc.tasks import (
    BernoulliGLMTask,
    GeneralizedExtremeValue,
    PsychometricTask,
)
from sbi_mcmc.tasks.ddm import CustomDDM, SimpleDDM

az.style.use("arviz-darkgrid")

print(f"Running on PyMC v{pm.__version__}")

parser = argparse.ArgumentParser(description="Run inference steps")
parser.add_argument(
    "--task_name", type=str, default="psychometric_curve", help="Task name"
)
parser.add_argument(
    "--observation_id", type=int, default=0, help="Observation ID"
)
parser.add_argument(
    "--train", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument("-r", nargs="+", type=int)
parser.add_argument(
    "--ids", nargs="+", type=int, help="Explicit list of observation IDs"
)

args = parser.parse_args()
# Priority: --ids > -r > --observation_id
if args.ids is not None:
    observation_ids = args.ids
elif args.r is not None:
    assert len(args.r) == 2
    observation_ids = list(range(args.r[0], args.r[1]))
else:
    observation_ids = [args.observation_id]

print(observation_ids)
# %%
sampler_kwargs = {"nuts_sampler": "numpyro", "target_accept": 0.99}
if args.task_name == "GEV":
    task = GeneralizedExtremeValue()
    sampler_kwargs.update({"nuts_sampler": "pymc", "target_accept": 0.9999999})
elif args.task_name == "BernoulliGLM":
    task = BernoulliGLMTask()
elif args.task_name == "psychometric_curve":
    task = PsychometricTask(overdispersion=True)
elif args.task_name == "CustomDDM":
    task = CustomDDM(dt=0.0001)
elif args.task_name == "SimpleDDM":
    task = SimpleDDM(simulator="hssm(fine)")
    sampler_kwargs.update(
        {
            "initvals": {"v": 2.0, "a": 0.5, "t": 0.1},
        }
    )
else:
    raise ValueError(f"Invalid task name {args.task_name}")

task_name = task.name
exp_id = "pymc_runs"
if args.train:
    dataset_name = "train_dataset"
else:
    dataset_name = "test_dataset_chunk_1"

result_save_dir = Path(f"./results/{task_name}/{exp_id}/{dataset_name}")
result_save_dir.mkdir(parents=True, exist_ok=True)

test_dataset_path = Path(f"./results/{task_name}/datasets/{dataset_name}.pkl")
test_dataset = read_from_file(test_dataset_path)
if isinstance(test_dataset, dict):
    observables = test_dataset["observables"]
else:
    assert isinstance(test_dataset, np.ndarray), (
        f"Invalid test_dataset type {type(test_dataset)}"
    )
    observables = test_dataset
    test_dataset = {"observables": observables}

# %%
for observation_id in observation_ids:
    print("=" * 80)
    print(f"Observation ID: {observation_id}")
    file_save_path = (
        result_save_dir / f"records/observation_{observation_id}.pkl"
    )
    if file_save_path.exists():
        print("File found. Skipping.")
        continue
    file_save_path.parent.mkdir(parents=True, exist_ok=True)
    if task.name == "BernoulliGLM":
        observation = test_dataset["observables_raw"][observation_id]
    else:
        observation = test_dataset["observables"][observation_id]
    pymc_model = task.setup_pymc_model(observation=observation)
    if "CustomDDM" in args.task_name:
        sampler_kwargs.update(
            {
                "initvals": {
                    "v": np.array([1.0, 1.0]),
                    "a": np.array([1.0, 1.0]),
                    "ndt_plus": task.min_rts / 2,
                    "ndt_minus": task.min_rts / 2,
                },
            }
        )
    # %%
    tic = time.time()
    idata_post = pm.sample(
        draws=2500,
        chains=4,
        model=pymc_model,
        **sampler_kwargs,
    )
    runtime = time.time() - tic
    print(f"Runtime: {runtime:.2f}s")
    # %%
    axes = az.plot_pair(
        idata_post,
        var_names=task.var_names,
        kind="kde",
        marginals=True,
        divergences=True,
    )
    fig = axes.ravel()[0].figure
    fig_save_path = (
        result_save_dir / f"figures/observation_{observation_id}.png"
    )
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path)

    # %%
    idata_rhat = az.rhat(idata_post, var_names=list(task.var_names))
    result_record = {
        "idata_post": idata_post,
        "idata_rhat": idata_rhat,
        "time": runtime,
    }
    print("max rhat:", np.max(idata_rhat.to_array()))
    print("rhat:", idata_rhat)

    save_to_file(result_record, file_save_path)
