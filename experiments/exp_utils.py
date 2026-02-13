from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_default_plot_settings():
    return plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,  # Thinner spines
            "grid.linewidth": 0.6,  # Thinner grid lines
            "lines.linewidth": 1.0,  # Line thickness
            "lines.markersize": 5,  # Default marker size
            "xtick.major.width": 0.8,  # Tick width
            "ytick.major.width": 0.8,
            "xtick.direction": "out",  # Ticks facing outward
            "ytick.direction": "out",
        }
    )


def read_reference_posterior(
    task,
    observation_id=0,
    source="pymc_runs",
    unconstrained=True,
    raise_error=False,
    return_all=False,
):
    import arviz as az
    from sbi_mcmc.utils.utils import read_from_file

    # assert source in ["pymc"]
    result_save_dir = Path(f"./results/{task.name}/")
    pymc_result_path = (
        result_save_dir / f"{source}/records/observation_{observation_id}.pkl"
    )
    try:
        pymc_result = read_from_file(pymc_result_path)
        if return_all:
            return pymc_result
    except FileNotFoundError:
        if raise_error:
            raise ValueError(f"{observation_id} not found")
        print(f"{observation_id} not found")
        return None
    idata_post_pymc = pymc_result["idata_post"]
    pymc_rhat = az.sel_utils.xarray_to_ndarray(pymc_result["idata_rhat"])[1]
    if not np.all(pymc_rhat < 1.01):
        print(f"{observation_id}: pymc rhat is {pymc_rhat} >= 1.01")
        if raise_error:
            raise ValueError("Rhat is too large")
    _, samples_pymc_constrained = az.sel_utils.xarray_to_ndarray(
        idata_post_pymc.posterior, var_names=task.var_names
    )
    samples_pymc_constrained = samples_pymc_constrained.T

    if unconstrained:
        samples_pymc_unconstrained = task.transform_to_unconstrained_space(
            samples_pymc_constrained
        )
        return samples_pymc_unconstrained
    return samples_pymc_constrained
