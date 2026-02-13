# Code submission

Code submission for review of "Amortized Bayesian Workflow".

## Installation

Create and activate a virtual environment using `conda`:
```bash
conda env create -f environment.yml
conda activate abw_review
```

Install the package with the following commands:
```bash
python -m ipykernel install --user --name abw_review  # to use the conda environment in Jupyter Notebook
pip install -e ."[cpu]"  # install the package in editable mode for CPU, mainly for testing (it can be run with the GEV problem)
# pip install -e ."[gpu]"  # install the package in editable mode for GPU
pip install -e ./BayesFlow
```
(P.S. The `BayesFlow` package used in our experiments is a snapshot of a publicly available version of the original package.)

## Run the entire workflow

The full workflow can be run sequentially by executing the following notebook in `experiments/`:

- `01_dataset_preparation.ipynb`: prepare the test datasets for the problems in the manuscript.
- `02_amortized_training.ipynb`: train the amortized estimator, "Training phase: simulation-based optimization"
- `03_inference_phase_ood.ipynb`: OOD diagnostics, "Step 1: Amortized posterior draws"
- `04_amortized_inference.ipynb`: generating amortized draws, "Step 1: Amortized posterior draws"
- `05_inference_phase_psis.ipynb`: "Step 2: Pareto-smoothed importance sampling"
- `06_chees_hmc.ipynb`: "Step 3: Many-chains MCMC with amortized initializations", using the CHEES-HMC algorithm.


## Note

Scripts used for experiments are in the `experiments/` folder.

- `run_pymc.py`: generate reference posterior samples using PyMC/numpyro.
- `check_results.ipynb`: check the results of the experiments and compute the metrics.
- `figure_box_plot.ipynb`: Figure 3 in the main text.
- `figure_amortized_init.ipynb`: Figure 6 in the main text and Figure 5 in the appendix.

Folder `shell_scripts/` contains the shell scripts used to run the experiments on the cluster.

The checkpoint files for the trained models are provided in `*.keras` format, under the corresponding folders for each problem. The checkpoints can be used to resume training or to run inference without retraining the model.