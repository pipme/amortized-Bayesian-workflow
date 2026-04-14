# Amortized Bayesian Workflow

This branch is the refactored, user-facing implementation of the amortized Bayesian workflow described in the paper [*Amortized Bayesian Workflow*](https://openreview.net/forum?id=osV7adJlKD).

The original experiment code remains on the `code-submission` branch. It is useful as a reference, but it mixes experiment scripts, path management, BayesFlow setup, and MCMC backends in ways that are hard to reuse.

## Project Philosophy (BayesFlow-First)

This package is intentionally a **thin workflow layer on top of BayesFlow**.

- **BayesFlow** handles amortized inference model construction and training
- **ABW (this package)** handles workflow orchestration after/beside amortization:
  - amortized draws
  - PSIS diagnostics
  - optional MCMC refinement
  - dataset-level status tracking
  - retrying failed datasets

This design keeps the package easier to maintain and easier to understand for non-expert users.

## Current Refactor Status

The `main` branch now provides a clean foundation for incremental migration:

- A typed package layout under `src/amortized_bayesian_workflow`
- A minimal workflow orchestrator (`WorkflowRunner`) + result objects (`WorkflowReport`)
- A backend plugin interface for MCMC samplers (optional, internal to workflow execution)
- Optional backends for:
  - `tfp_chees_hmc` (compatibility backend; implemented)
  - `blackjax_nuts` (implemented, optional `blackjax` dependency; JAX is required by the base package)
- A small test suite for the new core utilities
- A notebook-friendly workflow runner with dataset-level status reporting and retries
- A real ported example task: GEV (PyMC-based) under `amortized_bayesian_workflow.tasks.examples`
- A thin optional BayesFlow convenience builder for reducing notebook boilerplate

## Design goals

- Keep the workflow easy to use and easy to read
- Make TensorFlow Probability optional (not a hard dependency)
- Enable a gradual move to BlackJAX without changing user-facing APIs
- Stay compatible with modern BayesFlow releases via dependency pinning (`bayesflow>=2.0.8,<3`)

## Installation

Base package (includes JAX, which the workflow relies on):

```bash
pip install -e .
```

With BayesFlow:

```bash
pip install -e '.[bayesflow]'
```

With optional MCMC backends:

```bash
pip install -e '.[blackjax]'
pip install -e '.[tfp]'
```

## Logging Configuration

ABW defaults to WARNING-level logging. You can change this globally in notebooks,
scripts, or shell environments.

Notebook or script:

```python
import amortized_bayesian_workflow as abw

# ABW log level only
abw.configure_logging("INFO")

# ABW + noisy external loggers used in common workflows (for example PyMC forward sampling)
abw.configure_logging("WARNING", include_external_loggers=True)
```

Shell (affects import-time defaults for the process):

```bash
export ABW_LOG_LEVEL=ERROR
export ABW_LOG_INCLUDE_EXTERNAL=1
python your_script.py
```

For a clean API, logging is configured via `configure_logging(...)` (runtime)
or `ABW_LOG_LEVEL` / `ABW_LOG_INCLUDE_EXTERNAL` (process-level defaults).

## Where To Start

Start here if you are new to the package:

1. **Quick local workflow (recommended first)**  
   Open `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/quickstart_workflow.ipynb`
2. **Real task example (GEV + PyMC + BayesFlow 2.0.8)**  
   Open `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/gev_bayesflow_2_0_8_demo.ipynb`
3. **Fine control / debugging notebook (step-by-step pipeline)**  
   Open `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/workflow_step_by_step_debug.ipynb`
4. **Complete `JAXTask` example script**  
   Run `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/jax_task_complete_example.py`
5. **Complete `PyMCTask` example script**  
   Run `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/pymc_task_complete_example.py`

Recommended learning order:
- First run the quickstart notebook end-to-end on a small example
- Then run the GEV notebook
- Use the step-by-step debug notebook when you need finer control or diagnostics

Default execution mode:
- **Sequential local processing** (`parallel_mode="none"`) is the default and best-supported path
- Local parallel processing is optional
- SLURM is an advanced deployment option (see `scripts/slurm/`)

## Mental Model

Use the package in two layers:

1. **BayesFlow layer (training amortizer)**
   - Build adapter
   - Choose networks
   - Train `ContinuousApproximator`
2. **ABW layer (workflow execution)**
   - Wrap the trained BayesFlow approximator with `BayesFlowAmortizedPosterior`
   - Run `WorkflowRunner.run(...)`
   - Inspect statuses / diagnostics / failures
   - Retry difficult datasets if needed

## Complete Task Examples

Run these directly from the repo root:

```bash
python examples/jax_task_complete_example.py
python examples/pymc_task_complete_example.py
```

Both scripts demonstrate:
- defining a task (`JAXTask` or `PyMCTask`)
- fitting/running `WorkflowRunner` sequentially (default path)
- inspecting per-dataset status and diagnostics
- retrying failed datasets with updated config

## Quickstart shape (notebook-friendly)

```python
from amortized_bayesian_workflow import WorkflowConfig, WorkflowRunner
from amortized_bayesian_workflow.tasks import JAXTask

# Define a task (JAX simulator + log prior + log likelihood)
task = JAXTask(...)

# Wrap a trained BayesFlow approximator (or any object implementing sample_and_log_prob)
amortizer = ...

workflow = WorkflowRunner(
    task=task,
    approximator=amortizer,
    config=WorkflowConfig(),
    # optional: handcrafted summary statistics for Mahalanobis OOD diagnostics
    # diagnostic_summary_fn=lambda observations: my_summary_fn(observations),
)

workflow.fit()  # optional if amortizer is already trained
report = workflow.run(observations_batch)

report.summary_table()
report.failed_datasets()
draws = report.collect_posterior_draws()
```

Optional helper to reduce BayesFlow boilerplate (still BayesFlow under the hood):

```python
from amortized_bayesian_workflow.approximators import make_default_bayesflow_posterior

amortizer = make_default_bayesflow_posterior()
```

For full control, prefer constructing BayesFlow objects directly and wrapping them with:

```python
from amortized_bayesian_workflow.approximators import BayesFlowAmortizedPosterior
amortizer = BayesFlowAmortizedPosterior(bf_approximator)
```

Retry only failed datasets with different settings:

```python
report = workflow.retry_failed(
    report,
    observations_batch,
   config_override={
      "mcmc_backend": "tfp_chees_hmc",
      "mcmc_backend_options": {"num_warmup": 2000},
   },
)
```

### Amortized-draw OOD gate (Mahalanobis distance)

Before PSIS, the workflow can screen datasets using a Mahalanobis-distance OOD diagnostic
computed from training summary statistics.

Default behavior:
- ABW uses summary statistics extracted from the amortized approximator (e.g., BayesFlow learned
  summary/inference features) when available.
- If an approximator does not expose summary features, ABW falls back to flattened observables.

Advanced override:
- You can pass handcrafted summary statistics directly with
  `WorkflowRunner(..., diagnostic_summary_fn=...)`.

By default, datasets in the right `alpha=0.05` tail are flagged and sent to PSIS/MCMC.

```python
config = WorkflowConfig(
    mahalanobis_alpha=0.05,             # default
    force_psis_for_all_datasets=False,  # default
)
```

If you want to send every dataset to PSIS (i.e., reject amortized draws for all datasets),
set:

```python
config = WorkflowConfig(force_psis_for_all_datasets=True)
```

Caution:
- Setting `mahalanobis_alpha=1.0` does **not** mean "reject all datasets". It sets the
  threshold to the empirical minimum training distance.
- Future versions may add stronger amortized-draw diagnostics; the Mahalanobis gate is kept
  configurable for this reason.

### PyMC Model (Minimal Wrapping)

For PyMC users, the easiest path is to define a PyMC model builder and let `PyMCTask` wrap it:

```python
from amortized_bayesian_workflow.tasks import PyMCTask

def build_model(observation=None):
    import pymc as pm
    obs = observation if observation is not None else ...
    with pm.Model() as model:
        x = pm.Data("x", obs)
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("y", mu=theta, sigma=1, observed=x)
    return model

task = PyMCTask.from_model_builder(
    model_builder=build_model,
    observable_var_name="y",  # optional if your model has exactly one observed RV
)
```

`PyMCTask.from_model_builder(...)` automatically handles:
- parameter ordering / value-variable ordering
- prior predictive simulations
- transformation to PyMC unconstrained space
- JAXified posterior log-density for PSIS/MCMC

## Real Task Demo

- GEV (ported from `code-submission`): `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/gev_bayesflow_2_0_8_demo.ipynb`
- Task factory: `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/src/amortized_bayesian_workflow/tasks/examples/gev.py`
- Step-by-step debugging notebook (training -> amortized draws -> diagnostics -> PSIS -> MCMC): `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/workflow_step_by_step_debug.ipynb`

## Local Workflow (Recommended)

For practical applications, the recommended path is:

1. Train (or load) an amortized model locally
2. Run `workflow.run(observations_batch)` sequentially
3. Inspect `report.summary_table()` and `report.failed_datasets()`
4. Retry failed / `needs_review` datasets with `workflow.retry_failed(...)`
5. Only move to SLURM if the dataset batch is too large for a local run

This keeps the workflow simple and reproducible while preserving a clear path to advanced parallel execution when needed.

## Public API Surface (Minimal)

Most users only need:

- `WorkflowRunner`
- `WorkflowConfig`
- `WorkflowReport`
- `JAXTask` or `PyMCTask`
- `BayesFlowAmortizedPosterior`

Optional convenience (small wrappers around BayesFlow defaults):
- `make_default_bayesflow_posterior`
- `build_default_bayesflow_continuous_approximator`

## Advanced: SLURM Array Template (Optional)

The package is documented for **sequential local execution** first.

For HPC/SLURM users, minimal templates are provided under:
- `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/scripts/slurm/run_dataset.py`
- `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/scripts/slurm/run_array.sbatch.template`
- `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/scripts/slurm/merge_results.py`
- `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/factories.py` (copy/edit this first)

These are thin wrappers around the same core API (`WorkflowRunner.run(...)`) and are intentionally kept outside the main package surface.

Important:
- Keep `parallel_mode=\"none\"` inside each SLURM array task.
- Let SLURM provide dataset-level parallelism.
- Start by copying and editing `/Users/lichengk/project_results/amortized_bayesian_workflow/code-release/examples/factories.py` for your own task/checkpoint loading.

## Migration plan (next steps)

1. Move task definitions from `code-submission` into `src/.../tasks` with a stable task protocol.
2. Port BayesFlow adapter/training helpers into `src/.../approximators` with BayesFlow 2.x APIs.
3. Port PSIS and diagnostics into `src/.../diagnostics` with tests.
4. Add richer built-in diagnostics/plots and export helpers for downstream analysis.
5. Add CLI/examples for end-to-end runs on a small demo task.
