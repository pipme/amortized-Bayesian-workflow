# Project Guidelines

## Architecture
- This repository is a thin workflow layer around BayesFlow. Keep core orchestration inside `src/amortized_bayesian_workflow` and avoid duplicating BayesFlow internals.
- Central entry points:
	- `src/amortized_bayesian_workflow/workflow.py` (`WorkflowRunner`) for end-to-end execution.
	- `src/amortized_bayesian_workflow/report.py` (`WorkflowReport`, `DatasetResult`) for result aggregation.
	- `src/amortized_bayesian_workflow/config.py` (`WorkflowConfig`) for runtime behavior.
- Use protocol-oriented extension points:
	- Tasks implement the `WorkflowTask` protocol (`src/amortized_bayesian_workflow/tasks/base.py`).
	- Amortizers implement the `AmortizedPosterior` protocol (`src/amortized_bayesian_workflow/approximators/base.py`).
	- MCMC backends implement the sampler backend protocol and are resolved in `src/amortized_bayesian_workflow/backends/resolve.py`.

## Code Style
- Prefer simplicity and readability. Code should be straightforward to understand and maintain.
- Follow existing module style:
	- `from __future__ import annotations`
	- explicit NumPy/JAX array shape validation at boundaries
	- clear error messages when shape or backend assumptions fail
- Preserve the thin-wrapper philosophy: integrate with BayesFlow, do not re-implement BayesFlow behavior in this package.

## Build and Test
- Python: `>=3.10`
- Install:
	- `pip install -e .`
	- `pip install -e '.[dev]'`
	- Optional extras when needed: `.[bayesflow]`, `.[blackjax]`, `.[tfp]`, `.[pymc]`, or `.[all]`
- Test:
	- `pytest tests/`
	- Run targeted tests for changed modules first, then full `pytest tests/`.
- Correct python interpreter is required for test execution. For this project, I use `conda` env named `abw`.
  
## Conventions
- Keep public API changes minimal and intentional. Favor compatibility with the existing workflow surface (`WorkflowRunner`, task classes, report objects).
- Backend dependencies are optional. Guard optional imports and preserve graceful fallback/error behavior in backend resolution.
- For parallel execution, keep default sequential behavior stable (`parallel_mode="none"`) and treat concurrency changes as high-risk.
- Add or update tests in `tests/` for behavioral changes, especially around:
	- vectorized vs single log-posterior consistency
	- diagnostics and status transitions
	- backend resolution and optional dependency handling
	- keep only necessary tests in `tests/` and avoid unnecessary verbosity or duplication -- the test suit should be easy to run and maintain.

## Pitfalls
- BayesFlow compatibility is intentionally pinned (`bayesflow>=2.0.8,<3`); avoid broadening this range without validation.
- PyMC tasks and transformation utilities are sensitive to upstream API details and PyMC/PyTensor versions. Prefer small, well-tested edits in `src/amortized_bayesian_workflow/tasks/pymc_task.py` and `src/amortized_bayesian_workflow/tasks/pymc_utils.py`.
- Do not couple root package behavior to files inside the `BayesFlow/` subtree unless explicitly required.

## References
- Project overview and usage: `README.md`
- Core package code: `src/amortized_bayesian_workflow/`
- End-to-end usage examples: `examples/`
- BayesFlow upstream docs and contribution guidelines: `BayesFlow/README.md`, `BayesFlow/CONTRIBUTING.md`
