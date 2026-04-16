# Project Guidelines
Echo in the output: "I am reading the project guidelines in AGENTS.md to understand the coding standards and architectural principles for this repository."
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- No unnecessary fallbacks or edge cases. If a package is required, require it - don't add a fallback that adds complexity.
- No need to consider backward compatibility unless explicitly requested.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
- Don't edit README.md unless explicitly asked.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Architecture
- This repository is a thin workflow layer around BayesFlow. Keep core orchestration inside `src/amortized_bayesian_workflow` and avoid duplicating BayesFlow internals.
- Central entry points:
	- `src/amortized_bayesian_workflow/workflow.py` (`InferenceRunner`) for end-to-end execution.
	- `src/amortized_bayesian_workflow/report.py` (`WorkflowReport`, `DatasetResult`) for result aggregation.
	- `src/amortized_bayesian_workflow/config.py` (`InferenceConfig`) for runtime behavior.
- Use protocol-oriented extension points:
	- Tasks implement the `WorkflowTask` protocol (`src/amortized_bayesian_workflow/tasks/base.py`).
	- Amortizers implement the `AmortizedPosterior` protocol (`src/amortized_bayesian_workflow/approximators/base.py`).
	- MCMC backends implement the sampler backend protocol and are resolved in `src/amortized_bayesian_workflow/backends/resolve.py`.

## Code Style
- Prefer **simplicity** and **readability**. Code should be straightforward to understand and maintain.
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
    - No need to run tests unless making substantial changes
	- Run targeted tests only for changed modules
- Correct python interpreter is required for test execution. For this project, I use `conda` env named `abw`.
  
## Conventions
- Keep public API changes minimal and intentional. Avoid unnecessary complexity in the entire codebase for ease of maintenance.
- Backend dependencies are optional. Guard optional imports and preserve graceful fallback/error behavior in backend resolution.
- For parallel execution, keep default sequential behavior stable (`parallel_mode="none"`) and treat concurrency changes as high-risk.
- Add or update tests in `tests/` for significant changes, especially around:
	- vectorized vs single log-posterior consistency
	- diagnostics and status transitions
	- backend resolution and optional dependency handling
	- keep only absolutely necessary tests in `tests/` and avoid unnecessary verbosity or duplication -- the test suit should be easy to run and maintain.

## Pitfalls
- BayesFlow compatibility is intentionally pinned (`bayesflow>=2.0.10,<3`); avoid broadening this range without validation.
- PyMC tasks and transformation utilities are sensitive to upstream API details and PyMC/PyTensor versions. Prefer small, well-tested edits in `src/amortized_bayesian_workflow/tasks/pymc_task.py` and `src/amortized_bayesian_workflow/tasks/pymc_utils.py`.
- Do not couple root package behavior to files inside the `BayesFlow/` subtree unless explicitly required.

## References
- Project overview and usage: `README.md`
- Core package code: `src/amortized_bayesian_workflow/`
- End-to-end usage examples: `examples/`
- BayesFlow upstream docs and contribution guidelines: `BayesFlow/README.md`, `BayesFlow/CONTRIBUTING.md`
