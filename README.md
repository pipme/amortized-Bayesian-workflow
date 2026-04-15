# Amortized Bayesian Workflow

This repository contains a user-friendly and reusable implementation of [*Amortized Bayesian Workflow*](https://openreview.net/forum?id=osV7adJlKD). The paper presents a workflow for accelerating Bayesian posterior computation with three components: (1) amortized inference, (2) Pareto-smoothed importance sampling, and (3) many-chain MCMC initialized from amortized draws. 

(The original experiment code for the paper is available on the `code-submission` branch for reference.)

## Install

Recommended (includes optional backends and dev extras):

```bash
git clone https://github.com/pipme/amortized-Bayesian-workflow.git
cd amortized-Bayesian-workflow
pip install -e '.[all]'
```

Minimal install and add extras as needed:

```bash
pip install -e .
```

## Notebook

Please see [examples/workflow_step_by_step.ipynb](examples/workflow_step_by_step.ipynb) for a step-by-step demonstration of the workflow. It serves as a reusable template for new tasks and datasets.

## Citation

LI, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L., & Schmitt, M. (2026). Amortized Bayesian workflow. Transactions on Machine Learning Research.

### BibTeX
```bibtex
@article{liAmortizedBayesianWorkflow2026,
  title = {Amortized {{Bayesian}} Workflow},
  author = {LI, Chengkun and Vehtari, Aki and B{\"u}rkner, Paul-Christian and Radev, Stefan T. and Acerbi, Luigi and Schmitt, Marvin},
  year = 2026,
  journal = {Transactions on Machine Learning Research}
}
```
