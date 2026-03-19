from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amortized_bayesian_workflow.config import WorkflowConfig
from amortized_bayesian_workflow.report import DatasetResult
from amortized_bayesian_workflow.tasks.base import TaskMetadata
from amortized_bayesian_workflow.workflow import WorkflowRunner


class FakeTask:
    metadata = TaskMetadata(
        name="fake",
        parameter_names=("theta",),
        parameter_dims={"theta": 2},
    )

    def sample_prior_predictive(self, num_samples: int, *, seed: int):
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(num_samples, 2))
        x = theta + rng.normal(scale=0.1, size=(num_samples, 2))
        return {"parameters": theta, "observables": x}

    def vectorized_log_posterior_fn(self, observation: np.ndarray):
        obs = np.asarray(observation)

        def fn(theta_batch: np.ndarray) -> np.ndarray:
            diff = np.asarray(theta_batch) - obs
            return -0.5 * np.sum(diff**2, axis=1)

        return fn

    def single_log_posterior_fn(self, observation: np.ndarray):
        obs = np.asarray(observation)

        def fn(theta):
            diff = np.asarray(theta) - obs
            return -0.5 * np.sum(diff**2)

        return fn


class FakeAmortizer:
    def __init__(self):
        self.fit_calls = 0

    def fit(
        self,
        train_data=None,
        validation_data=None,
        num_epochs=None,
        batch_size=None,
    ):
        self.fit_calls += 1
        return {
            "train_n": len(train_data["parameters"]),
            "val_n": len(validation_data["parameters"]),
            "epochs": num_epochs,
            "batch_size": batch_size,
        }

    def sample_and_log_prob(self, observation, *, num_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        obs = np.asarray(observation)
        samples = obs + rng.normal(
            scale=0.5, size=(num_samples, obs.shape[-1])
        )
        log_prob = -0.5 * np.sum((samples - obs) ** 2, axis=1) - 0.1
        return type("Draws", (), {"samples": samples, "log_prob": log_prob})()


@dataclass(frozen=True)
class FakePSIS:
    pareto_k: float
    ess: float
    normalized_weights: np.ndarray
    smoothed_log_weights: np.ndarray

    def needs_mcmc(self, threshold: float) -> bool:
        return self.pareto_k > threshold


class FakeBackend:
    name = "fake_mcmc"

    def run(self, request):
        draws = np.repeat(
            request.initial_positions[:, None, :], request.num_samples, axis=1
        )
        return type(
            "Res",
            (),
            {
                "backend": self.name,
                "draws": draws,
                "diagnostics": {"nested_rhat": 1.01},
                "metadata": {"backend_note": "fake"},
            },
        )()


def make_fake_backend() -> FakeBackend:
    return FakeBackend()


def test_workflow_fit_uses_simulator_data():
    runner = WorkflowRunner(
        task=FakeTask(), approximator=FakeAmortizer(), config=WorkflowConfig()
    )
    out = runner.amortized_training()
    assert out["train_n"] == runner.config.num_train_simulations
    assert out["val_n"] == runner.config.num_validation_simulations


def test_workflow_run_psis_only(monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    def fake_compute_psis(*, log_target, log_proposal):
        n = len(log_target)
        return FakePSIS(
            pareto_k=0.2,
            ess=float(n),
            normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=20,
        mcmc_num_samples=10,
        force_psis_for_all_datasets=True,
        parallel_mode="none",
    )
    runner = WorkflowRunner(
        task=FakeTask(), approximator=FakeAmortizer(), config=cfg
    )
    report = runner.run([np.array([0.0, 0.0]), np.array([1.0, -1.0])])

    assert report.status_counts()["success"] == 2
    assert len(report.collect_posterior_draws()) == 2


def test_workflow_run_triggers_mcmc_and_retry_failed(monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    calls = {"n": 0}

    def fake_compute_psis(*, log_target, log_proposal):
        calls["n"] += 1
        n = len(log_target)
        pareto_k = 2.0 if calls["n"] == 1 else 0.1
        return FakePSIS(
            pareto_k=pareto_k,
            ess=float(n / 2),
            normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.linspace(0, -1, n),
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=16,
        mcmc_num_samples=5,
        mcmc_warmup=10,
        mcmc_backend="fake_mcmc",
        force_psis_for_all_datasets=True,
        parallel_mode="none",
    )
    runner = WorkflowRunner(
        task=FakeTask(),
        approximator=FakeAmortizer(),
        config=cfg,
        backend_factories={"fake_mcmc": make_fake_backend},
    )
    observations = [np.array([0.0, 0.0])]
    report = runner.run(observations)
    first = report.results[0]
    assert isinstance(first, DatasetResult)
    assert first.mcmc["backend"] == "fake_mcmc"

    # Force a failure, then ensure retry replaces the result.
    failed = report.with_replacements(
        {
            0: DatasetResult(
                dataset_id=0,
                status="failed",
                message="boom",
                error="test",
            )
        }
    )
    retried = runner.retry_failed(failed, observations)
    assert retried.results[0].status in {"success", "needs_review"}
