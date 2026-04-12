from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amortized_bayesian_workflow.backends.base import SamplerResult
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
        rng = np.random.default_rng(123)
        self.training_summaries = rng.normal(size=(256, 2))

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

    def summary_statistics(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations, dtype=float)


@dataclass(frozen=True)
class FakePSIS:
    pareto_k: float
    ess: float
    smoothed_normalized_weights: np.ndarray
    smoothed_log_weights: np.ndarray
    log_weights: np.ndarray | None = None
    num_proposal_samples: int = 1
    metadata: dict[str, float] | None = None

    def needs_mcmc(self) -> bool:
        return self.pareto_k > 0.7


class FakeBackend:
    name = "fake_mcmc"
    last_options = None

    def run(self, request):
        FakeBackend.last_options = dict(request.options)
        draws = np.repeat(
            request.initial_positions[:, None, :], request.num_samples, axis=1
        )
        return SamplerResult(
            backend=self.name,
            draws=draws,
            diagnostics={"rhat": 1.01},
            metadata={"backend_note": "fake"},
        )


class FakeCheesBackend:
    name = "blackjax_chees_hmc"
    last_options = None

    def run(self, request):
        FakeCheesBackend.last_options = dict(request.options)
        draws = np.repeat(
            request.initial_positions[:, None, :], request.num_samples, axis=1
        )
        return SamplerResult(
            backend=self.name,
            draws=draws,
            diagnostics={"nested_rhat": 1.01},
            metadata={},
        )


def make_fake_backend() -> FakeBackend:
    return FakeBackend()


def make_fake_chees_backend() -> FakeCheesBackend:
    return FakeCheesBackend()


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
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
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
    assert report.results[0].posterior_source == "psis_weighted"
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
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.linspace(0, -1, n),
            log_weights=np.linspace(0, -1, n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=16,
        mcmc_num_samples=5,
        mcmc_warmup=10,
        mcmc_sampler="fake_mcmc",
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
    assert first.posterior_source == "mcmc"
    assert first.mcmc["convergence_metric"] == "rhat"

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


def test_workflow_batch_mahalanobis_only_sends_ood_to_psis(monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    calls = {"n": 0}

    def fake_compute_psis(*, log_target, log_proposal):
        calls["n"] += 1
        n = len(log_target)
        return FakePSIS(
            pareto_k=0.1,
            ess=float(n),
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=30,
        mcmc_num_samples=8,
        parallel_mode="none",
        mahalanobis_alpha=0.05,
    )
    runner = WorkflowRunner(
        task=FakeTask(),
        approximator=FakeAmortizer(),
        config=cfg,
    )
    observations = [np.array([0.0, 0.0]), np.array([20.0, 20.0])]
    report = runner.run(observations)

    assert calls["n"] == 1
    assert report.results[0].amortized["ood_rejected"] is False
    assert report.results[1].amortized["ood_rejected"] is True


def test_workflow_mcmc_sampler_alias_and_chees_options(monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    def fake_compute_psis(*, log_target, log_proposal):
        n = len(log_target)
        return FakePSIS(
            pareto_k=2.0,
            ess=float(n / 2),
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.linspace(0, -1, n),
            log_weights=np.linspace(0, -1, n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=16,
        mcmc_num_samples=5,
        mcmc_warmup=10,
        mcmc_sampler="chees_hmc",
        mcmc_backend_options={
            "num_superchains": 3,
            "num_subchains_per_superchain": 2,
        },
        force_psis_for_all_datasets=True,
        parallel_mode="none",
    )
    runner = WorkflowRunner(
        task=FakeTask(),
        approximator=FakeAmortizer(),
        config=cfg,
        backend_factories={"blackjax_chees_hmc": make_fake_chees_backend},
    )

    report = runner.run([np.array([0.0, 0.0])])
    first = report.results[0]
    assert first.posterior_source == "mcmc"
    assert first.mcmc["backend"] == "blackjax_chees_hmc"
    assert first.mcmc["convergence_metric"] == "nested_rhat"
    assert FakeCheesBackend.last_options == {
        "num_superchains": 3,
        "subchains_per_superchain": 2,
    }


def test_psis_weighted_posterior_draws_are_resampled_not_raw(monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    def fake_compute_psis(*, log_target, log_proposal):
        n = len(log_target)
        weights = np.zeros(n, dtype=float)
        weights[0] = 1.0
        return FakePSIS(
            pareto_k=0.1,
            ess=1.0,
            smoothed_normalized_weights=weights,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=20,
        mcmc_num_samples=5,
        force_psis_for_all_datasets=True,
        parallel_mode="none",
    )
    runner = WorkflowRunner(
        task=FakeTask(), approximator=FakeAmortizer(), config=cfg
    )

    report = runner.run([np.array([0.0, 0.0])])
    result = report.results[0]
    assert result.posterior_source == "psis_weighted"
    draws = result.posterior_draws
    assert draws is not None
    assert draws.shape == (5, 2)
    # With all mass on index 0, all resampled rows are identical to first amortized draw.
    assert np.allclose(draws, np.repeat(draws[0:1], repeats=5, axis=0))


def test_workflow_report_save_and_load_roundtrip(tmp_path, monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod

    def fake_compute_psis(*, log_target, log_proposal):
        n = len(log_target)
        return FakePSIS(
            pareto_k=0.1,
            ess=float(n),
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=10,
        mcmc_num_samples=4,
        force_psis_for_all_datasets=True,
        parallel_mode="none",
    )
    runner = WorkflowRunner(
        task=FakeTask(), approximator=FakeAmortizer(), config=cfg
    )
    report = runner.run([np.array([0.0, 0.0])])

    out_dir = tmp_path / "report_artifact"
    report.save(out_dir)
    loaded = type(report).load(out_dir)

    assert loaded.results[0].dataset_id == report.results[0].dataset_id
    assert (
        loaded.results[0].posterior_source
        == report.results[0].posterior_source
    )
    assert np.allclose(
        loaded.results[0].amortized["draws"],
        report.results[0].amortized["draws"],
    )
    assert np.allclose(
        loaded.results[0].posterior_draws,
        report.results[0].posterior_draws,
    )
    assert (out_dir / "manifest.dill").exists()
    assert (out_dir / "datasets" / "dataset_0.dill").exists()


def test_workflow_persists_dataset_results_incrementally(
    tmp_path, monkeypatch
):
    import amortized_bayesian_workflow.workflow as wf_mod
    from amortized_bayesian_workflow.config import ArtifactLayout

    def fake_compute_psis(*, log_target, log_proposal):
        n = len(log_target)
        return FakePSIS(
            pareto_k=0.1,
            ess=float(n),
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    cfg = WorkflowConfig(
        num_amortized_draws=12,
        mcmc_num_samples=4,
        force_psis_for_all_datasets=True,
        parallel_mode="none",
        persist_dataset_results=True,
    )
    layout = ArtifactLayout(root=tmp_path, task_name="fake", run_name="r1")
    runner = WorkflowRunner(
        task=FakeTask(),
        approximator=FakeAmortizer(),
        config=cfg,
        layout=layout,
    )

    report = runner.run([np.array([0.0, 0.0]), np.array([1.0, -1.0])])

    assert len(report.results) == 2
    assert (layout.datasets_dir / "dataset_0.dill").exists()
    assert (layout.datasets_dir / "dataset_1.dill").exists()


def test_workflow_reuses_saved_dataset_results(tmp_path, monkeypatch):
    import amortized_bayesian_workflow.workflow as wf_mod
    from amortized_bayesian_workflow.config import ArtifactLayout
    from amortized_bayesian_workflow.utils import save_to_file

    calls = {"n": 0}

    def fake_compute_psis(*, log_target, log_proposal):
        calls["n"] += 1
        n = len(log_target)
        return FakePSIS(
            pareto_k=0.1,
            ess=float(n),
            smoothed_normalized_weights=np.ones(n) / n,
            smoothed_log_weights=np.zeros(n),
            log_weights=np.zeros(n),
            num_proposal_samples=n,
            metadata={"num_samples": float(n)},
        )

    monkeypatch.setattr(wf_mod, "compute_psis", fake_compute_psis)

    layout = ArtifactLayout(root=tmp_path, task_name="fake", run_name="r2")
    layout.ensure()
    cached = DatasetResult(
        dataset_id=0,
        status="success",
        message="cached",
        posterior_source="amortized",
        amortized={"draws": np.zeros((3, 2), dtype=float)},
    )
    save_to_file(
        cached,
        layout.datasets_dir / "dataset_0.dill",
        use_pickle=False,
    )

    cfg = WorkflowConfig(
        num_amortized_draws=12,
        mcmc_num_samples=4,
        force_psis_for_all_datasets=True,
        parallel_mode="none",
        persist_dataset_results=True,
    )
    runner = WorkflowRunner(
        task=FakeTask(),
        approximator=FakeAmortizer(),
        config=cfg,
        layout=layout,
    )

    report = runner.run([np.array([0.0, 0.0]), np.array([1.0, -1.0])])

    assert report.results[0].message == "cached"
    assert calls["n"] == 1
