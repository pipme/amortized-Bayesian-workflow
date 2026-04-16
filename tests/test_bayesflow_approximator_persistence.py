from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from amortized_bayesian_workflow.approximators import (
    BayesFlowAmortizedPosterior,
)


class _DummyApproximator:
    def __init__(self):
        self.saved_path: str | None = None

    def save(self, path: str) -> None:
        self.saved_path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("dummy")


class _BatchApproximator:
    def sample(self, *, num_samples: int, conditions, seed: int):
        _ = seed
        obs = np.asarray(conditions["x_obs"])
        num_datasets = obs.shape[0]
        theta = np.zeros((num_datasets, num_samples, 2), dtype=float)
        for i in range(num_datasets):
            theta[i, :, 0] = i
            theta[i, :, 1] = np.arange(num_samples, dtype=float)
        return {"theta": theta}

    def log_prob(self, payload):
        obs = np.asarray(payload["x_obs"])
        theta = np.asarray(payload["theta"])
        return theta.sum(axis=-1) + obs.sum(axis=-1)


def test_save_and_load_roundtrip(tmp_path):
    approx = _DummyApproximator()
    posterior = BayesFlowAmortizedPosterior(
        approximator=approx,
        observable_key="x_obs",
        training_summaries=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    out_dir = tmp_path / "posterior"
    posterior.save(out_dir)

    loaded_approx = _DummyApproximator()
    fake_keras = SimpleNamespace(
        saving=SimpleNamespace(load_model=lambda _: loaded_approx)
    )
    original_keras = sys.modules.get("keras")
    sys.modules["keras"] = fake_keras
    try:
        restored = BayesFlowAmortizedPosterior.load(out_dir)
    finally:
        if original_keras is not None:
            sys.modules["keras"] = original_keras
        else:
            del sys.modules["keras"]

    assert approx.saved_path == str(out_dir / "approximator.keras")
    assert restored.approximator is loaded_approx
    assert restored.observable_key == "x_obs"
    assert np.allclose(
        restored.training_summaries, posterior.training_summaries
    )


def test_load_raises_if_metadata_missing(tmp_path):
    out_dir = tmp_path / "posterior"
    out_dir.mkdir()
    (out_dir / "approximator.keras").write_text("dummy", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        BayesFlowAmortizedPosterior.load(out_dir)


def test_sample_and_log_prob_batch_returns_one_result_per_dataset():
    posterior = BayesFlowAmortizedPosterior(
        approximator=_BatchApproximator(),
        observable_key="x_obs",
    )
    observations = np.array([[10.0, 1.0], [20.0, 2.0]])

    draws = posterior.sample_and_log_prob_batch(
        observations,
        num_samples=3,
        seed=7,
    )

    assert len(draws) == 2
    assert draws[0].samples.shape == (3, 2)
    assert draws[1].samples.shape == (3, 2)
    assert np.allclose(draws[0].samples[:, 0], 0.0)
    assert np.allclose(draws[1].samples[:, 0], 1.0)
    assert np.allclose(draws[0].log_prob, np.array([11.0, 12.0, 13.0]))
    assert np.allclose(draws[1].log_prob, np.array([23.0, 24.0, 25.0]))
    assert draws[0].metadata["seed"] == 7
    assert draws[1].metadata["dataset_index"] == 1
