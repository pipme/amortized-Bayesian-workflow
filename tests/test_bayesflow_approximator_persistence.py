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
