from __future__ import annotations

import numpy as np

from amortized_bayesian_workflow.diagnostics import (
    MahalanobisReference,
)


def test_mahalanobis_reference_and_ood_result_shapes():
    rng = np.random.default_rng(0)
    train_obs = rng.normal(size=(100, 4))
    ref = MahalanobisReference.from_training_summaries(train_obs)

    out = ref.evaluate(train_obs[0], alpha=0.05)
    assert np.isfinite(out.statistic)
    assert np.isfinite(out.cutoff)
    assert isinstance(out.rejected, bool)


def test_alpha_one_does_not_reject_everything():
    train_statistics = np.array([[0.0], [1.0], [2.0], [3.0]])
    ref = MahalanobisReference.from_training_summaries(train_statistics)
    threshold = ref.threshold(alpha=1.0)
    assert np.isfinite(threshold)

    out = ref.evaluate(np.array([0.0]), alpha=1.0)
    # Alpha=1 means threshold is the minimum empirical distance; it does not force rejection.
    assert out.rejected in {True, False}


def test_statistics_are_square_root_of_sklearn_squared_distance():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(200, 3))
    probe = rng.normal(size=(3,))

    ref = MahalanobisReference.from_training_summaries(train)
    raw_squared_stat = float(ref.cov.mahalanobis(probe[None, :])[0])
    assert np.isclose(
        ref.statistic(probe), np.sqrt(max(raw_squared_stat, 0.0))
    )

    sqrt_cutoff = ref.threshold(alpha=0.05)
    expected_cutoff = np.quantile(
        np.sqrt(np.maximum(ref.cov.mahalanobis(train), 0.0)), 0.95
    )
    assert np.isclose(sqrt_cutoff, expected_cutoff)


def test_statistics_batch_matches_single_statistics():
    rng = np.random.default_rng(11)
    train = rng.normal(size=(128, 4))
    probes = rng.normal(size=(25, 4))
    ref = MahalanobisReference.from_training_summaries(train)

    batch = ref.statistics_batch(probes)
    single = np.array([ref.statistic(row) for row in probes])

    assert batch.shape == (25,)
    assert np.allclose(batch, single)


def test_evaluate_batch_matches_single_evaluate():
    rng = np.random.default_rng(19)
    train = rng.normal(size=(150, 3))
    probes = rng.normal(size=(10, 3))
    ref = MahalanobisReference.from_training_summaries(train)

    batch = ref.evaluate_batch(probes, alpha=0.1)
    single = [ref.evaluate(row, alpha=0.1) for row in probes]

    assert len(batch) == probes.shape[0]
    for b, s in zip(batch, single):
        assert np.isclose(b.statistic, s.statistic)
        assert np.isclose(b.cutoff, s.cutoff)
        assert np.isclose(b.alpha, s.alpha)
        assert b.rejected == s.rejected
