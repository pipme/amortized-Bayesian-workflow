import numpy as np

from amortized_bayesian_workflow.backends.utils import filter_initial_positions


def test_filter_initial_positions_drops_invalid_and_duplicates():
    positions = np.array([[0.0], [1.0], [1.0], [2.0]])

    def log_prob(x: np.ndarray) -> np.ndarray:
        out = -(x[:, 0] ** 2)
        out[x[:, 0] == 2.0] = -np.inf
        return out

    result = filter_initial_positions(positions, log_prob)
    assert result.positions.shape == (2, 1)
    assert np.allclose(result.positions[:, 0], np.array([0.0, 1.0]))
    assert np.all(np.isfinite(result.log_prob))


def test_filter_initial_positions_sorts_descending():
    positions = np.array([[0.0], [2.0], [1.0]])

    def log_prob(x: np.ndarray) -> np.ndarray:
        return np.array([0.2, 0.1, 0.9])

    result = filter_initial_positions(
        positions, log_prob, sort_descending=True, unique=False
    )
    assert np.allclose(result.log_prob, np.array([0.9, 0.2, 0.1]))

