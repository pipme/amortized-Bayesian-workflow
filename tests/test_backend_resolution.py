import pytest

from amortized_bayesian_workflow.backends import get_backend


def test_get_backend_unknown_raises_helpful_error():
    with pytest.raises(KeyError, match="Available"):
        get_backend("missing_backend")
