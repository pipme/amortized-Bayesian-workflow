import pytest

from amortized_bayesian_workflow.backends import get_backend


def test_get_backend_unknown_raises_helpful_error():
    with pytest.raises(KeyError, match="Available"):
        get_backend("missing_backend")


def test_get_backend_error_lists_blackjax_chees_hmc():
    with pytest.raises(KeyError) as exc:
        get_backend("missing_backend")
    assert "blackjax_chees_hmc" in str(exc.value)
