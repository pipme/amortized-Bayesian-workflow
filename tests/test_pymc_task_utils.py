from __future__ import annotations

from collections import OrderedDict

import numpy as np

from amortized_bayesian_workflow.tasks.pymc_task import _ndarray_values_as_dict


def test_ndarray_values_as_dict_scalar_dim_returns_scalar_for_single_row():
    values = np.array([[1.23, 4.56]], dtype=float)
    dims = OrderedDict([("theta", 1), ("beta", 1)])
    out = _ndarray_values_as_dict(values, dims)
    assert np.isscalar(out["theta"])
    assert np.isscalar(out["beta"])
    assert float(out["theta"]) == 1.23
    assert float(out["beta"]) == 4.56
