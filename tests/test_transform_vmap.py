"""Tests for the JAX-vmapped space-transform functions.

The old Python-loop implementations are kept here as local *reference*
implementations.  The canonical, production implementations live in
``pymc_utils`` as the ``*_vmap`` variants.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pymc as pm
import pytest

from amortized_bayesian_workflow.tasks.pymc_utils import (
    _infer_value_var_dims_from_initial_point,
    _transform_to_constrained_space_vmap,
    _transform_to_unconstrained_space_vmap,
    dict_values_as_ndarray,
    get_pymc_free_RVs_names,
    ndarray_values_as_dict,
    var_name_to_variable,
)

# ---------------------------------------------------------------------------
# Reference implementations (old loop-based versions kept for testing only)
# ---------------------------------------------------------------------------


def _transform_to_unconstrained_space_ref(
    params_values: np.ndarray,
    var_dims: OrderedDict,
    pymc_model: pm.Model,
    in_place: bool = True,
) -> np.ndarray:
    """Loop-based reference: constrained → unconstrained."""
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)

    m = pymc_model
    transformed_rvs = []
    ind_dict: dict[str, int] = {}
    for free_rv in m.free_RVs:
        transform = m.rvs_to_transforms.get(free_rv)
        if transform is None:
            transformed_rvs.append(free_rv)
        else:
            transformed_rvs.append(
                transform.forward(free_rv, *free_rv.owner.inputs)
            )
        ind_dict[free_rv.name] = len(transformed_rvs) - 1

    fn = m.compile_fn(inputs=m.free_RVs, outs=transformed_rvs)
    N_sims = params_values.shape[0]
    params_values_transformed = []
    for i in range(N_sims):
        value_dict = ndarray_values_as_dict(params_values[i], var_dims)
        value_unconstrained_list = fn(value_dict)
        value_unconstrained_dict = {
            var_name: value_unconstrained_list[ind_dict[var_name]]
            for var_name in var_dims.keys()
        }
        value_unconstrained = dict_values_as_ndarray(
            value_unconstrained_dict, var_dims=var_dims
        )
        if in_place:
            params_values[i] = value_unconstrained
        else:
            params_values_transformed.append(value_unconstrained)

    if not in_place:
        return np.concatenate(params_values_transformed, axis=0)
    return params_values


def _transform_to_constrained_space_ref(
    params_values: np.ndarray,
    value_var_dims: OrderedDict,
    var_dims: OrderedDict,
    pymc_model: pm.Model,
    in_place: bool = True,
) -> np.ndarray:
    """Loop-based reference: unconstrained → constrained."""
    if params_values.ndim == 1:
        params_values = np.atleast_2d(params_values)

    m = pymc_model
    inputs = [var_name_to_variable(name, m) for name in value_var_dims.keys()]

    outputs = m.unobserved_value_vars
    fn_inv = m.compile_fn(outs=outputs)
    ind_dict: dict[str, int] = {
        value_var.name: j
        for j, value_var in enumerate(outputs)
        if value_var.name in var_dims.keys()
    }

    N_sims = params_values.shape[0]
    params_values_transformed = []
    for i in range(N_sims):
        value_dict = ndarray_values_as_dict(params_values[i], value_var_dims)
        value_constrained_list = fn_inv(value_dict)
        value_constrained_dict = {
            name: value_constrained_list[ind_dict[name]]
            for name in var_dims.keys()
        }
        value_constrained = dict_values_as_ndarray(
            value_constrained_dict, var_dims=var_dims
        )
        if in_place:
            params_values[i] = value_constrained
        else:
            params_values_transformed.append(value_constrained)

    if not in_place:
        return np.concatenate(params_values_transformed, axis=0)
    return params_values


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_no_transform():
    """All scalars, no transforms (Normal only)."""
    with pm.Model() as m:
        pm.Normal("x", mu=0, sigma=1)
        pm.Normal("y", mu=5, sigma=2)
        pm.Normal("z", mu=-3, sigma=0.5)
    return m


@pytest.fixture(scope="module")
def model_mixed_transforms():
    """Mixed: unconstrained + log + logit transforms."""
    with pm.Model() as m:
        pm.Normal("x", mu=0, sigma=1)  # no transform
        pm.HalfNormal("y", sigma=1)  # log transform (y > 0)
        pm.Uniform("z", lower=0, upper=10)  # logit-scale transform
    return m


@pytest.fixture(scope="module")
def model_vector_param():
    """Vector-valued parameters."""
    with pm.Model() as m:
        pm.Normal("mu", mu=0, sigma=1, shape=3)  # 3-dim, no transform
        pm.HalfNormal("sigma", sigma=1, shape=2)  # 2-dim, log transform
    return m


@pytest.fixture(scope="module")
def model_all_transforms():
    """All parameters require transforms."""
    with pm.Model() as m:
        pm.HalfNormal("a", sigma=1)
        pm.HalfNormal("b", sigma=2)
        pm.Uniform("c", lower=-5, upper=5)
        pm.Uniform("d", lower=1, upper=3)
    return m


@pytest.fixture(scope="module")
def model_ordered():
    """Ordered transform (requires sorted constrained samples)."""
    with pm.Model() as m:
        pm.Normal(
            "cutpoints",
            mu=0,
            sigma=1,
            shape=3,
            transform=pm.distributions.transforms.ordered,
        )
        pm.Normal("mu", mu=0, sigma=1)
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dims(model):
    """Return (var_dims, value_var_dims) for a model."""
    _, value_var_names, _ = get_pymc_free_RVs_names(model)
    free_rv_names, _, _ = get_pymc_free_RVs_names(model)
    value_var_dims = _infer_value_var_dims_from_initial_point(
        model, value_var_names
    )
    # var_dims mirrors value_var_dims (same total dims, different keys)
    name_to_value = {
        rv.name: model.rvs_to_values[rv].name for rv in model.free_RVs
    }
    value_to_name = {v: k for k, v in name_to_value.items()}
    var_dims = OrderedDict(
        (value_to_name[vv], value_var_dims[vv]) for vv in value_var_dims
    )
    return var_dims, value_var_dims


def _sample_constrained(model, var_dims, n_samples, rng):
    """Draw feasible (constrained-space) samples for a model."""
    D = sum(var_dims.values())
    params = np.zeros((n_samples, D))
    start = 0
    for rv in model.free_RVs:
        name = rv.name
        if name not in var_dims:
            continue
        dim = var_dims[name]
        transform = model.rvs_to_transforms.get(rv)
        vals = rng.normal(0, 1, size=(n_samples, dim))
        if transform is not None:
            # HalfNormal / similar: need positive values
            try:
                # Check if it's a log transform (positive support)
                _ = transform.forward(np.array([0.5] * dim), *rv.owner.inputs)
                vals = np.abs(vals) + 0.01
            except Exception:
                pass
            # Uniform: need values in [lower, upper]
            try:
                test = transform.forward(
                    np.array([0.5] * dim), *rv.owner.inputs
                )
                if hasattr(rv.owner.inputs[1], "data"):
                    lower = float(rv.owner.inputs[1].data)
                    upper = float(rv.owner.inputs[2].data)
                    vals = np.clip(np.abs(vals), lower + 0.01, upper - 0.01)
            except Exception:
                pass
        params[:, start : start + dim] = vals
        start += dim
    return params


RNG = np.random.default_rng(42)


def _constrained_samples_for(model, var_dims, n_samples):
    """Return feasible constrained samples via pm.sample_prior_predictive.

    Automatically sorts values for variables with the ``Ordered`` transform,
    matching the fix already present in ``PyMCTask._sample_prior_predictive``.
    """
    import arviz as az

    free_rv_names, _, transforms_map = get_pymc_free_RVs_names(model)
    idata = pm.sample_prior_predictive(
        draws=n_samples, model=model, random_seed=0
    )
    _, arr = az.sel_utils.xarray_to_ndarray(
        idata.prior, var_names=free_rv_names
    )
    arr = arr.T  # (n_samples, D_total) — columns in free_rv_names order

    # Reorder to match var_dims order (which comes from value_var_names order)
    D = sum(var_dims.values())
    result = np.zeros((n_samples, D))
    name_to_col_start: dict[str, int] = {}
    col = 0
    for name in free_rv_names:
        dim = var_dims[name]
        name_to_col_start[name] = col
        col += dim

    out_col = 0
    for name, dim in var_dims.items():
        src = name_to_col_start[name]
        result[:, out_col : out_col + dim] = arr[:, src : src + dim]
        out_col += dim

    # Sort values for Ordered-transform variables so forward transform is valid
    # (same logic as PyMCTask._sample_prior_predictive).
    out_col = 0
    for name, dim in var_dims.items():
        transform = transforms_map.get(name)
        if transform is not None and isinstance(
            transform, pm.distributions.transforms.Ordered
        ):
            result[:, out_col : out_col + dim] = np.sort(
                result[:, out_col : out_col + dim], axis=1
            )
        out_col += dim

    return result


# ---------------------------------------------------------------------------
# Parametrized model configurations
# ---------------------------------------------------------------------------

MODEL_FIXTURES = [
    "model_no_transform",
    "model_mixed_transforms",
    "model_vector_param",
    "model_all_transforms",
    "model_ordered",
]


# ---------------------------------------------------------------------------
# Tests: forward (constrained → unconstrained)
# ---------------------------------------------------------------------------


class TestUnconstrained:
    """Test _transform_to_unconstrained_space_vmap vs reference."""

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    @pytest.mark.parametrize("n_samples", [1, 20, 200])
    def test_matches_reference(self, model_fixture, n_samples, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)

        constrained = _constrained_samples_for(model, var_dims, n_samples)

        ref = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )
        got = _transform_to_unconstrained_space_vmap(
            constrained.copy(), var_dims, model, in_place=False
        )

        assert got.shape == ref.shape, (
            f"Shape mismatch: got {got.shape}, expected {ref.shape}"
        )
        np.testing.assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-8,
            err_msg=f"{model_fixture} n={n_samples}",
        )

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_output_is_finite(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, _ = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 30)
        got = _transform_to_unconstrained_space_vmap(
            constrained.copy(), var_dims, model, in_place=False
        )
        assert np.isfinite(got).all(), (
            f"{model_fixture}: output contains non-finite values"
        )

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_in_place_true_mutates_input(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, _ = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 10)
        original_id = id(constrained)

        returned = _transform_to_unconstrained_space_vmap(
            constrained, var_dims, model, in_place=True
        )
        # Same object must be returned and mutated
        assert id(returned) == original_id
        # Content must differ from original constrained samples
        ref = _transform_to_unconstrained_space_ref(
            _constrained_samples_for(model, var_dims, 10),
            var_dims,
            model,
            in_place=False,
        )
        np.testing.assert_allclose(constrained, ref, atol=1e-10)

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_in_place_false_does_not_mutate_input(
        self, model_fixture, request
    ):
        model = request.getfixturevalue(model_fixture)
        var_dims, _ = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 10)
        original = constrained.copy()

        _transform_to_unconstrained_space_vmap(
            constrained, var_dims, model, in_place=False
        )
        np.testing.assert_array_equal(constrained, original)

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_1d_input_accepted(self, model_fixture, request):
        """A 1-D input (single sample) should be promoted to 2-D."""
        model = request.getfixturevalue(model_fixture)
        var_dims, _ = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 1)
        flat = constrained[0]  # 1-D

        got = _transform_to_unconstrained_space_vmap(
            flat.copy(), var_dims, model, in_place=False
        )
        assert got.ndim == 2 and got.shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: inverse (unconstrained → constrained)
# ---------------------------------------------------------------------------


class TestConstrained:
    """Test _transform_to_constrained_space_vmap vs reference."""

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    @pytest.mark.parametrize("n_samples", [1, 20, 200])
    def test_matches_reference(self, model_fixture, n_samples, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)

        constrained = _constrained_samples_for(model, var_dims, n_samples)
        # Work in the unconstrained domain by applying the forward transform first
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )

        ref = _transform_to_constrained_space_ref(
            unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )
        got = _transform_to_constrained_space_vmap(
            unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )

        assert got.shape == ref.shape
        np.testing.assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-8,
            err_msg=f"{model_fixture} n={n_samples}",
        )

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_output_is_finite(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 30)
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )
        got = _transform_to_constrained_space_vmap(
            unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )
        assert np.isfinite(got).all()

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_in_place_true_mutates_input(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 10)
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained, var_dims, model, in_place=False
        )
        arr = unconstrained.copy()
        original_id = id(arr)

        returned = _transform_to_constrained_space_vmap(
            arr, value_var_dims, var_dims, model, in_place=True
        )
        assert id(returned) == original_id

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_in_place_false_does_not_mutate_input(
        self, model_fixture, request
    ):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 10)
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained, var_dims, model, in_place=False
        )
        original = unconstrained.copy()
        _transform_to_constrained_space_vmap(
            unconstrained, value_var_dims, var_dims, model, in_place=False
        )
        np.testing.assert_array_equal(unconstrained, original)

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_1d_input_accepted(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        constrained = _constrained_samples_for(model, var_dims, 1)
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )
        flat = unconstrained[0]

        got = _transform_to_constrained_space_vmap(
            flat.copy(), value_var_dims, var_dims, model, in_place=False
        )
        assert got.ndim == 2 and got.shape[0] == 1


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Composing forward + backward must be (close to) identity."""

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    @pytest.mark.parametrize("n_samples", [1, 50, 300])
    def test_constrained_roundtrip(self, model_fixture, n_samples, request):
        """constrained → unconstrained (vmap) → constrained (vmap) ≈ identity."""
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)

        original = _constrained_samples_for(model, var_dims, n_samples)

        unconstrained = _transform_to_unconstrained_space_vmap(
            original.copy(), var_dims, model, in_place=False
        )
        recovered = _transform_to_constrained_space_vmap(
            unconstrained, value_var_dims, var_dims, model, in_place=False
        )

        np.testing.assert_allclose(
            recovered,
            original,
            atol=1e-9,
            rtol=1e-7,
            err_msg=f"Round-trip failed: {model_fixture} n={n_samples}",
        )

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    @pytest.mark.parametrize("n_samples", [1, 50, 300])
    def test_unconstrained_roundtrip(self, model_fixture, n_samples, request):
        """unconstrained → constrained (vmap) → unconstrained (vmap) ≈ identity."""
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)

        constrained_original = _constrained_samples_for(
            model, var_dims, n_samples
        )
        original_unconstrained = _transform_to_unconstrained_space_ref(
            constrained_original.copy(), var_dims, model, in_place=False
        )

        constrained = _transform_to_constrained_space_vmap(
            original_unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )
        recovered_unconstrained = _transform_to_unconstrained_space_vmap(
            constrained, var_dims, model, in_place=False
        )

        np.testing.assert_allclose(
            recovered_unconstrained,
            original_unconstrained,
            atol=1e-9,
            rtol=1e-7,
            err_msg=f"Unconstrained round-trip failed: {model_fixture} n={n_samples}",
        )

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_vmap_ref_roundtrip_consistency(self, model_fixture, request):
        """The vmap round-trip error is no worse than the reference loop round-trip."""
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        n = 50
        original = _constrained_samples_for(model, var_dims, n)

        # Reference round-trip
        unc_ref = _transform_to_unconstrained_space_ref(
            original.copy(), var_dims, model, in_place=False
        )
        con_ref = _transform_to_constrained_space_ref(
            unc_ref, value_var_dims, var_dims, model, in_place=False
        )
        err_ref = np.max(np.abs(original - con_ref))

        # Vmap round-trip
        unc_vmap = _transform_to_unconstrained_space_vmap(
            original.copy(), var_dims, model, in_place=False
        )
        con_vmap = _transform_to_constrained_space_vmap(
            unc_vmap, value_var_dims, var_dims, model, in_place=False
        )
        err_vmap = np.max(np.abs(original - con_vmap))

        # vmap error should be within 100 × machine-epsilon of reference error
        assert err_vmap <= err_ref + 1e-12, (
            f"{model_fixture}: vmap round-trip error ({err_vmap:.2e}) "
            f"exceeds reference ({err_ref:.2e})"
        )


# ---------------------------------------------------------------------------
# Numerical precision
# ---------------------------------------------------------------------------


class TestNumericalPrecision:
    """Ensure differences from reference are within floating-point tolerance."""

    MODELS = MODEL_FIXTURES
    ABS_TOL = 1e-10

    @pytest.mark.parametrize("model_fixture", MODELS)
    def test_forward_max_abs_diff(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, _ = _make_dims(model)
        n = 100
        constrained = _constrained_samples_for(model, var_dims, n)

        ref = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )
        got = _transform_to_unconstrained_space_vmap(
            constrained.copy(), var_dims, model, in_place=False
        )
        max_diff = np.max(np.abs(ref - got))
        assert max_diff <= self.ABS_TOL, (
            f"{model_fixture}: forward max diff {max_diff:.2e} > {self.ABS_TOL}"
        )

    @pytest.mark.parametrize("model_fixture", MODELS)
    def test_backward_max_abs_diff(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        n = 100
        constrained = _constrained_samples_for(model, var_dims, n)
        unconstrained = _transform_to_unconstrained_space_ref(
            constrained.copy(), var_dims, model, in_place=False
        )

        ref = _transform_to_constrained_space_ref(
            unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )
        got = _transform_to_constrained_space_vmap(
            unconstrained.copy(),
            value_var_dims,
            var_dims,
            model,
            in_place=False,
        )
        max_diff = np.max(np.abs(ref - got))
        assert max_diff <= self.ABS_TOL, (
            f"{model_fixture}: backward max diff {max_diff:.2e} > {self.ABS_TOL}"
        )

    @pytest.mark.parametrize("model_fixture", MODELS)
    def test_round_trip_max_abs_diff(self, model_fixture, request):
        model = request.getfixturevalue(model_fixture)
        var_dims, value_var_dims = _make_dims(model)
        n = 100
        original = _constrained_samples_for(model, var_dims, n)

        unc = _transform_to_unconstrained_space_vmap(
            original.copy(), var_dims, model, in_place=False
        )
        recovered = _transform_to_constrained_space_vmap(
            unc, value_var_dims, var_dims, model, in_place=False
        )
        max_diff = np.max(np.abs(original - recovered))
        # Allow slightly more tolerance for composed transforms
        assert max_diff <= 1e-9, (
            f"{model_fixture}: round-trip max diff {max_diff:.2e} > 1e-9"
        )
