"""C1 hardening (smoothness=1) and the hinge-form Pyomo exports."""
from __future__ import annotations

import numpy as np
import pytest

from difftree import DifferentiableTreeRegressor, export
from difftree import _hinge

TIED = dict(leaf_type="affine", tied_splits=True, continuous=True, hardening=True,
            n_hard_epochs=10, random_state=0)


@pytest.fixture(scope="module")
def sine():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (300, 1))
    y = np.sin(2 * X[:, 0]) + 0.03 * rng.normal(size=300)
    return X, y


@pytest.fixture(scope="module")
def sine_c0(sine):
    return DifferentiableTreeRegressor(depth=3, n_epochs=100, **TIED).fit(*sine)


@pytest.fixture(scope="module")
def sine_c1(sine):
    return DifferentiableTreeRegressor(depth=3, n_epochs=100, smoothness=1, **TIED).fit(*sine)


@pytest.fixture(scope="module")
def bump():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (300, 2))
    y = np.exp(-np.sum(X**2, axis=1)) + 0.01 * rng.normal(size=300)
    return X, y


@pytest.fixture(scope="module")
def bump_c1(bump):
    return DifferentiableTreeRegressor(depth=3, n_epochs=60, smoothness=1, **TIED).fit(*bump)


@pytest.fixture(scope="module")
def bump_c0(bump):
    return DifferentiableTreeRegressor(depth=3, n_epochs=60, **TIED).fit(*bump)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# The smoothed hinge itself
# ---------------------------------------------------------------------------


def test_phi_is_c1_and_reduces_to_relu():
    h = np.linspace(-2, 2, 4001)
    for eps in (0.1, 0.5, 1.0):
        v, dv = _hinge.phi(h, eps), _hinge.dphi(h, eps)
        np.testing.assert_allclose(v[np.abs(h) >= eps], np.maximum(h, 0)[np.abs(h) >= eps])
        np.testing.assert_allclose(np.gradient(v, h)[1:-1], dv[1:-1], atol=2e-3)
        assert np.max(np.abs(np.diff(dv))) <= (h[1] - h[0]) / (2 * eps) + 1e-12
    np.testing.assert_array_equal(_hinge.phi(h, 0.0), np.maximum(h, 0))


def test_pwl_terms_interpolate_phi():
    eps, K = 0.7, 5
    t, a = _hinge.pwl_terms(eps, K)
    h = np.linspace(-2, 2, 2001)
    approx = (np.maximum(h[:, None] - t[None], 0) * a).sum(axis=1)
    np.testing.assert_allclose(approx[np.abs(h) >= eps], np.maximum(h, 0)[np.abs(h) >= eps], atol=1e-12)
    np.testing.assert_allclose((np.maximum(t[:, None] - t[None], 0) * a).sum(axis=1), _hinge.phi(t, eps), atol=1e-12)
    err = approx - _hinge.phi(h, eps)
    assert err.min() >= -1e-12
    assert err.max() == pytest.approx(eps / (4 * K**2), rel=1e-3)


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------


def test_smoothness_validation(sine):
    X, y = sine
    with pytest.raises(ValueError, match="smoothness must be 0 or 1"):
        DifferentiableTreeRegressor(smoothness=2, n_epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="tied_splits"):
        DifferentiableTreeRegressor(leaf_type="affine", continuous=True, smoothness=1,
                                    n_epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="continuous"):
        DifferentiableTreeRegressor(leaf_type="affine", tied_splits=True, smoothness=1,
                                    n_epochs=1).fit(X, y)


def test_c1_gradient_is_continuous(sine_c0, sine_c1):
    xg = np.linspace(-2, 2, 4001)[:, None]
    J1 = sine_c1.predict_jacobian(xg, hard=True)[:, 0]
    J0 = sine_c0.predict_jacobian(xg, hard=True)[:, 0]
    # Neighbouring grid points: C1 gradient changes by at most curvature * dx.
    assert np.max(np.abs(np.diff(J1))) < 0.05
    assert np.max(np.abs(np.diff(J0))) > 0.5
    assert sine_c1.hardening_report_["max_gradient_jump"][0] == 0.0
    assert sine_c0.hardening_report_["max_gradient_jump"][0] > 0.5
    assert sine_c1.hardening_report_["max_boundary_jump"][0] < 1e-8
    assert sine_c1.hardening_report_["smoothness"] == 1


def test_c1_more_accurate_on_smooth_function(sine_c0, sine_c1):
    xg = np.linspace(-2, 2, 2001)[:, None]
    truth = np.sin(2 * xg[:, 0])
    assert _rmse(sine_c1.predict(xg, hard=True), truth) < 0.5 * _rmse(sine_c0.predict(xg, hard=True), truth)


def test_c1_jacobian_matches_finite_differences(bump_c1):
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.8, 1.8, (50, 2))
    J = bump_c1.predict_jacobian(X, hard=True)
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (bump_c1.predict(X + e, hard=True) - bump_c1.predict(X - e, hard=True)) / (2 * h)
        np.testing.assert_allclose(J[:, j], fd, atol=1e-5)


def test_c1_bands_and_leaf_models(bump_c1):
    m = bump_c1
    hw = m.hardening_report_["band_halfwidth"]
    assert hw.shape == (m.depth, 1)
    assert np.all(hw >= 0.05 - 1e-9) and np.all(hw <= 3.0 + 1e-9)
    # Outside every band the C1 model is the leaf model of the crisp tree.
    rng = np.random.default_rng(4)
    Xs = rng.uniform(-2, 2, (4000, 2))
    S = (Xs - m.x_mean_) / m.x_std_
    h = m.hinge_
    out = np.all(np.abs(S @ h["W"].T + h["b"]) > h["eps"][:, 0], axis=1)
    assert out.sum() > 100
    leaves = m.apply(Xs[out])
    P = m.params_
    leaf_pred = P["leaf_mu"][leaves, 0] + np.einsum("np,np->n", P["leaf_A"][leaves, 0], S[out])
    np.testing.assert_allclose(m.predict(Xs[out], hard=True), leaf_pred * m.y_std_[0] + m.y_mean_[0],
                               atol=1e-10)


def test_c0_unchanged_by_default(sine):
    """hinge_ridge=None keeps the C0 fit exactly as before (ridge on all
    coefficients = hard_ridge)."""
    X, y = sine
    kw = dict(depth=2, n_epochs=30, **TIED)
    a = DifferentiableTreeRegressor(**kw).fit(X, y)
    b = DifferentiableTreeRegressor(hinge_ridge=1e-6, **kw).fit(X, y)
    xg = np.linspace(-2, 2, 101)[:, None]
    np.testing.assert_allclose(a.predict(xg, hard=True), b.predict(xg, hard=True), atol=1e-10)
    assert a.smoothness_ == 0 and "band_halfwidth" not in a.hardening_report_


def test_harden_override_and_dydx(sine):
    X, y = sine
    J = (2 * np.cos(2 * X[:, 0]))[:, None]
    m = DifferentiableTreeRegressor(depth=2, n_epochs=30, leaf_type="affine", tied_splits=True,
                                    random_state=0).fit(X, y, dydx=J)
    m.harden(X, y, dydx=J, continuous=True, smoothness=1)
    assert m.smoothness_ == 1
    xg = np.linspace(-2, 2, 2001)[:, None]
    assert np.max(np.abs(np.diff(m.predict_jacobian(xg, hard=True)[:, 0]))) < 0.05
    m.harden(X, y, continuous=True, smoothness=0)
    assert m.smoothness_ == 0 and m.hinge_["eps"] is None


def test_multi_output_bands_per_output():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, (300, 2))
    Y = np.c_[np.exp(-np.sum(X**2, 1)), np.abs(X[:, 0]) + X[:, 1]] + 0.01 * rng.normal(size=(300, 2))
    m = DifferentiableTreeRegressor(depth=2, n_epochs=40, smoothness=1, **TIED).fit(X, Y)
    assert m.hardening_report_["band_halfwidth"].shape == (2, 2)
    assert m.predict_jacobian(X[:5], hard=True).shape == (5, 2, 2)
    # Only output 0 smoothed: output 1 keeps its own leaves, so no hinge form.
    m1 = DifferentiableTreeRegressor(depth=2, n_epochs=40, smoothness=1,
                                     **{**TIED, "continuous": [0]}).fit(X, Y)
    assert m1.hardening_report_["max_gradient_jump"][0] == 0.0
    with pytest.raises(ValueError, match="every output in hinge form"):
        export.pwl_error_bound(m1, 4)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_tree_exports_refuse_c1(bump_c1):
    for fn in (export.leaf_polytopes, export.to_spec, export.to_json):
        with pytest.raises(ValueError, match="C1"):
            fn(bump_c1)
    pytest.importorskip("pyomo.environ")
    for formulation in ("auto", "tree"):
        with pytest.raises(ValueError, match="formulation='pwl'"):
            export.to_pyomo(bump_c1, formulation=formulation)


def test_pwl_error_bound(bump_c0, bump_c1):
    b4, b8 = export.pwl_error_bound(bump_c1, 4), export.pwl_error_bound(bump_c1, 8)
    np.testing.assert_allclose(b8["over"], b4["over"] / 4)
    assert b4["bound"][0] == max(b4["over"][0], b4["under"][0]) > 0
    assert export.pwl_error_bound(bump_c0)["bound"][0] == 0.0
    with pytest.raises(ValueError, match="K must be a positive integer"):
        export.pwl_error_bound(bump_c1)


def _highs():
    pyo = pytest.importorskip("pyomo.environ")
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        pytest.skip("HiGHS (highspy) not available")
    return pyo, solver


def _pwl_at(pyo, solver, b, x):
    for v, xv in zip(b.x.values(), x):
        v.fix(xv)
    solver.solve(b.model())
    for v in b.x.values():
        v.unfix()
    return pyo.value(b.y[0])


def test_pwl_c0_is_exact(bump_c0):
    pyo, solver = _highs()
    b = export.to_pyomo(bump_c0, bounds=(-2, 2), formulation="pwl")
    m = b.model()
    assert sum(1 for v in b.on.values() if not v.fixed) <= bump_c0.depth
    m.obj = pyo.Objective(expr=b.y[0], sense=pyo.maximize)
    for x in np.random.default_rng(6).uniform(-2, 2, (8, 2)):
        assert _pwl_at(pyo, solver, b, x) == pytest.approx(bump_c0.predict(x[None], hard=True)[0], abs=1e-9)
    solver.solve(m)
    grid = np.stack(np.meshgrid(*[np.linspace(-2, 2, 201)] * 2), -1).reshape(-1, 2)
    assert pyo.value(m.obj) >= bump_c0.predict(grid, hard=True).max() - 1e-9
    x = np.array([pyo.value(v) for v in b.x.values()])
    assert export.leaf_from_pyomo(b) == bump_c0.apply(x[None])[0]


def test_pwl_c1_within_bound(bump_c1):
    pyo, solver = _highs()
    K = 4
    b = export.to_pyomo(bump_c1, bounds=(-2, 2), formulation="pwl", K=K)
    m = b.model()
    m.obj = pyo.Objective(expr=b.y[0], sense=pyo.maximize)
    eb = b.error_bound
    assert b.K == K and eb["bound"][0] == export.pwl_error_bound(bump_c1, K)["bound"][0]
    for x in np.random.default_rng(7).uniform(-2, 2, (10, 2)):
        d = _pwl_at(pyo, solver, b, x) - bump_c1.predict(x[None], hard=True)[0]
        assert -eb["under"][0] - 1e-9 <= d <= eb["over"][0] + 1e-9
    # The MILP optimum is within the bound of the C1 model's maximum.
    solver.solve(m)
    grid = np.stack(np.meshgrid(*[np.linspace(-2, 2, 201)] * 2), -1).reshape(-1, 2)
    c1_max = bump_c1.predict(grid, hard=True).max()
    assert pyo.value(m.obj) >= c1_max - eb["under"][0] - 1e-6
    assert pyo.value(m.obj) <= c1_max + eb["over"][0] + 1e-3  # grid resolution


def test_smooth_formulation_with_ipopt(bump_c1):
    pyo = pytest.importorskip("pyomo.environ")
    ipopt = pyo.SolverFactory("ipopt")
    if not ipopt.available(exception_flag=False):
        pytest.skip("IPOPT not available")
    b = export.to_pyomo(bump_c1, bounds=(-2, 2), formulation="smooth")
    m = b.model()
    m.obj = pyo.Objective(expr=b.y[0], sense=pyo.maximize)
    for start in ((1.2, -0.9), (-1.5, 1.5)):
        for v, s in zip(b.x.values(), start):
            v.set_value(s)
        res = ipopt.solve(m)
        assert str(res.solver.termination_condition) == "optimal"
        x = np.array([pyo.value(v) for v in b.x.values()])
        assert pyo.value(m.obj) == pytest.approx(bump_c1.predict(x[None], hard=True)[0], abs=1e-7)
        g = bump_c1.predict_jacobian(x[None], hard=True)[0]
        interior = (np.abs(x) < 2 - 1e-6)
        assert np.all(np.abs(g[interior]) < 1e-5)
