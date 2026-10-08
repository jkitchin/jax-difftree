"""Affine leaves, Jacobian matching, crisp mode / hardening, and export."""
from __future__ import annotations

import json

import numpy as np
import pytest

from difftree import DifferentiableTreeClassifier, DifferentiableTreeRegressor
from difftree import export


# Piecewise-linear, two outputs sharing one kink at x0 = 0; output 1 is
# piecewise constant.
def _f(X):
    y0 = np.where(X[:, 0] > 0, 2 * X[:, 0] - X[:, 1], -X[:, 0] + 0.5 * X[:, 1] + 1)
    y1 = np.where(X[:, 0] > 0, 1.0, -1.0)
    return np.stack([y0, y1], axis=1)


def _jac(X):
    J = np.zeros((X.shape[0], 2, 2))
    right = X[:, 0] > 0
    J[:, 0, 0] = np.where(right, 2.0, -1.0)
    J[:, 0, 1] = np.where(right, -1.0, 0.5)
    return J


@pytest.fixture(scope="module")
def pwl_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (600, 2))
    Y = _f(X) + 0.02 * rng.normal(size=(600, 2))
    X_te = rng.uniform(-2, 2, (400, 2))
    return X, Y, X_te, _f(X_te)


@pytest.fixture(scope="module")
def hardened(pwl_data):
    X, Y, _, _ = pwl_data
    m = DifferentiableTreeRegressor(
        depth=2, n_epochs=100, leaf_type="affine", constant_outputs=[1],
        hardening=True, n_hard_epochs=30, random_state=0,
    )
    return m.fit(X, Y)


def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


# ---------------------------------------------------------------------------
# Affine leaves (#1)
# ---------------------------------------------------------------------------


def test_affine_beats_constant(pwl_data):
    X, Y, X_te, Y_te = pwl_data
    kw = dict(depth=2, n_epochs=100, random_state=0)
    const = DifferentiableTreeRegressor(**kw).fit(X, Y[:, 0])
    aff = DifferentiableTreeRegressor(leaf_type="affine", **kw).fit(X, Y[:, 0])
    assert np.all(const.params_["leaf_A"] == 0)
    assert _rmse(aff.predict(X_te), Y_te[:, 0]) < 0.5 * _rmse(const.predict(X_te), Y_te[:, 0])
    mean, std = aff.predict(X_te, return_std=True)
    assert mean.shape == std.shape == (X_te.shape[0],)
    assert np.all(std > 0)


def test_constant_outputs_have_zero_slopes(hardened, pwl_data):
    _, _, X_te, Y_te = pwl_data
    A = hardened.params_["leaf_A"]
    assert A.shape == (4, 2, 2)
    np.testing.assert_array_equal(A[:, 1], 0.0)
    assert np.any(np.abs(A[:, 0]) > 0.1)
    # Output 1 is piecewise constant under crisp routing.
    pred = hardened.predict(X_te, hard=True)
    leaves = hardened.apply(X_te)
    for leaf in np.unique(leaves):
        vals = pred[leaves == leaf, 1]
        np.testing.assert_allclose(vals, vals[0])


def test_affine_multivariate_conditional_and_inverse(hardened, pwl_data):
    _, _, X_te, Y_te = pwl_data
    cond_mean, cond_cov, _ = hardened.predict_conditional(
        X_te[:20], observed={1: Y_te[:20, 1]}
    )
    assert cond_mean.shape == (20, 1) and cond_cov.shape == (20, 1, 1)
    x_best, _, _, _ = hardened.inverse_design({0: 1.0}, n_starts=4, n_steps=200)
    assert x_best.shape == (2,)


def test_jacobian_matching_improves_small_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (30, 2))
    y = _f(X)[:, 0]
    X_te = rng.uniform(-2, 2, (1000, 2))
    kw = dict(depth=1, n_epochs=300, batch_size=32, leaf_type="affine",
              hardening=True, random_state=0)
    plain = DifferentiableTreeRegressor(**kw).fit(X, y)
    sob = DifferentiableTreeRegressor(**kw).fit(X, y, dydx=_jac(X)[:, 0, :])
    err_plain = _rmse(plain.predict(X_te, hard=True), _f(X_te)[:, 0])
    err_sob = _rmse(sob.predict(X_te, hard=True), _f(X_te)[:, 0])
    assert err_sob < err_plain


def test_dydx_validation(pwl_data):
    X, Y, _, _ = pwl_data
    with pytest.raises(ValueError, match="leaf_type='affine'"):
        DifferentiableTreeRegressor(depth=1, n_epochs=1).fit(X, Y, dydx=_jac(X))
    with pytest.raises(ValueError, match="shape"):
        DifferentiableTreeRegressor(depth=1, n_epochs=1, leaf_type="affine").fit(
            X, Y, dydx=_jac(X)[:, :, :1]
        )
    with pytest.raises(ValueError, match="leaf_type"):
        DifferentiableTreeRegressor(leaf_type="linear").fit(X, Y)
    with pytest.raises(ValueError, match="constant_outputs"):
        DifferentiableTreeRegressor(
            n_epochs=1, leaf_type="affine", constant_outputs=[5]
        ).fit(X, Y)


# ---------------------------------------------------------------------------
# Crisp mode and hardening (#2)
# ---------------------------------------------------------------------------


def test_hard_predict_is_leaf_model(hardened, pwl_data):
    _, _, X_te, _ = pwl_data
    m = hardened
    leaves = m.apply(X_te)
    assert leaves.min() >= 0 and leaves.max() < 2**m.depth
    mean, std, ale, rou = m.predict(X_te, return_components=True, hard=True)
    np.testing.assert_array_equal(rou, 0.0)
    np.testing.assert_allclose(std, ale)
    # Leaf mean computed by hand in original units.
    P = m.params_
    xs = (X_te - m.x_mean_) / m.x_std_
    ys = P["leaf_mu"][leaves] + np.einsum("nyp,np->ny", P["leaf_A"][leaves], xs)
    np.testing.assert_allclose(mean, ys * m.y_std_ + m.y_mean_, rtol=1e-10)
    _, cov = m.predict_full_cov(X_te, hard=True)
    np.testing.assert_allclose(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), std)


def test_hardening_recovers_linear_truth_exactly():
    """With a noise-free linear target, the least-squares refit on the frozen
    partition reproduces the truth in every leaf."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (300, 3))
    y = 1.5 * X[:, 0] - 2 * X[:, 1] + 0.3 * X[:, 2] + 4
    m = DifferentiableTreeRegressor(
        depth=2, n_epochs=40, leaf_type="affine", n_hard_epochs=5, random_state=0
    ).fit(X, y)
    m.harden(X, y)
    assert m.hardened_
    X_te = rng.uniform(-1, 1, (200, 3))
    y_te = 1.5 * X_te[:, 0] - 2 * X_te[:, 1] + 0.3 * X_te[:, 2] + 4
    np.testing.assert_allclose(m.predict(X_te, hard=True), y_te, atol=1e-3)


def test_hardening_report(hardened, pwl_data):
    X, Y, X_te, Y_te = pwl_data
    rep = hardened.hardening_report_
    assert set(rep) == {"before", "after", "n_validation"}
    keys = {"rmse_soft", "rmse_hard", "rmse_gap", "r2_soft", "r2_hard",
            "nll_soft", "nll_hard", "mean_max_leaf_prob", "routing_agreement"}
    assert set(rep["before"]) == set(rep["after"]) == keys
    after = rep["after"]
    assert after["rmse_hard"].shape == (2,)
    np.testing.assert_allclose(after["rmse_gap"], after["rmse_hard"] - after["rmse_soft"])
    # Hardened routing is crisp, and the crisp model fits the noise floor.
    assert after["mean_max_leaf_prob"] > 0.95
    assert np.all(_rmse(hardened.predict(X_te, hard=True), Y_te) < 0.15)
    gap = hardened.soft_hard_gap(X_te, Y_te)
    assert 0.0 <= gap["routing_agreement"] <= 1.0


def test_classifier_hard_mode():
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    m = DifferentiableTreeClassifier(depth=3, n_epochs=40, random_state=0).fit(X, y)
    probs, ale, epi = m.predict_proba(X, return_components=True, hard=True)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_array_equal(epi, 0.0)
    assert (m.predict(X, hard=True) == y).mean() > 0.8
    assert m.apply(X).shape == (300,)


# ---------------------------------------------------------------------------
# Export (#3)
# ---------------------------------------------------------------------------


def test_leaf_polytopes_partition_box(hardened):
    m = hardened
    rng = np.random.default_rng(3)
    X = rng.uniform(m.x_min_, m.x_max_, (2000, 2))
    polys = export.leaf_polytopes(m)
    assert polys and all(lp.radius > 0 for lp in polys)
    inside = np.stack([lp.contains(X) for lp in polys], axis=1)
    # Every point is in exactly one leaf polytope (boundaries have measure 0),
    # and it is the leaf crisp routing picks.
    assert np.all(inside.sum(axis=1) == 1)
    leaf_ids = np.array([lp.leaf for lp in polys])[inside.argmax(axis=1)]
    np.testing.assert_array_equal(leaf_ids, m.apply(X))
    # Leaf models reproduce predict(hard=True) and its noise.
    pred = m.predict(X, hard=True)
    for j, lp in enumerate(polys):
        sel = inside[:, j]
        np.testing.assert_allclose(lp.predict(X[sel]), pred[sel], rtol=1e-9, atol=1e-9)
        assert lp.coef.shape == (2, 2) and lp.cov.shape == (2, 2)
        np.testing.assert_array_equal(lp.coef[1], 0.0)  # constant output


def test_prune_empty_leaves():
    """In 1D, a depth-3 tree has at most (number of distinct thresholds + 1)
    non-empty leaves, so some of its 8 leaves are empty within the box."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-3, 3, (400, 1))
    y = np.sin(2 * X[:, 0])
    m = DifferentiableTreeRegressor(depth=3, n_epochs=60, leaf_type="affine",
                                    random_state=0)
    m.fit(X, y)
    with pytest.warns(UserWarning, match="not hardened"):
        all_leaves = export.leaf_polytopes(m, prune_empty=False)
    with pytest.warns(UserWarning):
        kept = export.leaf_polytopes(m)
    assert len(all_leaves) == 8
    assert len(kept) < 8
    # Pruned spec still reproduces crisp predictions inside the box.
    with pytest.warns(UserWarning):
        spec = export.to_spec(m)
    X_te = rng.uniform(m.x_min_, m.x_max_, (500, 1))
    np.testing.assert_allclose(export.predict_spec(spec, X_te), m.predict(X_te, hard=True),
                               rtol=1e-9, atol=1e-9)
    assert spec["tree"]["node_count"] == 2 * len(kept) - 1


def test_prune_unreached(hardened):
    m = hardened
    rng = np.random.default_rng(5)
    X_left = rng.uniform([-2, -2], [-0.5, 2], (200, 2))
    polys = export.leaf_polytopes(m, X=X_left, prune_unreached=True)
    assert all(lp.n_samples > 0 for lp in polys)
    assert len(polys) < len(export.leaf_polytopes(m))
    with pytest.raises(ValueError, match="requires X"):
        export.leaf_polytopes(m, prune_unreached=True)


def test_spec_matches_jax_ldt_schema_and_predictions(hardened, tmp_path):
    m = hardened
    s = export.to_json(m, tmp_path / "tree.json")
    spec = json.loads((tmp_path / "tree.json").read_text())
    assert spec == json.loads(s)
    assert spec["version"] == 1 and spec["kind"] == "hyperplane_tree"
    assert set(spec["tree"]) == {
        "node_count", "is_leaf", "feature", "threshold", "left", "right", "leaf_params"
    }
    n_t = spec["n_features_transformed"]
    A = np.asarray(spec["transform_matrix"])
    assert A.shape == (2, n_t)
    np.testing.assert_array_equal(A[:, :2], np.eye(2))
    lp = np.asarray(spec["tree"]["leaf_params"])
    assert lp.shape == (spec["tree"]["node_count"], 1 + n_t, 2)
    np.testing.assert_array_equal(lp[:, 3:], 0.0)  # leaves use raw columns only
    rng = np.random.default_rng(6)
    X = rng.uniform(m.x_min_, m.x_max_, (1000, 2))
    np.testing.assert_allclose(export.predict_spec(s, X), m.predict(X, hard=True),
                               rtol=1e-9, atol=1e-9)


def test_spec_univariate_and_custom_bounds(pwl_data):
    X, Y, _, _ = pwl_data
    m = DifferentiableTreeRegressor(depth=2, n_epochs=40, leaf_type="affine",
                                    hardening=True, n_hard_epochs=5, random_state=0)
    m.fit(X, Y[:, 0])
    spec = export.to_spec(m, bounds=(-1.0, 1.0))
    assert spec["n_targets"] == 1
    assert spec["difftree"]["bounds"] == [[-1.0, -1.0], [1.0, 1.0]]
    X_in = np.random.default_rng(7).uniform(-1, 1, (300, 2))
    pred = export.predict_spec(spec, X_in)
    assert pred.shape == (300,)
    np.testing.assert_allclose(pred, m.predict(X_in, hard=True), rtol=1e-9, atol=1e-9)
