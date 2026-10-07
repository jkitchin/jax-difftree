"""Smoke tests for multivariate regression, conditional, inverse design."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from difftree import DifferentiableTreeRegressor


# Two correlated outputs with input-dependent mean and signal-dependent noise.
def _make_2d_data(n=900, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, n)
    x2 = rng.uniform(-2.0, 2.0, n)
    X = np.stack([x1, x2], axis=1)

    # True structure: y1 = sin(x1) + 0.3 x2; y2 = 0.6 y1 + cos(x2)
    y1_true = np.sin(x1) + 0.3 * x2
    y2_true = 0.6 * y1_true + np.cos(x2)

    sig1 = 0.05 + 0.10 * np.abs(y1_true)
    sig2 = 0.05 + 0.10 * np.abs(y2_true)
    # Independent noise, correlation between y1, y2 enters via the truth, not the noise.
    y1 = y1_true + rng.normal(0.0, sig1)
    y2 = y2_true + rng.normal(0.0, sig2)
    Y = np.stack([y1, y2], axis=1)
    return X, Y, np.stack([y1_true, y2_true], axis=1)


@pytest.fixture(scope="module")
def fitted_2d():
    X, Y, _ = _make_2d_data(n=900, seed=0)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=200, random_state=0)
    m = DifferentiableTreeRegressor(
        depth=4, n_epochs=120, n_calibration_steps=200,
        random_state=0, sigma_init=0.5,
    )
    m.fit(X_tr, Y_tr)
    return m, X_te, Y_te


def test_multivariate_fit_basic(fitted_2d):
    m, X_te, Y_te = fitted_2d
    assert m.n_outputs_ == 2
    assert not m.y_was_1d_
    pred = m.predict(X_te)
    assert pred.shape == Y_te.shape
    rmse = np.sqrt(np.mean((pred - Y_te) ** 2, axis=0))
    var = Y_te.var(axis=0)
    assert (rmse**2 < var).all(), f"per-output rmse² {rmse**2} not < var {var}"


def test_predict_return_std_shape(fitted_2d):
    m, X_te, _ = fitted_2d
    mean, std = m.predict(X_te, return_std=True)
    assert mean.shape == std.shape == (X_te.shape[0], 2)
    assert (std > 0).all()


def test_predict_components_decomposition(fitted_2d):
    m, X_te, _ = fitted_2d
    mean, std, ale, rou = m.predict(X_te, return_components=True)
    assert std.shape == ale.shape == rou.shape == (X_te.shape[0], 2)
    np.testing.assert_allclose(std**2, ale**2 + rou**2, rtol=1e-4, atol=1e-6)


def test_predict_full_cov_psd(fitted_2d):
    m, X_te, _ = fitted_2d
    mean, cov = m.predict_full_cov(X_te)
    assert mean.shape == (X_te.shape[0], 2)
    assert cov.shape == (X_te.shape[0], 2, 2)
    # Symmetric and PSD
    np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-6)
    eigs = np.linalg.eigvalsh(cov)
    assert (eigs > -1e-6).all()


def test_predict_conditional_tightens_uncertainty(fitted_2d):
    """Knowing y_1 should reduce the predictive variance on y_2 because the
    two outputs are correlated through the data-generating process."""
    m, X_te, Y_te = fitted_2d
    # marginal predictive variance on y_2
    _, std_marg = m.predict(X_te, return_std=True)
    var_marg = std_marg[:, 1] ** 2

    # conditional on the true y_1 (that's the upper bound on uncertainty
    # reduction; in practice you'd condition on a measured value)
    cond_mean, cond_cov, log_ev = m.predict_conditional(
        X_te, observed={0: Y_te[:, 0]}
    )
    assert cond_mean.shape == (X_te.shape[0], 1)
    assert cond_cov.shape == (X_te.shape[0], 1, 1)
    var_cond = cond_cov[:, 0, 0]
    # On average, conditioning should not increase uncertainty:
    assert var_cond.mean() <= var_marg.mean() + 1e-6, (
        f"conditional var {var_cond.mean():.4f} > marginal {var_marg.mean():.4f}"
    )
    # And it should track the truth at least as well as the marginal mean.
    err_marg = np.abs(m.predict(X_te)[:, 1] - Y_te[:, 1]).mean()
    err_cond = np.abs(cond_mean[:, 0] - Y_te[:, 1]).mean()
    assert err_cond <= err_marg + 0.1  # not strictly required but should be close


def test_sample_y_conditional_shape_and_mean(fitted_2d):
    m, X_te, Y_te = fitted_2d
    samples = m.sample_y_conditional(
        X_te[:30], observed={0: Y_te[:30, 0]},
        n_samples=200, random_state=0,
    )
    assert samples.shape == (200, 30, 1)
    cond_mean, _, _ = m.predict_conditional(X_te[:30], observed={0: Y_te[:30, 0]})
    np.testing.assert_allclose(samples.mean(axis=0), cond_mean, atol=0.2)


def test_inverse_design_recovers_target(fitted_2d):
    """Pick a target output the model can reach, run inverse design, and
    verify that the resulting x produces the target under the model."""
    m, X_te, Y_te = fitted_2d
    # Use a held-out point's true output as the target; inverse design should
    # find an x such that the model predicts something close to Y_te[i].
    target = {0: float(Y_te[0, 0]), 1: float(Y_te[0, 1])}
    x_best, log_lik, _, log_liks = m.inverse_design(
        target, n_starts=8, n_steps=400, learning_rate=0.05, random_state=0,
    )
    assert x_best.shape == (X_te.shape[1],)
    # Predict at the designed x and compare to target.
    pred_at_best = m.predict(x_best.reshape(1, -1))[0]
    err = np.abs(pred_at_best - np.array([target[0], target[1]]))
    # Very loose tolerance: the model's predictive uncertainty itself sets a floor.
    _, std_at_best = m.predict(x_best.reshape(1, -1), return_std=True)
    assert (err <= 4 * std_at_best[0] + 0.5).all(), (
        f"inverse-designed pred {pred_at_best} far from target {target}: "
        f"err={err}, std={std_at_best[0]}"
    )


def test_inverse_design_partial_target(fitted_2d):
    """Constrain only one output and let the other vary."""
    m, X_te, _ = fitted_2d
    target = {0: 0.5}  # ask for y_0 = 0.5, y_1 unconstrained
    x_best, log_lik, _, _ = m.inverse_design(
        target, n_starts=8, n_steps=300, random_state=1
    )
    pred = m.predict(x_best.reshape(1, -1))[0]
    assert abs(pred[0] - 0.5) < 0.4
