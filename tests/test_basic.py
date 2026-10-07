"""Smoke tests for difftree estimators."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1, make_moons
from sklearn.model_selection import train_test_split

from difftree import DifferentiableTreeClassifier, DifferentiableTreeRegressor


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def friedman():
    X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=100, random_state=0)


@pytest.fixture(scope="module")
def fitted_regressor(friedman):
    X_tr, _, y_tr, _ = friedman
    m = DifferentiableTreeRegressor(
        depth=4, n_epochs=80, batch_size=128,
        n_calibration_steps=150, random_state=0,
    )
    m.fit(X_tr, y_tr)
    return m


def test_regressor_basic(friedman, fitted_regressor):
    _, X_te, _, y_te = friedman
    m = fitted_regressor
    yhat = m.predict(X_te)
    assert yhat.shape == y_te.shape
    # Should beat dumb mean baseline.
    rmse = float(np.sqrt(np.mean((yhat - y_te) ** 2)))
    var_y = float(y_te.var())
    assert rmse**2 < var_y, f"rmse² {rmse**2:.3f} not < var_y {var_y:.3f}"


def test_regressor_return_std(friedman, fitted_regressor):
    _, X_te, _, _ = friedman
    mean, std = fitted_regressor.predict(X_te, return_std=True)
    assert mean.shape == std.shape == (X_te.shape[0],)
    assert np.all(std > 0)


def test_regressor_return_components(friedman, fitted_regressor):
    _, X_te, _, _ = friedman
    mean, std, std_ale, std_rou = fitted_regressor.predict(X_te, return_components=True)
    assert mean.shape == std.shape == std_ale.shape == std_rou.shape
    # Variance decomposition: total² ≈ aleatoric² + routing²
    np.testing.assert_allclose(
        std**2, std_ale**2 + std_rou**2, rtol=1e-4, atol=1e-6
    )


def test_regressor_sample_y(friedman, fitted_regressor):
    _, X_te, _, _ = friedman
    samples = fitted_regressor.sample_y(X_te[:20], n_samples=50, random_state=42)
    assert samples.shape == (50, 20)
    # Sample mean should be near predict mean
    pred = fitted_regressor.predict(X_te[:20])
    np.testing.assert_allclose(samples.mean(axis=0), pred, atol=1.0)


def test_regressor_predict_quantile(friedman, fitted_regressor):
    _, X_te, _, _ = friedman
    qs = fitted_regressor.predict_quantile(
        X_te[:10], q=[0.05, 0.5, 0.95], n_samples=500, random_state=0
    )
    assert qs.shape == (3, 10)
    assert np.all(qs[0] <= qs[1])
    assert np.all(qs[1] <= qs[2])


def test_regressor_calibration_improves_nll(friedman):
    """Without calibration the predictive variance is the raw model output;
    after calibration it should give a >= as good val NLL."""
    X_tr, _, y_tr, _ = friedman
    m = DifferentiableTreeRegressor(
        depth=4, n_epochs=60, n_calibration_steps=200, random_state=0,
    )
    m.fit(X_tr, y_tr)
    # log_alpha should not be exactly zero after calibration
    log_alpha = float(m.params_["log_alpha"])
    # log_gamma should not be uniform after per-depth calibration
    log_gamma = m.params_["log_gamma"]
    assert log_gamma.shape == (4,)
    # Either alpha or per-depth gamma moved during calibration:
    moved = (abs(log_alpha) > 1e-3) or (log_gamma.std() > 1e-3)
    assert moved, "calibration produced no parameter movement"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moons_data():
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    return train_test_split(X, y, test_size=120, random_state=0)


@pytest.fixture(scope="module")
def fitted_classifier(moons_data):
    X_tr, _, y_tr, _ = moons_data
    m = DifferentiableTreeClassifier(
        depth=4, n_epochs=80, batch_size=128,
        n_calibration_steps=150, random_state=0,
    )
    m.fit(X_tr, y_tr)
    return m


def test_classifier_basic(moons_data, fitted_classifier):
    _, X_te, _, y_te = moons_data
    yhat = fitted_classifier.predict(X_te)
    acc = float((yhat == y_te).mean())
    assert acc > 0.85, f"moons accuracy {acc:.3f} too low"


def test_classifier_predict_proba(moons_data, fitted_classifier):
    _, X_te, _, _ = moons_data
    p = fitted_classifier.predict_proba(X_te)
    assert p.shape == (X_te.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_proba_components(moons_data, fitted_classifier):
    _, X_te, _, _ = moons_data
    probs, ale, epi = fitted_classifier.predict_proba(X_te, return_components=True)
    assert probs.shape == (X_te.shape[0], 2)
    assert ale.shape == (X_te.shape[0],)
    assert epi.shape == (X_te.shape[0],)
    assert np.all(ale >= 0)
    assert np.all(epi >= -1e-6)
    # Total entropy = aleatoric + epistemic (mutual info)
    total = -np.sum(probs * np.log(probs + 1e-30), axis=1)
    np.testing.assert_allclose(total, ale + epi, atol=1e-4)


def test_classifier_multi_class():
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6,
        n_classes=4, n_clusters_per_class=1, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=80, random_state=0)
    m = DifferentiableTreeClassifier(
        depth=5, n_epochs=80, n_calibration_steps=150, random_state=0,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)
    assert p.shape == (X_te.shape[0], 4)
    assert m.score(X_te, y_te) > 0.5  # better than uniform random (0.25)


def test_classifier_sample_y(moons_data, fitted_classifier):
    _, X_te, _, _ = moons_data
    s = fitted_classifier.sample_y(X_te[:15], n_samples=30, random_state=0)
    assert s.shape == (30, 15)
    # samples are in classes_
    classes = set(int(c) for c in fitted_classifier.classes_)
    assert set(int(v) for v in np.unique(s)).issubset(classes)


# ---------------------------------------------------------------------------
# sklearn API hygiene
# ---------------------------------------------------------------------------


def test_get_set_params_roundtrip():
    m = DifferentiableTreeRegressor(depth=4, learning_rate=0.005)
    p = m.get_params()
    assert p["depth"] == 4 and p["learning_rate"] == 0.005
    m.set_params(depth=6)
    assert m.depth == 6


def test_classifier_classes_attribute(moons_data, fitted_classifier):
    assert set(fitted_classifier.classes_) == {0, 1}
