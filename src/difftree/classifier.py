"""Sklearn-compatible differentiable-tree classifier."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _model
from ._train import calibrate, train


def _classifier_loss(params, X, y):
    ll = _model.batched_classifier_log_lik(
        X,
        y,
        params["W"],
        params["b"],
        params["log_gamma"],
        params["leaf_logits"],
    )
    return -jnp.mean(ll)


class DifferentiableTreeClassifier(ClassifierMixin, BaseEstimator):
    """Soft oblique decision tree of fixed depth with categorical leaves.

    Predictive distribution at each x is a mixture of categoricals (one per
    leaf), weighted by the soft path probabilities. The total predictive
    entropy decomposes into an aleatoric (within-leaf) part and an epistemic
    (mutual-information / between-leaf-routing) part.

    Calibration is performed post-hoc on a validation slice by fitting the
    per-depth ``gamma`` (split steepness) by minimizing validation NLL.

    See ``DifferentiableTreeRegressor`` for shared hyperparameter docs.
    """

    def __init__(
        self,
        depth: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        n_epochs: int = 200,
        gamma_init: float = 1.0,
        gamma_max: float = 5.0,
        delta_gamma: float = 0.1,
        validation_fraction: float = 0.1,
        n_calibration_steps: int = 300,
        calibration_lr: float = 0.05,
        standardize: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.depth = depth
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma_init = gamma_init
        self.gamma_max = gamma_max
        self.delta_gamma = delta_gamma
        self.validation_fraction = validation_fraction
        self.n_calibration_steps = n_calibration_steps
        self.calibration_lr = calibration_lr
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64)
        self._label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder_.classes_
        y_int = self._label_encoder_.transform(y).astype(np.int32)
        n_classes = len(self.classes_)

        rng_np = np.random.default_rng(self.random_state)
        seed = int(rng_np.integers(0, 2**31 - 1))

        if self.standardize:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.x_std_ = np.ones(X.shape[1])

        Xs = (X - self.x_mean_) / self.x_std_

        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs, y_int, test_size=self.validation_fraction,
            stratify=y_int, random_state=seed,
        )
        X_tr_j = jnp.asarray(X_tr)
        y_tr_j = jnp.asarray(y_tr)
        X_val_j = jnp.asarray(X_val)
        y_val_j = jnp.asarray(y_val)

        key = jax.random.PRNGKey(seed)
        params = _model.init_classifier_params(
            key, self.depth, X.shape[1], n_classes, self.gamma_init
        )

        if self.verbose:
            print(f"[difftree] training on {X_tr.shape[0]} samples, "
                  f"{n_classes} classes...")
        params = train(
            params,
            _classifier_loss,
            X_tr_j,
            y_tr_j,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma_init=self.gamma_init,
            gamma_max=self.gamma_max,
            delta_gamma=self.delta_gamma,
            rng=rng_np,
            trainable_keys=("W", "b", "leaf_logits"),
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"[difftree] calibrating on {X_val.shape[0]} held-out samples...")
        params = calibrate(
            params,
            _classifier_loss,
            X_val_j,
            y_val_j,
            calibration_lr=self.calibration_lr,
            n_calibration_steps=self.n_calibration_steps,
            calib_keys=("log_gamma",),
            verbose=self.verbose,
        )

        self.params_ = {k: np.asarray(v) for k, v in params.items()}
        return self

    def _params_jax(self) -> dict:
        return {k: jnp.asarray(v) for k, v in self.params_.items()}

    def _standardize(self, X) -> jnp.ndarray:
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return jnp.asarray((X - self.x_mean_) / self.x_std_)

    def apply(self, X) -> np.ndarray:
        """Leaf index (0 .. 2**depth - 1, BFS leaf order) each row routes to
        under crisp routing: node ``i`` goes right iff ``w_i . x + b_i > 0``."""
        check_is_fitted(self, "params_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Xs = (X - self.x_mean_) / self.x_std_
        return _model.hard_leaf_index(Xs, self.params_["W"], self.params_["b"], self.depth)

    def predict_proba(self, X, return_components: bool = False, hard: bool = False):
        """Predict class probabilities.

        Parameters
        ----------
        return_components : bool
            If True, also return aleatoric and epistemic predictive entropies
            (per sample, both shape ``(n,)``).
        hard : bool
            If True, route each row crisply to one leaf (see :meth:`apply`)
            and return that leaf's class distribution. The epistemic
            (routing) entropy is then zero.
        """
        check_is_fitted(self, "params_")
        if hard:
            logits = np.asarray(self.params_["leaf_logits"], dtype=np.float64)[self.apply(X)]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            if return_components:
                ale = -np.sum(probs * np.log(probs + 1e-30), axis=1)
                return probs, ale, np.zeros_like(ale)
            return probs
        Xs = self._standardize(X)
        p = self._params_jax()
        if return_components:
            probs, ale, epi = _model.batched_classifier_proba_components(
                Xs, p["W"], p["b"], p["log_gamma"], p["leaf_logits"]
            )
            return np.asarray(probs), np.asarray(ale), np.asarray(epi)
        log_proba = _model.batched_classifier_log_proba(
            Xs, p["W"], p["b"], p["log_gamma"], p["leaf_logits"]
        )
        return np.asarray(jnp.exp(log_proba))

    def predict(self, X, hard: bool = False):
        proba = self.predict_proba(X, hard=hard)
        idx = np.argmax(proba, axis=1)
        return self._label_encoder_.inverse_transform(idx)

    def sample_y(self, X, n_samples: int = 1, random_state: int | None = None):
        """Monte Carlo class-label samples from the predictive distribution.

        Returns labels in the original class space, shape ``(n_samples, n)``.
        """
        check_is_fitted(self, "params_")
        Xs = self._standardize(X)
        p = self._params_jax()
        seed = random_state if random_state is not None else 0
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, n_samples * Xs.shape[0]).reshape(
            n_samples, Xs.shape[0], 2
        )
        out = np.empty((n_samples, Xs.shape[0]), dtype=int)
        for s in range(n_samples):
            samples_int = np.asarray(
                _model.batched_classifier_sample(
                    Xs, keys[s], p["W"], p["b"], p["log_gamma"], p["leaf_logits"]
                )
            )
            out[s] = self._label_encoder_.inverse_transform(samples_int)
        return out
