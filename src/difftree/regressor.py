"""Sklearn-compatible heteroscedastic differentiable-tree regressor.

Supports both univariate and multivariate targets. For multivariate y, leaves
are multivariate Gaussians, and additional methods expose:

  - conditional prediction p(y_unobs | x, y_obs) in closed form,
  - conditional sampling via leaf-weight reweighting + Schur complements,
  - inverse design: gradient-based optimization over x to match a target
    output vector under the trained likelihood.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _inverse, _model
from ._train import calibrate, train


def _regressor_loss(params, X, y):
    ll = _model.batched_regressor_log_lik(
        X, y,
        params["W"], params["b"], params["log_gamma"],
        params["leaf_mu"], params["leaf_chol_raw"], params["log_alpha"],
    )
    return -jnp.mean(ll)


class DifferentiableTreeRegressor(RegressorMixin, BaseEstimator):
    """Soft oblique decision tree of fixed depth with multivariate Gaussian leaves.

    Predictive distribution at each input x is a mixture of multivariate
    Gaussians, one per leaf, weighted by soft path probabilities mu_l(x). For
    univariate y this reduces to the heteroscedastic Gaussian model; for
    multivariate y it is a structured conditional GMM (input-dependent MDN).

    Parameters
    ----------
    depth : int, default=5
    learning_rate : float, default=0.01
    batch_size : int, default=256
    n_epochs : int, default=200
    gamma_init, gamma_max, delta_gamma : float
        Anneal split steepness 1 -> gamma_max during training.
    sigma_init : float, default=1.0
        Initial leaf marginal standard deviation (in standardized y units).
    warmup_fraction : float, default=0.25
    validation_fraction : float, default=0.1
        Fraction of training data held out for post-hoc gamma+alpha calibration.
    n_calibration_steps, calibration_lr : calibration hyperparams.
    standardize : bool, default=True
    random_state : int or None
    verbose : bool
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
        sigma_init: float = 1.0,
        warmup_fraction: float = 0.25,
        validation_fraction: float = 0.1,
        n_calibration_steps: int = 300,
        calibration_lr: float = 0.02,
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
        self.sigma_init = sigma_init
        self.warmup_fraction = warmup_fraction
        self.validation_fraction = validation_fraction
        self.n_calibration_steps = n_calibration_steps
        self.calibration_lr = calibration_lr
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        # Allow both 1D and 2D y. Track whether the user gave us 1D so that
        # predict() can return matching shape.
        y_arr = np.asarray(y)
        self.y_was_1d_ = y_arr.ndim == 1
        if self.y_was_1d_:
            y_arr = y_arr[:, None]

        X, y_arr = validate_data(self, X, y_arr, dtype=np.float64,
                                 multi_output=True, y_numeric=True)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        self.n_outputs_ = y_arr.shape[1]

        rng_np = np.random.default_rng(self.random_state)
        seed = int(rng_np.integers(0, 2**31 - 1))

        if self.standardize:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
            self.y_mean_ = y_arr.mean(axis=0)
            self.y_std_ = y_arr.std(axis=0)
            self.y_std_[self.y_std_ == 0] = 1.0
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.x_std_ = np.ones(X.shape[1])
            self.y_mean_ = np.zeros(self.n_outputs_)
            self.y_std_ = np.ones(self.n_outputs_)

        Xs = (X - self.x_mean_) / self.x_std_
        ys = (y_arr - self.y_mean_) / self.y_std_

        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs, ys, test_size=self.validation_fraction, random_state=seed
        )

        key = jax.random.PRNGKey(seed)
        params = _model.init_regressor_params(
            key, self.depth, X.shape[1], self.n_outputs_,
            self.gamma_init, self.sigma_init,
        )

        if self.verbose:
            print(f"[difftree] training on {X_tr.shape[0]} samples, "
                  f"d_y={self.n_outputs_}...")
        params = train(
            params, _regressor_loss,
            jnp.asarray(X_tr), jnp.asarray(y_tr),
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma_init=self.gamma_init,
            gamma_max=self.gamma_max,
            delta_gamma=self.delta_gamma,
            rng=rng_np,
            trainable_keys=("W", "b", "leaf_mu", "leaf_chol_raw"),
            warmup_keys=("W", "b", "leaf_mu"),
            warmup_fraction=self.warmup_fraction,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"[difftree] calibrating on {X_val.shape[0]} held-out samples...")
        params = calibrate(
            params, _regressor_loss,
            jnp.asarray(X_val), jnp.asarray(y_val),
            calibration_lr=self.calibration_lr,
            n_calibration_steps=self.n_calibration_steps,
            calib_keys=("log_gamma", "log_alpha"),
            verbose=self.verbose,
        )

        self.params_ = {k: np.asarray(v) for k, v in params.items()}
        return self

    # ------------------------------------------------------------------ helpers

    def _params_jax(self) -> dict:
        return {k: jnp.asarray(v) for k, v in self.params_.items()}

    def _standardize_x(self, X) -> jnp.ndarray:
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return jnp.asarray((X - self.x_mean_) / self.x_std_)

    def _standardize_y(self, y_arr: np.ndarray) -> np.ndarray:
        return (np.asarray(y_arr) - self.y_mean_) / self.y_std_

    def _unstandardize_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_std_ + self.y_mean_

    def _shape_y_out(self, arr: np.ndarray) -> np.ndarray:
        """Strip the trailing output dimension if the user fitted with 1D y."""
        if self.y_was_1d_ and arr.ndim >= 1 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    # ------------------------------------------------------------------ predict

    def predict(self, X, return_std: bool = False, return_components: bool = False):
        """Predict marginal mean (and optionally per-output marginal std).

        For multivariate y, ``std`` is the per-output marginal standard
        deviation (sqrt of diagonal of the predictive covariance). Use
        ``predict_full_cov`` for the full covariance matrix.
        """
        check_is_fitted(self, "params_")
        Xs = self._standardize_x(X)
        p = self._params_jax()
        mean, total, ale, rou = _model.batched_regressor_predict_moments(
            Xs, p["W"], p["b"], p["log_gamma"],
            p["leaf_mu"], p["leaf_chol_raw"], p["log_alpha"],
        )
        mean_o = self._unstandardize_y(np.asarray(mean))
        if not (return_std or return_components):
            return self._shape_y_out(mean_o)

        # Marginal stds per output (sqrt diag) and rescale to original units
        var_diag = np.asarray(jnp.diagonal(total, axis1=-2, axis2=-1))
        std_total = np.sqrt(var_diag) * self.y_std_
        if return_components:
            std_ale = np.sqrt(np.asarray(jnp.diagonal(ale, axis1=-2, axis2=-1))) * self.y_std_
            std_rou = np.sqrt(np.asarray(jnp.diagonal(rou, axis1=-2, axis2=-1))) * self.y_std_
            return (
                self._shape_y_out(mean_o),
                self._shape_y_out(std_total),
                self._shape_y_out(std_ale),
                self._shape_y_out(std_rou),
            )
        return self._shape_y_out(mean_o), self._shape_y_out(std_total)

    def predict_full_cov(self, X):
        """Predict mean and full predictive covariance per sample.

        Returns ``(mean, cov)`` of shapes ``(n, d_y)`` and ``(n, d_y, d_y)``,
        in original (un-standardized) y units.
        """
        check_is_fitted(self, "params_")
        Xs = self._standardize_x(X)
        p = self._params_jax()
        mean, total, _, _ = _model.batched_regressor_predict_moments(
            Xs, p["W"], p["b"], p["log_gamma"],
            p["leaf_mu"], p["leaf_chol_raw"], p["log_alpha"],
        )
        mean_o = self._unstandardize_y(np.asarray(mean))
        # Cov in original units: D Sigma_std D, where D = diag(y_std_)
        D = self.y_std_
        cov_o = np.asarray(total) * (D[:, None] * D[None, :])
        return self._shape_y_out(mean_o), cov_o

    # ------------------------------------------------------------------ sample

    def sample_y(self, X, n_samples: int = 1, random_state: int | None = None):
        """Marginal Monte Carlo draws from p(y|x).

        Returns shape ``(n_samples, n, d_y)`` for multivariate or
        ``(n_samples, n)`` for univariate.
        """
        check_is_fitted(self, "params_")
        Xs = self._standardize_x(X)
        p = self._params_jax()
        seed = random_state if random_state is not None else 0
        n = Xs.shape[0]
        out = np.empty((n_samples, n, self.n_outputs_))
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, n_samples * n).reshape(n_samples, n, 2)
        for s in range(n_samples):
            samples_std = _model.batched_regressor_sample(
                Xs, keys[s],
                p["W"], p["b"], p["log_gamma"],
                p["leaf_mu"], p["leaf_chol_raw"], p["log_alpha"],
            )
            out[s] = self._unstandardize_y(np.asarray(samples_std))
        return self._shape_y_out(out)

    def predict_quantile(
        self, X, q, n_samples: int = 1000, random_state: int | None = None
    ):
        """Approximate marginal quantiles by Monte Carlo. Univariate only."""
        if self.n_outputs_ != 1:
            raise ValueError("predict_quantile supports univariate y only; "
                             "use sample_y for multivariate.")
        samples = self.sample_y(X, n_samples=n_samples, random_state=random_state)
        q_arr = np.atleast_1d(q)
        out = np.quantile(samples, q_arr, axis=0)
        return out[0] if np.isscalar(q) else out

    # ------------------------------------------------------------------ conditional

    def _parse_observed(self, observed):
        """Normalize an ``observed`` argument into (obs_idx, y_obs_std).

        ``observed`` is a dict mapping output index -> value (scalar or
        per-row array). Returns standardized observation values broadcast
        appropriately.
        """
        if not observed:
            raise ValueError("`observed` must be a non-empty dict {output_idx: value}.")
        obs_idx = sorted(observed.keys())
        unobs_idx = [i for i in range(self.n_outputs_) if i not in observed]
        return obs_idx, unobs_idx

    def _broadcast_obs_values(self, observed: dict, n: int) -> np.ndarray:
        """Stack observed values into an (n, d_obs) array in standardized space."""
        obs_idx = sorted(observed.keys())
        values = []
        for i in obs_idx:
            v = np.asarray(observed[i], dtype=np.float64)
            if v.ndim == 0:
                v = np.full(n, float(v))
            elif v.shape[0] != n:
                raise ValueError(
                    f"observed[{i}] length {v.shape[0]} != n_samples {n}"
                )
            # Standardize using output i's mean/std
            v_std = (v - self.y_mean_[i]) / self.y_std_[i]
            values.append(v_std)
        return np.stack(values, axis=1)  # (n, d_obs)

    def predict_conditional(self, X, observed: dict):
        """Conditional moments p(y_unobs | x, y_obs).

        Parameters
        ----------
        X : (n, p) array
        observed : dict {output_idx: value}
            Each value is a scalar (broadcast across rows) or an (n,) array.

        Returns
        -------
        cond_mean : (n, d_unobs) array
        cond_cov : (n, d_unobs, d_unobs) array
        log_evidence : (n,) array
            log p(y_obs | x), useful for ranking how well x is consistent
            with the observation.
        """
        check_is_fitted(self, "params_")
        if self.n_outputs_ < 2:
            raise ValueError("predict_conditional requires multivariate y.")
        Xs = self._standardize_x(X)
        n = Xs.shape[0]
        obs_idx, unobs_idx = self._parse_observed(observed)
        if not unobs_idx:
            raise ValueError("All outputs are observed; nothing to predict.")
        y_obs_std = self._broadcast_obs_values(observed, n)

        p = self._params_jax()
        cond_mean, cond_cov, log_ev = _model.batched_conditional_moments(
            Xs, jnp.asarray(y_obs_std),
            jnp.asarray(obs_idx, dtype=jnp.int32),
            jnp.asarray(unobs_idx, dtype=jnp.int32),
            p["W"], p["b"], p["log_gamma"],
            p["leaf_mu"], p["leaf_chol_raw"], p["log_alpha"],
        )
        # Rescale to original y units
        D_u = self.y_std_[unobs_idx]
        m_u = self.y_mean_[unobs_idx]
        cond_mean_o = np.asarray(cond_mean) * D_u + m_u
        cond_cov_o = np.asarray(cond_cov) * (D_u[:, None] * D_u[None, :])
        return cond_mean_o, cond_cov_o, np.asarray(log_ev)

    def sample_y_conditional(
        self, X, observed: dict, n_samples: int = 1, random_state: int | None = None
    ):
        """Monte Carlo draws from p(y_unobs | x, y_obs).

        Returns shape ``(n_samples, n, d_unobs)`` in original y units.
        """
        check_is_fitted(self, "params_")
        if self.n_outputs_ < 2:
            raise ValueError("sample_y_conditional requires multivariate y.")
        Xs = self._standardize_x(X)
        n = Xs.shape[0]
        obs_idx, unobs_idx = self._parse_observed(observed)
        if not unobs_idx:
            raise ValueError("All outputs are observed; nothing to sample.")
        y_obs_std = self._broadcast_obs_values(observed, n)
        D_u = self.y_std_[unobs_idx]
        m_u = self.y_mean_[unobs_idx]

        p = self._params_jax()
        seed = random_state if random_state is not None else 0
        key = jax.random.PRNGKey(seed)

        out = np.empty((n_samples, n, len(unobs_idx)))
        for s in range(n_samples):
            keys_s = jax.random.split(jax.random.fold_in(key, s), n)
            samples_std = _model.batched_conditional_sample(
                Xs, jnp.asarray(y_obs_std), keys_s,
                jnp.asarray(obs_idx, dtype=jnp.int32),
                jnp.asarray(unobs_idx, dtype=jnp.int32),
                p["W"], p["b"], p["log_gamma"],
                p["leaf_mu"], p["leaf_chol_raw"], p["log_alpha"],
            )
            out[s] = np.asarray(samples_std) * D_u + m_u
        return out

    # ------------------------------------------------------------------ inverse design

    def inverse_design(
        self,
        targets: dict,
        *,
        n_starts: int = 16,
        n_steps: int = 400,
        learning_rate: float = 0.05,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        init_x: np.ndarray | None = None,
        random_state: int = 0,
    ):
        """Find x such that p(y_target | x) is maximized.

        Parameters
        ----------
        targets : dict {output_idx: value}
            Output values to match. Values are scalars in *original* y units.
            Subset of outputs is supported (constrained inverse design).
        n_starts : number of random restarts.
        n_steps : Adam steps per restart.
        learning_rate : Adam learning rate.
        bounds : tuple of (lo, hi) arrays in *original* x units, optional.
            Restricts the search to a feature box.
        init_x : (n_starts, p) array of starts in *original* x units, optional.
        random_state : int

        Returns
        -------
        x_best : (p,) ndarray in original x units.
        log_lik_best : float
            log p(y_target | x_best) under the trained likelihood.
        x_all : (n_starts, p) ndarray of all restart results.
        log_liks : (n_starts,) ndarray of all final log-likelihoods.
        """
        check_is_fitted(self, "params_")
        obs_idx = sorted(targets.keys())
        y_target_arr = np.array([targets[i] for i in obs_idx], dtype=np.float64)
        # Standardize target into model's y space
        y_target_std = (y_target_arr - self.y_mean_[obs_idx]) / self.y_std_[obs_idx]

        # Bounds and init in standardized x space
        if bounds is not None:
            lo = (np.asarray(bounds[0], dtype=np.float64) - self.x_mean_) / self.x_std_
            hi = (np.asarray(bounds[1], dtype=np.float64) - self.x_mean_) / self.x_std_
            bounds_std = (lo, hi)
        else:
            bounds_std = None
        if init_x is not None:
            init_x = (np.asarray(init_x, dtype=np.float64) - self.x_mean_) / self.x_std_

        params = self._params_jax()
        x_best_std, log_lik_best, x_all_std, log_liks = _inverse.inverse_design(
            params, y_target_std, obs_idx,
            n_starts=n_starts, n_steps=n_steps,
            learning_rate=learning_rate,
            bounds=bounds_std, init_x=init_x,
            random_state=random_state,
        )
        x_best = x_best_std * self.x_std_ + self.x_mean_
        x_all = x_all_std * self.x_std_ + self.x_mean_
        return x_best, log_lik_best, x_all, log_liks

    # ------------------------------------------------------------------ sklearn polish

    def score(self, X, y, sample_weight=None):
        """R^2 (univariate) or uniform-average R^2 across outputs (multivariate)."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight,
                        multioutput="uniform_average")
