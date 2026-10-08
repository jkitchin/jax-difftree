"""Sklearn-compatible heteroscedastic differentiable-tree regressor.

Supports both univariate and multivariate targets. For multivariate y, leaves
are multivariate Gaussians, and additional methods expose:

  - conditional prediction p(y_unobs | x, y_obs) in closed form,
  - conditional sampling via leaf-weight reweighting + Schur complements,
  - inverse design: gradient-based optimization over x to match a target
    output vector under the trained likelihood.

Leaves can be constant (``leaf_type="constant"``) or affine
(``leaf_type="affine"``, mean ``A_l x + b_l``), optionally trained with a
Jacobian-matching term when ``dydx`` is passed to ``fit``. ``predict(...,
hard=True)`` routes crisply, ``harden`` turns the soft tree into a crisp
piecewise model with least-squares leaves, and :mod:`difftree.export` exports
the crisp tree as leaf polytopes or a jax-ldt JSON spec.
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
from ._train import anneal, calibrate, train


def _reg_args(params) -> tuple:
    """Positional parameter tuple shared by the regressor functions in _model."""
    return (
        params["W"], params["b"], params["log_gamma"], params["leaf_mu"],
        params["leaf_A"], params["leaf_chol_raw"], params["log_alpha"],
    )


def _make_regressor_loss(slope_mask, jacobian_weight: float = 0.0):
    """Negative mean log-likelihood, plus an optional Jacobian-matching term.

    ``slope_mask`` has shape ``(d_y,)``: 1 for outputs with affine leaves and
    0 for outputs held constant per leaf (their slopes are zeroed). When the
    targets are a tuple ``(y, dydx)`` the loss adds

        jacobian_weight * mean_i || sum_l mu_l(x_i) A_l - dydx_i ||_F^2,

    restricted to the affine outputs. ``sum_l mu_l A_l`` is the Jacobian of the
    crisp tree within a leaf, averaged over the soft routing, so the term
    teaches the leaf slopes without penalizing the steep transitions between
    leaves that only exist because routing is soft.
    """
    mask = jnp.asarray(slope_mask)[:, None]  # (d_y, 1)

    def loss(params, X, y):
        J = None
        if isinstance(y, tuple):
            y, J = y
        params = {**params, "leaf_A": params["leaf_A"] * mask}
        ll = _model.batched_regressor_log_lik(X, y, *_reg_args(params))
        out = -jnp.mean(ll)
        if J is not None:
            S = _model.batched_expected_leaf_slope(
                X, params["W"], params["b"], params["log_gamma"], params["leaf_A"]
            )
            r = (S - J) * mask[None]
            out = out + jacobian_weight * jnp.mean(jnp.sum(r**2, axis=(1, 2)))
        return out

    return loss


_regressor_loss = _make_regressor_loss(jnp.ones((1,)))


def _chol_to_raw(L: np.ndarray) -> np.ndarray:
    """Inverse of ``_model.build_chol`` (with ``log_alpha = 0``).

    Diagonal entries are clipped just above ``MIN_DIAG``, the floor that
    ``build_chol`` imposes.
    """
    d = L.shape[-1]
    raw = np.tril(L, k=-1)
    diag = np.maximum(np.diag(L) - _model.MIN_DIAG, 1e-6)
    raw[np.arange(d), np.arange(d)] = np.log(np.expm1(diag))
    return raw


def _mahalanobis_sq(diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Row-wise ``diff_i^T cov_i^{-1} diff_i`` for (n, d) diff, (n, d, d) cov."""
    return np.einsum("nd,nd->n", diff, np.linalg.solve(cov, diff[..., None])[..., 0])


def _mvn_logpdf(diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = diff.shape[-1]
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + _mahalanobis_sq(diff, cov))


class DifferentiableTreeRegressor(RegressorMixin, BaseEstimator):
    """Soft oblique decision tree of fixed depth with multivariate Gaussian leaves.

    Predictive distribution at each input x is a mixture of multivariate
    Gaussians, one per leaf, weighted by soft path probabilities mu_l(x). For
    univariate y this reduces to the heteroscedastic Gaussian model; for
    multivariate y it is a structured conditional GMM (input-dependent MDN).
    All outputs share one partition (one set of splits).

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
    leaf_type : {"constant", "affine"}, default="constant"
        ``"constant"``: leaf ``l`` predicts the mean ``b_l``. ``"affine"``: leaf
        ``l`` predicts ``A_l x + b_l``. The Gaussian noise model is the same in
        both cases, so ``predict(return_std=True)`` works for either.
    constant_outputs : sequence of int or None, default=None
        With ``leaf_type="affine"``, output indices whose leaves stay constant
        (``A_l`` row fixed at zero), e.g. intensive properties that enter
        bilinear blending terms. Ignored for constant leaves.
    jacobian_weight : float, default=1.0
        Weight of the Jacobian-matching (Sobolev) term, used only when
        ``fit`` receives ``dydx``. The term is computed in standardized units.
    hardening : bool, default=False
        If True, ``fit`` ends with :meth:`harden` on the same train/validation
        split.
    gamma_hard : float, default=50.0
        Final split steepness reached while hardening.
    n_hard_epochs : int, default=50
        Training epochs used to anneal gamma from ``gamma_max`` to
        ``gamma_hard`` before the partition is frozen.
    hard_ridge : float, default=1e-6
        Ridge strength of the least-squares leaf refit. Coefficients shrink
        toward the soft leaf values, which only matters for leaves with too
        few training points to fit on their own.
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
        leaf_type: str = "constant",
        constant_outputs: Sequence[int] | None = None,
        jacobian_weight: float = 1.0,
        hardening: bool = False,
        gamma_hard: float = 50.0,
        n_hard_epochs: int = 50,
        hard_ridge: float = 1e-6,
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
        self.leaf_type = leaf_type
        self.constant_outputs = constant_outputs
        self.jacobian_weight = jacobian_weight
        self.hardening = hardening
        self.gamma_hard = gamma_hard
        self.n_hard_epochs = n_hard_epochs
        self.hard_ridge = hard_ridge
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ fit

    def fit(self, X, y, dydx=None):
        """Fit the tree.

        Parameters
        ----------
        X : (n, p) array
        y : (n,) or (n, d_y) array
        dydx : (n, d_y, p) array, optional
            Jacobian of y with respect to x at each training point, in
            original units. For 1D ``y`` a ``(n, p)`` array is also accepted.
            Requires ``leaf_type="affine"``; adds a Jacobian-matching term
            weighted by ``jacobian_weight``.
        """
        if self.leaf_type not in ("constant", "affine"):
            raise ValueError(
                f"leaf_type must be 'constant' or 'affine', got {self.leaf_type!r}"
            )
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
        self.slope_mask_ = self._make_slope_mask()

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
        # Training-data box: the default input domain for export.
        self.x_min_ = X.min(axis=0)
        self.x_max_ = X.max(axis=0)

        Xs = (X - self.x_mean_) / self.x_std_
        ys = (y_arr - self.y_mean_) / self.y_std_
        Js = self._standardize_dydx(dydx, X.shape[0])

        if Js is None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                Xs, ys, test_size=self.validation_fraction, random_state=seed
            )
            J_tr = None
        else:
            X_tr, X_val, y_tr, y_val, J_tr, _ = train_test_split(
                Xs, ys, Js, test_size=self.validation_fraction, random_state=seed
            )

        key = jax.random.PRNGKey(seed)
        params = _model.init_regressor_params(
            key, self.depth, X.shape[1], self.n_outputs_,
            self.gamma_init, self.sigma_init,
        )

        if self.verbose:
            print(f"[difftree] training on {X_tr.shape[0]} samples, "
                  f"d_y={self.n_outputs_}, leaf_type={self.leaf_type}...")
        train_loss, targets = self._train_loss_and_targets(y_tr, J_tr)
        mean_keys = self._mean_keys()
        params = train(
            params, train_loss,
            jnp.asarray(X_tr), targets,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma_init=self.gamma_init,
            gamma_max=self.gamma_max,
            delta_gamma=self.delta_gamma,
            rng=rng_np,
            trainable_keys=mean_keys + ("leaf_chol_raw",),
            warmup_keys=mean_keys,
            warmup_fraction=self.warmup_fraction,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"[difftree] calibrating on {X_val.shape[0]} held-out samples...")
        params = calibrate(
            params, _make_regressor_loss(self.slope_mask_),
            jnp.asarray(X_val), jnp.asarray(y_val),
            calibration_lr=self.calibration_lr,
            n_calibration_steps=self.n_calibration_steps,
            calib_keys=("log_gamma", "log_alpha"),
            verbose=self.verbose,
        )

        self._store_params(params)
        self.hardened_ = False
        if self.hardening:
            self._harden_std(X_tr, y_tr, J_tr, X_val, y_val, rng_np)
        return self

    # ------------------------------------------------------------------ fit helpers

    def _make_slope_mask(self) -> np.ndarray:
        """(d_y,) array: 1 where leaves are affine in x, 0 where constant."""
        if self.leaf_type == "constant":
            return np.zeros(self.n_outputs_)
        mask = np.ones(self.n_outputs_)
        for i in self.constant_outputs or ():
            if not 0 <= int(i) < self.n_outputs_:
                raise ValueError(
                    f"constant_outputs index {i} out of range for "
                    f"{self.n_outputs_} outputs"
                )
            mask[int(i)] = 0.0
        return mask

    def _mean_keys(self) -> tuple[str, ...]:
        keys = ("W", "b", "leaf_mu")
        return keys + ("leaf_A",) if self.leaf_type == "affine" else keys

    def _standardize_dydx(self, dydx, n: int):
        """Validate ``dydx`` and convert it to standardized units, or None."""
        if dydx is None:
            return None
        if self.leaf_type != "affine":
            raise ValueError("dydx requires leaf_type='affine'.")
        J = np.asarray(dydx, dtype=np.float64)
        p = self.n_features_in_
        if self.y_was_1d_ and J.ndim == 2:
            J = J[:, None, :]
        if J.shape != (n, self.n_outputs_, p):
            raise ValueError(
                f"dydx must have shape {(n, self.n_outputs_, p)}, got {J.shape}"
            )
        # d(y/sy)/d(x/sx) = J * sx / sy
        return J * self.x_std_[None, None, :] / self.y_std_[None, :, None]

    def _train_loss_and_targets(self, y_tr, J_tr):
        if J_tr is None:
            return _make_regressor_loss(self.slope_mask_), jnp.asarray(y_tr)
        loss = _make_regressor_loss(self.slope_mask_, self.jacobian_weight)
        return loss, (jnp.asarray(y_tr), jnp.asarray(J_tr))

    def _store_params(self, params):
        params = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        params["leaf_A"] = params["leaf_A"] * self.slope_mask_[None, :, None]
        self.params_ = params

    # ------------------------------------------------------------------ hardening

    def harden(self, X, y, dydx=None):
        """Turn the fitted soft tree into a crisp piecewise model.

        Steps:

        1. Keep training splits and leaves while gamma grows geometrically from
           ``gamma_max`` to ``gamma_hard`` over ``n_hard_epochs`` epochs, so the
           splits settle where crisp routing wants them.
        2. Freeze the partition: route each training point crisply
           (:meth:`apply`).
        3. Refit every leaf by least squares on the points it holds (affine
           outputs regress on ``[1, x]``, constant outputs take the mean;
           ``dydx`` rows join the least-squares system when given). Leaf noise
           covariances are refit from the residuals.
        4. Fit the variance scale ``alpha`` in closed form on the held-out
           validation slice under crisp routing, and set every per-depth gamma
           to ``gamma_hard``.

        ``X, y`` are split with ``validation_fraction`` exactly as in
        :meth:`fit`. Passing the training data used in ``fit`` is the usual
        choice. The result of comparing soft and hard predictions on the
        validation slice before and after hardening is stored in
        ``hardening_report_`` (see :meth:`soft_hard_gap`).

        Returns self.
        """
        check_is_fitted(self, "params_")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        Xs = self._standardize_x_np(X)
        ys = self._standardize_y(y_arr)
        Js = self._standardize_dydx(dydx, Xs.shape[0])
        rng_np = np.random.default_rng(self.random_state)
        seed = int(rng_np.integers(0, 2**31 - 1))
        if Js is None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                Xs, ys, test_size=self.validation_fraction, random_state=seed
            )
            J_tr = None
        else:
            X_tr, X_val, y_tr, y_val, J_tr, _ = train_test_split(
                Xs, ys, Js, test_size=self.validation_fraction, random_state=seed
            )
        self._harden_std(X_tr, y_tr, J_tr, X_val, y_val, rng_np)
        return self

    def _harden_std(self, X_tr, y_tr, J_tr, X_val, y_val, rng_np):
        """Hardening on standardized train/validation arrays."""
        X_val_o = X_val * self.x_std_ + self.x_mean_
        y_val_o = self._unstandardize_y(y_val)
        before = self.soft_hard_gap(X_val_o, y_val_o)

        if self.verbose:
            print(f"[difftree] hardening: gamma {self.gamma_max} -> {self.gamma_hard}")
        params = self._params_jax()
        if self.n_hard_epochs > 0:
            loss, targets = self._train_loss_and_targets(y_tr, J_tr)
            params = anneal(
                params, loss, jnp.asarray(X_tr), targets,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                n_epochs=self.n_hard_epochs,
                gamma_start=self.gamma_max,
                gamma_end=self.gamma_hard,
                rng=rng_np,
                trainable_keys=self._mean_keys() + ("leaf_chol_raw",),
                verbose=self.verbose,
            )
        self._store_params(params)
        p = self.params_

        leaves = self.apply_std(X_tr)
        new_mu, new_A, new_raw = self._refit_leaves(X_tr, y_tr, J_tr, leaves)
        p["leaf_mu"], p["leaf_A"], p["leaf_chol_raw"] = new_mu, new_A, new_raw
        p["log_alpha"] = np.zeros(())
        p["log_gamma"] = np.full(self.depth, np.log(self.gamma_hard))

        # Closed-form variance scale on validation under crisp routing:
        # alpha = mean squared Mahalanobis distance / d_y.
        mean_v, cov_v = self._hard_moments_std(X_val)
        z2 = _mahalanobis_sq(y_val - mean_v, cov_v)
        p["log_alpha"] = np.asarray(np.log(max(z2.mean() / self.n_outputs_, 1e-12)))

        self.hardened_ = True
        self.hardening_report_ = {
            "before": before,
            "after": self.soft_hard_gap(X_val_o, y_val_o),
            "n_validation": int(X_val.shape[0]),
        }
        return self

    def _refit_leaves(self, Xs, ys, Js, leaves):
        """Least-squares leaf refit on a frozen crisp partition (standardized).

        Returns new ``(leaf_mu, leaf_A, leaf_chol_raw)``. Coefficients are
        ridge-shrunk toward the current (soft) leaf values with strength
        ``hard_ridge``; empty leaves keep their current values.
        """
        p = self.params_
        n_lvs, d_y, n_feat = p["leaf_A"].shape
        mask = self.slope_mask_
        lam = float(self.hard_ridge)
        jw = float(self.jacobian_weight)
        L_prev = np.asarray(_model.build_chol(
            jnp.asarray(p["leaf_chol_raw"]), jnp.asarray(p["log_alpha"])
        ))
        Sigma_prev = L_prev @ np.swapaxes(L_prev, -1, -2)
        mu, A = p["leaf_mu"].copy(), p["leaf_A"].copy()
        raw = p["leaf_chol_raw"].copy()
        nu = d_y + 1.0  # prior weight (pseudo-points) on the previous covariance

        for l in range(n_lvs):
            idx = np.flatnonzero(leaves == l)
            n_l = idx.size
            if n_l == 0:
                continue
            Xl, yl = Xs[idx], ys[idx]
            # Affine fit: theta = [b; A^T], shape (p + 1, d_y).
            Z = np.hstack([np.ones((n_l, 1)), Xl])
            theta_prev = np.vstack([mu[l][None, :], A[l].T])
            G = Z.T @ Z + lam * np.eye(n_feat + 1)
            rhs = Z.T @ yl + lam * theta_prev
            if Js is not None:
                G[1:, 1:] += jw * n_l * np.eye(n_feat)
                rhs[1:] += jw * Js[idx].sum(axis=0).T
            theta = np.linalg.solve(G, rhs)
            # Constant fit for outputs whose leaves stay constant.
            const = (yl.sum(axis=0) + lam * mu[l]) / (n_l + lam)
            mu[l] = np.where(mask > 0, theta[0], const)
            A[l] = theta[1:].T * mask[:, None]

            resid = yl - (mu[l][None, :] + Xl @ A[l].T)
            S = (resid.T @ resid + nu * Sigma_prev[l]) / (n_l + nu)
            raw[l] = _chol_to_raw(np.linalg.cholesky(S))
        return mu, A, raw

    def soft_hard_gap(self, X, y):
        """Compare soft and hard (crisp) predictions on ``(X, y)``.

        Returns a dict with, per output, ``rmse_soft``, ``rmse_hard``,
        ``rmse_gap = rmse_hard - rmse_soft``, ``r2_soft``, ``r2_hard``; the
        mean negative log-likelihoods ``nll_soft`` and ``nll_hard`` (original
        y units); ``mean_max_leaf_prob``, the average probability of each
        point's most likely leaf under soft routing (1.0 means routing is
        already crisp); and ``routing_agreement``, the fraction of points
        whose most likely soft leaf is the crisp leaf. A large gap means the
        soft fit relies on blending leaves and is a misleading guide to the
        crisp (MILP-embeddable) model.
        """
        from sklearn.metrics import r2_score

        check_is_fitted(self, "params_")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        Xs_np = self._standardize_x_np(X)
        Xs = jnp.asarray(Xs_np)
        ys = self._standardize_y(y_arr)
        p = self._params_jax()

        mean_s, *_ = _model.batched_regressor_predict_moments(Xs, *_reg_args(p))
        mean_s = self._unstandardize_y(np.asarray(mean_s))
        mean_h_std, cov_h = self._hard_moments_std(Xs_np)
        mean_h = self._unstandardize_y(mean_h_std)

        log_jac = float(np.sum(np.log(self.y_std_)))
        ll_soft = np.asarray(_model.batched_regressor_log_lik(
            Xs, jnp.asarray(ys), *_reg_args(p)))
        ll_hard = _mvn_logpdf(ys - mean_h_std, cov_h)

        log_mu = np.asarray(_model.batched_log_path_probs(
            Xs, p["W"], p["b"], p["log_gamma"]))
        hard_leaf = self.apply_std(Xs_np)

        rmse_soft = np.sqrt(np.mean((mean_s - y_arr) ** 2, axis=0))
        rmse_hard = np.sqrt(np.mean((mean_h - y_arr) ** 2, axis=0))
        return {
            "rmse_soft": rmse_soft,
            "rmse_hard": rmse_hard,
            "rmse_gap": rmse_hard - rmse_soft,
            "r2_soft": np.atleast_1d(r2_score(y_arr, mean_s, multioutput="raw_values")),
            "r2_hard": np.atleast_1d(r2_score(y_arr, mean_h, multioutput="raw_values")),
            "nll_soft": float(-ll_soft.mean() + log_jac),
            "nll_hard": float(-ll_hard.mean() + log_jac),
            "mean_max_leaf_prob": float(np.exp(log_mu.max(axis=1)).mean()),
            "routing_agreement": float((log_mu.argmax(axis=1) == hard_leaf).mean()),
        }

    # ------------------------------------------------------------------ helpers

    def _params_jax(self) -> dict:
        return {k: jnp.asarray(v) for k, v in self.params_.items()}

    def _standardize_x_np(self, X) -> np.ndarray:
        """Standardized inputs as float64 NumPy (the crisp path stays in
        float64 so it matches the exported model to round-off)."""
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return (X - self.x_mean_) / self.x_std_

    def _standardize_x(self, X) -> jnp.ndarray:
        return jnp.asarray(self._standardize_x_np(X))

    def _standardize_y(self, y_arr: np.ndarray) -> np.ndarray:
        return (np.asarray(y_arr) - self.y_mean_) / self.y_std_

    def _unstandardize_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_std_ + self.y_mean_

    def _shape_y_out(self, arr: np.ndarray) -> np.ndarray:
        """Strip the trailing output dimension if the user fitted with 1D y."""
        if self.y_was_1d_ and arr.ndim >= 1 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    def apply_std(self, Xs) -> np.ndarray:
        """Crisp leaf index for standardized inputs (see :meth:`apply`)."""
        return _model.hard_leaf_index(Xs, self.params_["W"], self.params_["b"], self.depth)

    def apply(self, X) -> np.ndarray:
        """Leaf index (0 .. 2**depth - 1, BFS leaf order) each row routes to
        under crisp routing: node ``i`` goes right iff ``w_i . x + b_i > 0``."""
        check_is_fitted(self, "params_")
        return self.apply_std(self._standardize_x_np(X))

    def _hard_moments_std(self, Xs):
        """Crisp-routing mean (n, d_y) and covariance (n, d_y, d_y), standardized."""
        p = self.params_
        Xs = np.asarray(Xs, dtype=np.float64)
        leaves = self.apply_std(Xs)
        mean = p["leaf_mu"][leaves] + np.einsum("nyp,np->ny", p["leaf_A"][leaves], Xs)
        L = np.asarray(_model.build_chol(
            jnp.asarray(p["leaf_chol_raw"]), jnp.asarray(p["log_alpha"])
        ), dtype=np.float64)
        Sigma = L @ np.swapaxes(L, -1, -2)
        return mean, Sigma[leaves]

    # ------------------------------------------------------------------ predict

    def predict(
        self,
        X,
        return_std: bool = False,
        return_components: bool = False,
        hard: bool = False,
    ):
        """Predict marginal mean (and optionally per-output marginal std).

        For multivariate y, ``std`` is the per-output marginal standard
        deviation (sqrt of diagonal of the predictive covariance). Use
        ``predict_full_cov`` for the full covariance matrix.

        With ``hard=True`` each row is routed crisply to one leaf (see
        :meth:`apply`) and the prediction is that leaf's Gaussian: the std is
        the leaf noise only and the routing component is zero.
        """
        check_is_fitted(self, "params_")
        if hard:
            mean, ale = self._hard_moments_std(self._standardize_x_np(X))
            rou = np.zeros_like(ale)
            total = ale
        else:
            Xs = self._standardize_x(X)
            p = self._params_jax()
            mean, total, ale, rou = _model.batched_regressor_predict_moments(
                Xs, *_reg_args(p)
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

    def predict_full_cov(self, X, hard: bool = False):
        """Predict mean and full predictive covariance per sample.

        Returns ``(mean, cov)`` of shapes ``(n, d_y)`` and ``(n, d_y, d_y)``,
        in original (un-standardized) y units. ``hard=True`` uses crisp
        routing, as in :meth:`predict`.
        """
        check_is_fitted(self, "params_")
        if hard:
            mean, total = self._hard_moments_std(self._standardize_x_np(X))
        else:
            Xs = self._standardize_x(X)
            p = self._params_jax()
            mean, total, _, _ = _model.batched_regressor_predict_moments(
                Xs, *_reg_args(p)
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
                Xs, keys[s], *_reg_args(p)
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
            *_reg_args(p),
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
                *_reg_args(p),
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
