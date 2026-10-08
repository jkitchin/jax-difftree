"""Gradient-based inverse design for the differentiable-tree regressor.

Given target values for some subset of outputs, we use the trained tree as a
likelihood `p(y_target | x)` and search over `x` (in standardized feature space)
by Adam ascent. Multiple random restarts mitigate local optima from the soft
mixture surface; the best candidate by log-evidence is returned.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import _model


def _make_loss(params: dict, obs_idx: jnp.ndarray, y_target_obs: jnp.ndarray):
    def loss(x):
        return -_model.inverse_design_log_lik(
            x,
            y_target_obs,
            obs_idx,
            params["W"],
            params["b"],
            params["log_gamma"],
            params["leaf_mu"],
            params["leaf_A"],
            params["leaf_chol_raw"],
            params["log_alpha"],
        )
    return loss


def inverse_design(
    params: dict,
    y_target_obs: np.ndarray,
    obs_idx: Sequence[int],
    *,
    n_starts: int = 16,
    n_steps: int = 400,
    learning_rate: float = 0.05,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    init_x: np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Maximize log p(y_target_obs | x) over x.

    All operations are in *standardized feature space*; the caller is
    responsible for any inverse standardization on the returned x.

    Parameters
    ----------
    params : dict
        Fitted tree parameters (jax arrays).
    y_target_obs : (d_obs,) ndarray
        Target values for the observed output indices, in standardized y space.
    obs_idx : sequence of int
        Indices of constrained outputs in the multivariate y vector.
    n_starts : int
        Random restart count.
    n_steps : int
        Adam steps per restart.
    learning_rate : float
        Adam learning rate.
    bounds : tuple of (lo, hi) ndarrays, optional
        Per-feature bounds in standardized x space; clipped each step.
    init_x : (n_starts, p) ndarray, optional
        Custom starts; if None, drawn from standard normal.
    random_state : int

    Returns
    -------
    x_best : (p,) ndarray
        Best designed input (in standardized space).
    log_lik_best : float
        log p(y_target_obs | x_best).
    x_all : (n_starts, p) ndarray
        Optimized x from every restart.
    log_liks : (n_starts,) ndarray
        Final log-evidence for every restart.
    """
    obs_idx_j = jnp.asarray(obs_idx, dtype=jnp.int32)
    y_target_obs_j = jnp.asarray(y_target_obs)
    p = params["W"].shape[1]

    if init_x is None:
        rng = np.random.default_rng(random_state)
        init_x = rng.standard_normal((n_starts, p))
    else:
        init_x = np.asarray(init_x)
        if init_x.ndim == 1:
            init_x = init_x[None, :]
        n_starts = init_x.shape[0]

    if bounds is not None:
        lo, hi = jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
    else:
        lo = hi = None

    loss_fn = _make_loss(params, obs_idx_j, y_target_obs_j)

    @jax.jit
    def step_one(x, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(x)
        upd, opt_state = opt.update(g, opt_state, x)
        x = optax.apply_updates(x, upd)
        if lo is not None:
            x = jnp.clip(x, lo, hi)
        return x, opt_state, loss

    @jax.jit
    def run_one(x0):
        opt_state = opt.init(x0)

        def body(carry, _):
            x, opt_state = carry
            x, opt_state, loss = step_one(x, opt_state)
            return (x, opt_state), loss

        (x_final, _), losses = jax.lax.scan(body, (x0, opt_state), None, length=n_steps)
        return x_final, losses

    opt = optax.adam(learning_rate)

    x_all = []
    log_liks = []
    for s in range(n_starts):
        x_final, losses = run_one(jnp.asarray(init_x[s]))
        x_all.append(np.asarray(x_final))
        log_liks.append(-float(losses[-1]))

    x_all = np.stack(x_all, axis=0)
    log_liks = np.asarray(log_liks)
    best = int(np.argmax(log_liks))
    return x_all[best], float(log_liks[best]), x_all, log_liks
