"""Continuous fit of a tied-split tree as a hinging-hyperplanes model.

With tied splits, depth ``d`` uses one hyperplane ``h_d(x) = w_d . x + b_d``
everywhere, so the crisp partition is an arrangement of ``D`` hyperplanes.
A function that is affine on every cell and continuous across every
hyperplane must change slope by the same multiple of ``w_d`` all along
``h_d = 0``, so it is

    f(x) = theta_0 . [1, x] + sum_d c_d max(0, h_d(x))

(Breiman's hinging hyperplanes; a one-hidden-layer ReLU network with ``D``
units). Leaf ``l`` (sign pattern ``s_l``) then has the affine model
``theta_0 + sum_{d: s_ld = right} c_d [b_d; w_d]``. Everything here is in
standardized units.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import _geometry, _model


def _normal_equations(Xs, Y, J, Wd, bd, jw):
    """Gram matrix, right-hand side and constant of the least-squares problem

        ||Phi c - Y||^2 + jw * sum_i ||dPhi_i c - J_i^T||^2,

    with ``Phi = [1, x, max(0, h(x))]`` (n, q + D) and its x-Jacobian rows.
    ``Y`` is (n, m); ``J`` is (n, m, p) or None.
    """
    n, p = Xs.shape
    H = Xs @ Wd.T + bd
    act = (H > 0).astype(np.float64)  # (n, D)
    Phi = np.hstack([np.ones((n, 1)), Xs, np.maximum(H, 0.0)])
    G = Phi.T @ Phi
    rhs = Phi.T @ Y
    const = float(np.sum(Y**2))
    if J is not None:
        # dPhi_i = [0 | I_p | W_d^T act_id], so the summed blocks are
        # I-I: n I, I-hinge: W^T diag(counts), hinge-hinge: (act^T act) * (W W^T).
        q = p + 1
        G[1:q, 1:q] += jw * n * np.eye(p)
        cross = Wd.T * act.sum(axis=0)[None, :]  # (p, D)
        G[1:q, q:] += jw * cross
        G[q:, 1:q] += jw * cross.T
        G[q:, q:] += jw * (act.T @ act) * (Wd @ Wd.T)
        rhs[1:q] += jw * J.sum(axis=0).T
        rhs[q:] += jw * np.einsum("id,dp,imp->dm", act, Wd, J)
        const += jw * float(np.sum(J**2))
    return G, rhs, const


def fit(Xs, Y, J, Wd, bd, jw, ridge):
    """Least-squares hinge coefficients (q + D, m) and the objective value."""
    G, rhs, const = _normal_equations(Xs, Y, J, Wd, bd, jw)
    coef = np.linalg.solve(G + ridge * np.eye(G.shape[0]), rhs)
    sse = const - 2.0 * np.sum(coef * rhs) + np.sum(coef * (G @ coef))
    return coef, float(sse)


def fit_refine(Xs, Y, J, Wd, bd, jw, ridge, maxiter):
    """Move the hyperplanes (Powell, at most ``maxiter`` iterations) to
    minimize the least-squares objective, then fit the coefficients.

    The hyperplanes are scale-invariant here (the scale moves into ``c_d``),
    so they are optimized as unit normals and returned at their input norms.
    Returns ``(Wd, bd, coef)``.
    """
    D, p = Wd.shape
    norms = np.linalg.norm(Wd, axis=1)
    norms[norms == 0] = 1.0

    def unpack(v):
        W = v[: D * p].reshape(D, p)
        n = np.linalg.norm(W, axis=1)
        return W, v[D * p :], n

    def objective(v):
        W, b, n = unpack(v)
        if np.any(n < 1e-8):
            return 1e30
        return fit(Xs, Y, J, W / n[:, None], b / n, jw, ridge)[1]

    v0 = np.r_[(Wd / norms[:, None]).ravel(), bd / norms]
    if maxiter > 0:
        res = minimize(objective, v0, method="Powell",
                       options=dict(maxiter=int(maxiter), xtol=1e-6, ftol=1e-10))
        if res.fun <= objective(v0):
            v0 = res.x
    W, b, n = unpack(v0)
    W, b = W / n[:, None], b / n
    coef, _ = fit(Xs, Y, J, W, b, jw, ridge)
    # Return at the input norms (keeps soft-routing sharpness); c_d rescales.
    coef[p + 1 :] /= norms[:, None]
    return W * norms[:, None], b * norms, coef


def leaf_thetas(coef, Wd, bd, depth):
    """Per-leaf affine parameters ``[b_l; A_l^T]`` (n_leaves, q, m) of the
    hinge model, for the tree whose depth-``d`` nodes all split on
    ``(Wd[d], bd[d])``."""
    q = Wd.shape[1] + 1
    base = coef[:q]
    hinge = np.hstack([bd[:, None], Wd])  # (D, q): [b_d; w_d] per depth
    out = np.empty((_model.n_leaves(depth), q, coef.shape[1]))
    for leaf in range(_model.n_leaves(depth)):
        theta = base.copy()
        for d, (_, side) in enumerate(_geometry.leaf_path(depth, leaf)):
            if side == "right":
                theta += hinge[d][:, None] * coef[q + d][None, :]
        out[leaf] = theta
    return out
