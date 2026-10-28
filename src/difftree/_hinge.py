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


def fit(Xs, Y, J, Wd, bd, jw, ridge, hinge_ridge=None):
    """Least-squares hinge coefficients (q + D, m) and the objective value.

    ``ridge`` regularizes every coefficient; ``hinge_ridge``, if given,
    replaces it on the hinge coefficients ``c_d``.
    """
    G, rhs, const = _normal_equations(Xs, Y, J, Wd, bd, jw)
    r = np.full(G.shape[0], float(ridge))
    if hinge_ridge is not None:
        r[Xs.shape[1] + 1 :] = float(hinge_ridge)
    coef = np.linalg.solve(G + np.diag(r), rhs)
    sse = const - 2.0 * np.sum(coef * rhs) + np.sum(coef * (G @ coef))
    return coef, float(sse)


def fit_refine(Xs, Y, J, Wd, bd, jw, ridge, maxiter, hinge_ridge=None):
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
        return fit(Xs, Y, J, W / n[:, None], b / n, jw, ridge, hinge_ridge)[1]

    v0 = np.r_[(Wd / norms[:, None]).ravel(), bd / norms]
    if maxiter > 0:
        res = minimize(objective, v0, method="Powell",
                       options=dict(maxiter=int(maxiter), xtol=1e-6, ftol=1e-10))
        if res.fun <= objective(v0):
            v0 = res.x
    W, b, n = unpack(v0)
    W, b = W / n[:, None], b / n
    coef, _ = fit(Xs, Y, J, W, b, jw, ridge, hinge_ridge)
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


# ---------------------------------------------------------------------------
# Smoothed hinge (C1)
# ---------------------------------------------------------------------------
#
# Replacing max(0, h) by the C1 piecewise quadratic
#
#     phi_eps(h) = 0 (h <= -eps),  (h + eps)^2 / (4 eps) (|h| < eps),  h (h >= eps)
#
# gives a model that is the hinge model (the affine tree) outside a band
# |h_d| < eps_d around each hyperplane and quadratic inside it, with a
# continuous gradient everywhere. phi is positively homogeneous,
# phi_{s eps}(s h) = s phi_eps(h), so scaling (w_d, b_d, eps_d) by s and c_d by
# 1/s leaves the model unchanged. eps has shape (D, m): one band per depth and
# output.


def phi(H, eps):
    """Smoothed hinge; ``eps == 0`` gives ``max(0, H)``."""
    H, eps = np.broadcast_arrays(np.asarray(H, dtype=np.float64), eps)
    out = np.maximum(H, 0.0)
    band = np.abs(H) < eps
    out[band] = (H[band] + eps[band]) ** 2 / (4 * eps[band])
    return out


def dphi(H, eps):
    """Derivative of :func:`phi`; ``eps == 0`` gives the step ``H > 0``."""
    H, eps = np.broadcast_arrays(np.asarray(H, dtype=np.float64), eps)
    out = (H > 0).astype(np.float64)
    band = np.abs(H) < eps
    out[band] = (H[band] + eps[band]) / (2 * eps[band])
    return out


def smooth_predict(Xs, W, b, eps, coef):
    """Standardized outputs (n, m) of the smoothed hinge model."""
    p = Xs.shape[1]
    H = Xs @ W.T + b  # (n, D)
    Z = phi(H[:, :, None], eps[None])  # (n, D, m)
    return coef[0] + Xs @ coef[1 : p + 1] + np.einsum("ndm,dm->nm", Z, coef[p + 1 :])


def smooth_jacobian(Xs, W, b, eps, coef):
    """Standardized Jacobian (n, m, p) of the smoothed hinge model."""
    p = Xs.shape[1]
    H = Xs @ W.T + b
    S = dphi(H[:, :, None], eps[None])  # (n, D, m)
    return coef[1 : p + 1].T[None] + np.einsum("ndm,dm,dp->nmp", S, coef[p + 1 :], W)


def smooth_fit(Xs, Y, J, Wd, bd, jw, ridge, hinge_ridge, band_bounds, maxiter,
               eps0=(1.0, 0.05)):
    """Fit the smoothed hinge model, starting from the hyperplanes ``(Wd, bd)``.

    For fixed hyperplanes and band half-widths the model is linear in its
    coefficients, which are solved per output by least squares (``ridge`` on
    ``[1, x]``, ``hinge_ridge`` on the hinge coefficients, Jacobian rows
    weighted by ``jw`` when ``J`` is given). The unit normals, offsets and
    ``log eps`` are then optimized with L-BFGS-B (JAX gradients through the
    solve, float64), with ``eps`` in ``band_bounds`` (distance in standardized
    input space) and at most ``maxiter`` iterations (0 keeps the hyperplanes
    fixed and fits only the bands, in at most 100 iterations). The fit is nonconvex; it runs once per
    start in ``eps0`` (clipped to ``band_bounds``) and keeps the lowest
    objective. Returns ``(W, b, eps, coef)`` at the input norms of ``Wd``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    n, p = Xs.shape
    D, m = Wd.shape[0], Y.shape[1]
    q = p + 1
    norms = np.linalg.norm(Wd, axis=1)
    norms[norms == 0] = 1.0
    lo_e, hi_e = (float(v) for v in band_bounds)
    if not 0 < lo_e <= hi_e:
        raise ValueError(f"band_bounds must satisfy 0 < lo <= hi, got {band_bounds}.")
    ridge_diag = np.r_[np.full(q, ridge), np.full(D, hinge_ridge)]

    with jax.enable_x64(True):
        Xj, Yj = jnp.asarray(Xs), jnp.asarray(Y)
        Jj = None if J is None else jnp.asarray(J)
        R = jnp.diag(jnp.asarray(ridge_diag))

        def jphi(H, e):
            return jnp.where(H <= -e, 0.0, jnp.where(H >= e, H, (H + e) ** 2 / (4 * e)))

        def jdphi(H, e):
            return jnp.clip((H + e) / (2 * e), 0.0, 1.0)

        def unpack(v):
            U = v[: D * p].reshape(D, p)
            nu = jnp.linalg.norm(U, axis=1)
            return U / nu[:, None], v[D * p : D * p + D] / nu, jnp.exp(v[D * p + D :]).reshape(D, m)

        def solve_one(W, b, e, y, Jo):
            # Normal equations for one output with band half-widths e (D,).
            H = Xj @ W.T + b
            Phi = jnp.hstack([jnp.ones((n, 1)), Xj, jphi(H, e)])
            G = Phi.T @ Phi + R
            rhs = Phi.T @ y
            const = y @ y
            if Jo is not None:
                # dPhi_i = [0 | I_p | W^T diag(dphi_i)], (n, p, q + D)
                S = jdphi(H, e)
                dPhi = jnp.concatenate([
                    jnp.zeros((n, p, 1)),
                    jnp.broadcast_to(jnp.eye(p), (n, p, p)),
                    jnp.einsum("nd,dp->npd", S, W),
                ], axis=2)
                G = G + jw * jnp.einsum("npa,npb->ab", dPhi, dPhi)
                rhs = rhs + jw * jnp.einsum("npa,np->a", dPhi, Jo)
                const = const + jw * jnp.sum(Jo**2)
            c = jnp.linalg.solve(G, rhs)
            sse = const - 2 * c @ rhs + c @ (G - R) @ c
            return c, sse + c @ R @ c

        def objective(v):
            W, b, e = unpack(v)
            total = 0.0
            for o in range(m):
                Jo = None if Jj is None else Jj[:, o, :]
                total = total + solve_one(W, b, e[:, o], Yj[:, o], Jo)[1]
            return total

        vg = jax.jit(jax.value_and_grad(objective))
        fun = lambda v: tuple(np.asarray(a, dtype=np.float64) for a in vg(jnp.asarray(v)))
        base = np.r_[(Wd / norms[:, None]).ravel(), bd / norms]
        free = [(None, None)] * (D * p + D)
        if maxiter <= 0:
            free = [(v, v) for v in base]
        bounds = free + [(np.log(lo_e), np.log(hi_e))] * (D * m)
        best = None
        for e0 in eps0:
            v0 = np.r_[base, np.full(D * m, np.log(np.clip(e0, lo_e, hi_e)))]
            res = minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=int(maxiter) if maxiter > 0 else 100))
            if best is None or res.fun < best.fun:
                best = res
        W, b, e = unpack(jnp.asarray(best.x))
        coef = jnp.stack([
            solve_one(W, b, e[:, o], Yj[:, o], None if Jj is None else Jj[:, o, :])[0]
            for o in range(m)
        ], axis=1)
        W, b, e, coef = (np.array(a, dtype=np.float64) for a in (W, b, e, coef))
    # Back to the input norms (keeps soft-routing sharpness); c_d rescales.
    coef[q:] /= norms[:, None]
    return W * norms[:, None], b * norms, e * norms[:, None], coef


def pwl_terms(eps, K):
    """Breakpoints and weights of the K-segment interpolant of ``phi_eps``.

    The interpolant through ``K + 1`` equally spaced points on ``[-eps, eps]``
    is ``sum_k a_k max(0, h - t_k)``. Returns ``t`` (..., K + 1), same leading
    shape as ``eps``, and ``a`` (K + 1,), the same for every band.
    """
    s = np.arange(K + 1) / K
    slope = np.r_[0.0, (s[1:] + s[:-1]) / 2, 1.0]  # phi' at segment midpoints
    t = (2 * s - 1) * np.asarray(eps, dtype=np.float64)[..., None]
    return t, np.diff(slope)
