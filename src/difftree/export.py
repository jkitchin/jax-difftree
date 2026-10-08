"""Export a crisp (hardened) regressor for MILP embedding, in original units.

Under crisp routing (``predict(hard=True)``), internal node ``i`` sends ``x``
left iff ``a_i . x <= t_i``, with the split undone from standardized units:

    a_i = W_i / x_std,        t_i = a_i . x_mean - b_i.

Each leaf is then a polytope ``{x : G_l x <= h_l}``: the half-spaces along its
root-to-leaf path intersected with an input box. Its leaf model is affine,
``y = coef_l x + intercept_l`` (``coef_l`` has zero rows for outputs held
constant per leaf), with Gaussian noise covariance ``cov_l``.

Three entry points:

- :func:`leaf_polytopes` returns one :class:`LeafPolytope` per leaf, with
  empty leaves (one LP per leaf) and, optionally, unreached leaves pruned.
- :func:`to_spec` / :func:`to_json` emit the solver-neutral v1 JSON spec of
  jax-ldt (``jax_ldt.export.spec``), so ``jax_ldt.export.from_json`` and its
  discopt adapter can consume a difftree model. Each oblique split becomes
  one lifted column of ``transform_matrix``: the matrix is ``[I_p | a_1 ...
  a_k]``, the first ``p`` lifted features are the raw inputs (the leaf models
  use only these), and node ``i`` splits lifted feature ``p + j`` at ``t_i``.
- :func:`predict_spec` evaluates such a spec in NumPy, as a reference.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from sklearn.utils.validation import check_is_fitted

from . import _model

SPEC_VERSION = 1


@dataclass
class LeafPolytope:
    """One crisp leaf region and its model, in original units.

    Attributes
    ----------
    leaf : int
        Leaf index in BFS leaf order, as returned by ``model.apply``.
    path : list of (int, str)
        Internal nodes on the root-to-leaf path and the branch taken
        (``"left"`` or ``"right"``).
    G, h : ndarray
        Polytope ``{x : G x <= h}``: one row per path split, then the input
        box (``x <= hi`` and ``-x <= -lo``).
    coef : (d_y, p) ndarray
    intercept : (d_y,) ndarray
        Leaf mean ``coef @ x + intercept``.
    cov : (d_y, d_y) ndarray
        Leaf noise covariance (calibrated).
    radius : float
        Chebyshev radius of the polytope in standardized x units (radius of
        the largest ball inside it, capped at 1). Empty polytopes have 0.
    n_samples : int or None
        Rows of the ``X`` passed to :func:`leaf_polytopes` that route here.
    """

    leaf: int
    path: list = field(default_factory=list)
    G: np.ndarray = None
    h: np.ndarray = None
    coef: np.ndarray = None
    intercept: np.ndarray = None
    cov: np.ndarray = None
    radius: float = 0.0
    n_samples: int | None = None

    def contains(self, X, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of rows of ``X`` inside the (closed) polytope."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.all(X @ self.G.T <= self.h + tol, axis=1)

    def predict(self, X) -> np.ndarray:
        """Leaf mean at the rows of ``X``, shape (n, d_y)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return X @ self.coef.T + self.intercept


# ---------------------------------------------------------------------------
# Parameters in original units
# ---------------------------------------------------------------------------


def _check_model(model):
    check_is_fitted(model, "params_")
    if not hasattr(model, "y_mean_"):
        raise TypeError("export supports DifferentiableTreeRegressor only.")
    if not getattr(model, "hardened_", False):
        warnings.warn(
            "Exporting a tree that was not hardened: the export reproduces "
            "predict(hard=True), which can differ from the soft predict(). "
            "Call model.harden(X, y) (or fit with hardening=True) first, and "
            "check model.soft_hard_gap(X, y).",
            UserWarning,
            stacklevel=3,
        )


def _resolve_bounds(model, bounds):
    if bounds is None:
        lo, hi = model.x_min_, model.x_max_
    else:
        lo, hi = bounds
    p = model.n_features_in_
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (p,)).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (p,)).copy()
    if np.any(lo > hi):
        raise ValueError("bounds: every lower bound must be <= its upper bound.")
    return lo, hi


def _splits_original(model):
    """Split normals ``a`` (n_internal, p) and thresholds ``t`` (n_internal,):
    node ``i`` goes left iff ``a_i . x <= t_i``."""
    W = np.asarray(model.params_["W"], dtype=np.float64)
    b = np.asarray(model.params_["b"], dtype=np.float64)
    a = W / model.x_std_[None, :]
    t = a @ model.x_mean_ - b
    return a, t


def _leaves_original(model):
    """Leaf coefficients (L, d_y, p), intercepts (L, d_y), covariances
    (L, d_y, d_y) in original units."""
    import jax.numpy as jnp

    P = model.params_
    sy, my = model.y_std_, model.y_mean_
    sx, mx = model.x_std_, model.x_mean_
    coef = sy[None, :, None] * P["leaf_A"] / sx[None, None, :]
    intercept = my[None, :] + sy[None, :] * P["leaf_mu"] - coef @ mx
    L = np.asarray(
        _model.build_chol(jnp.asarray(P["leaf_chol_raw"]), jnp.asarray(P["log_alpha"])),
        dtype=np.float64,
    )
    cov = (L @ np.swapaxes(L, -1, -2)) * (sy[:, None] * sy[None, :])
    return coef, intercept, cov


def _leaf_path(depth: int, leaf: int) -> list:
    """[(internal node, 'left'|'right'), ...] from the root to ``leaf``."""
    node = _model.n_internal_nodes(depth) + leaf
    path = []
    while node > 0:
        parent = (node - 1) // 2
        path.append((parent, "left" if node == 2 * parent + 1 else "right"))
        node = parent
    return path[::-1]


def _chebyshev_radius(G, h) -> float:
    """Largest r in [0, 1] with a ball of radius r inside ``{x : G x <= h}``;
    0 when the polytope is empty."""
    norms = np.linalg.norm(G, axis=1)
    n_var = G.shape[1]
    c = np.zeros(n_var + 1)
    c[-1] = -1.0
    res = linprog(
        c,
        A_ub=np.hstack([G, norms[:, None]]),
        b_ub=h,
        bounds=[(None, None)] * n_var + [(0.0, 1.0)],
        method="highs",
    )
    if res.status != 0:
        return 0.0
    return float(res.x[-1])


# ---------------------------------------------------------------------------
# Leaf polytopes
# ---------------------------------------------------------------------------


def leaf_polytopes(
    model,
    bounds=None,
    *,
    prune_empty: bool = True,
    X=None,
    prune_unreached: bool = False,
    tol: float = 1e-7,
) -> list[LeafPolytope]:
    """Leaf polytopes ``{x : G_l x <= h_l}`` and leaf models in original units.

    Parameters
    ----------
    model : fitted DifferentiableTreeRegressor (ideally hardened).
    bounds : (lo, hi), optional
        Input box in original units (arrays of length p, or scalars).
        Defaults to the training-data box ``(model.x_min_, model.x_max_)``.
    prune_empty : bool, default True
        Drop leaves whose polytope has no interior inside the box (one LP per
        leaf).
    X : (n, p) array, optional
        Points to count per leaf (``n_samples``), e.g. the training inputs.
    prune_unreached : bool, default False
        Also drop leaves that no row of ``X`` reaches. Requires ``X``.
    tol : float
        A leaf is empty when its Chebyshev radius (standardized x units) is
        at most ``tol``.

    Returns
    -------
    list of LeafPolytope, in BFS leaf order.
    """
    _check_model(model)
    return _leaf_polytopes(model, bounds, prune_empty, X, prune_unreached, tol)


def _leaf_polytopes(model, bounds, prune_empty, X, prune_unreached, tol):
    if prune_unreached and X is None:
        raise ValueError("prune_unreached=True requires X.")
    lo, hi = _resolve_bounds(model, bounds)
    depth = model.depth
    p = model.n_features_in_
    a, t = _splits_original(model)
    coef, intercept, cov = _leaves_original(model)
    counts = None
    if X is not None:
        counts = np.bincount(model.apply(X), minlength=_model.n_leaves(depth))

    # LPs run in standardized units for conditioning; G, h are reported in
    # original units. With xs = (x - mx) / sx, G_s xs <= h_s is equivalent
    # to (G_s / sx) x <= h_s + (G_s / sx) . mx.
    sx, mx = model.x_std_, model.x_mean_
    W = np.asarray(model.params_["W"], dtype=np.float64)
    b = np.asarray(model.params_["b"], dtype=np.float64)
    eye = np.eye(p)
    box_G = np.vstack([eye, -eye])
    box_h = np.concatenate([hi, -lo])
    box_Gs = box_G * sx[None, :]
    box_hs = box_h - box_G @ mx

    out = []
    for leaf in range(_model.n_leaves(depth)):
        path = _leaf_path(depth, leaf)
        sign = np.array([1.0 if d == "left" else -1.0 for _, d in path])
        nodes = np.array([n for n, _ in path], dtype=int)
        Gs = np.vstack([sign[:, None] * W[nodes], box_Gs])
        hs = np.concatenate([-sign * b[nodes], box_hs])
        radius = _chebyshev_radius(Gs, hs)
        n_samples = None if counts is None else int(counts[leaf])
        if prune_empty and radius <= tol:
            continue
        if prune_unreached and n_samples == 0:
            continue
        out.append(LeafPolytope(
            leaf=leaf,
            path=[(int(n), d) for n, d in path],
            G=np.vstack([sign[:, None] * a[nodes], box_G]),
            h=np.concatenate([sign * t[nodes], box_h]),
            coef=coef[leaf],
            intercept=intercept[leaf],
            cov=cov[leaf],
            radius=radius,
            n_samples=n_samples,
        ))
    if not out:
        raise ValueError("Every leaf was pruned; check bounds and X.")
    return out


# ---------------------------------------------------------------------------
# jax-ldt v1 JSON spec
# ---------------------------------------------------------------------------


def to_spec(
    model,
    bounds=None,
    *,
    prune_empty: bool = True,
    X=None,
    prune_unreached: bool = False,
    tol: float = 1e-7,
) -> dict:
    """Serialize the crisp tree to jax-ldt's solver-neutral v1 spec (a dict).

    Arguments are as in :func:`leaf_polytopes`. Pruning collapses every
    internal node with a pruned side into its other child, so the spec only
    contains splits that separate kept leaves. Collapsing nodes whose pruned
    side is empty is exact inside the box; collapsing nodes whose side is only
    unreached (``prune_unreached``) reassigns that region to a neighbouring
    leaf.

    Layout: ``transform_matrix`` is ``[I_p | a_1 ... a_k]`` (one lifted column
    per kept internal node); internal nodes split lifted feature ``p + j`` at
    ``t`` and go left iff ``x'_f <= t``; ``leaf_params[node]`` is ``[bias;
    weights]`` over the lifted features, with weights only on the first ``p``
    (raw) columns. Extra difftree information (bounds, BFS leaf ids, leaf
    covariances) sits under the top-level ``"difftree"`` key, which jax-ldt
    ignores.
    """
    _check_model(model)
    leaves = _leaf_polytopes(model, bounds, prune_empty, X, prune_unreached, tol)
    lo, hi = _resolve_bounds(model, bounds)
    kept = {lp.leaf: lp for lp in leaves}
    depth = model.depth
    n_int = _model.n_internal_nodes(depth)
    p = model.n_features_in_
    d_y = model.n_outputs_
    a, t = _splits_original(model)

    # has_kept[node]: does the subtree under ``node`` hold a kept leaf?
    n_nodes = n_int + _model.n_leaves(depth)
    has_kept = np.zeros(n_nodes, dtype=bool)
    has_kept[n_int + np.array(sorted(kept))] = True
    for node in range(n_int - 1, -1, -1):
        has_kept[node] = has_kept[2 * node + 1] or has_kept[2 * node + 2]

    def effective(node):
        """Skip internal nodes with a pruned side."""
        while node < n_int:
            left, right = 2 * node + 1, 2 * node + 2
            if has_kept[left] and has_kept[right]:
                break
            node = left if has_kept[left] else right
        return node

    # Emit nodes in preorder; children ids are filled in after emission.
    records = []  # (bfs node, left slot, right slot)
    lifted = []   # BFS ids of kept internal nodes, in lifted-column order

    def emit(node):
        node = effective(node)
        rid = len(records)
        records.append([node, -1, -1])
        if node < n_int:
            lifted.append(node)
            records[rid][1] = emit(2 * node + 1)
            records[rid][2] = emit(2 * node + 2)
        return rid

    emit(0)
    n_t = p + len(lifted)
    col = {node: p + j for j, node in enumerate(lifted)}
    A = np.hstack([np.eye(p), a[lifted].T]) if lifted else np.eye(p)

    is_leaf, feature, threshold, left, right = [], [], [], [], []
    leaf_params = np.zeros((len(records), 1 + n_t, d_y))
    leaf_index, leaf_cov = [], []
    for rid, (node, l_id, r_id) in enumerate(records):
        if node < n_int:
            is_leaf.append(False)
            feature.append(col[node])
            threshold.append(float(t[node]))
            left.append(l_id)
            right.append(r_id)
            leaf_index.append(-1)
            leaf_cov.append(None)
        else:
            lp = kept[node - n_int]
            is_leaf.append(True)
            feature.append(-1)
            threshold.append(None)
            left.append(-1)
            right.append(-1)
            leaf_params[rid, 0] = lp.intercept
            leaf_params[rid, 1 : 1 + p] = lp.coef.T
            leaf_index.append(lp.leaf)
            leaf_cov.append(lp.cov.tolist())

    from . import __version__

    return {
        "version": SPEC_VERSION,
        "kind": "hyperplane_tree",
        "n_features_in": int(p),
        "n_features_transformed": int(n_t),
        "n_targets": int(d_y),
        "transform_matrix": A.tolist(),
        "categorical_features": [],
        "linear_features": list(range(p)),
        "tree": {
            "node_count": len(records),
            "is_leaf": is_leaf,
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "leaf_params": leaf_params.tolist(),
        },
        "difftree": {
            "version": __version__,
            "hardened": bool(getattr(model, "hardened_", False)),
            "bounds": [lo.tolist(), hi.tolist()],
            "leaf_index": leaf_index,
            "leaf_cov": leaf_cov,
        },
    }


def to_json(model, path: str | Path | None = None, **kwargs) -> str:
    """:func:`to_spec` as a JSON string, also written to ``path`` if given."""
    s = json.dumps(to_spec(model, **kwargs), indent=2, allow_nan=False)
    if path is not None:
        Path(path).write_text(s)
    return s


def predict_spec(spec: dict | str, X) -> np.ndarray:
    """Evaluate a v1 spec (dict or JSON string) at ``X`` with NumPy.

    Returns (n,) for single-target specs, else (n, n_targets).
    """
    if isinstance(spec, str):
        spec = json.loads(spec)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    T = spec["tree"]
    Xt = X @ np.asarray(spec["transform_matrix"], dtype=np.float64)
    is_leaf = np.asarray(T["is_leaf"], dtype=bool)
    feature = np.asarray(T["feature"], dtype=int)
    thr = np.array([np.nan if v is None else v for v in T["threshold"]])
    left = np.asarray(T["left"], dtype=int)
    right = np.asarray(T["right"], dtype=int)
    node = np.zeros(X.shape[0], dtype=int)
    rows = np.arange(X.shape[0])
    while not is_leaf[node].all():
        active = ~is_leaf[node]
        nd = node[active]
        go_left = Xt[rows[active], feature[nd]] <= thr[nd]
        node[active] = np.where(go_left, left[nd], right[nd])
    params = np.asarray(T["leaf_params"], dtype=np.float64)[node]  # (n, 1+n_t, T)
    X_aug = np.hstack([np.ones((X.shape[0], 1)), Xt])
    y = np.einsum("nf,nft->nt", X_aug, params)
    return y[:, 0] if spec["n_targets"] == 1 else y
