"""Polytope geometry of the crisp tree, in standardized x units.

Node ``i`` sends ``xs`` left iff ``W[i] @ xs + b[i] <= 0``. A leaf's region is
the intersection of its path half-spaces with an input box. Two leaves on
opposite sides of node ``i`` are *adjacent* across ``i`` when their regions
share a (p-1)-dimensional facet on the hyperplane ``W[i] @ xs + b[i] = 0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from . import _model


def leaf_path(depth: int, leaf: int) -> list:
    """[(internal node, 'left'|'right'), ...] from the root to ``leaf``."""
    node = _model.n_internal_nodes(depth) + leaf
    path = []
    while node > 0:
        parent = (node - 1) // 2
        path.append((parent, "left" if node == 2 * parent + 1 else "right"))
        node = parent
    return path[::-1]


def path_constraints(W, b, depth: int, leaf: int):
    """Half-spaces ``G xs <= h`` along the path to ``leaf`` (no box).

    Returns ``(nodes, sign, G, h)``; row ``j`` belongs to internal node
    ``nodes[j]`` and ``sign`` is +1 for a left branch, -1 for a right one.
    """
    path = leaf_path(depth, leaf)
    nodes = np.array([n for n, _ in path], dtype=int)
    sign = np.array([1.0 if d == "left" else -1.0 for _, d in path])
    return nodes, sign, sign[:, None] * W[nodes], -sign * b[nodes]


def box_constraints(lo, hi):
    """``G xs <= h`` for ``lo <= xs <= hi``."""
    eye = np.eye(len(lo))
    return np.vstack([eye, -eye]), np.concatenate([hi, -lo])


def chebyshev(G, h, A_eq=None, b_eq=None, cap: float = 1.0):
    """Center and radius of the largest ball (radius <= ``cap``) whose
    portion on ``A_eq x = b_eq`` lies inside ``{x : G x <= h}``.

    Returns ``(center, radius)``; ``(None, 0.0)`` when infeasible.
    """
    n_var = G.shape[1]
    norms = np.linalg.norm(G, axis=1)
    c = np.zeros(n_var + 1)
    c[-1] = -1.0
    A_eq_ = None if A_eq is None else np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    res = linprog(
        c,
        A_ub=np.hstack([G, norms[:, None]]),
        b_ub=h,
        A_eq=A_eq_,
        b_eq=b_eq,
        bounds=[(None, None)] * n_var + [(0.0, cap)],
        method="highs",
    )
    if res.status != 0:
        return None, 0.0
    return res.x[:-1], float(res.x[-1])


def subtree_leaves(depth: int, node: int) -> range:
    """BFS leaf indices under ``node``."""
    n_int = _model.n_internal_nodes(depth)
    lo = hi = node
    while lo < n_int:
        lo, hi = 2 * lo + 1, 2 * hi + 2
    return range(lo - n_int, hi - n_int + 1)


@dataclass
class Facet:
    """Shared facet between ``left`` and ``right`` leaves on node ``node``'s
    hyperplane. ``G, h`` (with ``W[node] @ xs + b[node] = 0``) describe it;
    ``center`` / ``radius`` are its Chebyshev ball within the hyperplane."""

    node: int
    left: int
    right: int
    G: np.ndarray
    h: np.ndarray
    center: np.ndarray
    radius: float


def shared_facets(W, b, depth: int, lo, hi, leaves=None, tol: float = 1e-7):
    """All adjacent leaf pairs inside the box ``[lo, hi]`` (standardized).

    ``leaves`` restricts the search to a set of leaf indices (e.g. the
    non-empty ones). One LP per candidate pair.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = set(range(_model.n_leaves(depth))) if leaves is None else set(leaves)
    box_G, box_h = box_constraints(np.asarray(lo, float), np.asarray(hi, float))
    cons = {l: path_constraints(W, b, depth, l) for l in keep}

    out = []
    for i in range(_model.n_internal_nodes(depth)):
        L = [l for l in subtree_leaves(depth, 2 * i + 1) if l in keep]
        R = [r for r in subtree_leaves(depth, 2 * i + 2) if r in keep]
        for l in L:
            nl, _, Gl, hl = cons[l]
            for r in R:
                nr, _, Gr, hr = cons[r]
                G = np.vstack([Gl[nl != i], Gr[nr != i], box_G])
                h = np.concatenate([hl[nl != i], hr[nr != i], box_h])
                center, radius = chebyshev(G, h, W[i][None, :], np.array([-b[i]]))
                if radius > tol:
                    out.append(Facet(i, l, r, G, h, center, radius))
    return out


def affine_range(c0, c, G, h, A_eq=None, b_eq=None):
    """Extremes of ``c0 + c @ x`` over ``{G x <= h, A_eq x = b_eq}``.

    Returns ``((min_value, argmin), (max_value, argmax))``; a value is NaN
    (and its point None) when the LP fails.
    """
    out = []
    for s in (1.0, -1.0):
        res = linprog(s * c, A_ub=G, b_ub=h, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * len(c), method="highs")
        if res.status == 0:
            out.append((c0 + c @ res.x, res.x))
        else:
            out.append((np.nan, None))
    return out[0], out[1]
