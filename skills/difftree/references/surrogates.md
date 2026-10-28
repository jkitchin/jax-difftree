# Tree surrogates for MILP and NLP optimization

Read this when the user wants to embed a difftree model in an optimization
problem (Pyomo, HiGHS, IPOPT, jax-ldt/discopt), harden a tree, remove jumps or
kinks, or export leaf polytopes.

## 1. Affine leaves and Jacobians

```python
from difftree import DifferentiableTreeRegressor

m = DifferentiableTreeRegressor(
    depth=4,
    leaf_type="affine",       # leaf l predicts A_l x + b_l
    constant_outputs=[1],     # optional: output 1 constant per leaf (e.g. intensive properties)
    jacobian_weight=1.0,      # weight of the Jacobian-matching term
    random_state=0,
).fit(X, Y, dydx=J)           # J: (n, d_y, p) dy/dx in original units, optional
```

All outputs share one partition, so one tree per unit operation means one set
of binaries. Jacobian data (`dydx`) helps most when samples are scarce.

## 2. Hardening: from soft tree to crisp model

`hardening=True` (constructor) or `m.harden(X, y, dydx=None, continuous=None,
smoothness=None, bounds=None)` anneals the splits to near-crisp, freezes the
partition, refits the leaves by least squares, refits noise covariances, and
recalibrates the variance on the validation slice.

```python
m.predict(X, hard=True)           # crisp model (what an optimizer sees)
m.apply(X)                        # crisp leaf index (BFS order)
m.hardening_report_               # dict, see below
m.soft_hard_gap(X, y)             # soft vs crisp on any data
```

`hardening_report_` keys: `before` / `after` (per-output `rmse_soft`,
`rmse_hard`, `rmse_gap`, `r2_*`, `nll_*`, `mean_max_leaf_prob`,
`routing_agreement`), `n_validation`, `continuous` (output indices),
`max_boundary_jump` and `max_gradient_jump` (per output, original units, over
facets inside the box), `smoothness`, and for C1 models `band_halfwidth`
(depth x continuous outputs, standardized input units).

Check `after["rmse_hard"]` against the accuracy you need before exporting.

## 3. Choosing the smoothness

| Variant | Constructor | Values | Gradient | Embedding |
|---|---|---|---|---|
| Discontinuous | `leaf_type="affine", hardening=True` | jump at splits | jumps | exact MILP |
| C0 | `+ tied_splits=True, continuous=True` | continuous | jumps | exact MILP (leaf or hinge form) |
| C1 | `+ smoothness=1` | continuous | continuous | MILP approximation within a bound, or NLP |

**Discontinuous** trees are often the most accurate per leaf, but a MILP
optimum can sit on the high side of a jump between leaves. Measure with
`export.boundary_jumps(m, bounds=(lo, hi))`.

**C0 (tied, continuous).** With `tied_splits=True` every node at depth `d`
uses one hyperplane `h_d(x) = w_d . x + b_d`, so the cells form a hyperplane
arrangement, and a continuous model on it is exactly a hinging-hyperplanes
model `f(x) = theta_0 . [1, x] + sum_d c_d max(0, h_d(x))` (a one-hidden-layer
ReLU network with `depth` units). Hardening fits this form directly and
refines the hyperplanes (`refine_splits` Powell iterations). It is continuous
everywhere, not only in the box. `continuous=True` without `tied_splits` also
works but forces leaves to merge where a deeper split meets a parent boundary,
and is usually much less accurate in more than one dimension.
`continuous=[0, 2]` makes only those outputs continuous.

**C1 (`smoothness=1`).** A piecewise-affine function with a continuous
gradient is globally affine, so C1 needs quadratic pieces. Each `max(0, h)` is
replaced by `phi_eps(h)`: 0 for `h <= -eps`, `(h + eps)^2 / (4 eps)` inside
the band, `h` for `h >= eps`. Outside the bands the model is the affine tree
(the leaf models still describe it there); inside, the gradient turns
smoothly. Settings:

- `band_bounds=(0.05, 3.0)`: band half-widths (standardized input distance).
  The lower bound caps curvature. Much narrower bands are C1 in name only and
  can stop IPOPT from converging. Lower it only to fit real kinks more sharply.
- `hinge_ridge` (default 1e-2 for C1): ridge on hinge coefficients; prevents
  nearly parallel hyperplanes with large cancelling coefficients, which would
  inflate big-M constants and the PWL error bound.
- `refine_splits`: L-BFGS-B iteration cap for the smoothed fit.

C1 is usually more accurate than C0 on smooth responses (the hinge model has
only `depth` kinks to bend with) and slightly less accurate on responses with
real kinks. `m.predict_jacobian(X, hard=True)` gives the continuous gradient.
The mean is C1; `return_std` still comes from crisp leaf covariances and can
jump at a split.

Requirements: `leaf_type="affine"`; C0/C1 tied forms need `tied_splits=True`;
`smoothness=1` needs `continuous`. Violations raise `ValueError` at `fit`.

## 4. Pyomo export

```python
import pyomo.environ as pyo
from difftree import export

M = pyo.ConcreteModel()
M.x = pyo.Var(range(p))
M.y = pyo.Var(range(d_y))
b = export.to_pyomo(m, M, x=M.x, y=M.y, bounds=(lo, hi), formulation="auto")
M.obj = pyo.Objective(expr=M.y[0], sense=pyo.minimize)
pyo.SolverFactory("appsi_highs").solve(M)
```

If `x` / `y` are omitted, variables are created on the block (`b.x`, `b.y`).
Existing bounds on `x` and `y` are only tightened.

`formulation`:

- `"auto"` (default): `"tree"` for piecewise-affine models; raises for C1.
- `"tree"`: a GDP disjunction with one `Disjunct` per non-empty leaf
  (polytope + affine model), transformed with `method="bigm"` (default),
  `"hull"`, or left untransformed (`None`). Output bounds are computed exactly
  by LP. One binary per leaf.
- `"pwl"`: the hinge form with one big-M binary per hinge term that the box
  cuts. Requires every output to be continuous and `tied_splits=True`. Exact
  for C0 with about `depth` binaries (instead of `2**depth`). For C1, needs
  `K` (segments per band) and approximates each smoothed hinge by its K-segment
  interpolant: `D * (K + 1)` hinge terms per output. `b.error_bound` holds the
  guarantee.
- `"smooth"`: the hinge form as an NLP with `Expr_if`, no binaries. C1 for a
  C1 model; give it to IPOPT. Same output requirements as `"pwl"`.

**PWL error bound.** `export.pwl_error_bound(m, K)` returns `over`, `under`
and `bound = max(over, under)` per output, in original units, valid at every
x: `-under <= f_pwl(x) - f_C1(x) <= over`. It scales as `1/K^2` and is attained
(tight). Choose `K` from the tolerance you need before building the model;
K = 8 to 16 is typical for curved responses. For C0 models it is 0.

**Recommended C1 workflow.** Solve the `"pwl"` MILP globally, then start IPOPT
on `"smooth"` from the MILP solution:

```python
K = 8
print(export.pwl_error_bound(m, K)["bound"])
b = export.to_pyomo(m, M, bounds=(lo, hi), formulation="pwl", K=K)
M.obj = pyo.Objective(expr=b.y[0], sense=pyo.maximize)
pyo.SolverFactory("appsi_highs").solve(M)
x0 = [pyo.value(v) for v in b.x.values()]

N = pyo.ConcreteModel()
s = export.to_pyomo(m, N, bounds=(lo, hi), formulation="smooth")
for v, xv in zip(s.x.values(), x0):
    v.set_value(xv)
N.obj = pyo.Objective(expr=s.y[0], sense=pyo.maximize)
pyo.SolverFactory("ipopt").solve(N)
```

**After a solve.** `export.leaf_from_pyomo(b)` gives the active leaf (for
hinge formulations, the crisp routing of the solved x). For the `"tree"`
formulation of a discontinuous tree the optimum can sit on a leaf boundary and
use either neighbour's model: `export.repair_solution(m, x, leaf, delta=1e-6,
bounds=(lo, hi))` moves x just inside the chosen leaf so that
`m.predict(x[None], hard=True)` reproduces the MILP value.

## 5. Other exports

- `export.leaf_polytopes(m, bounds=(lo, hi), X=X)`: list of `LeafPolytope`
  (`G x <= h`, `coef`, `intercept`, `cov`, `n_samples`, `predict`, `contains`).
  Empty leaves are pruned by default (`prune_empty`); `prune_unreached=True`
  also drops leaves no row of `X` reaches.
- `export.to_spec` / `export.to_json(m, "tree.json", bounds=...)`: the
  solver-neutral jax-ldt v1 spec (`kind = "hyperplane_tree"`), consumed by
  `jax_ldt.from_json` and its discopt adapter. `export.predict_spec(spec, X)`
  evaluates it; `export.leaf_from_binaries(spec, z)` maps solver binaries back
  to a leaf.

The leaf-based exports (`leaf_polytopes`, `to_spec`, `to_json`, `to_pyomo`
`"tree"`) refuse C1 models, because their leaf models describe the model only
outside the bands.

## 6. Practical advice

- Put `bounds` on the region the optimizer searches; build the model with
  data covering it. Outside the data the surrogate is unreliable.
- Optimizers find where the surrogate overestimates. Check the optimum with
  the true model, and if they disagree, sample more points in a smaller box
  around the solution, refit, and re-solve (trust-region refinement; see the
  `pyomo_embedding` notebook).
- `hull` gives a tighter LP relaxation than `bigm` for the `"tree"` form;
  this matters once the surrogate is part of a large model.
- The fits are nonconvex. If a C0 or C1 fit looks poor, try another
  `random_state`, a larger depth, or more data before tuning other knobs.
