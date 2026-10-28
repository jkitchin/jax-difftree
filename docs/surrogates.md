# Surrogates for MILP embedding

A trained soft tree can be turned into a crisp piecewise-affine model and
embedded exactly in a mixed-integer linear program (MILP). This page covers
affine leaves, hardening, export, the treatment of split boundaries,
continuous (C0) hardening, and smooth (C1) hardening for gradient-based
solvers. The [piecewise-linear MILP notebook](examples/piecewise_linear_milp)
walks through the whole workflow with plots, and the
[C0 and C1 notebook](examples/smooth_hinge_c1) covers the smooth variant.

## Affine leaves and Jacobian matching

```python
from difftree import DifferentiableTreeRegressor

model = DifferentiableTreeRegressor(
    depth=3,
    leaf_type="affine",       # leaf l predicts A_l x + b_l
    constant_outputs=[1],     # output 1 stays constant per leaf
    jacobian_weight=1.0,
)
model.fit(X_tr, Y_tr, dydx=J_tr)   # J_tr: (n, d_y, p) dy/dx, optional
mean, std = model.predict(X_te, return_std=True)
```

All outputs share one partition, so one tree per unit operation means one set
of binaries in a MILP. `constant_outputs` keeps selected outputs (e.g.
intensive properties that enter blending) constant per leaf so that bilinear
blend terms linearize exactly.

With `leaf_type="affine"` the leaf mean becomes
$m_\ell(x) = \mu_\ell + A_\ell x$ with $A_\ell \in \mathbb{R}^{d_y \times p}$;
rows of $A_\ell$ listed in `constant_outputs` are fixed at zero. The noise
model $\Sigma_\ell$ is unchanged, and every formula in [](theory.md) holds with
$\mu_\ell$ replaced by $m_\ell(x)$.

When `fit` receives Jacobians $J_i = \partial y / \partial x \,|_{x_i}$, the
loss gains a Sobolev term (in standardized units)

$$
\lambda \, \frac{1}{n} \sum_i \Bigl\| \textstyle\sum_\ell \mu_\ell(x_i) A_\ell - J_i \Bigr\|_F^2,
$$

with $\lambda$ = `jacobian_weight` and only the affine outputs included. The
path-weighted leaf slope $\sum_\ell \mu_\ell A_\ell$ is the Jacobian of the
crisp tree inside a leaf, so the term teaches leaf slopes without penalizing
the steep transitions between leaves that exist only because routing is soft.

## Crisp mode, hardening, and the soft/hard gap

```python
y_hard = model.predict(X_te, hard=True)   # route by the sign of each split
leaves = model.apply(X_te)                # crisp leaf index per row

model.harden(X_tr, Y_tr, dydx=J_tr)       # or construct with hardening=True
model.hardening_report_["before"]         # soft vs hard on validation, pre-hardening
model.hardening_report_["after"]          # ... and after
model.soft_hard_gap(X_te, Y_te)           # same report on any data
```

Crisp routing sends $x$ right at node $i$ iff $w_i^{\top} x + b_i > 0$. This
is the $\gamma \to \infty$ limit of the soft routing, where the path
probability concentrates on one leaf, and it partitions the input space into
polytopes. `predict(hard=True)` returns that leaf's Gaussian, so the routing
variance is zero.

`harden` (or `hardening=True` in the constructor) converts the soft tree into a crisp
model in four steps:

1. Continue training while $\gamma$ grows geometrically from `gamma_max` to
   `gamma_hard` over `n_hard_epochs` epochs.
2. Freeze the partition by routing each training point crisply.
3. Refit each leaf by least squares on its points (regression on $[1, x]$ for
   affine outputs, the mean for constant ones, with Jacobian rows added when
   `dydx` is given, and ridge shrinkage `hard_ridge` toward the soft values),
   then refit each leaf noise covariance from its residuals.
4. Fit $\alpha$ in closed form on the validation slice under crisp routing.

`hardening_report_` records `soft_hard_gap` on the validation slice before and
after: per-output RMSE and $R^2$ for soft and crisp predictions, their NLLs,
the average largest leaf probability, and how often the most likely soft leaf
is the crisp leaf. A large gap before hardening means the soft fit relies on
blending leaves and is a misleading guide to the crisp model.

## Export

```python
from difftree import export

polys = export.leaf_polytopes(model, bounds=(lo, hi), X=X_tr)
for lp in polys:            # {x : lp.G x <= lp.h}, y = lp.coef x + lp.intercept
    print(lp.leaf, lp.G.shape, lp.coef, lp.intercept, lp.cov)

s = export.to_json(model, "tree.json", bounds=(lo, hi))  # jax-ldt v1 spec
y = export.predict_spec(s, X_te)       # == model.predict(X_te, hard=True)

# With jax-ldt installed:
#   from jax_ldt import from_json
#   from jax_ldt.export import embed_in_discopt_model
#   tree = from_json(s)
```

In original units, node $i$ goes left iff $a_i^{\top} x \le t_i$ with
$a_i = w_i / s_x$ and $t_i = a_i^{\top} m_x - b_i$ (the standardization is
undone). Leaf $\ell$ is the polytope $\{x : G_\ell x \le h_\ell\}$ formed by
its path half-spaces and an input box, and its model is
$y = C_\ell x + c_\ell$ with noise covariance $D \Sigma_\ell D$.
`export.leaf_polytopes` returns these, after one LP per leaf (Chebyshev ball)
removes leaves with an empty interior, and optionally leaves that no row of a
given `X` reaches.

`export.to_spec` / `to_json` emit the v1 spec of jax-ldt
(`jax_ldt.export.spec`) with `kind = "hyperplane_tree"`. The transform matrix
is $[I_p \mid a_1 \cdots a_k]$, with one lifted column per kept internal node,
and the leaf parameters use only the first $p$ (raw) columns. Pruned splits
are collapsed into their kept child. The jax-ldt loader (`from_json` /
`predict_tree`) and its discopt adapter (`embed_in_discopt_model`) therefore
consume difftree models unchanged. Extra information (bounds, BFS leaf ids,
leaf covariances) goes under a top-level `"difftree"` key.

### Pyomo

```python
import pyomo.environ as pyo
from difftree import export

m = pyo.ConcreteModel()
m.x = pyo.Var(range(2))
m.y = pyo.Var(range(2))
tree = export.to_pyomo(model, m, x=m.x, y=m.y, bounds=(lo, hi), method="bigm")
m.obj = pyo.Objective(expr=m.y[0])
pyo.SolverFactory("appsi_highs").solve(m)

leaf = export.leaf_from_pyomo(tree)                 # BFS leaf index
x_ok = export.repair_solution(model, [pyo.value(v) for v in m.x.values()], leaf)
```

`export.to_pyomo` (needs `pip install difftree[pyomo]`) writes the crisp tree
as a generalized disjunctive program: leaf $\ell$ is a `Disjunct` holding its
path half-spaces $G_\ell x \le h_\ell$ and its model $y = C_\ell x + c_\ell$,
and a `Disjunction` makes exactly one leaf active. The input box becomes bounds
on `x`, and each output gets the exact range of its leaf models over their
polytopes (two LPs per leaf and output), so the big-M (`method="bigm"`) or
convex-hull (`method="hull"`) reformulation needs no hand-picked constants.
Existing bounds on `x` and `y` are only tightened. With `method=None` the
disjunction is left for you to transform with the rest of the model or to pass
to a GDP solver. The formulation uses one binary per kept leaf. The
[Pyomo notebook](examples/pyomo_embedding) optimizes a reactor with a tree
surrogate and refines it with a trust region. OMLT's
linear-tree formulations assume axis-aligned splits, so they do not apply to
difftree's oblique splits.

For a tied, continuous tree (see [continuous hardening](#continuous-hardening)),
`formulation="pwl"` writes the hinge form instead: one big-M binary per
hyperplane that crosses the box, so about `depth` binaries rather than one
per leaf, and the same optimum. The `"pwl"` and `"smooth"` formulations are
also how a C1 model is embedded ([below](#smooth-c1-hardening)).

## Split boundaries

```python
export.boundary_jumps(model, bounds=(lo, hi))   # largest jump per shared facet

# After a MILP solve (e.g. discopt via jax-ldt): result.x["x"], result.x["tree_z"]
leaf = export.leaf_from_binaries(s, z)           # leaf the solver chose
x_ok = export.repair_solution(model, x_milp, leaf, delta=1e-6)
model.predict(x_ok[None], hard=True)             # == the MILP objective (to O(delta))

# Jump-free hardened model: tie splits per depth, then harden continuously
model = DifferentiableTreeRegressor(depth=3, leaf_type="affine", tied_splits=True,
                                    continuous=True, hardening=True).fit(X_tr, Y_tr)
```

### Repairing MILP solutions

A crisp piecewise-affine tree can jump across a split boundary. A MILP treats
the leaf polytopes as closed sets, so its optimum can sit on a boundary and
use either neighbouring leaf's model there, while `predict(hard=True)` sends
boundary points left. The MILP objective is then the true infimum of the
crisp model, but the returned $x$ is on the wrong side of a measure-zero
boundary. `export.repair_solution` fixes this with one LP: it returns the
closest point (max-norm, standardized units) that lies at least $\delta$
inside every face of the leaf the solver chose (`export.leaf_from_binaries`
reads that leaf from the binaries), so crisp prediction there matches the
MILP objective to $O(\delta)$. Unlike a strict-inequality margin in the MILP
(jax-ldt's `boundary_eps`), it leaves the formulation unchanged and so cannot
create spurious infeasibility. If a surrogate output appears in a MILP
constraint, the repaired point can move it by $\delta$ times its slope.

### Measuring jumps

A large jump is usually fitting error, and an optimizer will seek it out.
`export.boundary_jumps` finds every pair of leaves sharing a
$(p-1)$-dimensional facet in the box (one LP per candidate pair) and
maximizes the difference of their models over the facet (two LPs per
output); `hardening_report_["max_boundary_jump"]` records the per-output
maximum.

### Continuous hardening

`harden(..., continuous=True)` removes the jumps. Across node $i$'s
hyperplane the two leaf models may differ only by a multiple of
$w_i^{\top} x + b_i$, which is $p$ linear equality constraints per shared
facet. How expensive that is depends on the tree structure.

**Untied splits (default).** The leaves are refit by one joint constrained
least-squares problem (or with a penalty, `continuity_weight`). This is
costly on an oblique tree for two reasons. First, where a deeper split meets
a parent boundary (a T-junction), both leaves of that split must agree with
the same neighbour, which forces them to be equal unless the two hyperplanes
coincide. Around any point where creases meet, the slope changes across them
must cancel, so a crease cannot simply end at another crease. Second,
continuity pins each kink to the learned split, so any error in where the
split sits shows up directly in the predictions.

**Tied splits (`tied_splits=True`).** All nodes at depth $d$ share one
hyperplane $h_d(x) = w_d^{\top} x + b_d$ (an oblivious oblique tree), so the
partition is an arrangement of $D$ hyperplanes and every junction is a
crossing. The functions that are continuous on such a partition are exactly
the hinging-hyperplanes models {cite:p}`breiman1993hinging`

$$
f(x) \;=\; \theta_0^{\top} [1, x] \;+\; \sum_{d=1}^{D} c_d \max\bigl(0,\, h_d(x)\bigr),
$$

a one-hidden-layer ReLU network with $D$ units. Leaf $\ell$'s model is
$\theta_0 + \sum_{d:\, \ell \text{ right of } h_d} c_d \, [b_d; w_d]$.
Hardening fits this form directly by least squares (with Jacobian rows when
`dydx` is given) and moves the hyperplanes to reduce the error (Powell, at
most `refine_splits` iterations, about a second). The result is continuous
everywhere, not only inside the box, and still exports as an ordinary tree.

Crisp RMSE on held-out data (max jump at round-off for the continuous
variants; depth 3 unless noted):

| Function | discontinuous | continuous, untied | continuous, tied |
|---|---|---|---|
| $\lvert x_0 \rvert + x_1 + \max(x_2, 0)$ | 0.049 | 0.583 | 0.004 |
| same, depth 4 | 0.005 | 0.291 | 0.005 |
| Gaussian bump (2 inputs) | 0.103 | 0.227 | 0.097 |
| same, depth 4 | 0.075 | 0.212 | 0.087 |
| $\sin 2x$ (1 input) | 0.216 | 0.275 | 0.297 |
| same, depth 4 | 0.202 | 0.280 | 0.215 |

Tied splits are the recommended way to get a jump-free surrogate. They are
much better than untied continuity and often as good as the discontinuous
tree, but with only $D$ hyperplanes they can be less flexible on smooth
functions without kink structure. The
[piecewise-linear MILP notebook](examples/piecewise_linear_milp) plots all
three variants across the worst split of the discontinuous tree.

### Smooth (C1) hardening

A continuous tree still has a jump in its *gradient* at every split. That is
the best a MILP can represent exactly: if two affine leaves also have equal
gradients across a facet, they are the same affine map, so a piecewise-affine
function with a continuous gradient is globally affine. Gradient-based solvers
(IPOPT, L-BFGS, Newton-type inverse design) prefer a continuous gradient, and
getting one needs pieces of degree at least 2. Two quadratic leaves meeting
on $h(x) = 0$ have a continuous gradient only if they differ by
$\alpha\, h(x)^2$.

`smoothness=1` (with `tied_splits=True` and `continuous`) builds such a model
from the hinge form by replacing each $\max(0, h)$ with

$$
\phi_\varepsilon(h) = \begin{cases} 0 & h \le -\varepsilon \\
(h + \varepsilon)^2 / (4\varepsilon) & |h| < \varepsilon \\
h & h \ge \varepsilon \end{cases}
$$

so the model equals the affine tree outside a band $|h_d| < \varepsilon_d$
around each hyperplane and is quadratic inside it. The leaf models still
describe the model outside the bands. Hardening first does the C0 fit, then
fits the hyperplanes, the band half-widths (one per depth and output, within
`band_bounds`) and the coefficients jointly (L-BFGS-B, at most
`refine_splits` iterations, starting from wide and from narrow bands).
`hinge_ridge` (default $10^{-2}$ for C1) keeps the fit from using nearly
parallel hyperplanes with large coefficients of opposite sign, which barely
change the function but inflate the error bound below. The lower end of
`band_bounds` (default 0.05, standardized input units) caps the curvature at
$|c_d| / (2\varepsilon_d)$. Narrower bands are C1 only in name, and IPOPT
fails to converge on them from distant starting points.

```python
model = DifferentiableTreeRegressor(depth=4, leaf_type="affine", tied_splits=True,
                                    continuous=True, smoothness=1,
                                    hardening=True).fit(X, y)
model.predict(X_new, hard=True)            # the C1 model
model.predict_jacobian(X_new, hard=True)   # its continuous gradient
model.hardening_report_["band_halfwidth"]  # (depth, n_continuous_outputs)
```

`predict(hard=True)` returns the C1 mean. Its `return_std` still comes from
the crisp leaf covariances, so the variance can jump at a split even though
the mean is C1.

**Optimizing a C1 model.** It has no exact MILP form, so the leaf-based
exports (`leaf_polytopes`, `to_spec`, `to_json` and `to_pyomo`'s default
formulation) raise an error rather than export the leaf models. Two Pyomo
formulations embed it instead:

- `to_pyomo(model, ..., formulation="pwl", K=K)` replaces each smoothed
  hinge by its piecewise-linear interpolant through $K + 1$ points across
  the band. The interpolant is itself a sum of $K + 1$ hinges, so the result
  is a MILP with one binary per hinge term that the box cuts. Since
  $\phi_\varepsilon$ is convex, the interpolant lies above it, and the error
  of output $k$ is in $[-\text{under}_k, \text{over}_k]$ everywhere, with
  $\text{over} = \sum_d \max(c_d, 0)\, \varepsilon_d / (4K^2)$ and
  $\text{under} = \sum_d \max(-c_d, 0)\, \varepsilon_d / (4K^2)$.
  `export.pwl_error_bound(model, K)` returns the bound before you build
  anything. It falls as $1/K^2$ and is attained at segment midpoints.
- `to_pyomo(model, ..., formulation="smooth")` writes the C1 model itself
  with `Expr_if` and no binaries, for IPOPT and similar NLP solvers.

A natural workflow is to solve the `"pwl"` MILP globally, then start IPOPT on
the `"smooth"` formulation from its solution. The
[C0 and C1 notebook](examples/smooth_hinge_c1) walks through both.

