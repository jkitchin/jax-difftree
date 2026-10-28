---
name: difftree
description: Fit and use differentiable oblique decision trees in JAX (the difftree package, DifferentiableTreeRegressor and DifferentiableTreeClassifier) with built-in uncertainty quantification, and turn them into surrogates for optimization. Use when the user mentions difftree or jax-difftree, wants a tree regressor with predictive std, aleatoric/routing variance, quantiles, multivariate conditioning p(y_unobs | x, y_obs) or inverse design, or wants a piecewise-affine, continuous (C0) or smooth (C1) tree surrogate embedded in a MILP or NLP with Pyomo, HiGHS, IPOPT or jax-ldt (hardening, leaf polytopes, boundary jumps, big-M export).
license: MIT
compatibility: Requires Python 3.11+ with jax, optax, scikit-learn, scipy and numpy. Pyomo export needs pyomo plus a solver (highspy for MILP, ipopt for NLP).
metadata:
  version: "0.1"
  repository: https://github.com/jkitchin/jax-difftree
---

# difftree

`difftree` trains soft, oblique-split binary trees of fixed depth end to end
with gradient descent in JAX, behind scikit-learn estimators. Each leaf holds a
Gaussian (a multivariate Gaussian for several outputs), so a prediction is a
mixture of Gaussians weighted by soft path probabilities. A trained tree can be
hardened into a crisp piecewise-affine model (optionally continuous, or with a
continuous gradient) and embedded in a MILP or NLP.

Docs: https://jkitchin.github.io/jax-difftree/

## Install

```bash
pip install git+https://github.com/jkitchin/jax-difftree.git
pip install "difftree[pyomo] @ git+https://github.com/jkitchin/jax-difftree.git" highspy   # Pyomo export
```

The import name is `difftree`. JAX installs its CPU build by default.

## Pick the workflow first

| The user wants | Use | Read |
|---|---|---|
| A regressor with error bars, quantiles, variance decomposition | `DifferentiableTreeRegressor` (defaults) | [uq.md](references/uq.md) |
| Several correlated outputs, conditioning on some, inverse design | same, 2D `y` | [uq.md](references/uq.md) |
| A classifier with aleatoric/epistemic entropy split | `DifferentiableTreeClassifier` | [uq.md](references/uq.md) |
| A surrogate inside a MILP (exact) | affine leaves + `hardening=True` (+ `tied_splits`, `continuous`) | [surrogates.md](references/surrogates.md) |
| A surrogate with a continuous gradient (NLP, IPOPT) | add `smoothness=1` | [surrogates.md](references/surrogates.md) |
| Constructor arguments and defaults | | [parameters.md](references/parameters.md) |

## Uncertainty quantification in five lines

```python
from difftree import DifferentiableTreeRegressor

model = DifferentiableTreeRegressor(depth=5, n_epochs=150, random_state=0).fit(X_tr, y_tr)
mean, std = model.predict(X_te, return_std=True)
mean, std, std_aleatoric, std_routing = model.predict(X_te, return_components=True)
q05, q50, q95 = model.predict_quantile(X_te, q=[0.05, 0.5, 0.95])   # univariate y only
```

`fit` holds out `validation_fraction` (default 0.1) of the data to calibrate
split steepness and a variance scale after training, so small datasets lose a
few points to calibration.

## Surrogate for optimization in one block

```python
from difftree import DifferentiableTreeRegressor, export

tree = DifferentiableTreeRegressor(
    depth=4, leaf_type="affine",         # leaf l predicts A_l x + b_l
    tied_splits=True, continuous=True,   # jump-free (C0) hinge model
    hardening=True, random_state=0,      # crisp piecewise-affine model at the end of fit
).fit(X, y)                              # fit(X, y, dydx=J) adds Jacobian matching
tree.predict(X_new, hard=True)           # the crisp model the MILP will see
tree.hardening_report_["after"]          # soft vs crisp accuracy on the validation slice

b = export.to_pyomo(tree, m, x=m.x, y=m.y, bounds=(lo, hi))   # exact MILP
```

The smoothness ladder, and what each step embeds as:

- **Discontinuous** (`leaf_type="affine", hardening=True`): exact MILP, but the
  solver can exploit jumps between leaves. Check with
  `hardening_report_["max_boundary_jump"]`.
- **C0** (`+ tied_splits=True, continuous=True`): continuous values, gradient
  jumps at every split. Exact MILP. The recommended default for MILP surrogates.
- **C1** (`+ smoothness=1`): continuous gradient, quadratic in a band around each
  split. No exact MILP; use `export.to_pyomo(..., formulation="pwl", K=K)`
  (MILP within `export.pwl_error_bound(tree, K)`) or `formulation="smooth"`
  (NLP for IPOPT).

## Rules that prevent the common mistakes

1. **Always use `hard=True` predictions to judge a surrogate.** `predict()` is
   the soft mixture; the MILP sees `predict(hard=True)`. Look at
   `hardening_report_` or `soft_hard_gap(X, y)` before trusting an export.
2. **Hardening is a constructor argument** (`hardening=True`), not a `fit`
   keyword. `model.harden(X, y)` hardens an already fitted model, and accepts
   `continuous=` and `smoothness=` overrides.
3. **`continuous`, `smoothness` and Pyomo export need `leaf_type="affine"`.**
   Constant leaves can only be continuous if globally constant.
4. **Use `tied_splits=True` with `continuous`.** Untied continuity on oblique
   trees in more than one dimension forces leaves to merge at T-junctions and
   costs a lot of accuracy.
5. **Set export `bounds` to the region the optimizer will search.** It defaults
   to the training-data box (`model.x_min_`, `model.x_max_`). Outside the data
   every input routes to an edge leaf, and neither the prediction nor its
   `std` can be trusted there (a single tree has no epistemic uncertainty).
6. **Evaluate the optimum with the true model.** An optimizer is drawn to where
   the surrogate overestimates; refine locally (trust region) if they disagree.
7. **Depth costs `2**depth - 1` split parameters.** Keep `depth <= 10` or so.
   Tied trees have only `depth` hyperplanes, so they are cheaper but less
   flexible; try depth 4 to 6.
8. **Results depend on `random_state`.** Training is nonconvex, and at small
   depth the error can vary several-fold across seeds. Fix the seed for
   reproducibility, and for important fits try a few seeds and keep the best
   on held-out data.
9. **Units.** All public inputs and outputs (predictions, Jacobians, bounds,
   export coefficients) are in original units. Internals (`params_`,
   `hinge_`, `band_halfwidth`) are in standardized units.

## Where things live

- `difftree.DifferentiableTreeRegressor`: `fit`, `predict`, `predict_full_cov`,
  `predict_quantile`, `predict_jacobian`, `sample_y`, `predict_conditional`,
  `sample_y_conditional`, `inverse_design`, `apply`, `harden`, `soft_hard_gap`,
  `score`.
- `difftree.DifferentiableTreeClassifier`: `fit`, `predict`, `predict_proba`,
  `sample_y`, `apply`.
- `difftree.export`: `leaf_polytopes`, `to_spec`, `to_json`, `predict_spec`,
  `to_pyomo`, `leaf_from_pyomo`, `pwl_error_bound`, `boundary_jumps`,
  `leaf_from_binaries`, `repair_solution`.

Worked notebooks (in the repository's `notebooks/`, rendered in the docs):
`heteroscedastic_demo`, `multivariate_inverse_design`, `piecewise_linear_milp`,
`pyomo_embedding`, `smooth_hinge_c1`, `scaling`, `failure_modes`.
