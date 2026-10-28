# Uncertainty quantification with difftree

Read this when the user wants error bars, calibrated variance, quantiles,
multi-output conditioning, inverse design or classification uncertainty.

## The model

A tree of depth `D` has `2**D - 1` oblique splits and `2**D` leaves. Node `i`
routes right with probability `sigmoid(gamma_d * (w_i . x + b_i))`, so each
input reaches every leaf with a path probability `mu_l(x)`. Leaf `l` holds a
Gaussian `N(m_l(x), Sigma_l)`, with `m_l` constant (default) or affine
(`leaf_type="affine"`). The predictive distribution is the mixture

    p(y | x) = sum_l mu_l(x) N(y; m_l(x), Sigma_l)

and its covariance splits into

    Cov[y | x] = sum_l mu_l Sigma_l          (aleatoric, within-leaf)
               + Cov_l[m_l(x)]               (routing, between-leaf)

Splits stay finitely soft at test time, so routing uncertainty is real: it is
large near split boundaries and where leaves disagree.

Training anneals `gamma` from `gamma_init` to `gamma_max`. After training,
`fit` calibrates the per-depth `gamma` and a scalar variance scale `alpha` on
a held-out `validation_fraction` of the data by minimizing NLL (temperature
scaling for trees). Inputs and outputs are standardized internally
(`standardize=True`).

## Univariate regression

```python
from difftree import DifferentiableTreeRegressor

m = DifferentiableTreeRegressor(depth=5, n_epochs=150, random_state=0).fit(X_tr, y_tr)
mean = m.predict(X_te)
mean, std = m.predict(X_te, return_std=True)
mean, std, std_ale, std_rou = m.predict(X_te, return_components=True)
samples = m.sample_y(X_te, n_samples=200)          # (n_samples, n)
q = m.predict_quantile(X_te, q=[0.05, 0.5, 0.95])   # Monte Carlo, univariate only
m.score(X_te, y_te)                                 # R^2, as in scikit-learn
```

Checking calibration: compare the empirical coverage of `mean +/- 1.96 std`
with 95%, or the standardized residuals `(y - mean) / std` with N(0, 1). The
`heteroscedastic_demo` notebook shows both.

## Multivariate outputs

Pass `y` of shape `(n, d_y)`. All outputs share one tree; each leaf has a full
covariance.

```python
m = DifferentiableTreeRegressor(depth=5, n_epochs=150).fit(X_tr, Y_tr)
mean, cov = m.predict_full_cov(X_te)                  # (n, d_y), (n, d_y, d_y)
cond_mean, cond_cov, log_evidence = m.predict_conditional(X_te, observed={0: Y_te[:, 0]})
draws = m.sample_y_conditional(X_te, observed={0: Y_te[:, 0]}, n_samples=500)
```

`observed` maps output index to a scalar (broadcast over rows) or an `(n,)`
array. Conditioning is exact per leaf (Schur complement) and reweights the
leaves by how well each explains the observed outputs.

## Inverse design

Find inputs that make a target output most likely under the trained model.

```python
x_best, log_lik, x_all, log_liks = m.inverse_design(
    targets={0: 1.8, 1: 1.0},          # original units; a subset of outputs is fine
    n_starts=24, n_steps=600, learning_rate=0.05,
    bounds=(lo, hi),                    # original units, optional but recommended
)
```

It is multi-start gradient ascent (Adam), so it is local. Use many starts,
always pass `bounds`, and inspect `x_all` / `log_liks` for several distinct
solutions. Without bounds it can wander outside the data, where the model
extrapolates.

## Gradients

`m.predict_jacobian(X)` returns `d mean / d x` of the soft mixture,
`(n, d_y, p)` or `(n, p)` for 1D `y`, in original units.

## Classification

```python
from difftree import DifferentiableTreeClassifier

c = DifferentiableTreeClassifier(depth=5, random_state=0).fit(X_tr, y_tr)
probs = c.predict_proba(X_te)
probs, H_aleatoric, H_epistemic = c.predict_proba(X_te, return_components=True)
labels = c.predict(X_te)
```

The total predictive entropy equals the expected within-leaf entropy
(aleatoric) plus the mutual information between label and leaf (epistemic).

## Pitfalls

- **No epistemic uncertainty away from the data.** A single tree has no
  parameter uncertainty: outside the training range every input routes to an
  edge leaf, so `std` stays the same while the error grows (in the
  `failure_modes` notebook, sigma stayed at 0.20 while the error went from
  0.12 to 0.79). Check that queries are inside the training box
  (`m.x_min_`, `m.x_max_`) before trusting an interval. An ensemble over seeds
  widens a little outside the data, but far less than the error.
- **Seed sensitivity.** Training is nonconvex. At small depth the RMSE can vary
  several-fold across `random_state`; deeper trees are more stable. Fit a few
  seeds and keep the best on held-out data.
- **Inverse design is multimodal.** Use `n_starts` of 16 or more and look at
  `x_all` for alternative designs.
- `predict_quantile` is univariate only; use `sample_y` for several outputs.
- Very small datasets: the calibration slice (`validation_fraction`) is taken
  from the training data. Reduce it, or accept a noisier calibration.
- Depth is capped in practice around 10 to 12 by the `2**D - 1` split
  parameters. The `scaling` notebook measures cost against depth.
