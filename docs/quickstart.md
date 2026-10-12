# Quickstart

Both estimators follow the scikit-learn API, so they work with pipelines,
cross-validation, and model selection utilities. The examples below use
`X_tr, y_tr` for training data and `X_te, y_te` for test data.

## Univariate regression

```python
from difftree import DifferentiableTreeRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=100, random_state=0)

model = DifferentiableTreeRegressor(depth=5, n_epochs=150, random_state=0)
model.fit(X_tr, y_tr)

mean = model.predict(X_te)
mean, std = model.predict(X_te, return_std=True)
mean, std, std_aleatoric, std_routing = model.predict(X_te, return_components=True)
samples = model.sample_y(X_te, n_samples=200)
q05, q50, q95 = model.predict_quantile(X_te, q=[0.05, 0.5, 0.95])
```

`return_components=True` splits the predictive standard deviation into its
within-leaf (aleatoric) and between-leaf (routing) parts, see
[](theory.md#variance-decomposition).

## Multivariate regression with cross-conditioning

Pass a two-dimensional target `Y_tr` of shape `(n, d_y)`. All outputs share
one tree, and each leaf holds a full multivariate Gaussian.

```python
model = DifferentiableTreeRegressor(depth=5, n_epochs=150).fit(X_tr, Y_tr)

# Joint mean and full predictive covariance
mean, cov = model.predict_full_cov(X_te)

# Conditional p(y_unobs | x, y_obs) in closed form
cond_mean, cond_cov, log_evidence = model.predict_conditional(
    X_te, observed={0: Y_te[:, 0]}
)
cond_samples = model.sample_y_conditional(
    X_te, observed={0: Y_te[:, 0]}, n_samples=500
)
```

## Inverse design

Optimize the input vector to maximize the log-likelihood of a target
multivariate output, with optional partial constraints and feature-box bounds:

```python
import numpy as np

x_best, log_lik, x_all, log_liks = model.inverse_design(
    targets={0: 1.8, 1: 1.0},  # or partial: {0: 1.8}
    n_starts=24, n_steps=600, learning_rate=0.05,
    bounds=(np.full(p, -1.0), np.full(p, 1.0)),
)
```

Inverse design is a local, multi-start gradient ascent; see
[](limitations.md).

## Classification

```python
from difftree import DifferentiableTreeClassifier

m = DifferentiableTreeClassifier(depth=5).fit(X_tr, y_tr)
probs = m.predict_proba(X_te)
probs, H_aleatoric, H_epistemic = m.predict_proba(X_te, return_components=True)
```

The entropy decomposition follows the standard mutual-information identity

$$
H[\mathbb{E}_\ell \pi_\ell] = \mathbb{E}_\ell H[\pi_\ell] + I[Y; \ell \mid x]
$$

used for separating aleatoric from epistemic uncertainty in Bayesian deep
learning {cite:p}`depeweg2018decomposition`.

## Next steps

To use a tree as a piecewise-affine surrogate inside a mixed-integer program,
continue with [](surrogates.md). The [examples](examples/heteroscedastic_demo)
are complete notebooks with plots.
