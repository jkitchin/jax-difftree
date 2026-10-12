# difftree

**Differentiable decision trees in JAX with built-in uncertainty quantification.**

`difftree` is a JAX implementation of soft, oblique-split, gradient-trainable
decision trees with scikit-learn compatible estimators. The package is built
around uncertainty quantification: regression leaves are heteroscedastic
Gaussians (or multivariate Gaussians for multi-output targets), and the
predictive distribution at each input is a mixture of Gaussians weighted by
soft path probabilities. This structure delivers a calibrated predictive
variance, a clean decomposition of uncertainty into within-leaf (aleatoric)
and between-leaf (routing) components, and, for multivariate outputs,
closed-form cross-conditioning and gradient-based inverse design as direct
consequences.

For surrogate optimization, leaves can be affine ($A_\ell x + b_\ell$, with
optional Jacobian matching), the soft tree can be hardened into a crisp
piecewise-affine model, and that model exports as leaf polytopes or as the
solver-neutral JSON spec of jax-ldt for exact MILP embedding.

The implementation is closest in spirit to the end-to-end deterministic-tree
formulation of {cite:t}`hehn2020end`, with two practical departures motivated
by UQ. First, splits are kept finitely soft at test time rather than annealed
to hard, so soft-routing uncertainty contributes to the predictive variance.
Second, a small post-hoc calibration step, in the same family as temperature
scaling for classifier logits {cite:p}`guo2017calibration`, jointly fits
per-depth steepness and a scalar variance correction on a held-out validation
slice by minimizing NLL. The broader design space is reviewed by
{cite:t}`costa2023recent`.

```python
from difftree import DifferentiableTreeRegressor

model = DifferentiableTreeRegressor(depth=5, random_state=0).fit(X_train, y_train)
mean, std = model.predict(X_test, return_std=True)
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
surrogates
```

```{toctree}
:maxdepth: 2
:caption: Background

background
theory
limitations
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/heteroscedastic_demo
examples/multivariate_inverse_design
examples/piecewise_linear_milp
examples/scaling
examples/failure_modes
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
references
```
