# Constructor parameters

Read this when choosing or explaining estimator settings. Both estimators are
scikit-learn compatible (`get_params`, `set_params`, pipelines, `clone`,
cross-validation).

## DifferentiableTreeRegressor

**Tree and training**

| Parameter | Default | Meaning |
|---|---|---|
| `depth` | 5 | Fixed depth `D`: `2**D - 1` oblique splits, `2**D` leaves. Practical limit about 10 to 12. |
| `learning_rate` | 0.01 | Adam learning rate. |
| `batch_size` | 256 | Minibatch size. |
| `n_epochs` | 200 | Training epochs. |
| `gamma_init`, `gamma_max`, `delta_gamma` | 1.0, 5.0, 0.1 | Split steepness is annealed from `gamma_init` to `gamma_max` during training. |
| `sigma_init` | 1.0 | Initial leaf std (standardized y units). |
| `warmup_fraction` | 0.25 | Fraction of epochs that train only the means before the noise model. |
| `validation_fraction` | 0.1 | Held out for post-hoc calibration of per-depth gamma and variance scale. |
| `n_calibration_steps`, `calibration_lr` | 300, 0.02 | Calibration optimizer. |
| `standardize` | True | Standardize x and y internally. Keep it on. |
| `random_state` | None | Seed. Training is nonconvex, so fix it for reproducibility. |
| `verbose` | False | Print progress. |

**Leaves**

| Parameter | Default | Meaning |
|---|---|---|
| `leaf_type` | `"constant"` | `"affine"`: leaf mean `A_l x + b_l`. Required for continuity, smoothness and useful MILP surrogates. |
| `constant_outputs` | None | With affine leaves, output indices kept constant per leaf. |
| `jacobian_weight` | 1.0 | Weight of the Jacobian-matching term, used only when `fit(..., dydx=J)`. |

**Hardening (crisp surrogate)**

| Parameter | Default | Meaning |
|---|---|---|
| `hardening` | False | End `fit` with `harden()` on the same train/validation split. |
| `gamma_hard` | 50.0 | Steepness reached while hardening. |
| `n_hard_epochs` | 50 | Epochs of annealing to `gamma_hard` before the partition is frozen. |
| `hard_ridge` | 1e-6 | Ridge of the least-squares leaf refit (toward the soft values). |
| `continuous` | False | True (all affine outputs) or a list of output indices that must be continuous across splits. |
| `continuity_weight` | None | Untied trees only: a penalty instead of exact continuity. |
| `tied_splits` | False | One hyperplane per depth (oblivious oblique tree). With `continuous`, the hardened model is a hinging-hyperplanes model. Recommended for continuity. |
| `refine_splits` | 50 | Tied + continuous: iterations used to move the hyperplanes during hardening (Powell for C0, L-BFGS-B for the C1 fit). 0 keeps them. |
| `smoothness` | 0 | 0: C0. 1: C1 (continuous gradient); needs `tied_splits=True` and `continuous`. |
| `band_bounds` | (0.05, 3.0) | C1 only: bounds on band half-widths, standardized input distance. |
| `hinge_ridge` | None | Ridge on hinge coefficients of the tied continuous fit. None: 1e-2 for C1, `hard_ridge` for C0. |

**Fitted attributes worth knowing**

`n_features_in_`, `n_outputs_`, `x_mean_`, `x_std_`, `y_mean_`, `y_std_`,
`x_min_`, `x_max_` (training box, the default export bounds), `params_`
(standardized parameters), `hardened_`, `hardening_report_`, `smoothness_`,
and `hinge_` (tied continuous models: `outputs`, `W`, `b`, `eps`, `coef` in
standardized units; `eps` is None for C0).

## DifferentiableTreeClassifier

`depth=5`, `learning_rate=0.01`, `batch_size=256`, `n_epochs=200`,
`gamma_init=1.0`, `gamma_max=5.0`, `delta_gamma=0.1`,
`validation_fraction=0.1`, `n_calibration_steps=300`, `calibration_lr=0.05`,
`standardize=True`, `random_state=None`, `verbose=False`. Leaves hold class
logits. Methods: `fit`, `predict`, `predict_proba(X, return_components=False,
hard=False)`, `sample_y`, `apply`.

## Quick recipes

- **Error bars on a 1D or tabular regression:** defaults, `depth` 4 to 6,
  `n_epochs` 150 to 300, a fixed `random_state`.
- **MILP surrogate:** `leaf_type="affine", tied_splits=True, continuous=True,
  hardening=True`, `depth` 3 to 6; pass `dydx` if derivatives are available.
- **NLP or gradient-based optimization:** the same plus `smoothness=1`.
- **Discontinuous response (real jumps):** `leaf_type="affine",
  hardening=True` without `continuous`, and check `boundary_jumps`.
