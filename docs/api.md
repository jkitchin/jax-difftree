# API reference

## Estimators

```{eval-rst}
.. autoclass:: difftree.DifferentiableTreeRegressor
   :members: fit, predict, predict_full_cov, predict_quantile, sample_y,
             predict_conditional, sample_y_conditional, inverse_design,
             predict_jacobian, apply, harden, soft_hard_gap, score

.. autoclass:: difftree.DifferentiableTreeClassifier
   :members: fit, predict, predict_proba, sample_y, apply
```

## Export

```{eval-rst}
.. automodule:: difftree.export
   :members: leaf_polytopes, LeafPolytope, to_spec, to_json, predict_spec,
             boundary_jumps, leaf_from_binaries, repair_solution, to_pyomo,
             leaf_from_pyomo, pwl_error_bound
```
