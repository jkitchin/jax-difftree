# Limitations and caveats

The fixed-depth full-binary structure makes JAX-friendly vectorization easy
but caps practical depth around $D \le 10$ to $12$ because of the
$2^D - 1$ split-parameter blowup. For larger problems, a grown-tree variant
in the style of {cite:t}`hehn2020end` could be added as a separate fitter that
reuses the same tree representation. The [scaling notebook](examples/scaling)
measures size, fit, and prediction cost against depth.

Splits are oblique linear by default. Non-linear splits (small CNNs or
feature-attention gates) fit the same training framework but are not yet
exposed as a public API.

Inverse design is local; multi-start ascent is the practical fallback for
multimodality. The [failure-modes notebook](examples/failure_modes) shows
this along with seed sensitivity and the behavior of the uncertainty under
extrapolation.

The exported leaf polytopes are closed, as is standard in big-M MILP
formulations, so a MILP optimum can sit on a split boundary and use either
neighbouring leaf's model there. `export.repair_solution` moves such a point
into the chosen leaf (see [](surrogates.md#repairing-milp-solutions)).

Continuity without tied splits costs accuracy on oblique trees in more than
one dimension, because T-junctions over-constrain the leaves and the kinks are
pinned to the learned splits. Use `tied_splits=True` with `continuous=True`
instead. Tied splits give up some flexibility, since a depth-$D$ tree then has
$D$ hyperplanes rather than $2^D - 1$.

A pruned spec matches `predict(hard=True)` only inside the export box.
Pruning unreached leaves (`prune_unreached=True`) also reassigns their regions
to neighbouring leaves.

A C1 model (`smoothness=1`) is quadratic near its splits, so it has no exact
MILP form. `to_pyomo(..., formulation="pwl", K=K)` gives a MILP that is
within `export.pwl_error_bound(model, K)` of it, and the leaf-based exports
refuse it. Its mean has a continuous gradient, but `predict(...,
return_std=True)` still uses the crisp leaf covariances, so the predicted
variance can jump at a split. The C1 fit is nonconvex and starts from the C0
solution, so its accuracy depends on that start and on the seed. The default
lower bound on the band half-width (`band_bounds`) rounds real kinks over a
band of about 0.05 standardized units; lowering it sharpens them but can stop
a smooth NLP solver from converging. The hinge-form exports (`"pwl"`,
`"smooth"`) need every output to be continuous with tied splits.
