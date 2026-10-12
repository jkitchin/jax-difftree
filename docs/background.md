# Why differentiable trees

Traditional decision trees {cite:p}`breiman1984cart,quinlan1986induction`
grow greedily and use hard, axis-aligned splits, which prevents joint
optimization of the whole tree and limits the splits to one feature at a
time. Several related lines of work make trees differentiable in some form.

Deep Neural Decision Forests {cite:p}`kontschieder2015deep` wrap a
probabilistic routing tree around a deep feature extractor, but inference
still evaluates every leaf for every sample, defeating the
conditional-data-flow advantage of trees. Soft decision trees distilled from
neural networks {cite:p}`frosst2017distilling` and adaptive neural trees
{cite:p}`tanno2019adaptive` retain soft routing throughout, and do not target
calibrated uncertainty directly. {cite:t}`hehn2020end` propose end-to-end
training of trees by annealing the steepness $\gamma$ during training so
that the final inference model is hard (single root-to-leaf path per sample)
while training remains differentiable. This recovers the tree's
$O(\log L)$ inference cost.

For UQ, the issue with annealing $\gamma \to \infty$ is that the
soft-routing component of predictive uncertainty is destroyed. `difftree`
keeps $\gamma$ finite at test time and exposes the routing component
explicitly, while still supporting hard inference (`predict(X, hard=True)`)
and an explicit hardening step (`harden`) for users who want crisp point
predictions or a MILP-embeddable model.

Non-greedy oblique training generally has been a productive direction
{cite:p}`norouzi2015efficient,murthy1994system`, and the soft-routing
formulation fits naturally inside it.
