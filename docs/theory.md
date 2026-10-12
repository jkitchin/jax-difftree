# Model

The tree has fixed depth $D$ with $2^D - 1$ internal nodes and $2^D$ leaves
in BFS order. Each internal node $i$ holds an oblique split parameterized by
$(w_i \in \mathbb{R}^p, b_i \in \mathbb{R})$. The split function is

\begin{equation*}
s_i(x) \;=\; \sigma\!\bigl(\gamma_d \, (w_i^{\top} x + b_i)\bigr),
\qquad
\sigma(z) = \frac{1}{1+e^{-z}},
\end{equation*}

where $\gamma_d$ is a per-depth steepness. The probability that input $x$
reaches leaf $\ell$ is the product of $s_i$ or $1-s_i$ along the path,
computed in log space for numerical stability.

## Leaves

For univariate regression, leaf $\ell$ stores $(\mu_\ell, \sigma_\ell^2)$.
For multivariate regression, it stores $(\mu_\ell \in \mathbb{R}^{d_y}, L_\ell)$
where $L_\ell$ is a Cholesky factor with positive-softplus diagonal so that
$\Sigma_\ell = L_\ell L_\ell^{\top}$ is positive-definite by construction. For
classification, leaves hold categorical logits. Affine leaves are described
in [](surrogates.md#affine-leaves-and-jacobian-matching).

## Predictive distribution

\begin{equation*}
p(y \mid x) \;=\; \sum_{\ell=1}^{2^D} \mu_\ell(x) \, \mathcal{N}\!\bigl(y;\, \mu_\ell, \alpha\, L_\ell L_\ell^{\top}\bigr),
\end{equation*}

a conditional Gaussian mixture with input-dependent mixture weights, in the
spirit of mixture density networks {cite:p}`bishop1994mdn` but with a
tree-structured parameterization of the mixture rather than a black-box
neural network. The scalar $\alpha = \exp(\log\alpha)$ is fitted post-hoc on a
held-out validation slice as an LLPR/temperature-scaling-style variance
correction.

## Variance decomposition

By the law of total variance,

\begin{equation*}
\mathrm{Cov}[y \mid x]
\;=\; \underbrace{\mathbb{E}_\ell\bigl[\Sigma_\ell\bigr]}_{\text{aleatoric}}
\;+\; \underbrace{\mathrm{Cov}_\ell\bigl[\mu_\ell\bigr]}_{\text{routing}},
\end{equation*}

where the expectation and covariance are taken with respect to leaf index
$\ell$ weighted by the path probabilities $\mu_\ell(x)$. Both pieces are
exposed by `predict(return_components=True)` and by `predict_full_cov`.

## Training

The loss is the negative marginal log-likelihood $-\log p(y \mid x)$ averaged
over the training set. Adam with global gradient-norm clipping is used. A
short warmup phase trains splits and leaf means with leaf variances frozen at
$\sigma_\ell^2 \approx 1$ to avoid the classic variance-collapse pathology of
heteroscedastic NLL training {cite:p}`lakshminarayanan2017simple`. The
steepness $\gamma$ is annealed from $\gamma_{\text{init}}$ to $\gamma_{\max}$
on a fixed per-epoch schedule during both phases, in the same spirit as
{cite:t}`hehn2020end` except that $\gamma$ is not driven to infinity,
preserving soft-routing UQ at test time.

## Calibration

After training, splits and leaf parameters are frozen, and a small set of
calibration parameters (per-depth $\log\gamma$ and, for regressors,
$\log\alpha$) are fitted by Adam on the held-out validation slice by
minimizing NLL. This is the tree analogue of temperature scaling
{cite:p}`guo2017calibration`, a low-dimensional post-hoc rescaling that
targets the predictive distribution's calibration without altering its mean
structure. Empirical coverage of the resulting prediction intervals tracks
nominal coverage closely on the test set, as demonstrated in the
[heteroscedastic-noise notebook](examples/heteroscedastic_demo).

## Cross-conditioning and inverse design

For multivariate $y$, the conditional
$p(y_{\text{unobs}} \mid x, y_{\text{obs}})$ is again a Gaussian mixture, with
each leaf's component obtained by Schur complement on $\Sigma_\ell$ and the
leaf weights re-normalized by the marginal density of $y_{\text{obs}}$ under
each leaf. This recipe is standard for Gaussian mixtures and gives
closed-form conditional mean and covariance and exact mixture sampling.

Inverse design exploits the fact that the tree's likelihood
$\log p(y_{\text{target}} \mid x)$ is end-to-end differentiable in $x$ since
splits and leaves are smooth. We optimize $x$ by Adam ascent with multiple
random restarts to handle the multimodality of the soft mixture surface.
Constrained variants where only a subset of outputs are targeted are
supported by passing a partial dict of target values.
