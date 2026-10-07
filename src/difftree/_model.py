"""JAX core: oblique-split soft binary tree of fixed depth D, multivariate y.

Internal nodes are stored in BFS order: node i has children 2*i+1 and 2*i+2.
At depth d (0-indexed), the BFS indices of internal nodes are
``2**d - 1 .. 2**(d+1) - 2``. Leaves are at depth D, BFS-indexed 0 .. 2**D - 1.

Leaves are multivariate Gaussian: ``leaf_mu`` of shape ``(n_leaves, d_y)`` and
``leaf_chol_raw`` of shape ``(n_leaves, d_y, d_y)``. The actual Cholesky
factor is constructed from ``leaf_chol_raw`` by taking its strictly lower
triangle and replacing the diagonal with ``softplus(diag) + MIN_DIAG`` to
guarantee positive-definiteness.

The univariate case (d_y=1) is recovered by setting ``d_y=1`` everywhere; all
matrices become ``1 x 1`` and the math reduces to the scalar Gaussian.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

MIN_DIAG = 0.05  # floor on Cholesky diagonal in standardized y units


def n_internal_nodes(depth: int) -> int:
    return 2**depth - 1


def n_leaves(depth: int) -> int:
    return 2**depth


def log_path_probs(x, W, b, log_gamma):
    """Log path probability to each leaf for a single input.

    x: (p,). W: (n_internal, p). b: (n_internal,). log_gamma: (D,).
    Returns log_mu of shape (n_leaves,) in BFS leaf order.
    """
    f = W @ x + b
    depth = log_gamma.shape[0]
    log_mu = jnp.zeros((1,))
    for d in range(depth):
        start = 2**d - 1
        end = 2 ** (d + 1) - 1
        z = jnp.exp(log_gamma[d]) * f[start:end]
        log_sigma = -jax.nn.softplus(-z)
        log_one_minus_sigma = -jax.nn.softplus(z)
        log_mu_left = log_mu + log_one_minus_sigma
        log_mu_right = log_mu + log_sigma
        log_mu = jnp.stack([log_mu_left, log_mu_right], axis=-1).reshape(-1)
    return log_mu


# ---------------------------------------------------------------------------
# Cholesky construction (positive-definite from raw parameters)
# ---------------------------------------------------------------------------


def build_chol(raw, log_alpha):
    """Build a lower-triangular Cholesky factor from raw parameters.

    raw: (..., d, d). The strict lower triangle is taken as-is; the diagonal
    is replaced by ``softplus(diag(raw)) + MIN_DIAG`` for positive-definiteness.
    The whole factor is then scaled by ``exp(log_alpha / 2)`` so that
    ``Sigma_calibrated = exp(log_alpha) * (L L^T)``.
    """
    d = raw.shape[-1]
    L = jnp.tril(raw, k=-1)
    raw_diag = jnp.diagonal(raw, axis1=-2, axis2=-1)
    diag_pos = jax.nn.softplus(raw_diag) + MIN_DIAG
    idx = jnp.arange(d)
    L = L.at[..., idx, idx].set(diag_pos)
    return L * jnp.exp(0.5 * log_alpha)


def _solve_lower(L, B):
    """Solve L @ X = B (L lower-triangular). Vectorized over leading dims."""
    return vmap(lambda L_, b_: solve_triangular(L_, b_, lower=True))(L, B)


def _mvn_log_density(diff, L):
    """log N(0; 0, L L^T) per leaf, where ``diff = y - mu`` is shape (n_leaves, d).

    Uses the Cholesky factor directly: log det(L L^T) = 2 sum log diag(L), and
    quadratic form via triangular solve.
    """
    z = _solve_lower(L, diff)
    log_diag = jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1))
    d = diff.shape[-1]
    return -0.5 * d * jnp.log(2.0 * jnp.pi) - log_diag.sum(axis=-1) - 0.5 * (z ** 2).sum(axis=-1)


# ---------------------------------------------------------------------------
# Regression: log-likelihood, predictive moments, marginal sample
# ---------------------------------------------------------------------------


def regressor_log_lik(x, y, W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha):
    """log p(y|x) for one sample.

    leaf_mu: (n_leaves, d_y). leaf_chol_raw: (n_leaves, d_y, d_y).
    Predictive density is a mixture of multivariate Gaussians weighted by path
    probabilities.
    """
    log_mu = log_path_probs(x, W, b, log_gamma)
    L = build_chol(leaf_chol_raw, log_alpha)
    diff = y[None, :] - leaf_mu  # (n_leaves, d_y)
    log_p = _mvn_log_density(diff, L)
    return logsumexp(log_mu + log_p)


def regressor_predict_moments(x, W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha):
    """Predictive moments for one sample under the marginal mixture p(y|x).

    Returns (mean, total_cov, aleatoric_cov, routing_cov) of shapes
    (d_y,), (d_y, d_y), (d_y, d_y), (d_y, d_y). The decomposition is

        Cov[y|x] = E_l[Sigma_l]                 (aleatoric, within-leaf)
                 + Cov_l[mu_l]                  (routing, between-leaf)
    """
    log_mu = log_path_probs(x, W, b, log_gamma)
    w = jnp.exp(log_mu)  # (n_leaves,)
    L = build_chol(leaf_chol_raw, log_alpha)
    Sigma = L @ jnp.swapaxes(L, -1, -2)  # (n_leaves, d, d)
    mean = jnp.sum(w[:, None] * leaf_mu, axis=0)
    aleatoric = jnp.sum(w[:, None, None] * Sigma, axis=0)
    delta = leaf_mu - mean  # (n_leaves, d)
    routing = jnp.sum(
        w[:, None, None] * delta[:, :, None] * delta[:, None, :], axis=0
    )
    total = aleatoric + routing
    return mean, total, aleatoric, routing


def regressor_sample(x, key, W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha):
    """Draw one sample y ~ p(y|x). Two-step: pick leaf by path prob, then MVN."""
    log_mu = log_path_probs(x, W, b, log_gamma)
    k1, k2 = jax.random.split(key)
    leaf_idx = jax.random.categorical(k1, log_mu)
    L = build_chol(leaf_chol_raw, log_alpha)
    L_l = L[leaf_idx]
    mu_l = leaf_mu[leaf_idx]
    eps = jax.random.normal(k2, mu_l.shape)
    return mu_l + L_l @ eps


# ---------------------------------------------------------------------------
# Conditional p(y_unobs | x, y_obs): Gaussian-mixture closed-form
# ---------------------------------------------------------------------------


def _conditional_per_leaf(y_obs, obs_idx, unobs_idx, leaf_mu, L):
    """Per-leaf conditional moments and marginal log-density of y_obs.

    Returns (cond_mu_per_leaf, cond_Sigma_per_leaf, log_p_obs_per_leaf).
    obs_idx and unobs_idx are integer 1-D arrays of static length.
    """
    Sigma = L @ jnp.swapaxes(L, -1, -2)
    mu_o = leaf_mu[:, obs_idx]
    mu_u = leaf_mu[:, unobs_idx]
    Sigma_oo = Sigma[:, obs_idx[:, None], obs_idx[None, :]]
    Sigma_uu = Sigma[:, unobs_idx[:, None], unobs_idx[None, :]]
    Sigma_uo = Sigma[:, unobs_idx[:, None], obs_idx[None, :]]

    # Conditional moments per leaf
    diff_o = y_obs[None, :] - mu_o  # (n_leaves, d_o)
    A = jnp.linalg.solve(Sigma_oo, diff_o[..., None])[..., 0]  # (n_leaves, d_o)
    cond_mu = mu_u + jnp.einsum("luo,lo->lu", Sigma_uo, A)
    Sigma_ou = jnp.swapaxes(Sigma_uo, -1, -2)
    B = jnp.linalg.solve(Sigma_oo, Sigma_ou)
    cond_Sigma = Sigma_uu - jnp.einsum("luo,lov->luv", Sigma_uo, B)

    # log N(y_obs; mu_o^l, Sigma_oo^l) per leaf via Cholesky of Sigma_oo
    L_oo = jnp.linalg.cholesky(Sigma_oo)
    log_p_obs = _mvn_log_density(diff_o, L_oo)
    return cond_mu, cond_Sigma, log_p_obs


def conditional_moments(
    x, y_obs, obs_idx, unobs_idx, W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha
):
    """Conditional predictive mean, total covariance and log-evidence.

    Returns (cond_mean, cond_total_cov, log_evidence) where log_evidence is
    log p(y_obs | x) (the marginal mixture density of the observed components).
    Total covariance includes both within-leaf and between-leaf terms after
    leaf-weight reweighting.
    """
    log_mu = log_path_probs(x, W, b, log_gamma)
    L = build_chol(leaf_chol_raw, log_alpha)
    cond_mu_l, cond_Sigma_l, log_p_obs_l = _conditional_per_leaf(
        y_obs, obs_idx, unobs_idx, leaf_mu, L
    )
    log_w = log_mu + log_p_obs_l
    log_evidence = logsumexp(log_w)
    w = jnp.exp(log_w - log_evidence)  # normalized

    cond_mean = jnp.sum(w[:, None] * cond_mu_l, axis=0)
    aleatoric = jnp.sum(w[:, None, None] * cond_Sigma_l, axis=0)
    delta = cond_mu_l - cond_mean
    routing = jnp.sum(
        w[:, None, None] * delta[:, :, None] * delta[:, None, :], axis=0
    )
    return cond_mean, aleatoric + routing, log_evidence


def conditional_sample(
    x, y_obs, key, obs_idx, unobs_idx,
    W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha,
):
    """Draw one sample y_unobs ~ p(y_unobs | x, y_obs)."""
    log_mu = log_path_probs(x, W, b, log_gamma)
    L = build_chol(leaf_chol_raw, log_alpha)
    cond_mu_l, cond_Sigma_l, log_p_obs_l = _conditional_per_leaf(
        y_obs, obs_idx, unobs_idx, leaf_mu, L
    )
    log_w = log_mu + log_p_obs_l
    k1, k2 = jax.random.split(key)
    leaf_idx = jax.random.categorical(k1, log_w)
    cond_mu = cond_mu_l[leaf_idx]
    cond_Sigma = cond_Sigma_l[leaf_idx]
    L_cond = jnp.linalg.cholesky(cond_Sigma)
    eps = jax.random.normal(k2, cond_mu.shape)
    return cond_mu + L_cond @ eps


# ---------------------------------------------------------------------------
# Inverse design: log p(y_target_obs | x) for use as a likelihood over x
# ---------------------------------------------------------------------------


def inverse_design_log_lik(
    x, y_target, obs_idx, W, b, log_gamma, leaf_mu, leaf_chol_raw, log_alpha
):
    """log p(y_target | x) restricted to observed components.

    Differentiable in x. Optimize x to maximize this scalar.
    """
    log_mu = log_path_probs(x, W, b, log_gamma)
    L = build_chol(leaf_chol_raw, log_alpha)
    Sigma = L @ jnp.swapaxes(L, -1, -2)
    mu_o = leaf_mu[:, obs_idx]
    Sigma_oo = Sigma[:, obs_idx[:, None], obs_idx[None, :]]
    L_oo = jnp.linalg.cholesky(Sigma_oo)
    diff_o = y_target[None, :] - mu_o
    log_p_obs_l = _mvn_log_density(diff_o, L_oo)
    return logsumexp(log_mu + log_p_obs_l)


# ---------------------------------------------------------------------------
# Batched (vmap) entry points
# ---------------------------------------------------------------------------

batched_log_path_probs = vmap(log_path_probs, in_axes=(0, None, None, None))

batched_regressor_log_lik = vmap(
    regressor_log_lik, in_axes=(0, 0, None, None, None, None, None, None)
)
batched_regressor_predict_moments = vmap(
    regressor_predict_moments, in_axes=(0, None, None, None, None, None, None)
)
batched_regressor_sample = vmap(
    regressor_sample, in_axes=(0, 0, None, None, None, None, None, None)
)

batched_conditional_moments = vmap(
    conditional_moments,
    in_axes=(0, 0, None, None, None, None, None, None, None, None),
)
batched_conditional_sample = vmap(
    conditional_sample,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
)


# ---------------------------------------------------------------------------
# Parameter initializers
# ---------------------------------------------------------------------------


def init_regressor_params(key, depth, p, d_y, gamma_init=1.0, sigma_init=1.0):
    """Initialize a multivariate regressor.

    leaf_chol_raw is initialized with small random off-diagonal and a diagonal
    chosen so that ``softplus(diag) + MIN_DIAG ~= sigma_init``, giving each
    leaf an isotropic covariance of approximately ``sigma_init**2`` initially.
    """
    n_int = n_internal_nodes(depth)
    n_lvs = n_leaves(depth)
    k_W, k_b, k_mu, k_chol = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(p)

    target_diag = jnp.log(jnp.exp(sigma_init - MIN_DIAG) - 1.0)
    raw = jax.random.normal(k_chol, (n_lvs, d_y, d_y)) * 0.01
    idx = jnp.arange(d_y)
    raw = raw.at[:, idx, idx].set(target_diag)

    return dict(
        W=jax.random.normal(k_W, (n_int, p)) * scale,
        b=jax.random.normal(k_b, (n_int,)) * 0.01,
        leaf_mu=jax.random.normal(k_mu, (n_lvs, d_y)) * 0.1,
        leaf_chol_raw=raw,
        log_gamma=jnp.full((depth,), jnp.log(gamma_init)),
        log_alpha=jnp.zeros(()),
    )


# ---------------------------------------------------------------------------
# Classifier (unchanged): categorical leaves
# ---------------------------------------------------------------------------


def classifier_log_proba(x, W, b, log_gamma, leaf_logits):
    log_mu = log_path_probs(x, W, b, log_gamma)
    log_pi = jax.nn.log_softmax(leaf_logits, axis=-1)
    return logsumexp(log_mu[:, None] + log_pi, axis=0)


def classifier_log_lik(x, y, W, b, log_gamma, leaf_logits):
    return classifier_log_proba(x, W, b, log_gamma, leaf_logits)[y]


def classifier_proba_components(x, W, b, log_gamma, leaf_logits):
    log_mu = log_path_probs(x, W, b, log_gamma)
    w = jnp.exp(log_mu)
    log_pi = jax.nn.log_softmax(leaf_logits, axis=-1)
    pi = jnp.exp(log_pi)
    probs = jnp.sum(w[:, None] * pi, axis=0)
    log_probs = jnp.log(probs + 1e-30)
    total = -jnp.sum(probs * log_probs)
    leaf_H = -jnp.sum(pi * log_pi, axis=-1)
    aleatoric = jnp.sum(w * leaf_H)
    epistemic = total - aleatoric
    return probs, aleatoric, epistemic


def classifier_sample(x, key, W, b, log_gamma, leaf_logits):
    log_mu = log_path_probs(x, W, b, log_gamma)
    k1, k2 = jax.random.split(key)
    leaf_idx = jax.random.categorical(k1, log_mu)
    return jax.random.categorical(k2, leaf_logits[leaf_idx])


batched_classifier_log_lik = vmap(
    classifier_log_lik, in_axes=(0, 0, None, None, None, None)
)
batched_classifier_log_proba = vmap(
    classifier_log_proba, in_axes=(0, None, None, None, None)
)
batched_classifier_proba_components = vmap(
    classifier_proba_components, in_axes=(0, None, None, None, None)
)
batched_classifier_sample = vmap(
    classifier_sample, in_axes=(0, 0, None, None, None, None)
)


def init_classifier_params(key, depth, p, n_classes, gamma_init=1.0):
    n_int = n_internal_nodes(depth)
    n_lvs = n_leaves(depth)
    k_W, k_b, k_logits = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(p)
    return dict(
        W=jax.random.normal(k_W, (n_int, p)) * scale,
        b=jax.random.normal(k_b, (n_int,)) * 0.01,
        leaf_logits=jax.random.normal(k_logits, (n_lvs, n_classes)) * 0.01,
        log_gamma=jnp.full((depth,), jnp.log(gamma_init)),
    )
