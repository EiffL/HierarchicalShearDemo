"""Per-pixel GMM variational inference for the Möbius lensing likelihood.

Fits a K-component Gaussian mixture model (GMM) to the per-pixel posterior
over reduced shear g = (g1, g2), using the exact Möbius lensing log-likelihood
and reparameterized ELBO optimization with optax.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .config import (
    INTERIM_PRIOR_SIGMA,
    N_ELBO_SAMPLES,
    N_GMM_COMPONENTS,
    N_VI_STEPS,
    SHAPE_NOISE,
    VI_LR,
)


class GMMParams(NamedTuple):
    """Parameters for a K-component 2-D Gaussian mixture.

    Attributes:
        log_weights: (K,) unnormalized log mixture weights.
        means: (K, 2) component means.
        log_scale_tril: (K, 3) parameterization of lower-triangular Cholesky
            factor [log L_00, L_10, log L_11].
    """

    log_weights: jnp.ndarray
    means: jnp.ndarray
    log_scale_tril: jnp.ndarray


def _tril_from_params(log_scale_tril: jnp.ndarray) -> jnp.ndarray:
    """Convert (3,) parameterization to (2, 2) lower-triangular matrix."""
    L = jnp.array([
        [jnp.exp(log_scale_tril[0]), 0.0],
        [log_scale_tril[1], jnp.exp(log_scale_tril[2])],
    ])
    return L


def lensing_log_lik(
    e1_obs: jnp.ndarray,
    e2_obs: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    sigma: float = SHAPE_NOISE,
) -> jnp.ndarray:
    """Single-galaxy Möbius lensing log-likelihood.

    log p(e_obs | g) = -|e_s|^2/(2 sigma^2) - log(2 pi sigma^2)
                       + 2 log(1 - |g|^2) - 2 log|1 - conj(g) e_obs|^2

    where e_s = (e_obs - g) / (1 - conj(g) e_obs).
    """
    e_obs = e1_obs + 1j * e2_obs
    g = g1 + 1j * g2

    denom = 1.0 - jnp.conj(g) * e_obs
    e_s = (e_obs - g) / denom

    log_p = (
        -jnp.abs(e_s) ** 2 / (2.0 * sigma**2)
        - jnp.log(2.0 * jnp.pi * sigma**2)
        + 2.0 * jnp.log(jnp.clip(1.0 - jnp.abs(g) ** 2, 1e-30))
        - 2.0 * jnp.log(jnp.clip(jnp.abs(denom) ** 2, 1e-30))
    )
    return log_p


def pixel_log_lik(
    eps1_pix: jnp.ndarray,
    eps2_pix: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    sigma: float = SHAPE_NOISE,
) -> jnp.ndarray:
    """Sum of single-galaxy log-likelihoods for one pixel (n_gal galaxies).

    Args:
        eps1_pix, eps2_pix: (n_gal,) observed ellipticities for this pixel.
        g1, g2: Scalar reduced shear for this pixel.
        sigma: Intrinsic shape noise.

    Returns:
        Scalar total log-likelihood.
    """
    ll = jax.vmap(lensing_log_lik, in_axes=(0, 0, None, None, None))(
        eps1_pix, eps2_pix, g1, g2, sigma
    )
    return jnp.sum(ll)


def interim_log_prior(
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    sigma_interim: float = INTERIM_PRIOR_SIGMA,
) -> jnp.ndarray:
    """Isotropic Gaussian interim prior on reduced shear.

    log p_interim(g) = -|g|^2 / (2 sigma^2) - log(2 pi sigma^2)
    """
    return (
        -(g1**2 + g2**2) / (2.0 * sigma_interim**2)
        - jnp.log(2.0 * jnp.pi * sigma_interim**2)
    )


def gmm_log_prob(params: GMMParams, g: jnp.ndarray) -> jnp.ndarray:
    """Evaluate log q(g) under the GMM.

    Args:
        params: GMM parameters.
        g: (2,) point [g1, g2].

    Returns:
        Scalar log q(g).
    """
    log_w = params.log_weights - jax.nn.logsumexp(params.log_weights)

    def _component_log_prob(log_w_k, mu_k, log_scale_tril_k):
        L = _tril_from_params(log_scale_tril_k)
        diff = g - mu_k
        # solve L z = diff => z = L^{-1} diff
        z = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        log_det = jnp.log(jnp.abs(L[0, 0])) + jnp.log(jnp.abs(L[1, 1]))
        log_p = -0.5 * jnp.dot(z, z) - log_det - jnp.log(2.0 * jnp.pi)
        return log_w_k + log_p

    log_probs = jax.vmap(_component_log_prob)(
        log_w, params.means, params.log_scale_tril
    )
    return jax.nn.logsumexp(log_probs)


def gmm_sample(
    params: GMMParams, key: jax.Array, n_samples: int
) -> jnp.ndarray:
    """Reparameterized sampling from the GMM.

    Args:
        params: GMM parameters.
        key: PRNG key.
        n_samples: Number of samples.

    Returns:
        (n_samples, 2) array of samples.
    """
    k1, k2 = jax.random.split(key)
    log_w = params.log_weights - jax.nn.logsumexp(params.log_weights)

    # Sample component indices
    indices = jax.random.categorical(k1, log_w, shape=(n_samples,))

    # Sample from each component via reparameterization
    eps = jax.random.normal(k2, (n_samples, 2))

    def _sample_one(idx, eps_i):
        mu = params.means[idx]
        L = _tril_from_params(params.log_scale_tril[idx])
        return mu + L @ eps_i

    samples = jax.vmap(_sample_one)(indices, eps)
    return samples


def _elbo(
    params: GMMParams,
    eps1_pix: jnp.ndarray,
    eps2_pix: jnp.ndarray,
    key: jax.Array,
    sigma: float = SHAPE_NOISE,
    n_samples: int = N_ELBO_SAMPLES,
    sigma_interim: float = INTERIM_PRIOR_SIGMA,
) -> jnp.ndarray:
    """Compute ELBO = E_q[log p(data|g) + log p_interim(g) - log q(g)].

    The VI target is p(data|g) * p_interim(g), so the GMM approximates
    the interim posterior. The interim prior is subtracted at the NUTS stage.

    Uses mixture-weighted decomposition with reparameterized samples.
    """
    log_w = params.log_weights - jax.nn.logsumexp(params.log_weights)
    K = params.means.shape[0]
    n_per_comp = n_samples // K

    def _component_elbo(k, key_k):
        mu_k = params.means[k]
        L_k = _tril_from_params(params.log_scale_tril[k])
        eps = jax.random.normal(key_k, (n_per_comp, 2))
        samples_k = eps @ L_k.T + mu_k  # (n_per_comp, 2)

        def _eval_one(g):
            ll = pixel_log_lik(eps1_pix, eps2_pix, g[0], g[1], sigma)
            lp = interim_log_prior(g[0], g[1], sigma_interim)
            lq = gmm_log_prob(params, g)
            return ll + lp - lq

        vals = jax.vmap(_eval_one)(samples_k)
        return jnp.exp(log_w[k]) * jnp.mean(vals)

    keys = jax.random.split(key, K)
    elbos = jax.vmap(_component_elbo)(jnp.arange(K), keys)
    return jnp.sum(elbos)


def _init_gmm_params(
    eps1_pix: jnp.ndarray,
    eps2_pix: jnp.ndarray,
    key: jax.Array,
    K: int = N_GMM_COMPONENTS,
) -> GMMParams:
    """Initialize GMM parameters near the per-pixel sample mean."""
    mean_g1 = jnp.mean(eps1_pix)
    mean_g2 = jnp.mean(eps2_pix)

    # Perturb means slightly for each component
    perturbations = jax.random.normal(key, (K, 2)) * 0.005
    means = jnp.stack([mean_g1, mean_g2]) + perturbations

    log_weights = jnp.zeros(K)
    # Initialize covariance near the expected posterior width:
    # sigma ~ sigma_eps / sqrt(n_gal) ≈ 0.047
    init_sigma = 0.04
    log_scale_tril = jnp.tile(
        jnp.array([jnp.log(init_sigma), 0.0, jnp.log(init_sigma)]), (K, 1)
    )
    return GMMParams(log_weights=log_weights, means=means, log_scale_tril=log_scale_tril)


def fit_pixel_gmm(
    eps1_pix: jnp.ndarray,
    eps2_pix: jnp.ndarray,
    key: jax.Array,
    sigma: float = SHAPE_NOISE,
    K: int = N_GMM_COMPONENTS,
    n_steps: int = N_VI_STEPS,
    lr: float = VI_LR,
    n_samples: int = N_ELBO_SAMPLES,
) -> GMMParams:
    """Fit a GMM to one pixel's posterior over reduced shear.

    Uses optax.adam + jax.lax.scan for efficient optimization.

    Args:
        eps1_pix, eps2_pix: (n_gal,) observed ellipticities.
        key: PRNG key.
        sigma: Shape noise.
        K: Number of GMM components.
        n_steps: Optimization steps.
        lr: Learning rate.
        n_samples: ELBO MC samples per step.

    Returns:
        Optimized GMMParams.
    """
    k_init, k_opt = jax.random.split(key)
    init_params = _init_gmm_params(eps1_pix, eps2_pix, k_init, K)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)

    def _step(carry, key_i):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(
            lambda p: -_elbo(p, eps1_pix, eps2_pix, key_i, sigma, n_samples)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    keys = jax.random.split(k_opt, n_steps)
    (final_params, _), losses = jax.lax.scan(_step, (init_params, opt_state), keys)
    return final_params


def fit_all_pixels(
    eps1: jnp.ndarray,
    eps2: jnp.ndarray,
    key: jax.Array,
    sigma: float = SHAPE_NOISE,
) -> GMMParams:
    """Fit per-pixel GMMs across the full (n, n) grid.

    Uses jax.lax.map over rows and jax.vmap over columns to balance
    memory usage and speed.

    Args:
        eps1, eps2: (n, n, n_gal) observed ellipticities.
        key: PRNG key.
        sigma: Shape noise.

    Returns:
        GMMParams with shapes (n, n, K, ...).
    """
    n = eps1.shape[0]
    keys = jax.random.split(key, n * n).reshape(n, n, 2)

    def _fit_row(row_idx):
        def _fit_col(col_idx):
            return fit_pixel_gmm(
                eps1[row_idx, col_idx],
                eps2[row_idx, col_idx],
                keys[row_idx, col_idx],
                sigma,
            )
        return jax.vmap(_fit_col)(jnp.arange(n))

    # Sequential over rows, vmap over columns
    result = jax.lax.map(_fit_row, jnp.arange(n))
    return result


def gmm_log_prob_grid(
    gmm_params: GMMParams, g1: jnp.ndarray, g2: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate GMM log-probability for all pixels.

    Args:
        gmm_params: GMMParams with shapes (n, n, K, ...).
        g1, g2: (n, n) reduced shear field.

    Returns:
        (n, n) array of log q(g) values.
    """

    def _eval_pixel(params_pixel, g1_pixel, g2_pixel):
        pixel_params = GMMParams(
            log_weights=params_pixel.log_weights,
            means=params_pixel.means,
            log_scale_tril=params_pixel.log_scale_tril,
        )
        return gmm_log_prob(pixel_params, jnp.array([g1_pixel, g2_pixel]))

    def _eval_row(params_row, g1_row, g2_row):
        return jax.vmap(_eval_pixel)(params_row, g1_row, g2_row)

    return jax.vmap(_eval_row)(gmm_params, g1, g2)


def gmm_log_prob_minus_interim_grid(
    gmm_params: GMMParams,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    sigma_interim: float = INTERIM_PRIOR_SIGMA,
) -> jnp.ndarray:
    """Evaluate log q(g) - log p_interim(g) for all pixels.

    This recovers the data likelihood (up to a constant) by removing
    the interim prior that was included during the VI fit.

    Args:
        gmm_params: GMMParams with shapes (n, n, K, ...).
        g1, g2: (n, n) reduced shear field.
        sigma_interim: Interim prior width.

    Returns:
        (n, n) array of log q(g) - log p_interim(g).
    """
    log_q = gmm_log_prob_grid(gmm_params, g1, g2)
    log_p_interim = interim_log_prior(g1, g2, sigma_interim)
    return log_q - log_p_interim
