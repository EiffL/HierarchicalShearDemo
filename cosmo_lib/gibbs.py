"""Pipeline B: IS-marginalized Metropolis-Hastings cosmology sampler.

Per-pixel interim posterior samples (independent across pixels) are
pre-drawn, FFT'd, and then used for per-Fourier-mode importance
sampling following Schneider et al. (2015, Eq. 15).

Because i.i.d.-across-pixel noise produces independent Fourier modes,
the marginal likelihood factorises over modes:

    log p(data|θ) = Σ_k log [ (1/K) Σ_s p(γ̃_k^s | θ) ]

Each 1-D IS average is well-estimated with modest K, and the sum over
~n² modes gives a low-variance estimate of the full marginal.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import (
    GRID_SIZE,
    MH_PROPOSAL_SIGMA,
    N_GIBBS,
    PIXEL_SCALE,
)
from .power_spectrum import cl_model


class GibbsState(NamedTuple):
    """Carry state for the MH scan loop."""

    omega_m: float
    sigma_8: float
    n_accepted: int


class GibbsOutput(NamedTuple):
    """Per-iteration output stored by the sampler."""

    omega_m: float
    sigma_8: float


def compute_patch_posteriors(
    eps1: jnp.ndarray,
    eps2: jnp.ndarray,
    sigma_eps: float,
    n_gal: int,
) -> dict:
    """Compute analytic Gaussian per-patch posteriors under flat interim prior.

    For each pixel p, the observed mean shear eps_bar_p is sufficient:
        g_p | data ~ N(eps_bar_p, sigma_eps^2 / n_gal)

    Args:
        eps1: (n, n, n_gal) observed eps1.
        eps2: (n, n, n_gal) observed eps2.
        sigma_eps: Per-component shape noise.
        n_gal: Number of galaxies per pixel.

    Returns:
        Dictionary with:
          - mean1, mean2: (n, n) posterior means.
          - sigma_patch: scalar posterior standard deviation.
    """
    mean1 = jnp.mean(eps1, axis=-1)
    mean2 = jnp.mean(eps2, axis=-1)
    sigma_patch = sigma_eps / jnp.sqrt(n_gal)
    return {"mean1": mean1, "mean2": mean2, "sigma_patch": float(sigma_patch)}


def draw_patch_samples(
    key: jax.Array,
    mean: jnp.ndarray,
    sigma_patch: float,
    n_samples: int,
) -> jnp.ndarray:
    """Draw samples from per-patch Gaussian posteriors.

    Args:
        key: JAX PRNG key.
        mean: (n, n) posterior means.
        sigma_patch: Posterior standard deviation.
        n_samples: Number of samples per patch.

    Returns:
        (n, n, n_samples) array of shear samples.
    """
    n1, n2 = mean.shape
    noise = jax.random.normal(key, (n1, n2, n_samples)) * sigma_patch
    return mean[..., None] + noise


def precompute_sample_power(
    samples1: jnp.ndarray,
    samples2: jnp.ndarray,
    n: int,
    delta: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute Fourier power spectra of all K joint field samples.

    Each joint field sample γ^s is formed by taking the s-th draw at
    each pixel (drawn independently from the per-pixel data posterior).

    Args:
        samples1: (n, n, K) per-pixel posterior samples for gamma1.
        samples2: (n, n, K) per-pixel posterior samples for gamma2.
        n: Grid size.
        delta: Pixel scale in arcmin.

    Returns:
        Tuple (power1, power2), each (K, n, n), containing
        |FFT(γ^s)|² · area / n² for each sample.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    # (n, n, K) -> (K, n, n) for batched FFT
    s1 = jnp.transpose(samples1, (2, 0, 1))
    s2 = jnp.transpose(samples2, (2, 0, 1))

    fft1 = jnp.fft.fft2(s1)
    fft2 = jnp.fft.fft2(s2)

    norm = area / (n**2)
    power1 = jnp.abs(fft1) ** 2 * norm
    power2 = jnp.abs(fft2) ** 2 * norm

    return power1, power2


def marginal_log_likelihood(
    power1: jnp.ndarray,
    power2: jnp.ndarray,
    omega_m: float,
    sigma_8: float,
    ell2d: jnp.ndarray,
    mask: jnp.ndarray,
) -> float:
    """Bias-corrected per-mode IS estimate of the marginal log-likelihood.

    Exploits the fact that Fourier modes of i.i.d.-across-pixel
    samples are independent, so the marginal factorises:

        log p(data|θ) = Σ_k log[ (1/K) Σ_s p(γ̃_k^s | θ) ]

    A second-order Jensen bias correction is applied per mode:
        correction_k = Var(X_k) / (2K · mean(X_k)²)
    reducing the bias from O(1/K) to O(1/K²).

    Args:
        power1: (K, n, n) pre-computed FFT power for gamma1 samples.
        power2: (K, n, n) pre-computed FFT power for gamma2 samples.
        omega_m: Omega_m.
        sigma_8: sigma_8.
        ell2d: (n, n) multipole grid.
        mask: (n, n) weight mask (Hermitian-corrected).

    Returns:
        Scalar log-marginal-likelihood (up to additive constant).
    """
    K = power1.shape[0]
    log_K = jnp.log(jnp.float32(K))

    cl2d = cl_model(jnp.clip(ell2d, 1.0, None), omega_m, sigma_8)
    cl2d = cl2d.at[0, 0].set(1.0)
    cl_half = cl2d / 2.0

    inv_cl = 1.0 / cl_half       # (n, n)
    log_cl = jnp.log(cl_half)    # (n, n)

    # Per-sample, per-mode log field prior (up to constant)
    f1 = -power1 * inv_cl[None, :, :] - log_cl[None, :, :]  # (K, n, n)
    f2 = -power2 * inv_cl[None, :, :] - log_cl[None, :, :]  # (K, n, n)

    # Per-mode logsumexp over K samples
    lse1 = jax.scipy.special.logsumexp(f1, axis=0)  # (n, n)
    lse2 = jax.scipy.special.logsumexp(f2, axis=0)

    # Second-order Jensen bias correction: +Var/(2K mean²)
    # Var/mean² = E[X²]/mean² - 1 = exp(logsumexp(2f) - 2·logsumexp(f) + log K) - 1
    lse2f1 = jax.scipy.special.logsumexp(2.0 * f1, axis=0)
    lse2f2 = jax.scipy.special.logsumexp(2.0 * f2, axis=0)

    var_over_mean_sq_1 = jnp.exp(lse2f1 - 2.0 * lse1 + log_K) - 1.0
    var_over_mean_sq_2 = jnp.exp(lse2f2 - 2.0 * lse2 + log_K) - 1.0

    correction1 = var_over_mean_sq_1 / (2.0 * K)
    correction2 = var_over_mean_sq_2 / (2.0 * K)

    # Sum over weighted modes (lse includes log K, which cancels in MH ratio)
    return jnp.sum(mask * (lse1 + correction1 + lse2 + correction2))


def _mh_step(
    carry: GibbsState,
    key: jax.Array,
    power1: jnp.ndarray,
    power2: jnp.ndarray,
    ell2d: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[GibbsState, GibbsOutput]:
    """Single MH step targeting p(θ|data) with per-mode IS likelihood."""
    omega_m, sigma_8, n_accepted = carry

    log_like_current = marginal_log_likelihood(
        power1, power2, omega_m, sigma_8, ell2d, mask
    )

    # Propose new cosmology
    proposal = jax.random.normal(key, (2,)) * MH_PROPOSAL_SIGMA
    omega_m_prop = omega_m + proposal[0]
    sigma_8_prop = sigma_8 + proposal[1]

    # Uniform prior bounds
    in_bounds = (
        (omega_m_prop > 0.1)
        & (omega_m_prop < 0.6)
        & (sigma_8_prop > 0.4)
        & (sigma_8_prop < 1.2)
    )

    log_like_prop = jnp.where(
        in_bounds,
        marginal_log_likelihood(
            power1, power2, omega_m_prop, sigma_8_prop, ell2d, mask
        ),
        -jnp.inf,
    )

    log_alpha = log_like_prop - log_like_current
    u = jax.random.uniform(jax.random.fold_in(key, 1))
    accept = jnp.log(u) < log_alpha

    omega_m_new = jnp.where(accept, omega_m_prop, omega_m)
    sigma_8_new = jnp.where(accept, sigma_8_prop, sigma_8)
    n_accepted_new = n_accepted + accept.astype(jnp.int32)

    new_state = GibbsState(
        omega_m=omega_m_new,
        sigma_8=sigma_8_new,
        n_accepted=n_accepted_new,
    )
    output = GibbsOutput(omega_m=omega_m_new, sigma_8=sigma_8_new)
    return new_state, output


def gibbs_sampler(
    key: jax.Array,
    samples1: jnp.ndarray,
    samples2: jnp.ndarray,
    ell2d: jnp.ndarray,
    n: int = GRID_SIZE,
    delta: float = PIXEL_SCALE,
    n_iter: int = N_GIBBS,
) -> tuple[GibbsState, GibbsOutput]:
    """Run MH chain with per-mode IS marginal likelihood.

    Pre-computes FFT power of all K joint field samples (formed from
    per-pixel independent interim posterior draws), then runs MH where
    each step evaluates the marginal likelihood as a sum of per-mode
    logsumexp terms.

    Args:
        key: JAX PRNG key.
        samples1: (n, n, K) per-pixel posterior samples for gamma1.
        samples2: (n, n, K) per-pixel posterior samples for gamma2.
        ell2d: (n, n) multipole grid.
        n: Grid size.
        delta: Pixel scale.
        n_iter: Number of MH iterations.

    Returns:
        Tuple of (final_state, chain_output).
    """
    power1, power2 = precompute_sample_power(samples1, samples2, n, delta)

    # Weight mask accounting for Hermitian symmetry of real-input FFT.
    # Modes (i,j) and (n-i, n-j) are conjugate pairs with identical
    # |FFT|², so each independent mode should be counted once.
    # Conjugate-paired modes get weight 0.5; self-conjugate modes get 1.0.
    n2 = n // 2
    weight = jnp.ones((n, n)) * 0.5
    weight = weight.at[0, 0].set(0.0)       # DC excluded
    weight = weight.at[0, n2].set(1.0)       # self-conjugate
    weight = weight.at[n2, 0].set(1.0)       # self-conjugate
    weight = weight.at[n2, n2].set(1.0)      # self-conjugate
    mask = weight

    keys = jax.random.split(key, n_iter)

    init_state = GibbsState(omega_m=0.3, sigma_8=0.8, n_accepted=0)

    step_fn = partial(
        _mh_step,
        power1=power1,
        power2=power2,
        ell2d=ell2d,
        mask=mask,
    )

    final_state, chain = jax.lax.scan(step_fn, init_state, keys)
    return final_state, chain


def wiener_filter_field(
    data_mean: jnp.ndarray,
    ell2d: jnp.ndarray,
    omega_m: float,
    sigma_8: float,
    sigma_patch: float,
    n: int = GRID_SIZE,
    delta: float = PIXEL_SCALE,
) -> jnp.ndarray:
    """Wiener-filter the pixel-averaged data to reconstruct the shear field.

    For each Fourier mode k:
        γ̂_k = S_k / (S_k + N_k) · d̃_k

    where S_k = C_l(k)/2 and N_k = σ_patch² · area.

    Args:
        data_mean: (n, n) pixel-averaged shear (one component).
        ell2d: (n, n) multipole grid.
        omega_m: Omega_m for the signal model.
        sigma_8: sigma_8 for the signal model.
        sigma_patch: Per-pixel noise standard deviation.
        n: Grid size.
        delta: Pixel scale in arcmin.

    Returns:
        (n, n) Wiener-filtered shear field.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    cl2d = cl_model(jnp.clip(ell2d, 1.0, None), omega_m, sigma_8)
    cl2d = cl2d.at[0, 0].set(0.0)
    signal_power = cl2d / 2.0
    noise_power = sigma_patch**2 * area

    data_fft = jnp.fft.fft2(data_mean)

    wiener = jnp.where(
        signal_power > 0,
        signal_power / (signal_power + noise_power),
        0.0,
    )

    return jnp.real(jnp.fft.ifft2(data_fft * wiener))
