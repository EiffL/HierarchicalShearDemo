"""Pipeline B: Joint field-level NUTS sampler.

Non-centered parameterization of the convergence field:
    z ~ N(0, I)                          iid standard normal (n x n)
    kappa_fft = sqrt(C_l(theta) / area) * fft2(z)   color by power spectrum
    gamma_fft = D(k) * kappa_fft                     Kaiser-Squires
    patch_mean ~ N(gamma, sigma_patch^2)             observed data

Total parameter space: n*n + 2  (z field + omega_m + sigma_8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .config import (
    GRID_SIZE,
    JOINT_MAX_TREE_DEPTH,
    JOINT_TARGET_ACCEPT,
    N_JOINT_SAMPLES,
    N_JOINT_WARMUP,
    PIXEL_SCALE,
)
from .power_spectrum import cl_model
from .vi_gmm import fit_all_pixels, gmm_log_prob_minus_interim_grid


def _fourier_setup(n: int, delta: float) -> dict:
    """Pre-compute Fourier-space grids and Kaiser-Squires kernel.

    Args:
        n: Grid size (pixels per side).
        delta: Pixel scale in arcmin.

    Returns:
        Dictionary with ell2d, D_ell, area, delta_rad, n.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)

    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    return {
        "ell2d": ell2d,
        "D_ell": D_ell,
        "area": area,
        "delta_rad": delta_rad,
        "n": n,
    }


def z_to_shear(
    z: jnp.ndarray,
    omega_m: float,
    sigma_8: float,
    fourier: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable forward model: white noise z -> (kappa, gamma1, gamma2, g1, g2).

    Args:
        z: (n, n) iid standard normal field.
        omega_m: Matter density parameter.
        sigma_8: RMS matter fluctuation amplitude.
        fourier: Pre-computed Fourier grids from _fourier_setup.

    Returns:
        Tuple (kappa, gamma1, gamma2, g1, g2), each (n, n).
    """
    n = fourier["n"]
    area = fourier["area"]
    ell2d = fourier["ell2d"]
    D_ell = fourier["D_ell"]

    # Power spectrum on 2D grid
    cl2d = cl_model(jnp.clip(ell2d, 1.0, None), omega_m, sigma_8)
    cl2d = cl2d.at[0, 0].set(0.0)

    # Color the white noise.  fft2(z) has E[|.|^2] = n^2 per mode
    # (z is real iid N(0,1)), so we need sqrt(C_l / area) to match the
    # simulation convention E[|kappa_fft[k]|^2] = C_l * n^2 / area.
    sigma_fourier = jnp.sqrt(cl2d / area)
    z_fft = jnp.fft.fft2(z)
    kappa_fft = sigma_fourier * z_fft

    kappa = jnp.real(jnp.fft.ifft2(kappa_fft))

    # Kaiser-Squires directly from kappa_fft (already Hermitian since z real)
    gamma_fft = D_ell * kappa_fft
    gamma = jnp.fft.ifft2(gamma_fft)
    gamma1 = jnp.real(gamma)
    gamma2 = jnp.imag(gamma)

    # Reduced shear
    g1 = gamma1 / (1.0 - kappa)
    g2 = gamma2 / (1.0 - kappa)

    return kappa, gamma1, gamma2, g1, g2


def hierarchical_model(
    gmm_params,
    fourier: dict,
    n: int,
) -> None:
    """NumPyro model for joint field-level inference with GMM likelihood.

    Samples z (white noise field) and cosmological parameters jointly,
    maps to reduced shear, and evaluates the pre-fitted per-pixel GMM
    log-probability as the likelihood.

    Args:
        gmm_params: Pre-fitted GMMParams with shapes (n, n, K, ...).
        fourier: Pre-computed Fourier grids.
        n: Grid size.
    """
    s_8 = numpyro.sample("S_8", dist.Uniform(0.3, 1.2))
    omega_m = numpyro.sample("omega_m", dist.Uniform(0.1, 0.6))
    sigma_8 = s_8 / jnp.sqrt(omega_m / 0.3)
    numpyro.deterministic("sigma_8", sigma_8)

    z = numpyro.sample("z", dist.Normal(jnp.zeros((n, n)), 1.0).to_event(2))

    kappa, gamma1, gamma2, g1, g2 = z_to_shear(z, omega_m, sigma_8, fourier)
    numpyro.deterministic("kappa", kappa)

    # log q(g) - log p_interim(g) recovers the data log-likelihood
    # (up to a constant), removing the interim prior used during VI
    log_lik = gmm_log_prob_minus_interim_grid(gmm_params, g1, g2)
    numpyro.factor("gmm_likelihood", jnp.sum(log_lik))


def run_hierarchical_pipeline(
    eps1: jnp.ndarray,
    eps2: jnp.ndarray,
    key: jax.Array,
    fourier: dict | None = None,
    n: int = GRID_SIZE,
    delta: float = PIXEL_SCALE,
    sigma: float | None = None,
    num_warmup: int = N_JOINT_WARMUP,
    num_samples: int = N_JOINT_SAMPLES,
    target_accept: float = JOINT_TARGET_ACCEPT,
    max_tree_depth: int = JOINT_MAX_TREE_DEPTH,
) -> dict:
    """Run joint field-level NUTS inference with GMM VI pre-step.

    First fits per-pixel GMMs to approximate the Möbius lensing posterior,
    then runs NUTS with the GMM likelihood.

    Args:
        eps1, eps2: (n, n, n_gal) observed ellipticities.
        key: JAX PRNG key.
        fourier: Pre-computed Fourier grids (computed if None).
        n: Grid size.
        delta: Pixel scale in arcmin.
        sigma: Shape noise (uses config default if None).
        num_warmup: NUTS warmup iterations.
        num_samples: NUTS post-warmup samples.
        target_accept: Target acceptance probability.
        max_tree_depth: Maximum tree depth for NUTS.

    Returns:
        Dictionary with 'omega_m', 'sigma_8', 'S_8', 'kappa' arrays.
        (z samples are dropped to save memory.)
    """
    from .config import SHAPE_NOISE

    if sigma is None:
        sigma = SHAPE_NOISE
    if fourier is None:
        fourier = _fourier_setup(n, delta)

    k_vi, k_nuts = jax.random.split(key)

    # Step 1: Fit per-pixel GMMs
    print("  Fitting per-pixel GMMs (VI)...")
    gmm_params = fit_all_pixels(eps1, eps2, k_vi, sigma)
    print("  GMM VI complete.")

    # Step 2: Run NUTS with GMM likelihood
    kernel = NUTS(
        hierarchical_model,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=True,
    )
    mcmc.run(k_nuts, gmm_params, fourier, n)
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    # Drop z to save memory, keep everything else
    return {
        "omega_m": samples["omega_m"],
        "sigma_8": samples["sigma_8"],
        "S_8": samples["S_8"],
        "kappa": samples["kappa"],
    }


def posterior_sample_shear(
    samples: dict,
    fourier: dict,
    index: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute reconstructed shear from a single posterior kappa sample.

    Takes one posterior kappa sample, applies Kaiser-Squires to get shear.

    Args:
        samples: Dictionary with 'kappa' array of shape (n_samples, n, n).
        fourier: Pre-computed Fourier grids.
        index: Which posterior sample to use.

    Returns:
        Tuple (gamma1_recon, gamma2_recon), each (n, n).
    """
    kappa_sample = samples["kappa"][index]
    D_ell = fourier["D_ell"]

    kappa_fft = jnp.fft.fft2(kappa_sample)
    gamma_fft = D_ell * kappa_fft
    gamma = jnp.fft.ifft2(gamma_fft)

    return jnp.real(gamma), jnp.imag(gamma)
