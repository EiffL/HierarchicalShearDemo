"""Gaussian convergence field simulation and galaxy catalog generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    GRID_SIZE,
    N_GAL_PER_PIX,
    OMEGA_M_TRUE,
    PIXEL_SCALE,
    SHAPE_NOISE,
    SIGMA_8_TRUE,
)
from .power_spectrum import cl_model


def _ell_grid(n: int, delta: float) -> jnp.ndarray:
    """Compute 2-D multipole grid from pixel grid parameters.

    Args:
        n: Grid size (pixels per side).
        delta: Pixel scale in arcmin.

    Returns:
        2-D array of multipole magnitudes l.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)
    return ell2d


def simulate_shear_catalog(
    key: jax.Array,
    n: int = GRID_SIZE,
    delta: float = PIXEL_SCALE,
    n_gal: int = N_GAL_PER_PIX,
    sigma_eps: float = SHAPE_NOISE,
    omega_m: float = OMEGA_M_TRUE,
    sigma_8: float = SIGMA_8_TRUE,
) -> dict:
    """Generate a Gaussian convergence field and noisy galaxy catalog.

    Steps:
      1. Draw Gaussian kappa on 2-D grid with power spectrum C_l(theta_true).
      2. Convert kappa -> (gamma1, gamma2) via Kaiser-Squires (pure E-mode).
      3. For each pixel, draw n_gal galaxy ellipticities:
         eps_i = g + noise, noise ~ N(0, sigma_eps^2).

    Args:
        key: JAX PRNG key.
        n: Grid size (pixels per side).
        delta: Pixel scale in arcmin.
        n_gal: Number of galaxies per pixel.
        sigma_eps: Per-component shape noise.
        omega_m: True Omega_m for the simulation.
        sigma_8: True sigma_8 for the simulation.

    Returns:
        Dictionary with keys:
          - kappa: (n, n) convergence field.
          - gamma1, gamma2: (n, n) shear field.
          - eps1, eps2: (n, n, n_gal) observed ellipticities.
          - ell2d: (n, n) multipole grid.
          - cl_true: (n, n) true C_l on the grid.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Multipole grid
    ell2d = _ell_grid(n, delta)

    # True power spectrum on 2-D grid (set DC to zero)
    cl2d = cl_model(jnp.clip(ell2d, 1.0, None), omega_m, sigma_8)
    cl2d = cl2d.at[0, 0].set(0.0)

    # Draw Gaussian kappa in Fourier space
    delta_rad = jnp.deg2rad(delta / 60.0)
    npix = n * n
    area = (n * delta_rad) ** 2
    sigma_fourier = jnp.sqrt(cl2d * npix / area)
    sigma_fourier = sigma_fourier.at[0, 0].set(0.0)

    # Complex Gaussian draw
    real_part = jax.random.normal(k1, (n, n)) * sigma_fourier
    imag_part = jax.random.normal(k2, (n, n)) * sigma_fourier
    kappa_fft = real_part + 1j * imag_part

    # Enforce Hermitian symmetry for real output
    kappa = jnp.real(jnp.fft.ifft2(kappa_fft))

    # Kaiser-Squires: gamma_tilde(l) = D(l) * kappa_tilde(l)
    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    gamma_fft = D_ell * kappa_fft
    gamma = jnp.fft.ifft2(gamma_fft)
    gamma1 = jnp.real(gamma)
    gamma2 = jnp.imag(gamma)

    # Galaxy catalog: eps_i = g_p + noise
    noise1 = jax.random.normal(k3, (n, n, n_gal)) * sigma_eps
    k4 = jax.random.fold_in(k3, 1)
    noise2 = jax.random.normal(k4, (n, n, n_gal)) * sigma_eps
    eps1 = gamma1[..., None] + noise1
    eps2 = gamma2[..., None] + noise2

    return {
        "kappa": kappa,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "eps1": eps1,
        "eps2": eps2,
        "ell2d": ell2d,
        "cl_true": cl2d,
    }
