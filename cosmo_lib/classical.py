"""Pipeline A: Classical pseudo-Cl estimation and NUTS inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .config import N_ELL_BINS, N_NUTS_SAMPLES, N_NUTS_WARMUP
from .power_spectrum import cl_model


def estimate_pseudo_cl(
    eps1: jnp.ndarray,
    eps2: jnp.ndarray,
    n: int,
    delta: float,
    n_gal: int,
    sigma_eps: float,
    n_bins: int = N_ELL_BINS,
) -> dict:
    """Estimate binned E-mode pseudo-Cl from galaxy ellipticities.

    Steps:
      1. Average ellipticities per pixel -> noisy shear map.
      2. FFT -> E/B decomposition.
      3. Bin power spectrum in l and subtract noise bias.

    Args:
        eps1: (n, n, n_gal) observed eps1.
        eps2: (n, n, n_gal) observed eps2.
        n: Grid size.
        delta: Pixel scale in arcmin.
        n_gal: Galaxies per pixel.
        sigma_eps: Shape noise per component.
        n_bins: Number of l-bins.

    Returns:
        Dictionary with keys:
          - ell_bins: (n_bins,) bin centers.
          - cl_hat: (n_bins,) noise-subtracted E-mode pseudo-Cl.
          - cl_noise: scalar noise bias per mode.
          - n_modes: (n_bins,) number of modes per bin.
          - gamma1_obs, gamma2_obs: (n, n) mean shear maps.
    """
    # Mean shear per pixel
    gamma1_obs = jnp.mean(eps1, axis=-1)
    gamma2_obs = jnp.mean(eps2, axis=-1)

    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    # FFT
    g1_fft = jnp.fft.fft2(gamma1_obs)
    g2_fft = jnp.fft.fft2(gamma2_obs)

    # E/B decomposition
    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    gamma_fft = g1_fft + 1j * g2_fft
    kappa_E_fft = jnp.conj(D_ell) * gamma_fft

    # E-mode power spectrum
    ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)
    power2d = jnp.abs(kappa_E_fft) ** 2 * area / (n**2)

    # Noise bias
    noise_per_mode = 2.0 * sigma_eps**2 / n_gal * area

    # Bin in l
    ell_flat = ell2d.ravel()
    power_flat = power2d.ravel()

    ell_min = 2.0 * jnp.pi / (n * delta_rad)
    ell_max = jnp.pi / delta_rad
    bin_edges = jnp.linspace(jnp.log(ell_min), jnp.log(ell_max), n_bins + 1)
    bin_edges = jnp.exp(bin_edges)

    ell_centers = []
    cl_hat_list = []
    n_modes_list = []

    # Use numpy for binning (not on the JIT path)
    ell_flat_np = np.array(ell_flat)
    power_flat_np = np.array(power_flat)
    bin_edges_np = np.array(bin_edges)

    for i in range(n_bins):
        mask = (ell_flat_np >= bin_edges_np[i]) & (ell_flat_np < bin_edges_np[i + 1])
        mask[0] = False  # exclude DC
        nm = mask.sum()
        if nm > 0:
            ell_centers.append(np.mean(ell_flat_np[mask]))
            cl_hat_list.append(np.mean(power_flat_np[mask]) - float(noise_per_mode))
            n_modes_list.append(nm)
        else:
            ell_centers.append(
                float(np.sqrt(bin_edges_np[i] * bin_edges_np[i + 1]))
            )
            cl_hat_list.append(0.0)
            n_modes_list.append(1)

    return {
        "ell_bins": jnp.array(ell_centers),
        "cl_hat": jnp.array(cl_hat_list),
        "cl_noise": noise_per_mode,
        "n_modes": jnp.array(n_modes_list),
        "gamma1_obs": gamma1_obs,
        "gamma2_obs": gamma2_obs,
    }


def compute_eb_power(
    gamma1: jnp.ndarray,
    gamma2: jnp.ndarray,
    n: int,
    delta: float,
    n_bins: int = N_ELL_BINS,
) -> dict:
    """Compute binned E-mode and B-mode power spectra from a shear field.

    Uses the spin-2 E/B decomposition:
        psi(l) = D*(l) * FFT(gamma1 + i*gamma2)
        kappa_E(l) = [psi(l) + conj(psi(-l))] / 2
        kappa_B(l) = [psi(l) - conj(psi(-l))] / (2i)

    Args:
        gamma1: (n, n) shear component 1.
        gamma2: (n, n) shear component 2.
        n: Grid size.
        delta: Pixel scale in arcmin.
        n_bins: Number of ell bins.

    Returns:
        Dictionary with keys:
          - ell_bins: (n_bins,) bin centers.
          - cl_E: (n_bins,) binned E-mode power.
          - cl_B: (n_bins,) binned B-mode power.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    # Spin-2 FFT
    gamma_fft = jnp.fft.fft2(gamma1 + 1j * gamma2)

    # Kaiser-Squires kernel
    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    # E/B decomposition
    psi = jnp.conj(D_ell) * gamma_fft
    psi_neg = jnp.roll(jnp.flip(psi), (1, 1), axis=(0, 1))
    kappa_E_fft = (psi + jnp.conj(psi_neg)) / 2.0
    kappa_B_fft = (psi - jnp.conj(psi_neg)) / 2j

    # Power spectra
    power_E = jnp.abs(kappa_E_fft) ** 2 * area / n**2
    power_B = jnp.abs(kappa_B_fft) ** 2 * area / n**2

    # Ell grid
    ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)

    # Bin in ell (same binning as estimate_pseudo_cl)
    ell_flat = np.array(ell2d.ravel())
    power_E_flat = np.array(power_E.ravel())
    power_B_flat = np.array(power_B.ravel())

    ell_min = 2.0 * np.pi / (n * delta_rad)
    ell_max = np.pi / delta_rad
    bin_edges = np.exp(np.linspace(np.log(ell_min), np.log(ell_max), n_bins + 1))

    ell_centers = []
    cl_E_list = []
    cl_B_list = []

    for i in range(n_bins):
        mask = (ell_flat >= bin_edges[i]) & (ell_flat < bin_edges[i + 1])
        mask[0] = False
        nm = mask.sum()
        if nm > 0:
            ell_centers.append(np.mean(ell_flat[mask]))
            cl_E_list.append(np.mean(power_E_flat[mask]))
            cl_B_list.append(np.mean(power_B_flat[mask]))
        else:
            ell_centers.append(float(np.sqrt(bin_edges[i] * bin_edges[i + 1])))
            cl_E_list.append(0.0)
            cl_B_list.append(0.0)

    return {
        "ell_bins": jnp.array(ell_centers),
        "cl_E": jnp.array(cl_E_list),
        "cl_B": jnp.array(cl_B_list),
    }


def classical_model(
    cl_hat: jnp.ndarray,
    n_modes: jnp.ndarray,
    ell_bins: jnp.ndarray,
    cl_noise: float,
) -> None:
    """NumPyro model for pseudo-Cl Gaussian likelihood.

    Likelihood:
        log p(C_hat_b | theta) = -1/2 sum_b (C_hat_b - C_b(theta))^2 / Var_b
    where Var_b = 2*(C_b(theta) + N_b)^2 / n_modes_b.

    Args:
        cl_hat: Noise-subtracted binned pseudo-Cl.
        n_modes: Number of modes per bin.
        ell_bins: Bin center multipoles.
        cl_noise: Noise power per mode.
    """
    omega_m = numpyro.sample("omega_m", dist.Uniform(0.1, 0.6))
    sigma_8 = numpyro.sample("sigma_8", dist.Uniform(0.4, 1.2))

    cl_theory = cl_model(ell_bins, omega_m, sigma_8)
    variance = 2.0 * (cl_theory + cl_noise) ** 2 / n_modes

    numpyro.sample(
        "cl_obs", dist.Normal(cl_theory, jnp.sqrt(variance)), obs=cl_hat
    )


def run_classical_pipeline(
    cl_hat: jnp.ndarray,
    n_modes: jnp.ndarray,
    ell_bins: jnp.ndarray,
    cl_noise: float,
    key: jax.Array,
) -> dict:
    """Run classical pseudo-Cl inference via NUTS.

    Args:
        cl_hat: Noise-subtracted binned pseudo-Cl.
        n_modes: Number of modes per bin.
        ell_bins: Bin center multipoles.
        cl_noise: Noise power per mode.
        key: JAX PRNG key.

    Returns:
        Dictionary with 'omega_m' and 'sigma_8' posterior sample arrays.
    """
    kernel = NUTS(classical_model)
    mcmc = MCMC(
        kernel,
        num_warmup=N_NUTS_WARMUP,
        num_samples=N_NUTS_SAMPLES,
        num_chains=1,
        progress_bar=True,
    )
    mcmc.run(key, cl_hat, n_modes, ell_bins, cl_noise)
    mcmc.print_summary()
    return mcmc.get_samples()
