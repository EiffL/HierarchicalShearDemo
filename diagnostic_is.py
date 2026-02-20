#!/usr/bin/env python3
"""Diagnostic: Compare IS marginal log-likelihood with exact Gaussian marginal."""
import jax
import jax.numpy as jnp

from cosmo_lib.config import (
    GRID_SIZE, PIXEL_SCALE, N_GAL_PER_PIX, SHAPE_NOISE, SEED,
    N_PATCH_SAMPLES,
)
from cosmo_lib.simulation import simulate_shear_catalog
from cosmo_lib.gibbs import (
    compute_patch_posteriors, draw_patch_samples,
    precompute_sample_power, marginal_log_likelihood,
)
from cosmo_lib.power_spectrum import cl_model

key = jax.random.PRNGKey(SEED)
k_sim, k_s1, k_s2 = jax.random.split(key, 3)

# Simulate
sim = simulate_shear_catalog(k_sim)
patch = compute_patch_posteriors(sim["eps1"], sim["eps2"], SHAPE_NOISE, N_GAL_PER_PIX)

# Draw IS samples
samples1 = draw_patch_samples(k_s1, patch["mean1"], patch["sigma_patch"], N_PATCH_SAMPLES)
samples2 = draw_patch_samples(k_s2, patch["mean2"], patch["sigma_patch"], N_PATCH_SAMPLES)

# Precompute FFT power of samples
n = GRID_SIZE
delta = PIXEL_SCALE
delta_rad = jnp.deg2rad(delta / 60.0)
area = (n * delta_rad) ** 2
ell2d = sim["ell2d"]

power1, power2 = precompute_sample_power(samples1, samples2, n, delta)

# Hermitian weight mask
n2 = n // 2
weight = jnp.ones((n, n)) * 0.5
weight = weight.at[0, 0].set(0.0)
weight = weight.at[0, n2].set(1.0)
weight = weight.at[n2, 0].set(1.0)
weight = weight.at[n2, n2].set(1.0)

# Data power (per component)
data_fft1 = jnp.fft.fft2(patch["mean1"])
data_fft2 = jnp.fft.fft2(patch["mean2"])
data_power1 = jnp.abs(data_fft1) ** 2 * area / n**2
data_power2 = jnp.abs(data_fft2) ** 2 * area / n**2

noise_power = patch["sigma_patch"] ** 2 * area
sigma_eta_sq = n**2 * patch["sigma_patch"] ** 2  # FFT noise variance from sampling

print(f"Grid: {n}x{n}, pixel scale: {delta} arcmin")
print(f"sigma_patch = {patch['sigma_patch']:.6f}")
print(f"noise_power (N_ell) = {noise_power:.2e}")
print(f"K = {N_PATCH_SAMPLES}")
print()

# Check average per-component data power vs model
cl_true = cl_model(jnp.clip(ell2d, 1.0, None), 0.3, 0.8)
cl_true = cl_true.at[0, 0].set(0.0)
mask_dc = (ell2d > 0).astype(jnp.float32)
n_modes = float(jnp.sum(mask_dc))

avg_data_pow1 = float(jnp.sum(mask_dc * data_power1) / n_modes)
avg_data_pow2 = float(jnp.sum(mask_dc * data_power2) / n_modes)
avg_cl = float(jnp.sum(mask_dc * cl_true) / n_modes)

print(f"Average per-component data power: {(avg_data_pow1 + avg_data_pow2)/2:.2e}")
print(f"Average C_ell (model at truth):   {avg_cl:.2e}")
print(f"Average C_ell/2:                  {avg_cl/2:.2e}")
print(f"Average noise power:              {noise_power:.2e}")
print(f"Ratio data_power / (C_ell + noise): {(avg_data_pow1+avg_data_pow2)/2 / (avg_cl + noise_power):.3f}")
print(f"Ratio data_power / (C_ell/2 + noise): {(avg_data_pow1+avg_data_pow2)/2 / (avg_cl/2 + noise_power):.3f}")
print()


def exact_marginal(omega_m, sigma_8, signal_scale=0.5):
    """Exact Gaussian marginal using C_ell * signal_scale per component."""
    cl2d = cl_model(jnp.clip(ell2d, 1.0, None), omega_m, sigma_8)
    cl2d = cl2d.at[0, 0].set(0.0)
    signal = cl2d * signal_scale  # per-component signal power
    total = signal + noise_power
    total = jnp.where(total > 0, total, 1.0)
    log_l1 = -jnp.log(total) - data_power1 / total
    log_l2 = -jnp.log(total) - data_power2 / total
    return float(jnp.sum(weight * (log_l1 + log_l2)))


# Compare at several parameter points
test_points = [
    (0.30, 0.80, "truth"),
    (0.39, 0.82, "classical mean"),
    (0.46, 0.86, "gibbs mean"),
    (0.20, 0.60, "low"),
    (0.50, 1.00, "high"),
]

print(f"{'Point':>20s}  {'IS (C_l/2)':>12s}  {'Exact (C_l/2)':>14s}  {'Exact (C_l)':>12s}  {'IS-Exact(C_l/2)':>16s}  {'IS-Exact(C_l)':>14s}")
print("-" * 105)
for om, s8, label in test_points:
    is_val = float(marginal_log_likelihood(power1, power2, om, s8, ell2d, weight))
    ex_half = exact_marginal(om, s8, signal_scale=0.5)
    ex_full = exact_marginal(om, s8, signal_scale=1.0)
    print(f"{label:>20s}  {is_val:12.2f}  {ex_half:14.2f}  {ex_full:12.2f}  {is_val - ex_half:16.2f}  {is_val - ex_full:14.2f}")

# Find the peak of each log-likelihood surface along the S_8 direction
print("\n--- S_8 scan (Omega_m = 0.35) ---")
print(f"{'sigma_8':>8s}  {'IS':>12s}  {'Exact(C_l/2)':>14s}  {'Exact(C_l)':>12s}")
for s8 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
    is_val = float(marginal_log_likelihood(power1, power2, 0.35, s8, ell2d, weight))
    ex_half = exact_marginal(0.35, s8, 0.5)
    ex_full = exact_marginal(0.35, s8, 1.0)
    print(f"{s8:8.2f}  {is_val:12.2f}  {ex_half:14.2f}  {ex_full:12.2f}")
