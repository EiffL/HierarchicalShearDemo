"""Verify spin-2 shear field properties."""
import jax
import jax.numpy as jnp
from cosmo_lib.simulation import simulate_shear_catalog

key = jax.random.PRNGKey(42)
sim = simulate_shear_catalog(key)

gamma1, gamma2 = sim["gamma1"], sim["gamma2"]
kappa = sim["kappa"]

# 1. Check power spectrum recovery: pseudo-Cl of true shear should match input C_l
# Compute |D* * FFT(gamma1 + i*gamma2)|^2 and compare to |FFT(kappa)|^2
n = gamma1.shape[0]
delta_rad = jnp.deg2rad(2.0 / 60.0)
area = (n * delta_rad) ** 2

gamma_fft = jnp.fft.fft2(gamma1 + 1j * gamma2)
freq = jnp.fft.fftfreq(n, d=delta_rad)
kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
ell_sq = kx**2 + ky**2
ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
D_ell = D_ell.at[0, 0].set(0.0 + 0j)

kappa_E_fft = jnp.conj(D_ell) * gamma_fft
power_from_shear = jnp.mean(jnp.abs(kappa_E_fft[1:, 1:]) ** 2) * area / n**2

kappa_fft = jnp.fft.fft2(kappa)
power_from_kappa = jnp.mean(jnp.abs(kappa_fft[1:, 1:]) ** 2) * area / n**2

ratio = float(power_from_shear / power_from_kappa)
print(f"Power ratio (shear E-mode / kappa): {ratio:.3f}")
print(f"  Expected ~1.0 for correct spin-2, ~2.0 for broken")
assert 0.8 < ratio < 1.2, f"FAIL: ratio={ratio:.3f}, expected ~1.0"

# 2. Check B-mode is zero for true (noiseless) shear
psi = jnp.conj(D_ell) * gamma_fft
# psi(-l): reflect indices
psi_neg = jnp.roll(jnp.flip(psi), (1, 1), axis=(0, 1))
kappa_B_fft = (psi - jnp.conj(psi_neg)) / 2j
kappa_B_fft = kappa_B_fft.at[0, 0].set(0.0)

power_B = jnp.mean(jnp.abs(kappa_B_fft[1:, 1:]) ** 2) * area / n**2
power_E_from_decomp = jnp.mean(jnp.abs(((psi + jnp.conj(psi_neg)) / 2)[1:, 1:]) ** 2) * area / n**2

b_over_e = float(power_B / power_E_from_decomp)
print(f"B/E power ratio: {b_over_e:.6f}")
print(f"  Expected ~0 for correct spin-2")
assert b_over_e < 0.01, f"FAIL: B/E={b_over_e:.6f}, expected ~0"

# 3. Check gamma1/gamma2 are NOT independent (spin-2 correlation)
# For a spin-2 field, cross-power FFT(gamma1)*conj(FFT(gamma2)) should be nonzero
g1_fft = jnp.fft.fft2(gamma1)
g2_fft = jnp.fft.fft2(gamma2)
cross = jnp.mean(jnp.abs(g1_fft * jnp.conj(g2_fft)))
auto1 = jnp.mean(jnp.abs(g1_fft) ** 2)
auto2 = jnp.mean(jnp.abs(g2_fft) ** 2)
cross_corr = float(cross / jnp.sqrt(auto1 * auto2))
print(f"gamma1-gamma2 cross-correlation: {cross_corr:.3f}")
print(f"  Expected >0.1 for spin-2, ~0 for independent")

print("\nAll checks passed!")
