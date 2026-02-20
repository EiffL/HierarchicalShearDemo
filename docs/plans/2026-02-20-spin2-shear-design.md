# Proper Spin-2 Shear Simulation + Validation

## Problem

The simulation in `cosmo_lib/simulation.py` generates shear by applying Kaiser-Squires to a non-Hermitian `kappa_fft` (a raw complex Gaussian draw). This produces gamma1 and gamma2 that are independent isotropic Gaussian fields rather than a correlated spin-2 pair. Consequences:

1. Shear maps lack the tangential/radial patterns expected from weak lensing.
2. The pseudo-Cl estimator recovers `2 * C_l` instead of `C_l` (factor-of-2 bias), because `|D* * D * a|^2 = |a|^2` has twice the variance of the correct `|kappa_H|^2`.
3. NUTS inference is biased upward in (Omega_m, sigma_8).

## Fix

After generating `kappa = Re(IFFT(kappa_fft))` (which is correct), FFT it back to get Hermitian Fourier coefficients before applying Kaiser-Squires:

```python
kappa = jnp.real(jnp.fft.ifft2(kappa_fft))
kappa_fft_H = jnp.fft.fft2(kappa)
gamma_fft = D_ell * kappa_fft_H
gamma = jnp.fft.ifft2(gamma_fft)
gamma1, gamma2 = gamma.real, gamma.imag
```

## Additional Validation

### B-mode power spectrum diagnostic

Proper E/B decomposition from the spin-2 field:

```
psi(l) = D*(l) * FFT(gamma1 + i*gamma2)
kappa_E(l) = [psi(l) + conj(psi(-l))] / 2
kappa_B(l) = [psi(l) - conj(psi(-l))] / (2i)
```

For a pure lensing field, B-mode power should be consistent with zero. Add this as both a plot and a verification check.

### Shear whisker plot

Overlay shear sticks on the convergence map. Each pixel gets a line segment with length ~ |gamma| and orientation angle = arctan2(gamma2, gamma1) / 2.

## Files Touched

| File | Change |
|------|--------|
| `cosmo_lib/simulation.py` | Fix KS to use Hermitian kappa_fft (~3 lines) |
| `cosmo_lib/classical.py` | Add `compute_eb_power()` for E/B decomposition |
| `cosmo_lib/plotting.py` | Add B-mode to power spectrum plot; add whisker plot |
| `cosmo_lib/verification.py` | Add B-mode consistency check |
| `cosmology_inference_demo.py` | Wire in E/B diagnostic and whisker plot |

## Out of Scope

The Gibbs sampler (`gibbs.py`) treats gamma1/gamma2 as independent isotropic fields with power C_l/2 each. This is inconsistent with spin-2 structure and will need updating separately.
