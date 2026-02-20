# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
.venv/bin/python cosmology_inference_demo.py
```

The virtualenv is a symlink to `../repo/SHINE/.venv` and contains JAX, NumPyro, matplotlib, scipy, and related scientific packages. Always use `.venv/bin/python` to run code.

Plots are saved to `plots/`. Bias test scripts (`test_pcl_bias.py`, `test_pcl_bias_multi.py`) are standalone and can be run directly.

## Architecture

This is a cosmological weak-lensing inference demo comparing two approaches to constraining (Omega_m, sigma_8) from simulated galaxy shear data. The entry point is `cosmology_inference_demo.py`, which orchestrates everything by importing from `cosmo_lib/`.

**`cosmo_lib/` module layout:**

- `config.py` — All constants (grid size, noise levels, true cosmology, sampler settings). Every other module imports from here rather than defining its own constants.
- `power_spectrum.py` — `cl_model()`: parametric angular power spectrum C_l(Omega_m, sigma_8). Used by both pipelines and plotting.
- `simulation.py` — Generates a Gaussian convergence field kappa, derives spin-2 shear via Kaiser-Squires (using Hermitian FFT of the real kappa field), and produces a noisy galaxy ellipticity catalog.
- `classical.py` — **Pipeline A**: Estimates E-mode pseudo-Cl power spectrum with proper E/B decomposition, bins with mode-count filtering (`n_min_modes=5`), then runs NumPyro NUTS to sample (S_8, Omega_m) with uniform priors. Also contains `compute_eb_power()` for E/B diagnostic.
- `gibbs.py` — **Pipeline B**: Per-mode IS-marginalized Metropolis-Hastings sampler using `jax.lax.scan`. Pre-computes FFT power of K joint field samples, then evaluates the per-mode factorized IS marginal likelihood with second-order Jensen bias correction. Contains `GibbsState`/`GibbsOutput` NamedTuples for the scan carry/output. Also provides `wiener_filter_field()` for shear reconstruction.
- `plotting.py` — Corner plots with density contours (68%/95%), shear map comparisons, power spectrum plots, whisker plots. Saves PNGs to `plots/`.
- `verification.py` — Quantitative checks: truth-in-CI, cross-method S_8 agreement, MH acceptance rate, B-mode null test.

**Key dependency flow:** `config` <- `power_spectrum` <- `simulation`, `classical`, `gibbs` <- `plotting`, `verification`.

## Key Implementation Details

**Simulation (spin-2 shear):** The convergence field kappa is drawn as a complex Gaussian in Fourier space, then IFFT'd to real space. The Kaiser-Squires kernel D(l) is applied to FFT(kappa) (the Hermitian FFT of the real field), NOT to the original complex draw. This produces proper spin-2 correlated (gamma1, gamma2).

**Pseudo-Cl E-mode extraction:** Uses the identity kappa_E(l) = [psi(l) + conj(psi(-l))]/2 where psi = D*(l) * gamma_tilde(l), with psi(-l) computed via `jnp.roll(jnp.flip(psi), (1,1))`. The noise bias is E-mode only: N = sigma_eps^2 / n_gal * area.

**Classical model parametrization:** Samples (S_8, Omega_m) directly with uniform priors, derives sigma_8 = S_8 / sqrt(Omega_m / 0.3). This avoids prior-induced bias on S_8 that arises from sampling (Omega_m, sigma_8) with uniform priors.

## Known Issues

- The **Gibbs sampler** (Pipeline B) shows a systematic offset in Omega_m (biased high). MH acceptance rate is ~70%, which is reasonable, but the S_8 constraint, while better-centered, is much tighter than the classical pipeline. Diagnosing and fixing the Gibbs IS likelihood is the next development priority.
- The classical pipeline is **verified unbiased** via multi-seed testing (20 realizations at 128x128, 100 gal/pix): S_8 bias < 0.003, 20/20 coverage at 95% CI, zero NUTS divergences.
