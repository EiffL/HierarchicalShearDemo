# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
.venv/bin/python cosmology_inference_demo.py
```

The virtualenv is a symlink to `../repo/SHINE/.venv` and contains JAX, NumPyro, optax, matplotlib, scipy, and related scientific packages. Always use `.venv/bin/python` to run code.

Plots are saved to `plots/`. Bias test scripts (`test_pcl_bias.py`, `test_pcl_bias_multi.py`) are standalone and can be run directly.

## Architecture

This is a cosmological weak-lensing inference demo comparing two approaches to constraining (Omega_m, sigma_8) from simulated galaxy shear data. The entry point is `cosmology_inference_demo.py`, which orchestrates everything by importing from `cosmo_lib/`.

**`cosmo_lib/` module layout:**

- `config.py` — All constants (grid size, noise levels, true cosmology, sampler settings). Every other module imports from here rather than defining its own constants.
- `power_spectrum.py` — `cl_model()`: parametric angular power spectrum C_l(Omega_m, sigma_8). Used by both pipelines and plotting.
- `simulation.py` — Generates a Gaussian convergence field kappa, derives spin-2 shear via Kaiser-Squires (using Hermitian FFT of the real kappa field), computes reduced shear g = γ/(1-κ), and produces a galaxy ellipticity catalog using the Möbius lensing transformation e_obs = (e_int + g)/(1 + conj(g) e_int).
- `classical.py` — **Pipeline A**: Estimates E-mode pseudo-Cl power spectrum with proper E/B decomposition, bins with mode-count filtering (`n_min_modes=5`), then runs NumPyro NUTS to sample (S_8, Omega_m) with uniform priors. Also contains `compute_eb_power()` for E/B diagnostic. Operates on per-pixel mean ellipticities (valid since g ≈ γ in the weak-lensing regime).
- `vi_gmm.py` — Per-pixel GMM variational inference. Fits a K-component Gaussian mixture to each pixel's interim posterior q(g) ∝ p(data|g) × p_interim(g), where p(data|g) is the exact Möbius lensing likelihood and p_interim is an isotropic Gaussian prior N(0, σ_interim²). Uses reparameterized ELBO optimization with optax.adam + jax.lax.scan. Provides `gmm_log_prob_minus_interim_grid()` which evaluates log q(g) - log p_interim(g) to recover the data likelihood for NUTS.
- `hierarchical.py` — **Pipeline B**: Two-stage field-level inference. Stage 1: fits per-pixel GMMs via `vi_gmm.fit_all_pixels()` (jax.lax.map over rows, vmap over columns). Stage 2: joint NUTS sampler with non-centered parameterization — samples z (n×n) and (S_8, Omega_m) jointly, forward-models to reduced shear g, and uses the GMM-minus-interim log-probability as the likelihood via `numpyro.factor`. Also provides `posterior_sample_shear()` for map reconstruction.
- `plotting.py` — Corner plots with density contours (68%/95%), shear map comparisons, power spectrum plots, whisker plots. Saves PNGs to `plots/`.
- `verification.py` — Quantitative checks: truth-in-CI, cross-method S_8 agreement, B-mode null test.

**Key dependency flow:** `config` <- `power_spectrum` <- `simulation`, `classical`, `vi_gmm` <- `hierarchical` <- `plotting`, `verification`.

## Key Implementation Details

**Simulation (Möbius lensing):** The convergence field kappa is drawn as a complex Gaussian in Fourier space, then IFFT'd to real space. Kaiser-Squires gives spin-2 shear (gamma1, gamma2) from FFT(kappa) (the Hermitian FFT of the real field). Reduced shear g = γ/(1-κ) is computed, and observed ellipticities use the full Möbius transformation: e_obs = (e_int + g)/(1 + conj(g) e_int), where e_int ~ N(0, σ_e² I) per component (Rayleigh magnitude + uniform orientation).

**Möbius lensing log-likelihood:** For a single galaxy: log p(e_obs|g) = -|e_s|²/(2σ²) - log(2πσ²) + 2 log(1-|g|²) - 2 log|1-conj(g)e_obs|², where e_s = (e_obs - g)/(1 - conj(g) e_obs) is the inferred intrinsic ellipticity.

**Pseudo-Cl E-mode extraction:** Uses the identity kappa_E(l) = [psi(l) + conj(psi(-l))]/2 where psi = D*(l) * gamma_tilde(l), with psi(-l) computed via `jnp.roll(jnp.flip(psi), (1,1))`. The noise bias is E-mode only: N = sigma_eps^2 / n_gal * area.

**Classical model parametrization:** Samples (S_8, Omega_m) directly with uniform priors, derives sigma_8 = S_8 / sqrt(Omega_m / 0.3). This avoids prior-induced bias on S_8 that arises from sampling (Omega_m, sigma_8) with uniform priors. Uses per-pixel mean ellipticities, which is a valid approximation since g ≈ γ for small κ.

**Per-pixel GMM VI with interim prior:** Each pixel's shear posterior is approximated by a K-component GMM fitted to the target p(data|g) × p_interim(g), where p_interim = N(0, σ_interim² I) is an isotropic Gaussian interim prior. The interim prior regularizes the VI optimization (faster convergence, correct posterior widths). At the NUTS stage, log p_interim is subtracted: the effective per-pixel log-likelihood is log q(g) - log p_interim(g), which recovers the data likelihood up to a constant. This interim-prior trick is essential — without it the GMM components are too narrow and bias the cosmological posteriors.

**Field-level non-centered parameterization:** The latent field z ~ N(0, I) is colored in Fourier space: kappa_fft = sqrt(C_l / area) * fft2(z). The sqrt(C_l / area) normalization (not sqrt(C_l * n^2 / area)) is correct because fft2(z) has E[|.|^2] = n^2 per mode for real iid z, yielding E[|kappa_fft|^2] = C_l * n^2 / area matching the simulation convention. The forward model extends to reduced shear g = γ/(1-κ), which is evaluated against the per-pixel GMMs. Both pipelines use the same (S_8, Omega_m) uniform priors.

## Known Issues

- The classical pipeline is **verified unbiased** via multi-seed testing (20 realizations at 128x128, 100 gal/pix): S_8 bias < 0.003, 20/20 coverage at 95% CI, zero NUTS divergences.
- Both pipelines agree on S_8 to ~1%, with zero NUTS divergences in both.
- The Möbius nonlinearity is negligible at the current signal level (|κ| ~ 0.01), so the per-pixel posteriors are essentially Gaussian. The GMM VI machinery is designed to handle stronger lensing regimes (cluster cores, higher A0) where the nonlinearity matters.
- The interim prior σ_interim = 0.15 is chosen to be ~3× wider than the per-pixel posterior width (σ_eps/√n_gal ≈ 0.047). This is broad enough not to bias the posteriors while providing sufficient regularization for VI convergence.
- The `vi_gmm.fit_all_pixels` uses `jax.lax.map` over rows (sequential) and `jax.vmap` over columns to manage memory. For larger grids, this may need further batching.
- The demo uses a **simplified power spectrum model** (not a Boltzmann code), so absolute parameter values should not be compared to real survey results.
