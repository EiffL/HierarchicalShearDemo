# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
.venv/bin/python cosmology_inference_demo.py
```

The virtualenv is a symlink to `../repo/SHINE/.venv` and contains JAX, NumPyro, matplotlib, GalSim, and related scientific packages. Always use `.venv/bin/python` to run code.

## Architecture

This is a cosmological weak-lensing inference demo comparing two approaches to constraining (Omega_m, sigma_8) from simulated galaxy shear data. The entry point is `cosmology_inference_demo.py`, which orchestrates everything by importing from `cosmo_lib/`.

**`cosmo_lib/` module layout:**

- `config.py` — All constants (grid size, noise levels, true cosmology, sampler settings). Every other module imports from here rather than defining its own constants.
- `power_spectrum.py` — `cl_model()`: parametric angular power spectrum C_l(Omega_m, sigma_8). Used by both pipelines and plotting.
- `simulation.py` — Generates a Gaussian convergence field kappa, derives shear via Kaiser-Squires, and produces a noisy galaxy ellipticity catalog. All done in Fourier space with JAX.
- `classical.py` — **Pipeline A**: Bins ellipticities into pixels, estimates E-mode pseudo-Cl power spectrum, then runs NumPyro NUTS to sample (Omega_m, sigma_8).
- `gibbs.py` — **Pipeline B**: Hierarchical Gibbs sampler using `jax.lax.scan`. Alternates between IS-reweighting per-patch shear samples (importance sampling from interim posteriors into field-conditional posteriors) and Metropolis-Hastings cosmology updates. Contains `GibbsState`/`GibbsOutput` NamedTuples for the scan carry/output.
- `plotting.py` — Corner plots, shear map comparisons, power spectrum plots. Saves PNGs to the working directory.
- `verification.py` — Quantitative checks: truth-in-CI, cross-method S_8 agreement, MH acceptance rate.

**Key dependency flow:** `config` ← `power_spectrum` ← `simulation`, `classical`, `gibbs` ← `plotting`, `verification`.

## Known Issues

The Gibbs sampler (Pipeline B) currently has poor mixing (~1.4% MH acceptance rate) and gets stuck at the prior boundary. The `SIGMA_INTERIM` and `N_PATCH_SAMPLES` constants in `config.py` were inferred from context (not present in the original monolithic script) and likely need tuning.
