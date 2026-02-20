#!/usr/bin/env python3
"""Cosmological Inference Comparison Demo.

Demonstrates that per-patch shear posteriors from SHINE can be combined into
cosmological constraints via Gibbs sampling + importance sampling (IS), and
that these constraints are consistent with a classical pseudo-Cl pipeline.

Following Schneider et al. (2015), this script:
  1. Simulates a Gaussian convergence field kappa and derives shear gamma via
     Kaiser-Squires, then generates a noisy galaxy ellipticity catalog.
  2. Pipeline A (Classical Pseudo-Cl): bins ellipticities into pixels,
     estimates E-mode power spectrum, runs NUTS over (Omega_m, sigma_8).
  3. Pipeline B (Hierarchical Gibbs): runs a Gibbs sampler that alternates
     Wiener-filter shear sampling with Metropolis cosmology updates.
  4. Compares both posteriors via corner plots, shear maps, and power spectra.

Usage:
    python cosmology_inference_demo.py

Produces:
    cosmo_comparison_corner.png
    cosmo_comparison_shear_maps.png
    cosmo_comparison_power_spectrum.png
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from cosmo_lib.config import (
    GRID_SIZE,
    N_ELL_BINS,
    N_GAL_PER_PIX,
    N_GIBBS,
    N_GIBBS_BURN,
    N_NUTS_SAMPLES,
    N_NUTS_WARMUP,
    N_PATCH_SAMPLES,
    PIXEL_SCALE,
    SEED,
    SHAPE_NOISE,
)
from cosmo_lib.simulation import simulate_shear_catalog
from cosmo_lib.classical import estimate_pseudo_cl, compute_eb_power, run_classical_pipeline
from cosmo_lib.gibbs import (
    compute_patch_posteriors,
    draw_patch_samples,
    gibbs_sampler,
    wiener_filter_field,
)
from cosmo_lib.plotting import (
    plot_corner_comparison,
    plot_power_spectrum,
    plot_shear_maps,
    plot_shear_whiskers,
)
from cosmo_lib.verification import run_verification


def main() -> None:
    """Run the full cosmological inference comparison demo."""
    print("=" * 70)
    print("SHINE — Cosmological Inference Comparison Demo")
    print("Pseudo-Cℓ (NUTS) vs Gibbs + Importance Sampling")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    k_sim, k_cl, k_gibbs, k_s1, k_s2 = jax.random.split(key, 5)

    # ------------------------------------------------------------------
    # 1. Simulate
    # ------------------------------------------------------------------
    print(f"\nSimulating: {GRID_SIZE}x{GRID_SIZE} grid, "
          f"{N_GAL_PER_PIX} gal/pix, sigma_eps={SHAPE_NOISE}")
    t0 = time.time()
    sim = simulate_shear_catalog(k_sim)
    print(f"  kappa rms = {float(jnp.std(sim['kappa'])):.6f}")
    print(f"  gamma rms = {float(jnp.std(sim['gamma1'])):.6f}")
    print(f"  Simulation done in {time.time() - t0:.1f}s")

    # E/B diagnostic on true (noiseless) shear
    eb_data = compute_eb_power(
        sim["gamma1"], sim["gamma2"], GRID_SIZE, PIXEL_SCALE
    )
    print(f"  E/B check: mean B/E = {float(jnp.mean(jnp.abs(eb_data['cl_B'])) / jnp.mean(eb_data['cl_E'])):.6f}")

    # ------------------------------------------------------------------
    # 2. Pipeline A: Classical Pseudo-Cl
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pipeline A: Classical Pseudo-Cℓ + NUTS")
    print("-" * 70)

    t0 = time.time()
    cl_data = estimate_pseudo_cl(
        sim["eps1"],
        sim["eps2"],
        GRID_SIZE,
        PIXEL_SCALE,
        N_GAL_PER_PIX,
        SHAPE_NOISE,
    )
    print(f"  Pseudo-Cℓ estimated ({len(cl_data['ell_bins'])} bins kept of {N_ELL_BINS})")

    print(f"  Running NUTS ({N_NUTS_WARMUP} warmup + {N_NUTS_SAMPLES} samples)...")
    classical_samples = run_classical_pipeline(
        cl_data["cl_hat"],
        cl_data["n_modes"],
        cl_data["ell_bins"],
        cl_data["cl_noise"],
        k_cl,
    )
    print(f"  Classical pipeline done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Pipeline B: Gibbs + IS
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pipeline B: Hierarchical Gibbs + Importance Sampling")
    print("-" * 70)

    t0 = time.time()

    # Per-patch posteriors
    patch = compute_patch_posteriors(
        sim["eps1"], sim["eps2"], SHAPE_NOISE, N_GAL_PER_PIX
    )
    print(f"  Patch posterior sigma = {patch['sigma_patch']:.6f}")

    # Draw K independent samples per pixel from interim posteriors
    print(f"  Drawing {N_PATCH_SAMPLES} interim posterior samples per pixel...")
    samples1 = draw_patch_samples(k_s1, patch["mean1"], patch["sigma_patch"], N_PATCH_SAMPLES)
    samples2 = draw_patch_samples(k_s2, patch["mean2"], patch["sigma_patch"], N_PATCH_SAMPLES)

    # Run MH sampler with per-mode IS marginal likelihood
    print(f"  Running MH sampler ({N_GIBBS} iterations, "
          f"burn-in {N_GIBBS_BURN})...")
    print("  (First call includes JIT compilation)")
    gibbs_state, gibbs_chain = gibbs_sampler(
        k_gibbs,
        samples1,
        samples2,
        sim["ell2d"],
    )
    # Block until done
    jax.block_until_ready(gibbs_chain.omega_m)
    print(f"  MH sampler done in {time.time() - t0:.1f}s")
    acc_rate = float(gibbs_state.n_accepted) / N_GIBBS * 100
    print(f"  MH acceptance rate: {acc_rate:.1f}%")

    # Wiener-filtered shear reconstruction for visualization
    om_post = float(jnp.mean(gibbs_chain.omega_m[N_GIBBS_BURN:]))
    s8_post = float(jnp.mean(gibbs_chain.sigma_8[N_GIBBS_BURN:]))
    gamma1_recon = wiener_filter_field(
        patch["mean1"], sim["ell2d"], om_post, s8_post, patch["sigma_patch"]
    )
    gamma2_recon = wiener_filter_field(
        patch["mean2"], sim["ell2d"], om_post, s8_post, patch["sigma_patch"]
    )

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Generating comparison plots")
    print("-" * 70)

    plot_corner_comparison(classical_samples, gibbs_chain)
    plot_shear_maps(
        sim,
        cl_data["gamma1_obs"],
        cl_data["gamma2_obs"],
        gamma1_recon,
        gamma2_recon,
    )
    plot_power_spectrum(cl_data, sim["ell2d"], eb_data=eb_data)
    plot_shear_whiskers(sim)

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    run_verification(classical_samples, gibbs_chain, gibbs_state, eb_data=eb_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
