#!/usr/bin/env python3
"""Cosmological Inference Comparison Demo.

Demonstrates two approaches to constraining (Omega_m, sigma_8) from simulated
galaxy shear data:

  1. Pipeline A (Classical Pseudo-Cl): bins ellipticities into pixels,
     estimates E-mode power spectrum, runs NUTS over (Omega_m, sigma_8).
  2. Pipeline B (Field-level NUTS): jointly samples a non-centered convergence
     field z and cosmological parameters using NUTS in 1026 dimensions.

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
    N_JOINT_SAMPLES,
    N_JOINT_WARMUP,
    N_NUTS_SAMPLES,
    N_NUTS_WARMUP,
    PIXEL_SCALE,
    SEED,
    SHAPE_NOISE,
)
from cosmo_lib.simulation import simulate_shear_catalog
from cosmo_lib.classical import estimate_pseudo_cl, compute_eb_power, run_classical_pipeline
from cosmo_lib.hierarchical import (
    _fourier_setup,
    posterior_sample_shear,
    run_hierarchical_pipeline,
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
    print("Pseudo-Cℓ (NUTS) vs Field-level NUTS")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    k_sim, k_cl, k_hier = jax.random.split(key, 3)

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
    # 3. Pipeline B: Field-level NUTS
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pipeline B: Field-level NUTS (joint z + cosmology)")
    print("-" * 70)

    t0 = time.time()

    # Pre-compute Fourier grids
    fourier = _fourier_setup(GRID_SIZE, PIXEL_SCALE)

    print(f"  Step 1: Per-pixel GMM VI on full galaxy catalog")
    print(f"  Step 2: Joint NUTS ({N_JOINT_WARMUP} warmup + "
          f"{N_JOINT_SAMPLES} samples, {GRID_SIZE}x{GRID_SIZE}+2 = "
          f"{GRID_SIZE*GRID_SIZE+2} dims)")
    print("  (First call includes JIT compilation)")
    hierarchical_samples = run_hierarchical_pipeline(
        sim["eps1"],
        sim["eps2"],
        k_hier,
        fourier=fourier,
    )
    jax.block_until_ready(hierarchical_samples["omega_m"])
    print(f"  Field-level pipeline done in {time.time() - t0:.1f}s")

    # Posterior mean shear reconstruction for visualization
    gamma1_recon, gamma2_recon = posterior_sample_shear(hierarchical_samples, fourier, index=-1)

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Generating comparison plots")
    print("-" * 70)

    plot_corner_comparison(classical_samples, hierarchical_samples)
    plot_shear_maps(
        sim,
        cl_data["gamma1_obs"],
        cl_data["gamma2_obs"],
        gamma1_recon,
        gamma2_recon,
    )
    plot_power_spectrum(cl_data, sim["ell2d"], eb_data=eb_data)
    plot_shear_whiskers(sim, stride=max(1, GRID_SIZE // 32))

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    run_verification(classical_samples, hierarchical_samples, eb_data=eb_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
