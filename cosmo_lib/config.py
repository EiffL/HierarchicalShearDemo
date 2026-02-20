"""Configuration constants for the cosmological inference demo."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Grid / survey settings
# ---------------------------------------------------------------------------
GRID_SIZE = 32  # pixels per side
PIXEL_SCALE = 2.0  # arcmin / pixel
N_GAL_PER_PIX = 30  # galaxies per pixel
SHAPE_NOISE = 0.26  # per-component intrinsic shape noise

# ---------------------------------------------------------------------------
# True cosmology
# ---------------------------------------------------------------------------
OMEGA_M_TRUE = 0.3
SIGMA_8_TRUE = 0.8

# Power spectrum model amplitude
A0 = 1e-4

# ---------------------------------------------------------------------------
# Gibbs sampler settings
# ---------------------------------------------------------------------------
N_GIBBS = 3000
N_GIBBS_BURN = 500
MH_PROPOSAL_SIGMA = jnp.array([0.02, 0.02])  # (Omega_m, sigma_8) proposal widths

# Interim prior width (matches per-patch posterior sigma)
SIGMA_INTERIM = SHAPE_NOISE / np.sqrt(N_GAL_PER_PIX)

# Number of interim samples per patch for importance sampling
N_PATCH_SAMPLES = 2000

# ---------------------------------------------------------------------------
# Classical pipeline NUTS settings
# ---------------------------------------------------------------------------
N_NUTS_WARMUP = 500
N_NUTS_SAMPLES = 2000

# Ell binning
N_ELL_BINS = 15

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
SEED = 42
