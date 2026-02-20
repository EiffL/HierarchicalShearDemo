"""Configuration constants for the cosmological inference demo."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Grid / survey settings
# ---------------------------------------------------------------------------
GRID_SIZE = 128     # pixels per side
PIXEL_SCALE = 1.0   # arcmin / pixel
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
# Joint field-level NUTS settings (Pipeline B)
# ---------------------------------------------------------------------------
N_JOINT_WARMUP = 500
N_JOINT_SAMPLES = 2000
JOINT_TARGET_ACCEPT = 0.8
JOINT_MAX_TREE_DEPTH = 8

# ---------------------------------------------------------------------------
# Per-pixel GMM variational inference settings
# ---------------------------------------------------------------------------
N_GMM_COMPONENTS = 2
N_VI_STEPS = 2000
VI_LR = 1e-3
N_ELBO_SAMPLES = 32
INTERIM_PRIOR_SIGMA = 0.15  # isotropic Gaussian interim prior on g per component

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
SEED = 137
