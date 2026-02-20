"""Parametric angular power spectrum model."""

from __future__ import annotations

import jax.numpy as jnp

from .config import A0


def cl_model(
    ell: jnp.ndarray, omega_m: float, sigma_8: float
) -> jnp.ndarray:
    """Differentiable angular power spectrum C_l(theta).

    A simple parametric model that captures the S_8 degeneracy:
        C_l = A0 * sigma_8^2 * (Omega_m / 0.3)^2 * (l / 100)^{-1} / (1 + (l / 1000)^2)

    Args:
        ell: Multipole moments (1-D array).
        omega_m: Matter density parameter.
        sigma_8: RMS matter fluctuation amplitude.

    Returns:
        Power spectrum values at each l.
    """
    return (
        A0
        * sigma_8**2
        * (omega_m / 0.3) ** 2
        * (ell / 100.0) ** (-1.0)
        / (1.0 + (ell / 1000.0) ** 2)
    )
