"""Comparison plots: corner, shear maps, and power spectra."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import OMEGA_M_TRUE, SIGMA_8_TRUE
from .power_spectrum import cl_model


def _plot_contours(ax, x, y, color, label, n_bins=40, xy_range=None):
    """Plot 68% and 95% density contours from 2D samples."""
    from scipy.ndimage import gaussian_filter

    # 2D histogram
    if xy_range is not None:
        x_range, y_range = xy_range
    else:
        x_range = (np.percentile(x, 0.5), np.percentile(x, 99.5))
        y_range = (np.percentile(y, 0.5), np.percentile(y, 99.5))
    H, xedges, yedges = np.histogram2d(
        x, y, bins=n_bins, range=[x_range, y_range],
    )
    H = gaussian_filter(H, sigma=1.5)

    # Find levels enclosing 68% and 95% of the density
    H_sorted = np.sort(H.ravel())[::-1]
    H_cumsum = np.cumsum(H_sorted) / H_sorted.sum()
    level_68 = H_sorted[np.searchsorted(H_cumsum, 0.68)]
    level_95 = H_sorted[np.searchsorted(H_cumsum, 0.95)]

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    ax.contourf(xc, yc, H.T, levels=[level_95, level_68, H.max()],
                colors=[color], alpha=[0.15, 0.35])
    ax.contour(xc, yc, H.T, levels=[level_95, level_68],
               colors=[color], linewidths=1.2)
    # Invisible line for legend
    ax.plot([], [], color=color, lw=2, label=label)


def plot_corner_comparison(
    classical_samples: dict,
    hierarchical_samples: dict,
) -> None:
    """Overlay (Omega_m, sigma_8, S_8) posteriors from both methods.

    Args:
        classical_samples: Dict with 'omega_m' and 'sigma_8' arrays from NUTS.
        hierarchical_samples: Dict with 'omega_m' and 'sigma_8' arrays from
            the field-level NUTS sampler.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    om_cl = np.array(classical_samples["omega_m"])
    s8_cl = np.array(classical_samples["sigma_8"])
    om_hi = np.array(hierarchical_samples["omega_m"])
    s8_hi = np.array(hierarchical_samples["sigma_8"])

    # Derived S_8
    s8_derived_cl = s8_cl * np.sqrt(om_cl / 0.3)
    s8_derived_hi = s8_hi * np.sqrt(om_hi / 0.3)
    s8_true_val = SIGMA_8_TRUE * np.sqrt(OMEGA_M_TRUE / 0.3)

    # --- Top row: marginals ---

    # Omega_m marginal
    ax = axes[0, 0]
    ax.hist(om_cl, bins=40, density=True, alpha=0.6, color="C0", label="Pseudo-Cℓ")
    ax.hist(om_hi, bins=40, density=True, alpha=0.6, color="C1", label="Field-level NUTS")
    ax.axvline(OMEGA_M_TRUE, color="k", ls="--", lw=1.5, label="Truth")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(r"$\Omega_m$ marginal")

    # S_8 marginal
    ax = axes[0, 1]
    ax.hist(s8_derived_cl, bins=40, density=True, alpha=0.6, color="C0",
            label="Pseudo-Cℓ")
    ax.hist(s8_derived_hi, bins=40, density=True, alpha=0.6, color="C1",
            label="Field-level NUTS")
    ax.axvline(s8_true_val, color="k", ls="--", lw=1.5, label="Truth")
    ax.set_xlabel(r"$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(r"$S_8$ marginal")

    # --- Bottom row: joint contours ---
    # Compute shared axis ranges from both sample sets
    om_all = np.concatenate([om_cl, om_hi])
    s8_all = np.concatenate([s8_cl, s8_hi])
    s8d_all = np.concatenate([s8_derived_cl, s8_derived_hi])

    om_s8_range = (
        (np.percentile(om_all, 0.5), np.percentile(om_all, 99.5)),
        (np.percentile(s8_all, 0.5), np.percentile(s8_all, 99.5)),
    )
    om_s8d_range = (
        (np.percentile(om_all, 0.5), np.percentile(om_all, 99.5)),
        (np.percentile(s8d_all, 0.5), np.percentile(s8d_all, 99.5)),
    )

    # Omega_m vs sigma_8
    ax = axes[1, 0]
    _plot_contours(ax, om_cl, s8_cl, "C0", "Pseudo-Cℓ", xy_range=om_s8_range)
    _plot_contours(ax, om_hi, s8_hi, "C1", "Field-level NUTS", xy_range=om_s8_range)
    ax.plot(OMEGA_M_TRUE, SIGMA_8_TRUE, "k+", ms=15, mew=2, label="Truth")
    om_line = np.linspace(0.10, 0.60, 100)
    s8_line = s8_true_val / np.sqrt(om_line / 0.3)
    ax.plot(om_line, s8_line, "k-", alpha=0.4, lw=1, label=r"$S_8$ degeneracy")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.legend(fontsize=9)
    ax.set_title(r"$\Omega_m$ vs $\sigma_8$")

    # S_8 vs Omega_m
    ax = axes[1, 1]
    _plot_contours(ax, om_cl, s8_derived_cl, "C0", "Pseudo-Cℓ", xy_range=om_s8d_range)
    _plot_contours(ax, om_hi, s8_derived_hi, "C1", "Field-level NUTS", xy_range=om_s8d_range)
    ax.plot(OMEGA_M_TRUE, s8_true_val, "k+", ms=15, mew=2, label="Truth")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$S_8$")
    ax.legend(fontsize=9)
    ax.set_title(r"$\Omega_m$ vs $S_8$")

    plt.suptitle(
        "Cosmological Posterior Comparison: Pseudo-Cℓ vs Field-level NUTS",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("plots/cosmo_comparison_corner.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/cosmo_comparison_corner.png")


def plot_shear_maps(
    sim: dict,
    gamma1_obs: jnp.ndarray,
    gamma2_obs: jnp.ndarray,
    gamma1_recon: jnp.ndarray,
    gamma2_recon: jnp.ndarray,
) -> None:
    """Plot true, observed, and reconstructed shear maps.

    Args:
        sim: Simulation dictionary (contains gamma1, gamma2 true fields).
        gamma1_obs: (n, n) pixel-averaged observed gamma1.
        gamma2_obs: (n, n) pixel-averaged observed gamma2.
        gamma1_recon: (n, n) reconstructed gamma1.
        gamma2_recon: (n, n) reconstructed gamma2.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmin1 = float(jnp.min(sim["gamma1"]))
    vmax1 = float(jnp.max(sim["gamma1"]))
    vmin2 = float(jnp.min(sim["gamma2"]))
    vmax2 = float(jnp.max(sim["gamma2"]))

    # gamma1 row
    im = axes[0, 0].imshow(
        np.array(sim["gamma1"]), origin="lower", cmap="RdBu_r", vmin=vmin1, vmax=vmax1
    )
    axes[0, 0].set_title(r"True $\gamma_1$")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(
        np.array(gamma1_obs), origin="lower", cmap="RdBu_r", vmin=vmin1, vmax=vmax1
    )
    axes[0, 1].set_title(r"Observed $\bar{\gamma}_1$")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].imshow(
        np.array(gamma1_recon),
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin1,
        vmax=vmax1,
    )
    axes[0, 2].set_title(r"NUTS Sample $\gamma_1$")
    plt.colorbar(im, ax=axes[0, 2])

    # gamma2 row
    im = axes[1, 0].imshow(
        np.array(sim["gamma2"]), origin="lower", cmap="RdBu_r", vmin=vmin2, vmax=vmax2
    )
    axes[1, 0].set_title(r"True $\gamma_2$")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(
        np.array(gamma2_obs), origin="lower", cmap="RdBu_r", vmin=vmin2, vmax=vmax2
    )
    axes[1, 1].set_title(r"Observed $\bar{\gamma}_2$")
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].imshow(
        np.array(gamma2_recon),
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin2,
        vmax=vmax2,
    )
    axes[1, 2].set_title(r"NUTS Sample $\gamma_2$")
    plt.colorbar(im, ax=axes[1, 2])

    plt.suptitle(
        "Shear Field: True | Observed | NUTS Sample",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("plots/cosmo_comparison_shear_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/cosmo_comparison_shear_maps.png")


def plot_power_spectrum(
    cl_data: dict,
    ell2d: jnp.ndarray,
    eb_data: dict | None = None,
) -> None:
    """Plot true Cl, estimated pseudo-Cl, noise level, and optional E/B diagnostic.

    Args:
        cl_data: Dictionary from estimate_pseudo_cl.
        ell2d: (n, n) multipole grid.
        eb_data: Optional dict from compute_eb_power with cl_E, cl_B, ell_bins.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ell_bins = np.array(cl_data["ell_bins"])
    cl_hat = np.array(cl_data["cl_hat"])
    cl_noise = float(cl_data["cl_noise"])

    # True power spectrum curve
    ell_smooth = np.logspace(np.log10(ell_bins.min()), np.log10(ell_bins.max()), 200)
    cl_true = np.array(cl_model(jnp.array(ell_smooth), OMEGA_M_TRUE, SIGMA_8_TRUE))

    ax.loglog(ell_smooth, cl_true, "k-", lw=2, label=r"True $C_\ell$")
    ax.loglog(ell_bins, np.maximum(cl_hat, 1e-20), "C0o", ms=6, label=r"Pseudo-$C_\ell$")
    ax.axhline(cl_noise, color="C3", ls="--", lw=1.5, label=f"Noise ($N_\\ell$)")

    if eb_data is not None:
        eb_ell = np.array(eb_data["ell_bins"])
        cl_E = np.array(eb_data["cl_E"])
        cl_B = np.array(eb_data["cl_B"])
        ax.loglog(eb_ell, np.maximum(cl_E, 1e-20), "C2s", ms=5, alpha=0.7,
                  label=r"True $C_\ell^{EE}$")
        ax.loglog(eb_ell, np.maximum(cl_B, 1e-20), "C4^", ms=5, alpha=0.7,
                  label=r"True $C_\ell^{BB}$")

    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$C_\ell$", fontsize=13)
    ax.set_title("E-mode Power Spectrum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("plots/cosmo_comparison_power_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/cosmo_comparison_power_spectrum.png")


def plot_shear_whiskers(
    sim: dict,
    stride: int = 1,
) -> None:
    """Plot shear sticks overlaid on the convergence field.

    Each pixel gets a headless line segment with length proportional to |gamma|
    and orientation angle = arctan2(gamma2, gamma1) / 2 (spin-2).

    Args:
        sim: Simulation dictionary (contains kappa, gamma1, gamma2).
        stride: Plot every stride-th pixel (for readability on dense grids).
    """
    kappa = np.array(sim["kappa"])
    gamma1 = np.array(sim["gamma1"])
    gamma2 = np.array(sim["gamma2"])
    n = kappa.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Convergence background
    im = ax.imshow(kappa, origin="lower", cmap="RdBu_r", alpha=0.6,
                   extent=[0, n, 0, n])
    plt.colorbar(im, ax=ax, label=r"$\kappa$", shrink=0.8)

    # Shear sticks
    mag = np.sqrt(gamma1**2 + gamma2**2)
    angle = 0.5 * np.arctan2(gamma2, gamma1)  # spin-2: half angle

    # Normalize stick length for visibility
    scale = 0.8 * stride / np.max(mag) if np.max(mag) > 0 else 1.0

    for i in range(0, n, stride):
        for j in range(0, n, stride):
            dx = mag[i, j] * scale * np.cos(angle[i, j])
            dy = mag[i, j] * scale * np.sin(angle[i, j])
            cx, cy = j + 0.5, i + 0.5  # pixel center
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                    "k-", lw=0.8, alpha=0.8)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_title(r"Shear whiskers on $\kappa$ field", fontsize=14, fontweight="bold")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")

    plt.tight_layout()
    plt.savefig("plots/cosmo_shear_whiskers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/cosmo_shear_whiskers.png")
