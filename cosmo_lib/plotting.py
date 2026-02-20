"""Comparison plots: corner, shear maps, and power spectra."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import N_GIBBS_BURN, OMEGA_M_TRUE, SIGMA_8_TRUE
from .gibbs import GibbsOutput
from .power_spectrum import cl_model


def plot_corner_comparison(
    classical_samples: dict,
    gibbs_chain: GibbsOutput,
    n_burn: int = N_GIBBS_BURN,
) -> None:
    """Overlay (Omega_m, sigma_8) posteriors from both methods with S_8 degeneracy.

    Args:
        classical_samples: Dict with 'omega_m' and 'sigma_8' arrays from NUTS.
        gibbs_chain: GibbsOutput from the Gibbs sampler.
        n_burn: Number of burn-in samples to discard from Gibbs chain.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    om_cl = np.array(classical_samples["omega_m"])
    s8_cl = np.array(classical_samples["sigma_8"])
    om_gi = np.array(gibbs_chain.omega_m[n_burn:])
    s8_gi = np.array(gibbs_chain.sigma_8[n_burn:])

    # Derived S_8
    s8_derived_cl = s8_cl * np.sqrt(om_cl / 0.3)
    s8_derived_gi = s8_gi * np.sqrt(om_gi / 0.3)

    # Omega_m marginal
    ax = axes[0, 0]
    ax.hist(om_cl, bins=40, density=True, alpha=0.6, color="C0", label="Pseudo-Cℓ")
    ax.hist(om_gi, bins=40, density=True, alpha=0.6, color="C1", label="Gibbs+IS")
    ax.axvline(OMEGA_M_TRUE, color="k", ls="--", lw=1.5, label="Truth")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(r"$\Omega_m$ marginal")

    # sigma_8 marginal
    ax = axes[0, 1]
    ax.hist(s8_cl, bins=40, density=True, alpha=0.6, color="C0", label="Pseudo-Cℓ")
    ax.hist(s8_gi, bins=40, density=True, alpha=0.6, color="C1", label="Gibbs+IS")
    ax.axvline(SIGMA_8_TRUE, color="k", ls="--", lw=1.5, label="Truth")
    ax.set_xlabel(r"$\sigma_8$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(r"$\sigma_8$ marginal")

    # Joint contour
    ax = axes[1, 0]
    ax.scatter(om_cl, s8_cl, s=1, alpha=0.1, color="C0", rasterized=True)
    ax.scatter(om_gi, s8_gi, s=1, alpha=0.1, color="C1", rasterized=True)
    ax.plot(OMEGA_M_TRUE, SIGMA_8_TRUE, "k+", ms=15, mew=2, label="Truth")

    # S_8 degeneracy line
    om_line = np.linspace(0.15, 0.55, 100)
    s8_true_derived = SIGMA_8_TRUE * np.sqrt(OMEGA_M_TRUE / 0.3)
    s8_line = s8_true_derived / np.sqrt(om_line / 0.3)
    ax.plot(om_line, s8_line, "k-", alpha=0.4, lw=1, label=r"$S_8$ degeneracy")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.legend(fontsize=9)
    ax.set_title(r"$\Omega_m$ vs $\sigma_8$")
    ax.set_xlim(0.1, 0.6)
    ax.set_ylim(0.4, 1.2)

    # S_8 marginal
    ax = axes[1, 1]
    ax.hist(
        s8_derived_cl,
        bins=40,
        density=True,
        alpha=0.6,
        color="C0",
        label="Pseudo-Cℓ",
    )
    ax.hist(
        s8_derived_gi,
        bins=40,
        density=True,
        alpha=0.6,
        color="C1",
        label="Gibbs+IS",
    )
    s8_true_val = SIGMA_8_TRUE * np.sqrt(OMEGA_M_TRUE / 0.3)
    ax.axvline(s8_true_val, color="k", ls="--", lw=1.5, label="Truth")
    ax.set_xlabel(r"$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(r"$S_8$ marginal")

    plt.suptitle(
        "Cosmological Posterior Comparison: Pseudo-Cℓ vs Gibbs+IS",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("cosmo_comparison_corner.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cosmo_comparison_corner.png")


def plot_shear_maps(
    sim: dict,
    gamma1_obs: jnp.ndarray,
    gamma2_obs: jnp.ndarray,
    gamma1_recon: jnp.ndarray,
    gamma2_recon: jnp.ndarray,
) -> None:
    """Plot true, observed, and Wiener-filtered shear maps.

    Args:
        sim: Simulation dictionary (contains gamma1, gamma2 true fields).
        gamma1_obs: (n, n) pixel-averaged observed gamma1.
        gamma2_obs: (n, n) pixel-averaged observed gamma2.
        gamma1_recon: (n, n) reconstructed gamma1 (Wiener-filtered).
        gamma2_recon: (n, n) reconstructed gamma2 (Wiener-filtered).
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
    axes[0, 2].set_title(r"Wiener-filtered $\gamma_1$")
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
    axes[1, 2].set_title(r"Wiener-filtered $\gamma_2$")
    plt.colorbar(im, ax=axes[1, 2])

    plt.suptitle(
        "Shear Field: True | Observed | Gibbs-Reconstructed",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("cosmo_comparison_shear_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cosmo_comparison_shear_maps.png")


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
    plt.savefig("cosmo_comparison_power_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cosmo_comparison_power_spectrum.png")
