"""Generate verification plots for the spin-2 shear simulation.

Produces a suite of diagnostic figures in verification_plots/ that validate:
  1. Spin-2 shear field correctness (E/B decomposition, power recovery)
  2. Kaiser-Squires consistency (kappa <-> shear round-trip)
  3. Noise properties (pseudo-Cl bias subtraction)
  4. Visual field maps and whisker patterns

Run:  PYTHONPATH=. python tests/generate_verification_plots.py
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmo_lib.config import (
    GRID_SIZE,
    N_ELL_BINS,
    N_GAL_PER_PIX,
    OMEGA_M_TRUE,
    PIXEL_SCALE,
    SHAPE_NOISE,
    SIGMA_8_TRUE,
)
from cosmo_lib.classical import compute_eb_power, estimate_pseudo_cl
from cosmo_lib.power_spectrum import cl_model
from cosmo_lib.simulation import simulate_shear_catalog

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "verification_plots")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
print("Simulating shear field...")
key = jax.random.PRNGKey(42)
sim = simulate_shear_catalog(key)

kappa = sim["kappa"]
gamma1 = sim["gamma1"]
gamma2 = sim["gamma2"]
n = GRID_SIZE
delta_rad = jnp.deg2rad(PIXEL_SCALE / 60.0)
area = (n * delta_rad) ** 2

# Fourier-space quantities
freq = jnp.fft.fftfreq(n, d=delta_rad)
kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)
ell_sq = kx**2 + ky**2
ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
D_ell = D_ell.at[0, 0].set(0.0 + 0j)

kappa_fft = jnp.fft.fft2(kappa)
gamma_fft = jnp.fft.fft2(gamma1 + 1j * gamma2)

# E/B decomposition in Fourier space
psi = jnp.conj(D_ell) * gamma_fft
psi_neg = jnp.roll(jnp.flip(psi), (1, 1), axis=(0, 1))
kappa_E_fft = (psi + jnp.conj(psi_neg)) / 2.0
kappa_B_fft = (psi - jnp.conj(psi_neg)) / 2j

# Binned spectra
eb_data = compute_eb_power(gamma1, gamma2, n=n, delta=PIXEL_SCALE)

# Bin kappa power the same way
power_kappa_2d = jnp.abs(kappa_fft) ** 2 * area / n**2
ell_flat_np = np.array(ell2d.ravel())
power_kappa_flat = np.array(power_kappa_2d.ravel())
ell_min = 2.0 * np.pi / (n * float(delta_rad))
ell_max = np.pi / float(delta_rad)
bin_edges = np.exp(np.linspace(np.log(ell_min), np.log(ell_max), N_ELL_BINS + 1))

kappa_ell_centers = []
cl_kappa_list = []
for i in range(N_ELL_BINS):
    mask = (ell_flat_np >= bin_edges[i]) & (ell_flat_np < bin_edges[i + 1])
    mask[0] = False
    nm = mask.sum()
    if nm > 0:
        kappa_ell_centers.append(np.mean(ell_flat_np[mask]))
        cl_kappa_list.append(np.mean(power_kappa_flat[mask]))
    else:
        kappa_ell_centers.append(float(np.sqrt(bin_edges[i] * bin_edges[i + 1])))
        cl_kappa_list.append(0.0)

ell_kappa = np.array(kappa_ell_centers)
cl_kappa = np.array(cl_kappa_list)

# Theory curve
ell_smooth = np.logspace(np.log10(ell_min), np.log10(ell_max), 300)
cl_theory = np.array(cl_model(jnp.array(ell_smooth), OMEGA_M_TRUE, SIGMA_8_TRUE))

# Pseudo-Cl from noisy data
cl_data = estimate_pseudo_cl(
    sim["eps1"], sim["eps2"], n, PIXEL_SCALE, N_GAL_PER_PIX, SHAPE_NOISE
)


# ===================================================================
# PLOT 1: Power Spectrum Recovery
# Three-way comparison: theory C_l, kappa power, shear E-mode power
# ===================================================================
print("Plot 1: Power spectrum recovery...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.05})

eb_ell = np.array(eb_data["ell_bins"])
cl_E = np.array(eb_data["cl_E"])

ax1.loglog(ell_smooth, cl_theory, "k-", lw=2, label=r"Theory $C_\ell(\Omega_m, \sigma_8)$")
ax1.loglog(ell_kappa, cl_kappa, "C0o", ms=7, label=r"$|\tilde{\kappa}(\ell)|^2$ (input field)")
ax1.loglog(eb_ell, cl_E, "C2s", ms=6, mfc="none", mew=1.5,
           label=r"$C_\ell^{EE}$ from shear (E-mode)")
ax1.set_ylabel(r"$C_\ell$", fontsize=13)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title("Power Spectrum Recovery: Theory vs Kappa vs Shear E-mode",
              fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")

# Ratio panel — skip first bin (few modes, cosmic variance dominated)
valid = cl_kappa > 0
ratio = np.where(valid, cl_E / cl_kappa, np.nan)
# Plot only bins 1+ (skip the lowest-ell bin which has few modes)
ax2.semilogx(eb_ell[1:], ratio[1:], "C3o-", ms=5, lw=1.2)
ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
ax2.fill_between([ell_min, ell_max], 0.9, 1.1, alpha=0.1, color="green")
ax2.set_ylabel(r"$C_\ell^{EE} / C_\ell^{\kappa\kappa}$", fontsize=12)
ax2.set_xlabel(r"$\ell$", fontsize=13)
ax2.set_ylim(0.95, 1.05)
ax2.grid(True, alpha=0.3, which="both")
mean_ratio = np.nanmean(ratio[1:])
ax2.text(0.02, 0.92, f"Mean ratio (bins 1+) = {mean_ratio:.6f}\n10% band shown",
         transform=ax2.transAxes, fontsize=9, color="green", va="top")

plt.savefig(os.path.join(OUTDIR, "01_power_spectrum_recovery.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 2: E/B Mode Separation
# E-mode should match theory; B-mode should be zero (machine precision)
# ===================================================================
print("Plot 2: E/B mode separation...")
cl_B = np.array(eb_data["cl_B"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.05})

ax1.loglog(ell_smooth, cl_theory, "k-", lw=2, label=r"Theory $C_\ell$")
ax1.loglog(eb_ell, np.maximum(cl_E, 1e-30), "C2o", ms=6,
           label=r"$C_\ell^{EE}$ (E-mode)")
ax1.loglog(eb_ell, np.maximum(np.abs(cl_B), 1e-30), "C4^", ms=5,
           label=r"$|C_\ell^{BB}|$ (B-mode)")

# Mark the dynamic range
if np.any(cl_B != 0):
    max_ratio = np.max(cl_E) / np.max(np.abs(cl_B[cl_B != 0])) if np.any(cl_B != 0) else np.inf
    ax1.text(0.02, 0.05, f"E/B dynamic range: {max_ratio:.0e}",
             transform=ax1.transAxes, fontsize=10, color="C4",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="C4", alpha=0.8))

ax1.set_ylabel(r"$C_\ell$", fontsize=13)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title("E/B Mode Decomposition: B-mode Should Be Zero",
              fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")

# B/E ratio panel
b_over_e = np.abs(cl_B) / np.maximum(cl_E, 1e-30)
ax2.semilogy(eb_ell, np.maximum(b_over_e, 1e-20), "C4o-", ms=5, lw=1.2)
ax2.axhline(0.01, color="red", ls="--", lw=1, alpha=0.7, label="1% threshold")
ax2.set_ylabel(r"$|C_\ell^{BB}| / C_\ell^{EE}$", fontsize=12)
ax2.set_xlabel(r"$\ell$", fontsize=13)
ax2.set_ylim(1e-20, 1)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which="both")

plt.savefig(os.path.join(OUTDIR, "02_eb_mode_separation.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 3: Field Maps (kappa, gamma1, gamma2)
# Visual check that gamma1/gamma2 show spin-2 structure around kappa peaks
# ===================================================================
print("Plot 3: Field maps...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

kappa_np = np.array(kappa)
g1_np = np.array(gamma1)
g2_np = np.array(gamma2)

vmax_k = np.max(np.abs(kappa_np))
vmax_g = max(np.max(np.abs(g1_np)), np.max(np.abs(g2_np)))

im0 = axes[0].imshow(kappa_np, origin="lower", cmap="RdBu_r",
                      vmin=-vmax_k, vmax=vmax_k)
axes[0].set_title(r"$\kappa$ (convergence)", fontsize=12, fontweight="bold")
plt.colorbar(im0, ax=axes[0], shrink=0.85)

im1 = axes[1].imshow(g1_np, origin="lower", cmap="RdBu_r",
                      vmin=-vmax_g, vmax=vmax_g)
axes[1].set_title(r"$\gamma_1$ (shear)", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes[1], shrink=0.85)

im2 = axes[2].imshow(g2_np, origin="lower", cmap="RdBu_r",
                      vmin=-vmax_g, vmax=vmax_g)
axes[2].set_title(r"$\gamma_2$ (shear)", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes[2], shrink=0.85)

for ax in axes:
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")

fig.suptitle(f"Simulated Fields ({n}x{n} grid, "
             r"$\Omega_m$=" + f"{OMEGA_M_TRUE}, "
             r"$\sigma_8$=" + f"{SIGMA_8_TRUE})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "03_field_maps.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 4: Shear Whisker Map with Convergence Contours
# Whiskers should show tangential alignment around kappa peaks
# ===================================================================
print("Plot 4: Shear whiskers...")
fig, ax = plt.subplots(figsize=(7, 7))

im = ax.imshow(kappa_np, origin="lower", cmap="RdBu_r", alpha=0.5,
               extent=[0, n, 0, n])
plt.colorbar(im, ax=ax, label=r"$\kappa$", shrink=0.8)

# Overlay contours
levels = np.linspace(-vmax_k, vmax_k, 8)
ax.contour(np.arange(n) + 0.5, np.arange(n) + 0.5, kappa_np,
           levels=levels, colors="gray", linewidths=0.5, alpha=0.5)

# Whiskers
mag = np.sqrt(g1_np**2 + g2_np**2)
angle = 0.5 * np.arctan2(g2_np, g1_np)
scale = 0.8 / np.max(mag) if np.max(mag) > 0 else 1.0

for i in range(n):
    for j in range(n):
        dx = mag[i, j] * scale * np.cos(angle[i, j])
        dy = mag[i, j] * scale * np.sin(angle[i, j])
        cx, cy = j + 0.5, i + 0.5
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                "k-", lw=0.9, alpha=0.85)

ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_aspect("equal")
ax.set_title(r"Shear Whiskers on $\kappa$ (tangential pattern expected)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "04_shear_whiskers.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 5: Cross-Power Spectrum gamma1 x gamma2
# Non-zero cross-power confirms spin-2 coupling (not independent)
# ===================================================================
print("Plot 5: Cross-power spectrum...")
g1_fft = jnp.fft.fft2(gamma1)
g2_fft = jnp.fft.fft2(gamma2)

# Cross-power and auto-power in 2D
cross_2d = jnp.real(g1_fft * jnp.conj(g2_fft)) * area / n**2
auto1_2d = jnp.abs(g1_fft) ** 2 * area / n**2
auto2_2d = jnp.abs(g2_fft) ** 2 * area / n**2

# Bin cross-power
cross_flat = np.array(cross_2d.ravel())
auto1_flat = np.array(auto1_2d.ravel())
auto2_flat = np.array(auto2_2d.ravel())

cross_binned = []
auto1_binned = []
auto2_binned = []
for i in range(N_ELL_BINS):
    mask = (ell_flat_np >= bin_edges[i]) & (ell_flat_np < bin_edges[i + 1])
    mask[0] = False
    nm = mask.sum()
    if nm > 0:
        cross_binned.append(np.mean(cross_flat[mask]))
        auto1_binned.append(np.mean(auto1_flat[mask]))
        auto2_binned.append(np.mean(auto2_flat[mask]))
    else:
        cross_binned.append(0.0)
        auto1_binned.append(0.0)
        auto2_binned.append(0.0)

cross_binned = np.array(cross_binned)
auto1_binned = np.array(auto1_binned)
auto2_binned = np.array(auto2_binned)
denom = np.sqrt(auto1_binned * auto2_binned)
r_coeff = np.where(denom > 0, cross_binned / denom, 0.0)

# Theory prediction for cross-correlation coefficient
# For spin-2 shear from pure E-mode: gamma = D * kappa
# gamma1 = Re(gamma), gamma2 = Im(gamma)
# Cross-power = Re(D) * Im(D) * |kappa|^2 = (lx^2-ly^2)(2*lx*ly) / l^4 * C_l
# The angle-averaged cross-correlation coefficient is non-trivial

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[1, 1],
                                sharex=True, gridspec_kw={"hspace": 0.1})

ax1.semilogx(eb_ell, cross_binned, "C5o-", ms=5, lw=1.2, label=r"$\langle \tilde{\gamma}_1 \tilde{\gamma}_2^* \rangle$")
ax1.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax1.set_ylabel(r"Cross-power $C_\ell^{\gamma_1 \gamma_2}$", fontsize=12)
ax1.legend(fontsize=10)
ax1.set_title(r"$\gamma_1$-$\gamma_2$ Cross-Power (non-zero $\Rightarrow$ spin-2 coupling)",
              fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")

ax2.semilogx(eb_ell, r_coeff, "C3o-", ms=5, lw=1.2)
ax2.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax2.set_ylabel(r"$r(\ell) = C_\ell^{12} / \sqrt{C_\ell^{11} C_\ell^{22}}$", fontsize=12)
ax2.set_xlabel(r"$\ell$", fontsize=13)
ax2.set_ylim(-1, 1)
ax2.grid(True, alpha=0.3, which="both")
r_valid = r_coeff[np.isfinite(r_coeff)]
mean_r = float(np.mean(np.abs(r_valid))) if len(r_valid) > 0 else 0.0
ax2.text(0.02, 0.92, f"Mean |r| = {mean_r:.3f}", transform=ax2.transAxes,
         fontsize=10, va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="C3", alpha=0.8))

plt.savefig(os.path.join(OUTDIR, "05_cross_power.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 6: Noisy Pseudo-Cl Recovery
# Shows that noise-subtracted pseudo-Cl from galaxy catalog matches theory
# ===================================================================
print("Plot 6: Pseudo-Cl recovery from noisy data...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.05})

cl_ell = np.array(cl_data["ell_bins"])
cl_hat = np.array(cl_data["cl_hat"])
cl_noise = float(cl_data["cl_noise"])

ax1.loglog(ell_smooth, cl_theory, "k-", lw=2, label=r"Theory $C_\ell$")
# Split positive and negative pseudo-Cl for proper log-scale display
pos_mask = cl_hat > 0
neg_mask = cl_hat <= 0
if np.any(pos_mask):
    ax1.loglog(cl_ell[pos_mask], cl_hat[pos_mask], "C0o", ms=6,
               label=r"Pseudo-$C_\ell$ (noise-subtracted)")
if np.any(neg_mask):
    ax1.loglog(cl_ell[neg_mask], np.abs(cl_hat[neg_mask]) + 1e-20, "C0v", ms=6,
               mfc="none", mew=1.5, label=r"Negative $C_\ell$ (|value|)")
ax1.loglog(eb_ell, cl_E, "C2s", ms=5, mfc="none", mew=1.5, alpha=0.7,
           label=r"True $C_\ell^{EE}$ (noiseless)")
ax1.axhline(cl_noise, color="C3", ls="--", lw=1.5, alpha=0.7,
            label=f"Noise bias $N_\\ell$ = {cl_noise:.2e}")
ax1.set_ylabel(r"$C_\ell$", fontsize=13)
ax1.legend(fontsize=9, loc="upper right")
ax1.set_title("Pseudo-Cl Recovery from Noisy Galaxy Catalog",
              fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")

# Fractional residual vs noiseless truth — only where signal > noise
cl_theory_at_bins = np.array(cl_model(jnp.array(cl_ell), OMEGA_M_TRUE, SIGMA_8_TRUE))
signal_dominated = cl_theory_at_bins > cl_noise
frac_res = np.where(signal_dominated, (cl_hat - cl_theory_at_bins) / cl_theory_at_bins, np.nan)
ax2.semilogx(cl_ell[signal_dominated], frac_res[signal_dominated], "C0o-", ms=5, lw=1.2)
if np.any(~signal_dominated):
    ax2.semilogx(cl_ell[~signal_dominated], np.zeros(np.sum(~signal_dominated)),
                 "C3x", ms=7, mew=1.5, alpha=0.5, label="Noise-dominated (excluded)")
    ax2.legend(fontsize=8)
ax2.axhline(0, color="k", ls="--", lw=1, alpha=0.7)
ax2.fill_between([ell_min, ell_max], -0.5, 0.5, alpha=0.08, color="blue",
                 label="50% band")
ax2.set_ylabel(r"$(C_\ell^{\rm obs} - C_\ell^{\rm th}) / C_\ell^{\rm th}$", fontsize=11)
ax2.set_xlabel(r"$\ell$", fontsize=13)
ax2.set_ylim(-2, 2)
ax2.grid(True, alpha=0.3, which="both")

plt.savefig(os.path.join(OUTDIR, "06_pseudo_cl_recovery.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ===================================================================
# PLOT 7: Summary Statistics Dashboard
# Single-panel summary of key numbers
# ===================================================================
print("Plot 7: Summary dashboard...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

# Compute summary statistics (skip first bin for power ratio — few modes)
ratio_valid = ratio[1:]  # skip lowest-ell bin
ratio_valid = ratio_valid[np.isfinite(ratio_valid)]
power_ratio_mean = float(np.mean(ratio_valid)) if len(ratio_valid) > 0 else float("nan")
power_ratio_std = float(np.std(ratio_valid)) if len(ratio_valid) > 0 else float("nan")
be_ratio = float(np.mean(np.abs(cl_B)) / np.mean(cl_E)) if np.mean(cl_E) > 0 else float("inf")
cross_corr_mean = mean_r  # already computed above (NaN-safe)
kappa_rms = float(jnp.std(kappa))
gamma_rms = float(jnp.std(gamma1))
g1g2_ratio = float(jnp.std(gamma1) / jnp.std(gamma2))

checks = [
    ("E-mode power / kappa power (mean)", f"{power_ratio_mean:.4f} +/- {power_ratio_std:.4f}",
     "~1.0", 0.9 < power_ratio_mean < 1.1),
    ("B/E power ratio", f"{be_ratio:.2e}", "< 0.01", be_ratio < 0.01),
    (r"gamma1-gamma2 |cross-correlation|", f"{cross_corr_mean:.3f}", "> 0.1", cross_corr_mean > 0.1),
    ("kappa rms", f"{kappa_rms:.6f}", "> 0", kappa_rms > 0),
    ("gamma rms", f"{gamma_rms:.6f}", "> 0", gamma_rms > 0),
    ("gamma1_rms / gamma2_rms", f"{g1g2_ratio:.3f}", "~1.0", 0.5 < g1g2_ratio < 2.0),
]

header = f"Spin-2 Shear Verification Summary  ({n}x{n} grid, seed=42)"
ax.text(0.5, 0.95, header, transform=ax.transAxes, fontsize=14,
        fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8e8e8", ec="black"))

row_y = 0.82
ax.text(0.02, row_y, "Check", fontsize=11, fontweight="bold", va="top")
ax.text(0.45, row_y, "Value", fontsize=11, fontweight="bold", va="top")
ax.text(0.70, row_y, "Expected", fontsize=11, fontweight="bold", va="top")
ax.text(0.88, row_y, "Status", fontsize=11, fontweight="bold", va="top")
ax.axhline(y=0.79, xmin=0.02, xmax=0.98, color="black", lw=0.8)

for i, (name, value, expected, passed) in enumerate(checks):
    y = 0.75 - i * 0.10
    status = "PASS" if passed else "FAIL"
    color = "#2d7d2d" if passed else "#cc0000"
    ax.text(0.02, y, name, fontsize=10, va="top")
    ax.text(0.45, y, value, fontsize=10, va="top", family="monospace")
    ax.text(0.70, y, expected, fontsize=10, va="top")
    ax.text(0.88, y, status, fontsize=11, fontweight="bold", va="top", color=color)

all_pass = all(c[3] for c in checks)
footer_color = "#2d7d2d" if all_pass else "#cc0000"
footer_text = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
ax.text(0.5, 0.08, footer_text, transform=ax.transAxes, fontsize=14,
        fontweight="bold", ha="center", va="top", color=footer_color,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=footer_color, lw=2))

plt.savefig(os.path.join(OUTDIR, "07_summary_dashboard.png"),
            dpi=150, bbox_inches="tight")
plt.close()


print(f"\nAll verification plots saved to {OUTDIR}/")
print("Files:")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(".png"):
        print(f"  {f}")
