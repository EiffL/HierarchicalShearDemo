# Spin-2 Shear Simulation Fix + Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the shear field simulation to produce a proper spin-2 field, add E/B power spectrum diagnostics, and add a shear whisker plot — so the pseudo-Cl pipeline does correct end-to-end inference.

**Architecture:** The convergence field kappa is already generated correctly as a real Gaussian field. The fix is to FFT this real field (getting Hermitian Fourier coefficients) before applying Kaiser-Squires, producing a proper spin-2 shear. New E/B decomposition and whisker plot functions validate the result.

**Tech Stack:** JAX, NumPy, matplotlib. Run everything with `.venv/bin/python`. No pytest — verification uses inline assertion scripts.

**Design doc:** `docs/plans/2026-02-20-spin2-shear-design.md`

---

### Task 1: Fix spin-2 shear generation in simulation.py

**Files:**
- Modify: `cosmo_lib/simulation.py:88-105`

**Step 1: Write a verification script that exposes the current bug**

Create `tests/test_spin2.py`:

```python
"""Verify spin-2 shear field properties."""
import jax
import jax.numpy as jnp
from cosmo_lib.simulation import simulate_shear_catalog

key = jax.random.PRNGKey(42)
sim = simulate_shear_catalog(key)

gamma1, gamma2 = sim["gamma1"], sim["gamma2"]
kappa = sim["kappa"]

# 1. Check power spectrum recovery: pseudo-Cl of true shear should match input C_l
# Compute |D* * FFT(gamma1 + i*gamma2)|^2 and compare to |FFT(kappa)|^2
n = gamma1.shape[0]
delta_rad = jnp.deg2rad(2.0 / 60.0)
area = (n * delta_rad) ** 2

gamma_fft = jnp.fft.fft2(gamma1 + 1j * gamma2)
freq = jnp.fft.fftfreq(n, d=delta_rad)
kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
ell_sq = kx**2 + ky**2
ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
D_ell = D_ell.at[0, 0].set(0.0 + 0j)

kappa_E_fft = jnp.conj(D_ell) * gamma_fft
power_from_shear = jnp.mean(jnp.abs(kappa_E_fft[1:, 1:]) ** 2) * area / n**2

kappa_fft = jnp.fft.fft2(kappa)
power_from_kappa = jnp.mean(jnp.abs(kappa_fft[1:, 1:]) ** 2) * area / n**2

ratio = float(power_from_shear / power_from_kappa)
print(f"Power ratio (shear E-mode / kappa): {ratio:.3f}")
print(f"  Expected ~1.0 for correct spin-2, ~2.0 for broken")
assert 0.8 < ratio < 1.2, f"FAIL: ratio={ratio:.3f}, expected ~1.0"

# 2. Check B-mode is zero for true (noiseless) shear
psi = jnp.conj(D_ell) * gamma_fft
# psi(-l): reflect indices
psi_neg = jnp.roll(jnp.flip(psi), (1, 1), axis=(0, 1))
kappa_B_fft = (psi - jnp.conj(psi_neg)) / 2j
kappa_B_fft = kappa_B_fft.at[0, 0].set(0.0)

power_B = jnp.mean(jnp.abs(kappa_B_fft[1:, 1:]) ** 2) * area / n**2
power_E_from_decomp = jnp.mean(jnp.abs(((psi + jnp.conj(psi_neg)) / 2)[1:, 1:]) ** 2) * area / n**2

b_over_e = float(power_B / power_E_from_decomp)
print(f"B/E power ratio: {b_over_e:.6f}")
print(f"  Expected ~0 for correct spin-2")
assert b_over_e < 0.01, f"FAIL: B/E={b_over_e:.6f}, expected ~0"

# 3. Check gamma1/gamma2 are NOT independent (spin-2 correlation)
# For a spin-2 field, cross-power FFT(gamma1)*conj(FFT(gamma2)) should be nonzero
g1_fft = jnp.fft.fft2(gamma1)
g2_fft = jnp.fft.fft2(gamma2)
cross = jnp.mean(jnp.abs(g1_fft * jnp.conj(g2_fft)))
auto1 = jnp.mean(jnp.abs(g1_fft) ** 2)
auto2 = jnp.mean(jnp.abs(g2_fft) ** 2)
cross_corr = float(cross / jnp.sqrt(auto1 * auto2))
print(f"gamma1-gamma2 cross-correlation: {cross_corr:.3f}")
print(f"  Expected >0.1 for spin-2, ~0 for independent")

print("\nAll checks passed!")
```

**Step 2: Run the verification script to confirm it fails with the current code**

Run: `.venv/bin/python tests/test_spin2.py`
Expected: FAIL on the power ratio check (~2.0 instead of ~1.0)

**Step 3: Apply the fix in simulation.py**

In `cosmo_lib/simulation.py`, replace lines 92-105:

```python
    # Enforce Hermitian symmetry for real output
    kappa = jnp.real(jnp.fft.ifft2(kappa_fft))

    # Kaiser-Squires: gamma_tilde(l) = D(l) * kappa_tilde(l)
    # IMPORTANT: use FFT of the real kappa field (Hermitian) so that
    # the resulting shear is a proper spin-2 field with correlated
    # gamma1/gamma2 components.
    kappa_fft_hermitian = jnp.fft.fft2(kappa)

    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    gamma_fft = D_ell * kappa_fft_hermitian
    gamma = jnp.fft.ifft2(gamma_fft)
    gamma1 = jnp.real(gamma)
    gamma2 = jnp.imag(gamma)
```

**Step 4: Run the verification script to confirm it passes**

Run: `.venv/bin/python tests/test_spin2.py`
Expected: All three checks pass (ratio ~1.0, B/E ~0, cross-correlation >0.1)

**Step 5: Commit**

```bash
git add cosmo_lib/simulation.py tests/test_spin2.py
git commit -m "fix: use Hermitian kappa FFT for proper spin-2 shear generation"
```

---

### Task 2: Add E/B power spectrum decomposition to classical.py

**Files:**
- Modify: `cosmo_lib/classical.py` (add function after `estimate_pseudo_cl`)

**Step 1: Add `compute_eb_power()` function**

Add after the `estimate_pseudo_cl` function in `cosmo_lib/classical.py`:

```python
def compute_eb_power(
    gamma1: jnp.ndarray,
    gamma2: jnp.ndarray,
    n: int,
    delta: float,
    n_bins: int = N_ELL_BINS,
) -> dict:
    """Compute binned E-mode and B-mode power spectra from a shear field.

    Uses the spin-2 E/B decomposition:
        psi(l) = D*(l) * FFT(gamma1 + i*gamma2)
        kappa_E(l) = [psi(l) + conj(psi(-l))] / 2
        kappa_B(l) = [psi(l) - conj(psi(-l))] / (2i)

    Args:
        gamma1: (n, n) shear component 1.
        gamma2: (n, n) shear component 2.
        n: Grid size.
        delta: Pixel scale in arcmin.
        n_bins: Number of ell bins.

    Returns:
        Dictionary with keys:
          - ell_bins: (n_bins,) bin centers.
          - cl_E: (n_bins,) binned E-mode power.
          - cl_B: (n_bins,) binned B-mode power.
    """
    delta_rad = jnp.deg2rad(delta / 60.0)
    area = (n * delta_rad) ** 2

    # Spin-2 FFT
    gamma_fft = jnp.fft.fft2(gamma1 + 1j * gamma2)

    # Kaiser-Squires kernel
    freq = jnp.fft.fftfreq(n, d=delta_rad)
    kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
    ell_sq = kx**2 + ky**2
    ell_sq_safe = jnp.where(ell_sq > 0, ell_sq, 1.0)
    D_ell = (kx**2 - ky**2 + 2j * kx * ky) / ell_sq_safe
    D_ell = D_ell.at[0, 0].set(0.0 + 0j)

    # E/B decomposition
    psi = jnp.conj(D_ell) * gamma_fft
    psi_neg = jnp.roll(jnp.flip(psi), (1, 1), axis=(0, 1))
    kappa_E_fft = (psi + jnp.conj(psi_neg)) / 2.0
    kappa_B_fft = (psi - jnp.conj(psi_neg)) / 2j

    # Power spectra
    power_E = jnp.abs(kappa_E_fft) ** 2 * area / n**2
    power_B = jnp.abs(kappa_B_fft) ** 2 * area / n**2

    # Ell grid
    ell2d = 2.0 * jnp.pi * jnp.sqrt(kx**2 + ky**2)

    # Bin in ell (same binning as estimate_pseudo_cl)
    ell_flat = np.array(ell2d.ravel())
    power_E_flat = np.array(power_E.ravel())
    power_B_flat = np.array(power_B.ravel())

    ell_min = 2.0 * np.pi / (n * delta_rad)
    ell_max = np.pi / delta_rad
    bin_edges = np.exp(np.linspace(np.log(ell_min), np.log(ell_max), n_bins + 1))

    ell_centers = []
    cl_E_list = []
    cl_B_list = []

    for i in range(n_bins):
        mask = (ell_flat >= bin_edges[i]) & (ell_flat < bin_edges[i + 1])
        mask[0] = False
        nm = mask.sum()
        if nm > 0:
            ell_centers.append(np.mean(ell_flat[mask]))
            cl_E_list.append(np.mean(power_E_flat[mask]))
            cl_B_list.append(np.mean(power_B_flat[mask]))
        else:
            ell_centers.append(float(np.sqrt(bin_edges[i] * bin_edges[i + 1])))
            cl_E_list.append(0.0)
            cl_B_list.append(0.0)

    return {
        "ell_bins": jnp.array(ell_centers),
        "cl_E": jnp.array(cl_E_list),
        "cl_B": jnp.array(cl_B_list),
    }
```

**Step 2: Verify the E/B decomposition works with the fixed simulation**

Add to `tests/test_spin2.py` (append):

```python
# Test compute_eb_power function
from cosmo_lib.classical import compute_eb_power
eb = compute_eb_power(gamma1, gamma2, n=32, delta=2.0)
mean_E = float(jnp.mean(eb["cl_E"]))
mean_B = float(jnp.mean(eb["cl_B"]))
print(f"\ncompute_eb_power check:")
print(f"  Mean E-mode power: {mean_E:.2e}")
print(f"  Mean B-mode power: {mean_B:.2e}")
print(f"  B/E ratio: {mean_B/mean_E:.6f}")
assert mean_B / mean_E < 0.01, f"FAIL: B/E too high"
print("  compute_eb_power check passed!")
```

Run: `.venv/bin/python tests/test_spin2.py`
Expected: All checks pass, B/E ~ 0.

**Step 3: Commit**

```bash
git add cosmo_lib/classical.py tests/test_spin2.py
git commit -m "feat: add E/B power spectrum decomposition"
```

---

### Task 3: Add B-mode to power spectrum plot

**Files:**
- Modify: `cosmo_lib/plotting.py` — update `plot_power_spectrum` signature and body

**Step 1: Update `plot_power_spectrum` to accept and plot E/B data**

In `cosmo_lib/plotting.py`, replace the `plot_power_spectrum` function:

```python
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
```

**Step 2: Commit**

```bash
git add cosmo_lib/plotting.py
git commit -m "feat: add E/B mode display to power spectrum plot"
```

---

### Task 4: Add shear whisker plot

**Files:**
- Modify: `cosmo_lib/plotting.py` — add `plot_shear_whiskers` function

**Step 1: Add the whisker plot function**

Add to `cosmo_lib/plotting.py`:

```python
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
    plt.savefig("cosmo_shear_whiskers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cosmo_shear_whiskers.png")
```

**Step 2: Commit**

```bash
git add cosmo_lib/plotting.py
git commit -m "feat: add shear whisker plot overlaid on convergence"
```

---

### Task 5: Add B-mode verification check

**Files:**
- Modify: `cosmo_lib/verification.py` — add B-mode check to `run_verification`

**Step 1: Update run_verification to accept and check E/B data**

In `cosmo_lib/verification.py`, add an `eb_data` parameter and a B-mode check section.

Add to the function signature:
```python
def run_verification(
    classical_samples: dict,
    gibbs_chain: GibbsOutput,
    gibbs_state: GibbsState,
    eb_data: dict | None = None,
    n_burn: int = N_GIBBS_BURN,
    n_total: int = N_GIBBS,
) -> None:
```

Add before the final `print("=" * 70)`:

```python
    if eb_data is not None:
        print("\n4. B-mode consistency (true shear field):")
        cl_E = np.array(eb_data["cl_E"])
        cl_B = np.array(eb_data["cl_B"])
        mean_E = np.mean(cl_E[cl_E > 0])
        mean_B = np.mean(np.abs(cl_B))
        ratio = mean_B / mean_E if mean_E > 0 else float("inf")
        status = "PASS" if ratio < 0.05 else "WARN"
        print(f"  [{status}] Mean |B|/E ratio: {ratio:.4f} (expect < 0.05)")
```

**Step 2: Commit**

```bash
git add cosmo_lib/verification.py
git commit -m "feat: add B-mode consistency check to verification"
```

---

### Task 6: Wire everything into the main script

**Files:**
- Modify: `cosmology_inference_demo.py`

**Step 1: Add imports and wire in E/B diagnostic and whisker plot**

Add to imports:
```python
from cosmo_lib.classical import estimate_pseudo_cl, compute_eb_power, run_classical_pipeline
```

Add to plotting imports:
```python
from cosmo_lib.plotting import (
    plot_corner_comparison,
    plot_power_spectrum,
    plot_shear_maps,
    plot_shear_whiskers,
)
```

After the simulation block (after `print(f"  Simulation done in...")`), add:
```python
    # E/B diagnostic on true (noiseless) shear
    eb_data = compute_eb_power(
        sim["gamma1"], sim["gamma2"], GRID_SIZE, PIXEL_SCALE
    )
    print(f"  E/B check: mean B/E = {float(jnp.mean(jnp.abs(eb_data['cl_B'])) / jnp.mean(eb_data['cl_E'])):.6f}")
```

Update the `plot_power_spectrum` call to pass `eb_data`:
```python
    plot_power_spectrum(cl_data, sim["ell2d"], eb_data=eb_data)
```

Add the whisker plot call after the other plots:
```python
    plot_shear_whiskers(sim)
```

Update the `run_verification` call:
```python
    run_verification(classical_samples, gibbs_chain, gibbs_state, eb_data=eb_data)
```

**Step 2: Run the full demo end-to-end**

Run: `.venv/bin/python cosmology_inference_demo.py`
Expected:
- kappa rms and gamma rms should be similar order of magnitude
- E/B check should show B/E ~ 0
- Power spectrum plot shows true E-mode matching C_l, B-mode near zero
- Whisker plot shows tangential shear around kappa peaks
- Pseudo-Cl recovery is unbiased (no factor-of-2)
- NUTS posterior should contain the truth

**Step 3: Commit**

```bash
git add cosmology_inference_demo.py
git commit -m "feat: wire in E/B diagnostic and whisker plot to main demo"
```

---

### Task 7: Final verification — run full pipeline and check outputs

**Step 1: Run the full pipeline**

Run: `.venv/bin/python cosmology_inference_demo.py`

Check:
- All verification checks pass (truth in 95% CI for classical pipeline)
- B-mode ratio < 0.05
- Power spectrum plot shows E-mode data points matching true C_l curve
- Shear whisker plot saved

**Step 2: Run the spin-2 test suite**

Run: `.venv/bin/python tests/test_spin2.py`
Expected: All assertions pass.

**Step 3: Final commit with any remaining tweaks**

```bash
git add -A
git commit -m "chore: final verification of spin-2 shear pipeline"
```
