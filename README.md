# Hierarchical Shear Inference Demo

Comparing two approaches to cosmological parameter inference from weak gravitational lensing shear data: a classical **pseudo-C&#8467;** power spectrum pipeline and a **joint field-level NUTS** sampler that infers the convergence field and cosmological parameters simultaneously.

Both pipelines operate on the same simulated galaxy shear catalog and produce posterior constraints on the matter density parameter &Omega;<sub>m</sub> and the fluctuation amplitude &sigma;<sub>8</sub>, as well as the derived combination S<sub>8</sub> = &sigma;<sub>8</sub>&radic;(&Omega;<sub>m</sub>/0.3).

<p align="center">
  <img src="plots/cosmo_comparison_corner.png" width="85%"/>
</p>

## Overview

Weak gravitational lensing distorts the shapes of distant galaxies by a small amount (~1%), encoding information about the intervening matter distribution. Measuring these distortions across many galaxies constrains the cosmological parameters that govern the growth of structure.

This demo implements the full pipeline from simulation to posterior inference:

```mermaid
flowchart LR
    subgraph Simulation
        A[Draw Gaussian κ field] --> B["Kaiser-Squires: κ → (γ₁, γ₂)"]
        B --> C[Add shape noise → galaxy catalog]
    end
    subgraph Pipeline A: Pseudo-Cℓ
        C --> D[Pixel-average ellipticities]
        D --> E[FFT → E/B decomposition]
        E --> F[Bin E-mode power in ℓ]
        F --> G["NUTS sampling of (S₈, Ωₘ)"]
    end
    subgraph Pipeline B: Field-level NUTS
        C --> H[Per-pixel interim posteriors]
        H --> I["Joint NUTS over (z, S₈, Ωₘ)"]
        I --> J[Posterior mean κ → shear reconstruction]
    end
    G --> M[Compare posteriors]
    J --> M
```

## The Physics

### Shear as a Spin-2 Field

Gravitational lensing shear (&gamma;<sub>1</sub>, &gamma;<sub>2</sub>) is a **spin-2** quantity: the two components are not independent but are related to the scalar convergence &kappa; through the Kaiser-Squires relation in Fourier space:

$$\tilde{\gamma}(\boldsymbol{\ell}) = D(\boldsymbol{\ell}) \, \tilde{\kappa}(\boldsymbol{\ell}), \qquad D(\boldsymbol{\ell}) = \frac{\ell_x^2 - \ell_y^2 + 2i\,\ell_x \ell_y}{|\boldsymbol{\ell}|^2}$$

This ensures that a pure convergence field produces **only E-mode** shear (no B-modes), analogous to the curl-free property of a gradient field. The simulation generates shear correctly via this relation, and B-mode power serves as a null test.

<p align="center">
  <img src="plots/cosmo_shear_whiskers.png" width="50%"/>
</p>

### Power Spectrum Model

We use a parametric angular power spectrum that captures the essential S<sub>8</sub> degeneracy:

$$C_\ell = A_0 \, \sigma_8^2 \left(\frac{\Omega_m}{0.3}\right)^2 \left(\frac{\ell}{100}\right)^{-1} \frac{1}{1 + (\ell/1000)^2}$$

This is a simplified model (not a full Boltzmann code), but it exhibits the correct scaling with cosmological parameters and the characteristic S<sub>8</sub> degeneracy direction in the (&Omega;<sub>m</sub>, &sigma;<sub>8</sub>) plane.

## Pipeline A: Pseudo-C&#8467; + NUTS

The classical approach compresses the shear data into a binned power spectrum, then performs Bayesian inference:

```mermaid
flowchart TD
    A["Galaxy ellipticities ε(x)"] --> B["Pixel-average → γ̄(x)"]
    B --> C["FFT: γ̃(ℓ) = FFT(γ̄₁ + iγ̄₂)"]
    C --> D["E/B separation via ψ(ℓ) = D*(ℓ)·γ̃(ℓ)"]
    D --> E["κ̃_E(ℓ) = [ψ(ℓ) + conj(ψ(−ℓ))] / 2"]
    E --> F["Power: P(ℓ) = |κ̃_E|² · area / n²"]
    F --> G["Bin in ℓ, subtract noise bias N_ℓ"]
    G --> H["Gaussian likelihood: ĈL_b ~ N(C_b(θ), 2(C_b+N)²/ν_b)"]
    H --> I["NUTS → posterior on (S₈, Ωₘ)"]
```

**Key features:**
- Proper E/B decomposition isolates the lensing signal from potential systematics
- Bins with fewer than 5 Fourier modes are excluded (unreliable at low &#8467;)
- Parametrized as (S<sub>8</sub>, &Omega;<sub>m</sub>) with uniform priors to avoid prior-induced bias on S<sub>8</sub>
- Uses `target_accept_prob=0.9` for zero-divergence HMC sampling along the S<sub>8</sub> degeneracy

<p align="center">
  <img src="plots/cosmo_comparison_power_spectrum.png" width="65%"/>
</p>

## Pipeline B: Field-level NUTS

The field-level approach jointly samples the convergence field and cosmological parameters using a **non-centered parameterization** that is amenable to gradient-based NUTS sampling in ~1000 dimensions.

### From interim posteriors to field-level inference

The key insight from [Schneider et al. (2015)](https://doi.org/10.1088/0004-637X/807/1/87) is that per-patch (per-pixel) posteriors on shear can serve as **sufficient data summaries** for hierarchical cosmological inference. In this demo, each pixel has $n_\mathrm{gal}$ observed ellipticities $\varepsilon_i = \gamma_p + n_i$ with i.i.d. noise $n_i \sim \mathcal{N}(0, \sigma_\varepsilon^2)$. The analytic interim posterior on the shear in pixel $p$ is:

$$\gamma_p \mid \{\varepsilon_i\} \sim \mathcal{N}\!\left(\bar{\varepsilon}_p,\; \sigma_\mathrm{patch}^2\right), \qquad \sigma_\mathrm{patch} = \frac{\sigma_\varepsilon}{\sqrt{n_\mathrm{gal}}}$$

where $\bar{\varepsilon}_p$ is the per-pixel mean ellipticity. This Gaussian interim posterior is the likelihood for the field-level stage: the pixel-averaged data $\bar{\varepsilon}_p$ are sufficient statistics, and the field-level model conditions on them directly.

### Non-centered generative model

Rather than sampling the convergence field $\kappa$ directly (which would be strongly correlated with the cosmological parameters through the power spectrum), we use a **non-centered parameterization** with a white-noise latent field $z$:

$$z_{ij} \sim \mathcal{N}(0, 1) \quad \text{(i.i.d. standard normal on the } n \times n \text{ grid)}$$

$$\tilde{\kappa}(\boldsymbol{\ell}) = \sqrt{\frac{C_\ell(\Omega_m, \sigma_8)}{\mathcal{A}}} \; \tilde{z}(\boldsymbol{\ell})$$

$$\tilde{\gamma}(\boldsymbol{\ell}) = D(\boldsymbol{\ell}) \, \tilde{\kappa}(\boldsymbol{\ell})$$

$$\bar{\varepsilon}_p \sim \mathcal{N}\!\left(\gamma_p,\; \sigma_\mathrm{patch}^2\right)$$

where $\tilde{z} = \mathrm{FFT}(z)$, $\mathcal{A} = (n \, \Delta)^2$ is the survey area, and $\gamma_p = \mathrm{IFFT}(\tilde{\gamma})$ evaluated at pixel $p$. The normalization $\sqrt{C_\ell / \mathcal{A}}$ accounts for the fact that $|\tilde{z}(\boldsymbol{\ell})|^2$ has expectation $n^2$ (from the unnormalized FFT of $n^2$ i.i.d. unit normals), producing the correct field power $\mathbb{E}[|\tilde{\kappa}(\boldsymbol{\ell})|^2] = C_\ell \, n^2 / \mathcal{A}$.

The total parameter space is $n^2 + 2 = 1026$ dimensions (for $n = 32$). The non-centered parameterization decouples the prior on $z$ from the cosmological parameters, making the posterior geometry more amenable to NUTS. Cosmology enters only through the deterministic coloring $\sqrt{C_\ell}$, so the sampler can efficiently adapt its mass matrix to the different scales of the $z$ field and the cosmological parameters.

### NUTS configuration

The joint sampler uses NumPyro's NUTS with:
- `target_accept_prob=0.8` (slightly lower than the pseudo-C&#8467; pipeline due to the higher dimensionality)
- `max_tree_depth=8` (sufficient for the moderate correlations in the non-centered parameterization)
- Same (S<sub>8</sub>, &Omega;<sub>m</sub>) uniform priors as Pipeline A, with &sigma;<sub>8</sub> derived as S<sub>8</sub> / &radic;(&Omega;<sub>m</sub>/0.3)

## Shear Field Reconstruction

The field-level sampler produces **posterior samples of the convergence field** $\kappa$ as a byproduct. The posterior mean $\langle \kappa \rangle$ is then mapped to shear via Kaiser-Squires, providing an optimally denoised reconstruction:

<p align="center">
  <img src="plots/cosmo_comparison_shear_maps.png" width="85%"/>
</p>

## Running

```bash
.venv/bin/python cosmology_inference_demo.py
```

The virtualenv (symlinked to the parent SHINE project) contains JAX, NumPyro, matplotlib, and scipy. The demo takes ~1 minute on a modern CPU.

**Output:** plots saved to `plots/`, verification checks printed to stdout.

## Module Structure

```mermaid
flowchart BT
    config["config.py\n─────────\nGrid, noise, cosmology\nconstants"]
    ps["power_spectrum.py\n─────────\ncl_model(ℓ, Ωₘ, σ₈)"]
    sim["simulation.py\n─────────\nGaussian κ → spin-2 γ\n→ noisy galaxy catalog"]
    cl["classical.py\n─────────\nPseudo-Cℓ estimation\nE/B decomposition\nNUTS inference"]
    hi["hierarchical.py\n─────────\nNon-centered z → κ → γ\nJoint NUTS inference\nPosterior mean shear"]
    pl["plotting.py\n─────────\nCorner, shear maps\npower spectra, whiskers"]
    ve["verification.py\n─────────\nCI checks, S₈ agreement\nB-mode null test"]
    demo["cosmology_inference_demo.py\n─────────\nEntry point: run both\npipelines and compare"]

    config --> ps
    ps --> sim
    ps --> cl
    ps --> hi
    ps --> pl
    sim --> demo
    cl --> demo
    hi --> demo
    pl --> demo
    ve --> demo
```

| Module | Role |
|--------|------|
| `config.py` | All constants: grid size, noise levels, true cosmology, sampler settings |
| `power_spectrum.py` | Parametric C<sub>&#8467;</sub>(&Omega;<sub>m</sub>, &sigma;<sub>8</sub>) model |
| `simulation.py` | Gaussian &kappa; field, Kaiser-Squires shear, noisy galaxy catalog |
| `classical.py` | E-mode pseudo-C<sub>&#8467;</sub> estimation, Gaussian likelihood, NUTS |
| `hierarchical.py` | Non-centered field-level model, joint NUTS over (z, S<sub>8</sub>, &Omega;<sub>m</sub>), posterior mean shear |
| `plotting.py` | Density contour corner plots, shear maps, power spectra, whisker plots |
| `verification.py` | Truth-in-CI checks, cross-method S<sub>8</sub> agreement, B-mode null test |

## Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid size | 32 &times; 32 | Pixels per side |
| Pixel scale | 2.0 arcmin | Angular size per pixel |
| Galaxies/pixel | 30 | Number density |
| Shape noise | 0.26 per component | Intrinsic ellipticity dispersion |
| &Omega;<sub>m</sub> (true) | 0.3 | Matter density |
| &sigma;<sub>8</sub> (true) | 0.8 | Fluctuation amplitude |
| NUTS samples (pseudo-C&#8467;) | 500 warmup + 4000 | Classical pipeline |
| NUTS samples (field-level) | 500 warmup + 1000 | Hierarchical pipeline (1026 dims) |

## References

- Schneider, M. D., et al. (2015). "Hierarchical Bayesian weak lensing: advantages and implementation." *The Astrophysical Journal*, 807(1), 87. [doi:10.1088/0004-637X/807/1/87](https://doi.org/10.1088/0004-637X/807/1/87)
- Kaiser, N. & Squires, G. (1993). "Mapping the dark matter with weak gravitational lensing." *The Astrophysical Journal*, 404, 441.
