# Schneider et al. (2015) — Hierarchical Probabilistic Inference of Cosmic Shear

**Reference:** Schneider, Hogg, Marshall, Dawson, Meyers, Bard & Lang, *ApJ* 807:87 (2015)

## Overview

The paper presents a forward-modeling, hierarchical Bayesian framework for inferring cosmic shear from galaxy imaging surveys. The key innovation is a **divide-and-conquer** strategy: fit each galaxy independently to obtain "interim posterior" samples of its parameters, then combine those per-galaxy samples into a global inference on the shear field and cosmology using **importance sampling**. This avoids the intractable joint fit of all galaxies simultaneously while still propagating the full per-galaxy uncertainty into the cosmological constraints.

---

## The Three-Level Hierarchy

The full joint posterior (Equation 18 in the paper) factorizes into three conditionally independent branches:

1. **Cosmological parameters** $\boldsymbol{\theta}$ (e.g. $\Omega_m$, $\sigma_8$) — set the angular power spectrum $C_\ell$ of the lensing potential $\psi$.
2. **Observational PSFs** $\Pi_{n,i}$ — one per galaxy $n$ per epoch $i$, drawn from epoch-level observing conditions $\Omega_i$.
3. **Intrinsic galaxy properties** $\omega_n$ (ellipticity, size, brightness) — drawn from a population-level distribution with hyperparameters $\alpha$ (modeled via a Dirichlet Process).

The lens potential $\psi_s(\mathbf{x})$ connects branch 1 to the galaxy-level likelihood: it is the 2D projection of the 3D matter potential $\Psi^{\rm LT}$, which depends on cosmology $\boldsymbol{\theta}$ and initial conditions $\Psi^{\rm IC}$.

---

## Step 1: Interim Posterior Sampling (the "Reaper")

For each galaxy $n$ observed across epochs $i = 1, \ldots, n_{\rm epoch}$, run an MCMC sampler on the per-galaxy likelihood of the pixel data $\mathbf{d}_n$:

$$\ln \Pr(\mathbf{d}_n) = -\frac{1}{2} \sum_i \frac{(d_{n,i} - \hat{d}(\omega_n, \psi, \Pi_i))^2}{\sigma_{{\rm pix},n,i}^2} + \text{const.}$$

This produces $K$ samples $\{\omega_{nk}, \Pi_{nk}\}_{k=1}^K$ from the **interim posterior**:

$$\Pr(\omega_n, \Pi_n | \mathbf{d}_n) = \frac{1}{Z_n} \Pr(\mathbf{d}_n | \omega_n, \Pi_n) \Pr(\omega_n | I_0) \Pr(\Pi_n | I_0)$$

where $I_0$ denotes the interim prior assumptions (chosen for computational convenience — broad or flat priors are fine). Crucially, the shear/lensing potential is **not** fit at this stage; the interim prior on $\psi$ can be any convenient Gaussian whose covariance mimics cosmological spatial correlations (Equation 9), but $\psi$ need not be tied to a specific cosmology yet.

**Key point:** Each galaxy is fit independently and in parallel. The interim samples capture the full degeneracy between intrinsic shape $\omega_n$ and applied shear $\psi$. Only ~10 samples per galaxy are needed.

---

## Step 2: Importance Sampling to Build the Global Likelihood (the "Thresher")

The goal is to evaluate the **marginalized likelihood** for the data of all galaxies, given the shear field $\psi$, the galaxy population hyperparameters $\alpha$, and the observing conditions $\Omega$:

$$\Pr(\mathbf{d} | \alpha, \Omega, \psi) = \prod_{n=1}^{n_{\rm gal}} \Pr(\mathbf{d}_n | \alpha, \Omega, \psi)$$

where each per-galaxy term involves an integral over the nuisance parameters $\omega_n$ and $\Pi_n$:

$$\Pr(\mathbf{d}_n | \alpha, \Omega, \psi) \propto \int d\omega_n \int d\Pi_n \; \frac{\Pr(\omega_n | \alpha) \Pr(\Pi_n | \Omega)}{\Pr(\omega_n | I_0) \Pr(\Pi_n | I_0)} \Pr(\mathbf{d}_n | \omega_n, \psi, \Pi_n) \Pr(\omega_n | I_0) \Pr(\Pi_n | I_0)$$

### The importance-sampling identity

Using the identity $\int p(x) f(x) dx = \int p(x) g(x) \frac{f(x)}{g(x)} dx$ to rewrite this integral as an expectation over the interim posterior, the marginalized likelihood becomes (Equation 15):

$$\boxed{\Pr(\mathbf{d}_n | \alpha, \Omega, \psi) \approx \frac{Z_n}{K} \sum_{k=1}^{K} \frac{\Pr(\omega_{nk} | \alpha, \psi) \; \Pr(\Pi_{nk} | \Omega)}{\Pr(\omega_{nk} | I_0) \; \Pr(\Pi_{nk} | I_0)}}$$

where the sum runs over the $K$ interim posterior samples $\{\omega_{nk}, \Pi_{nk}\}$ drawn in Step 1.

**What this achieves:** The expensive pixel-level likelihood $\Pr(\mathbf{d}_n | \omega_n, \psi, \Pi_n)$ was already evaluated during interim sampling. Now, to evaluate the likelihood under a *different* model (different $\alpha$, $\psi$, or $\Omega$), we only need to re-weight the existing samples by the ratio of the new prior to the interim prior. This is extremely fast — just evaluate prior ratios on stored samples.

**Why it works:** The interim posterior samples cover the region of parameter space that matters for each galaxy. By re-weighting them, we can "reinterpret" those samples under any new model for the shear and galaxy population, without re-running any pixel-level fits.

---

## Step 3: From Shear Field to Cosmology (the "Winnower")

Given interim samples of the lens potential $\psi_s$ from the Thresher, cosmological parameter constraints follow by another layer of importance sampling (Equation 17):

$$\Pr(\mathbf{d}_n | \Psi^{\rm IC}, \boldsymbol{\theta}, W) \propto \frac{1}{N} \sum_{k=1}^{N} \frac{\prod_s \Pr(\hat{\psi}_{s,k} | \Psi^{\rm IC}, \boldsymbol{\theta}, W)}{\prod_s \Pr(\hat{\psi}_{s,k} | I)}$$

where the sum is over $N$ lens-potential samples $\hat{\psi}_{s,k}$ from the Thresher output. The numerator evaluates the probability of each sampled $\psi$ under a cosmological model (parameterized by $\boldsymbol{\theta}$ and initial conditions $\Psi^{\rm IC}$), and the denominator is the interim prior used when sampling $\psi$.

This connects the inferred shear field to the matter power spectrum and thereby to $(\Omega_m, \sigma_8)$.

---

## The Role of the Dirichlet Process (DP) for Galaxy Properties

A critical ingredient is the hierarchical model for intrinsic galaxy properties $\omega_n$. Rather than assuming all galaxies share one ellipticity distribution, the paper uses a **Dirichlet Process mixture model**:

$$\omega_n \sim \mathcal{N}(0, \alpha_n), \quad \alpha_n \sim G(\alpha_n | \mathcal{A}), \quad G \sim \text{DP}(\mathcal{A}, G_0)$$

This automatically clusters galaxies into latent classes (e.g., round vs. elongated populations) and learns the number of classes from the data. The DP model:
- Reduces shear bias by correctly modeling the intrinsic ellipticity distribution (especially when it is multimodal).
- Enables per-galaxy weighting: galaxies from intrinsically rounder populations contribute more precise shear constraints.
- Is updated via Gibbs sampling of the class assignments $c_n$ and hyperparameters $\alpha$, using the importance samples to evaluate marginal likelihoods (Equations 26-30).

---

## Computational Pipeline Summary (Figure 9)

| Stage | Name | Input | Output |
|-------|------|-------|--------|
| 1 | **Reaper** | Pixel cutouts per galaxy | $K \sim 10$ interim posterior samples $(\omega_n, \Pi_n)$ per galaxy (embarrassingly parallel) |
| 2 | **Thresher** | All interim samples for a field | MCMC samples of: (a) population-level distribution parameters $\alpha$, (b) shear field $\psi$ on a grid |
| 3 | **Winnower** | Shear field samples from Thresher | Cosmological parameter constraints $(\Omega_m, \sigma_8, \ldots)$ via importance sampling against cosmological models |

---

## Why This Matters

1. **No point estimators needed.** Traditional pipelines compress each galaxy to a single ellipticity estimate, losing information. This framework keeps the full per-galaxy posterior.
2. **Proper marginalization over nuisances.** Intrinsic shapes, PSF uncertainties, and population-level distributions are all marginalized, not fixed or assumed.
3. **Scalable.** The expensive step (pixel-level fitting) is embarrassingly parallel. The global inference reuses stored samples via cheap importance-weight evaluations.
4. **Reduced bias.** Toy models in the paper show that hierarchical inference of the intrinsic ellipticity distribution (rather than assuming a fixed form) substantially reduces shear bias, especially when the galaxy population has multiple morphological classes.
