#!/usr/bin/env python3
"""Diagnostic script for the Gibbs+IS sampler with poor mixing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cosmo_lib.config import (
    GRID_SIZE,
    N_GAL_PER_PIX,
    N_PATCH_SAMPLES,
    PIXEL_SCALE,
    SEED,
    SHAPE_NOISE,
    SIGMA_INTERIM,
    MH_PROPOSAL_SIGMA,
)
from cosmo_lib.simulation import simulate_shear_catalog
from cosmo_lib.gibbs import (
    compute_conditional_params,
    compute_patch_posteriors,
    draw_patch_samples,
    field_log_likelihood,
    importance_resample_patches,
    gibbs_step,
    GibbsState,
)
from cosmo_lib.power_spectrum import cl_model

# ---- reproducibility: same key splits as main script ----
key = jax.random.PRNGKey(SEED)
k_sim, k_cl, k_gibbs, k_s1, k_s2 = jax.random.split(key, 5)

# ==================================================================
# 1. Simulate data (same as main script, same seed)
# ==================================================================
print("=" * 70)
print("1. Simulating data (same seed as main script)")
print("=" * 70)
sim = simulate_shear_catalog(k_sim)
gamma1_true = sim["gamma1"]
gamma2_true = sim["gamma2"]
ell2d = sim["ell2d"]

print(f"  Grid size         : {GRID_SIZE}x{GRID_SIZE}")
print(f"  kappa rms         : {float(jnp.std(sim['kappa'])):.6f}")
print(f"  gamma1 rms        : {float(jnp.std(gamma1_true)):.6f}")
print(f"  gamma2 rms        : {float(jnp.std(gamma2_true)):.6f}")
print(f"  SIGMA_INTERIM     : {SIGMA_INTERIM:.6f}")
print(f"  SHAPE_NOISE       : {SHAPE_NOISE}")
print(f"  N_GAL_PER_PIX     : {N_GAL_PER_PIX}")
print(f"  N_PATCH_SAMPLES   : {N_PATCH_SAMPLES}")
print(f"  MH_PROPOSAL_SIGMA : {MH_PROPOSAL_SIGMA}")

# ==================================================================
# 2. Field log-likelihood at TRUE vs BOUNDARY cosmology
# ==================================================================
print("\n" + "=" * 70)
print("2. Field log-likelihood: TRUE vs BOUNDARY cosmology")
print("=" * 70)

ll_true = field_log_likelihood(
    gamma1_true, gamma2_true, 0.3, 0.8, ell2d, GRID_SIZE, PIXEL_SCALE
)
ll_boundary = field_log_likelihood(
    gamma1_true, gamma2_true, 0.6, 1.2, ell2d, GRID_SIZE, PIXEL_SCALE
)

print(f"  log L(true field | omega_m=0.3, sigma_8=0.8)  = {float(ll_true):.4f}")
print(f"  log L(true field | omega_m=0.6, sigma_8=1.2)  = {float(ll_boundary):.4f}")
print(f"  Delta log L (boundary - true)                  = {float(ll_boundary - ll_true):.4f}")

if float(ll_boundary) > float(ll_true):
    print("  >>> WARNING: Likelihood at boundary is HIGHER than at truth — possible likelihood bug!")
else:
    print("  >>> OK: Likelihood at truth is higher (as expected).")

# ==================================================================
# 3. IS weights for a single patch [32,32]
# ==================================================================
print("\n" + "=" * 70)
print("3. IS weight diagnostics for pixel [32,32]")
print("=" * 70)

# Compute conditional params at true cosmology for true field
cl2d_true = cl_model(jnp.clip(ell2d, 1.0, None), 0.3, 0.8)
cl2d_true = cl2d_true.at[0, 0].set(1e-10)

mu1_cond, sigma1_cond = compute_conditional_params(gamma1_true, cl2d_true, GRID_SIZE, PIXEL_SCALE)

# Draw interim samples around the patch posterior mean
patch = compute_patch_posteriors(sim["eps1"], sim["eps2"], SHAPE_NOISE, N_GAL_PER_PIX)
samples1 = draw_patch_samples(k_s1, patch["mean1"], patch["sigma_patch"], N_PATCH_SAMPLES)

# Extract samples for pixel [32,32]
px, py = 32, 32
s = samples1[px, py, :]  # (N_PATCH_SAMPLES,)
mu_c = mu1_cond[px, py]
sig_c = float(sigma1_cond)

print(f"  Patch posterior mean at [{px},{py}] : {float(patch['mean1'][px,py]):.6f}")
print(f"  Conditional mean (mu_cond)         : {float(mu_c):.6f}")
print(f"  sigma_cond                         : {sig_c:.6e}")
print(f"  SIGMA_INTERIM                      : {SIGMA_INTERIM:.6e}")
print(f"  Ratio sigma_cond / SIGMA_INTERIM   : {sig_c / SIGMA_INTERIM:.6e}")

# Compute IS log-weights for this patch
log_target = -0.5 * ((s - mu_c) / sig_c) ** 2
log_interim = -0.5 * (s / SIGMA_INTERIM) ** 2
log_w = log_target - log_interim
# Normalize for ESS computation
log_w_shifted = log_w - jnp.max(log_w)
w = jnp.exp(log_w_shifted)
w_norm = w / jnp.sum(w)
ess = 1.0 / jnp.sum(w_norm ** 2)

print(f"  log_w range                        : [{float(jnp.min(log_w)):.4f}, {float(jnp.max(log_w)):.4f}]")
print(f"  log_w span                         : {float(jnp.max(log_w) - jnp.min(log_w)):.4f}")
print(f"  ESS (out of {N_PATCH_SAMPLES})                  : {float(ess):.4f}")

if float(ess) < 5.0:
    print("  >>> WARNING: ESS ~ 1 means IS weights are DEGENERATE. Only ~1 sample matters!")
else:
    print(f"  >>> ESS looks reasonable ({float(ess):.1f} / {N_PATCH_SAMPLES}).")

# ==================================================================
# 4. sigma_cond vs SIGMA_INTERIM comparison
# ==================================================================
print("\n" + "=" * 70)
print("4. sigma_cond vs SIGMA_INTERIM")
print("=" * 70)

print(f"  sigma_cond (field conditional std)  : {sig_c:.6e}")
print(f"  SIGMA_INTERIM (interim prior width) : {SIGMA_INTERIM:.6e}")
print(f"  Ratio sigma_cond / SIGMA_INTERIM    : {sig_c / SIGMA_INTERIM:.6e}")

if sig_c < SIGMA_INTERIM * 0.1:
    print("  >>> PROBLEM: sigma_cond << SIGMA_INTERIM")
    print("      The IS target (field conditional) is MUCH narrower than the proposal")
    print("      (interim prior), so most proposal samples fall outside the target,")
    print("      leading to degenerate weights.")
elif sig_c > SIGMA_INTERIM * 10:
    print("  >>> sigma_cond >> SIGMA_INTERIM — target is wider than proposal.")
else:
    print("  >>> The two scales are comparable.")

# Also check for gamma2
mu2_cond, sigma2_cond = compute_conditional_params(gamma2_true, cl2d_true, GRID_SIZE, PIXEL_SCALE)
print(f"\n  sigma_cond for gamma2               : {float(sigma2_cond):.6e}")
print(f"  (Should be same as gamma1 — it depends only on C_l and grid geometry)")

# ==================================================================
# 5. Run 50 Gibbs steps and print per-step acceptance + trajectory
# ==================================================================
print("\n" + "=" * 70)
print("5. Running 50 Gibbs steps — per-step acceptance and trajectory")
print("=" * 70)

from functools import partial

# Prepare interim samples (same as main script)
samples2 = draw_patch_samples(k_s2, patch["mean2"], patch["sigma_patch"], N_PATCH_SAMPLES)

N_DIAG = 50
keys = jax.random.split(k_gibbs, N_DIAG + 1)
gibbs_keys = keys[1:]  # first key was used for splitting

state = GibbsState(
    gamma1=patch["mean1"],
    gamma2=patch["mean2"],
    omega_m=0.3,
    sigma_8=0.8,
    n_accepted=0,
)

omega_m_traj = [0.3]
sigma_8_traj = [0.8]
accepted_list = []

# We need to run step-by-step (not via scan) to capture per-step acceptance
step_fn = partial(
    gibbs_step,
    samples1=samples1,
    samples2=samples2,
    ell2d=ell2d,
    n=GRID_SIZE,
    delta=PIXEL_SCALE,
    sigma_interim=SIGMA_INTERIM,
)

# JIT compile the step function
step_fn_jit = jax.jit(step_fn)

prev_n_accepted = 0
for i in range(N_DIAG):
    state, output = step_fn_jit(state, gibbs_keys[i])
    # Block to read values
    om_val = float(output.omega_m)
    s8_val = float(output.sigma_8)
    n_acc = int(state.n_accepted)
    accepted = n_acc > prev_n_accepted
    prev_n_accepted = n_acc
    accepted_list.append(accepted)
    omega_m_traj.append(om_val)
    sigma_8_traj.append(s8_val)

print(f"\n  {'Step':>4s}  {'Accept':>7s}  {'omega_m':>9s}  {'sigma_8':>9s}")
print(f"  {'----':>4s}  {'-------':>7s}  {'---------':>9s}  {'---------':>9s}")
for i in range(N_DIAG):
    acc_str = "YES" if accepted_list[i] else "no"
    print(f"  {i+1:4d}  {acc_str:>7s}  {omega_m_traj[i+1]:9.5f}  {sigma_8_traj[i+1]:9.5f}")

n_total_accepted = sum(accepted_list)
print(f"\n  Total accepted: {n_total_accepted} / {N_DIAG} = {100*n_total_accepted/N_DIAG:.1f}%")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"  Field log-likelihood at truth         : {float(ll_true):.4f}")
print(f"  Field log-likelihood at boundary       : {float(ll_boundary):.4f}")
print(f"  sigma_cond                             : {sig_c:.6e}")
print(f"  SIGMA_INTERIM                          : {SIGMA_INTERIM:.6e}")
print(f"  sigma_cond / SIGMA_INTERIM             : {sig_c / SIGMA_INTERIM:.6e}")
print(f"  ESS at pixel [32,32]                   : {float(ess):.4f} / {N_PATCH_SAMPLES}")
print(f"  MH acceptance (50 steps)               : {n_total_accepted}/{N_DIAG} = {100*n_total_accepted/N_DIAG:.1f}%")
print(f"  omega_m range in chain                 : [{min(omega_m_traj):.5f}, {max(omega_m_traj):.5f}]")
print(f"  sigma_8 range in chain                 : [{min(sigma_8_traj):.5f}, {max(sigma_8_traj):.5f}]")
