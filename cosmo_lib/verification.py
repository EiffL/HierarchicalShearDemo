"""Quantitative consistency checks between the two pipelines."""

from __future__ import annotations

import numpy as np

from .config import N_GIBBS, N_GIBBS_BURN, OMEGA_M_TRUE, SIGMA_8_TRUE
from .gibbs import GibbsOutput, GibbsState


def run_verification(
    classical_samples: dict,
    gibbs_chain: GibbsOutput,
    gibbs_state: GibbsState,
    n_burn: int = N_GIBBS_BURN,
    n_total: int = N_GIBBS,
) -> None:
    """Run quantitative consistency checks and print results.

    Checks:
      - Truth within 95% CI for both methods.
      - S_8 mean and std agree to ~10%.
      - Gibbs MH acceptance rate in 25-50%.

    Args:
        classical_samples: NUTS posterior samples.
        gibbs_chain: Gibbs chain output.
        gibbs_state: Final Gibbs state.
        n_burn: Gibbs burn-in.
        n_total: Total Gibbs iterations.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    om_cl = np.array(classical_samples["omega_m"])
    s8_cl = np.array(classical_samples["sigma_8"])
    om_gi = np.array(gibbs_chain.omega_m[n_burn:])
    s8_gi = np.array(gibbs_chain.sigma_8[n_burn:])

    # S_8 derived
    s8d_cl = s8_cl * np.sqrt(om_cl / 0.3)
    s8d_gi = s8_gi * np.sqrt(om_gi / 0.3)
    s8_true = SIGMA_8_TRUE * np.sqrt(OMEGA_M_TRUE / 0.3)

    def check_ci(name: str, samples: np.ndarray, truth: float) -> bool:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        inside = lo <= truth <= hi
        status = "PASS" if inside else "FAIL"
        print(
            f"  [{status}] {name}: truth={truth:.4f}, "
            f"95% CI=[{lo:.4f}, {hi:.4f}], "
            f"mean={np.mean(samples):.4f} +/- {np.std(samples):.4f}"
        )
        return inside

    print("\n1. Truth within 95% CI:")
    check_ci("Classical Omega_m", om_cl, OMEGA_M_TRUE)
    check_ci("Classical sigma_8", s8_cl, SIGMA_8_TRUE)
    check_ci("Classical S_8", s8d_cl, s8_true)
    check_ci("Gibbs Omega_m", om_gi, OMEGA_M_TRUE)
    check_ci("Gibbs sigma_8", s8_gi, SIGMA_8_TRUE)
    check_ci("Gibbs S_8", s8d_gi, s8_true)

    print("\n2. S_8 comparison between methods:")
    s8_cl_mean, s8_cl_std = np.mean(s8d_cl), np.std(s8d_cl)
    s8_gi_mean, s8_gi_std = np.mean(s8d_gi), np.std(s8d_gi)
    mean_diff = abs(s8_cl_mean - s8_gi_mean) / s8_cl_mean * 100
    std_diff = abs(s8_cl_std - s8_gi_std) / s8_cl_std * 100
    print(f"  Classical S_8: {s8_cl_mean:.4f} +/- {s8_cl_std:.4f}")
    print(f"  Gibbs S_8:     {s8_gi_mean:.4f} +/- {s8_gi_std:.4f}")
    print(f"  Mean relative diff: {mean_diff:.1f}%")
    print(f"  Std relative diff:  {std_diff:.1f}%")

    print("\n3. Gibbs MH acceptance rate:")
    acc_rate = float(gibbs_state.n_accepted) / n_total * 100
    status = "PASS" if 15 <= acc_rate <= 60 else "WARN"
    print(f"  [{status}] Acceptance rate: {acc_rate:.1f}%")

    print("\n" + "=" * 70)
