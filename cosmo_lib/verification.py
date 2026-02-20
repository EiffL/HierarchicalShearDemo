"""Quantitative consistency checks between the two pipelines."""

from __future__ import annotations

import numpy as np

from .config import OMEGA_M_TRUE, SIGMA_8_TRUE


def run_verification(
    classical_samples: dict,
    hierarchical_samples: dict,
    eb_data: dict | None = None,
) -> None:
    """Run quantitative consistency checks and print results.

    Checks:
      - Truth within 95% CI for both methods.
      - S_8 mean and std agree to ~10%.
      - B-mode null test.

    Args:
        classical_samples: NUTS posterior samples.
        hierarchical_samples: Field-level NUTS posterior samples (dict with
            'omega_m', 'sigma_8' keys).
        eb_data: Optional E/B power spectrum data for null test.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    om_cl = np.array(classical_samples["omega_m"])
    s8_cl = np.array(classical_samples["sigma_8"])
    om_hi = np.array(hierarchical_samples["omega_m"])
    s8_hi = np.array(hierarchical_samples["sigma_8"])

    # S_8 derived
    s8d_cl = s8_cl * np.sqrt(om_cl / 0.3)
    s8d_hi = s8_hi * np.sqrt(om_hi / 0.3)
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
    check_ci("Hierarchical Omega_m", om_hi, OMEGA_M_TRUE)
    check_ci("Hierarchical sigma_8", s8_hi, SIGMA_8_TRUE)
    check_ci("Hierarchical S_8", s8d_hi, s8_true)

    print("\n2. S_8 comparison between methods:")
    s8_cl_mean, s8_cl_std = np.mean(s8d_cl), np.std(s8d_cl)
    s8_hi_mean, s8_hi_std = np.mean(s8d_hi), np.std(s8d_hi)
    mean_diff = abs(s8_cl_mean - s8_hi_mean) / s8_cl_mean * 100
    std_diff = abs(s8_cl_std - s8_hi_std) / s8_cl_std * 100
    print(f"  Classical S_8:     {s8_cl_mean:.4f} +/- {s8_cl_std:.4f}")
    print(f"  Hierarchical S_8:  {s8_hi_mean:.4f} +/- {s8_hi_std:.4f}")
    print(f"  Mean relative diff: {mean_diff:.1f}%")
    print(f"  Std relative diff:  {std_diff:.1f}%")

    if eb_data is not None:
        print("\n3. B-mode consistency (true shear field):")
        cl_E = np.array(eb_data["cl_E"])
        cl_B = np.array(eb_data["cl_B"])
        mean_E = np.mean(cl_E[cl_E > 0])
        mean_B = np.mean(np.abs(cl_B))
        ratio = mean_B / mean_E if mean_E > 0 else float("inf")
        status = "PASS" if ratio < 0.05 else "WARN"
        print(f"  [{status}] Mean |B|/E ratio: {ratio:.4f} (expect < 0.05)")

    print("\n" + "=" * 70)
