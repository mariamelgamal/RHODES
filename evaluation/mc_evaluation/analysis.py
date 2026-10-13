"""Statistics for comparing the nominal and robust designs."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Design
from .distributions import SAMPLERS, Scenario


def find_design(designs: pd.DataFrame, design: Design) -> int:
    """Row index of `design` in the design space."""
    match = (designs["total_config"] == design.label) & (designs["GPU_Clock"] == design.gpu_clock)
    idx = np.flatnonzero(match.to_numpy())
    if len(idx) == 0:
        clocks = sorted(designs.loc[designs["total_config"] == design.label, "GPU_Clock"].unique())
        raise ValueError(f"Design {design} not in the design space; clocks available for {design.label}: {clocks}")
    return int(idx[0])


def describe(samples: np.ndarray) -> dict[str, float]:
    p05, p50, p95 = np.percentile(samples, [5, 50, 95])
    return {"mean": samples.mean(), "std": samples.std(), "p05": p05, "p50": p50, "p95": p95}


def paired_bootstrap_ci(x_nominal: np.ndarray, x_robust: np.ndarray, statistic, rng,
                        n_boot: int = 1000, level: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI of statistic(x_nominal, x_robust), resampling the same draws for both designs."""
    idx = rng.integers(0, len(x_nominal), size=(n_boot, len(x_nominal)))
    values = statistic(x_nominal[idx], x_robust[idx])
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def compare_nominal_robust(x_nominal: np.ndarray, x_robust: np.ndarray, rng, n_boot: int = 1000) -> dict[str, float]:
    """Mean/P95 of both designs, the robust design's improvement, and a 95% CI on the P95 ratio."""
    n, r = describe(x_nominal), describe(x_robust)
    p95_ratio = lambda a, b: np.percentile(a, 95, axis=-1) / np.percentile(b, 95, axis=-1)
    ci_lo, ci_hi = paired_bootstrap_ci(x_nominal, x_robust, p95_ratio, rng, n_boot)
    return {
        "Nominal_Mean": n["mean"], "Nominal_Std": n["std"], "Nominal_P95": n["p95"],
        "Robust_Mean": r["mean"], "Robust_Std": r["std"], "Robust_P95": r["p95"],
        "P95_Improvement_%": (n["p95"] - r["p95"]) / n["p95"] * 100,
        "Mean_Improvement_%": (n["mean"] - r["mean"]) / n["mean"] * 100,
        "P95_Ratio": n["p95"] / r["p95"],
        "P95_Ratio_CI95_low": ci_lo,
        "P95_Ratio_CI95_high": ci_hi,
    }


def violates_constraints(designs: pd.DataFrame, area_max: float, tC_threshold: float, carbon_col: str) -> pd.Series:
    """True where a design exceeds the area limit or the carbon threshold (on `carbon_col`)."""
    return (designs["Total_area"] > area_max) | (designs[carbon_col] > tC_threshold)


def tcdp_report(designs: pd.DataFrame, idx_nominal: int, idx_robust: int, carbon_col: str = "Total_p95") -> dict[str, float]:
    """Time, carbon, and tCDP (carbon x time) ratios of the nominal over the robust design."""
    n, r = designs.iloc[idx_nominal], designs.iloc[idx_robust]
    tcdp_n = n[carbon_col] * n["TotalTime (s)"]
    tcdp_r = r[carbon_col] * r["TotalTime (s)"]
    return {
        "time_ratio": n["TotalTime (s)"] / r["TotalTime (s)"],
        "carbon_ratio": n[carbon_col] / r[carbon_col],
        "tCDP_nominal": tcdp_n,
        "tCDP_robust": tcdp_r,
        "tCDP_ratio": tcdp_n / tcdp_r,
    }


def verify_scenarios(scenarios: dict[str, Scenario], rng, n_samples: int = 50_000) -> pd.DataFrame:
    """Realized mean/std/min of every sampled parameter vs its target (mu, sigma).

    Differences come from the bounds: resampling below min_val raises the mean and
    shrinks the std. Cauchy has no finite mean/std, so only its location and min are shown.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Bernoulli shift warnings are expected here
        for family, scenario in scenarios.items():
            for name, (dist, spec) in scenario.items():
                x = SAMPLERS[dist](rng, n_samples, spec)
                heavy_tailed = dist == "Cauchy"
                rows.append({
                    "Distribution": family, "Parameter": name, "Sampler": dist,
                    "Target_Mean": spec.mu, "Actual_Mean": x.mean(),
                    "Mean_Error_%": (x.mean() - spec.mu) / spec.mu * 100,
                    "Target_Std": np.nan if heavy_tailed else spec.sigma,
                    "Actual_Std": x.std(ddof=1),
                    "Std_Error_%": np.nan if heavy_tailed or spec.sigma == 0 else (x.std(ddof=1) - spec.sigma) / spec.sigma * 100,
                    "Min": x.min(),
                })
    return pd.DataFrame(rows)
