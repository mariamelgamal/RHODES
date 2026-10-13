"""Probability distributions for the uncertain carbon parameters.

Each sampler draws from a parent distribution with mean `mu` and standard deviation
`sigma`, then enforces the bounds (`min_val`, `max_val`) by rejection (resampling). When a
bound lies within about 1σ of the mean, the realized mean and std differ from (mu, sigma);
`analysis.verify_scenarios` reports by how much.

All samplers take an explicit random generator (np.random.Generator, or a legacy
np.random.RandomState) so that runs are reproducible.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import A_W, CarbonParameters

# Order in which parameters are sampled (kept fixed so a seed always gives the same draws)
PARAMETERS = ("CIfab", "CIuse", "EPW_c", "EPW_cIO", "EPW_g", "GPW_c", "GPW_cIO", "GPW_g", "MPW")


@dataclass(frozen=True)
class ParameterSpec:
    """Parent-distribution parameters and bounds for one uncertain parameter."""

    mu: float
    sigma: float
    min_val: float | None = None
    max_val: float | None = None


def _resample_out_of_bounds(samples, draw, min_val=None, max_val=None):
    """Redraw (in place) every sample outside [min_val, max_val] until all are inside."""
    def out_of_bounds():
        bad = np.zeros(samples.shape, dtype=bool)
        if min_val is not None:
            bad |= samples < min_val
        if max_val is not None:
            bad |= samples > max_val
        return bad

    bad = out_of_bounds()
    while bad.any():
        samples[bad] = draw(bad.sum())
        bad = out_of_bounds()
    return samples


def sample_normal(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """Normal N(mu, sigma), resampled below min_val."""
    draw = lambda n: rng.normal(loc=spec.mu, scale=spec.sigma, size=n)
    return _resample_out_of_bounds(draw(size), draw, spec.min_val)


def sample_lognormal(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """Log-normal with mean mu and std sigma, resampled below min_val."""
    sigma_ln = np.sqrt(np.log(1 + (spec.sigma / spec.mu) ** 2))
    mu_ln = np.log(spec.mu) - 0.5 * sigma_ln**2
    draw = lambda n: rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)
    return _resample_out_of_bounds(draw(size), draw, spec.min_val)


def sample_exponential(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """Shifted exponential mu - sigma + Exp(scale=sigma) (mean mu, std sigma), resampled below min_val."""
    draw = lambda n: spec.mu - spec.sigma + rng.exponential(scale=spec.sigma, size=n)
    return _resample_out_of_bounds(draw(size), draw, spec.min_val)


def sample_uniform(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """U(min_val, max_val) if both bounds are given, else U(mu - √3σ, mu + √3σ) with the low end raised to min_val."""
    if spec.min_val is not None and spec.max_val is not None:
        return rng.uniform(low=spec.min_val, high=spec.max_val, size=size)
    half_width = np.sqrt(3) * spec.sigma
    low = spec.mu - half_width
    if spec.min_val is not None:
        low = max(low, spec.min_val)
    return rng.uniform(low=low, high=spec.mu + half_width, size=size)


def sample_bernoulli(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """Two-point distribution on {mu - sigma, mu + sigma}, shifted up if mu - sigma < min_val."""
    samples = spec.mu - spec.sigma + 2 * spec.sigma * rng.binomial(n=1, p=0.5, size=size)
    if spec.min_val is not None and spec.mu - spec.sigma < spec.min_val:
        shift = spec.min_val - (spec.mu - spec.sigma)
        warnings.warn(f"Bernoulli lower point {spec.mu - spec.sigma:.4g} < min_val {spec.min_val:.4g}; "
                      f"shifting the distribution up by {shift:.4g}")
        samples = samples + shift
    return samples


def sample_cauchy(rng, size: int, spec: ParameterSpec) -> np.ndarray:
    """Cauchy with location mu and scale sigma, truncated to [min_val, max_val] by resampling."""
    # inverse CDF, tan(pi * (U - 1/2)), the same draws as scipy.stats.cauchy.rvs
    draw = lambda n: spec.mu + spec.sigma * np.tan(np.pi * (rng.uniform(size=n) - 0.5))
    return _resample_out_of_bounds(draw(size), draw, spec.min_val, spec.max_val)


SAMPLERS: dict[str, Callable] = {
    "Normal": sample_normal,
    "LogNormal": sample_lognormal,
    "Exponential": sample_exponential,
    "Uniform": sample_uniform,
    "Bernoulli": sample_bernoulli,
    "Cauchy": sample_cauchy,
}


def base_specs(p: CarbonParameters) -> dict[str, ParameterSpec]:
    """Mean, std, and lower bound of each parameter (std = full spread of the uncertainty range)."""
    return {
        "CIfab":   ParameterSpec(p.CIfab, 0.1, min_val=0.041),  # 0.041 kg CO2e/kWh = solar
        "CIuse":   ParameterSpec(p.CIuse, 0.1, min_val=0.041),
        "EPW_c":   ParameterSpec(p.EPW_c, 345 / A_W, min_val=1000 / A_W),  # 345 for 7 nm EUV, 375 for 7 nm DUV
        "EPW_cIO": ParameterSpec(p.EPW_cIO, 150 / A_W, min_val=800 / A_W),
        "EPW_g":   ParameterSpec(p.EPW_g, 345 / A_W, min_val=1000 / A_W),
        "GPW_c":   ParameterSpec(p.GPW_c, 102 / A_W, min_val=66 / A_W),  # abated GPW
        "GPW_cIO": ParameterSpec(p.GPW_cIO, 45 / A_W, min_val=53 / A_W),
        "GPW_g":   ParameterSpec(p.GPW_g, 102 / A_W, min_val=66 / A_W),
    }


def cauchy_specs(p: CarbonParameters) -> dict[str, ParameterSpec]:
    """Cauchy location, scale, and truncation bounds (heavy tails are cut at 2x the mean)."""
    return {
        "CIfab":   ParameterSpec(p.CIfab, 0.05, min_val=0.041, max_val=p.CIfab * 2),
        "CIuse":   ParameterSpec(p.CIuse, 0.05, min_val=0.041, max_val=p.CIuse * 2),
        "EPW_c":   ParameterSpec(p.EPW_c, 172 / A_W, min_val=1000 / A_W, max_val=p.EPW_c * 2),
        "EPW_cIO": ParameterSpec(p.EPW_cIO, 75 / A_W, min_val=800 / A_W, max_val=p.EPW_cIO * 2),
        "EPW_g":   ParameterSpec(p.EPW_g, 172 / A_W, min_val=1000 / A_W, max_val=p.EPW_g * 2),
        "GPW_c":   ParameterSpec(p.GPW_c, 51 / A_W, min_val=66 / A_W, max_val=p.GPW_c * 2),
        "GPW_cIO": ParameterSpec(p.GPW_cIO, 22.5 / A_W, min_val=53 / A_W, max_val=p.GPW_cIO * 2),
        "GPW_g":   ParameterSpec(p.GPW_g, 51 / A_W, min_val=66 / A_W, max_val=p.GPW_g * 2),
    }


Scenario = dict[str, tuple[str, ParameterSpec]]  # parameter -> (distribution name, spec)


def build_scenarios(p: CarbonParameters) -> dict[str, Scenario]:
    """One scenario per distribution family; MPW is uniform within ±5% in every scenario."""
    mpw = ("Uniform", ParameterSpec(p.MPW, 0.0, min_val=p.MPW * 0.95, max_val=p.MPW * 1.05))
    scenarios = {}
    for family in SAMPLERS:
        specs = cauchy_specs(p) if family == "Cauchy" else base_specs(p)
        scenarios[family] = {name: (family, spec) for name, spec in specs.items()} | {"MPW": mpw}
    return scenarios


def sample_parameters(scenario: Scenario, n_samples: int, rng) -> dict[str, np.ndarray]:
    """Draw `n_samples` values of every parameter in `scenario`, in the fixed PARAMETERS order."""
    return {name: SAMPLERS[scenario[name][0]](rng, n_samples, scenario[name][1]) for name in PARAMETERS}
