"""Propagate sampled carbon parameters through the designs.

Every design is evaluated on the same parameter samples (common random numbers), so the
nominal-vs-robust comparison is paired: differences come from the designs, not from the draws.
Designs are processed in chunks, so memory stays bounded even for all ~45k designs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CarbonParameters
from .design_space import KWH_PER_WS

PERCENTILES = (5, 50, 95)


@dataclass(frozen=True)
class DesignArrays:
    """The design quantities the MC needs, as 1-D arrays (one entry per design)."""

    A_c: np.ndarray        # CPU core area (before yield)
    A_cIO: np.ndarray      # CPU IO area (before yield)
    A_g: np.ndarray        # GPU area (yield-adjusted)
    E_cpu_kWh: np.ndarray  # CPU energy per run
    E_gpu_kWh: np.ndarray  # GPU energy per run

    @classmethod
    def from_design_space(cls, designs: pd.DataFrame, p: CarbonParameters) -> DesignArrays:
        cores = designs["CPU_cores"].to_numpy(dtype=float)
        A_c, A_cIO = cores * p.A_cpu_core, cores * p.A_cpu_io  # CPU die area before yield
        cpu_time = (designs["CPU_SetupTime (s)"] + designs["CPU_KernelTime (s)"] + designs["CPU_TeardownTime (s)"]).to_numpy(dtype=float)
        return cls(
            A_c=A_c,
            A_cIO=A_cIO,
            A_g=designs["Area_g"].to_numpy(dtype=float),
            E_cpu_kWh=p.P_cpu_core * cores * cpu_time * KWH_PER_WS,
            E_gpu_kWh=(designs["GPU_Power Usage (W)"] * designs["GPU_KernelTime (s)"]).to_numpy(dtype=float) * KWH_PER_WS,
        )

    def subset(self, cols) -> DesignArrays:
        return DesignArrays(*(getattr(self, f)[cols] for f in self.__dataclass_fields__))


def carbon_samples(arrays: DesignArrays, samples: dict[str, np.ndarray], Nwork: float) -> dict[str, np.ndarray]:
    """Carbon of each design under each parameter sample, as (n_samples, n_designs) arrays."""
    s = {name: values[:, None] for name, values in samples.items()}  # (S, 1) to broadcast over designs
    A_c, A_cIO, A_g = arrays.A_c[None, :], arrays.A_cIO[None, :], arrays.A_g[None, :]

    embodied_c = (
        s["CIfab"] * (s["EPW_c"] * A_c + s["EPW_cIO"] * A_cIO)
        + s["GPW_c"] * A_c + s["GPW_cIO"] * A_cIO + s["MPW"] * (A_c + A_cIO)
    )
    embodied_g = s["CIfab"] * (s["EPW_g"] * A_g) + s["GPW_g"] * A_g + s["MPW"] * A_g
    operational_perrun = s["CIuse"] * arrays.E_cpu_kWh[None, :] + s["CIuse"] * arrays.E_gpu_kWh[None, :]

    embodied = embodied_c + embodied_g
    operational = operational_perrun * Nwork
    return {
        "Embodied": embodied,
        "Operational_perrun": operational_perrun,
        "Operational": operational,
        "Total": embodied + operational,
    }


def summarize_designs(arrays: DesignArrays, samples: dict[str, np.ndarray], Nwork: float,
                      chunk_size: int = 2048) -> pd.DataFrame:
    """Mean and 5th/50th/95th percentiles of each carbon component, for every design."""
    n_designs = len(arrays.A_c)
    columns: dict[str, list[np.ndarray]] = {}
    for start in range(0, n_designs, chunk_size):
        cols = slice(start, min(start + chunk_size, n_designs))
        for component, values in carbon_samples(arrays.subset(cols), samples, Nwork).items():
            # keep the original column names, e.g. Total_p95 and Operational_p95_perrun
            name, suffix = (("Operational", "_perrun") if component == "Operational_perrun" else (component, ""))
            pct = np.percentile(values, PERCENTILES, axis=0)
            for stat, row in [("mean", values.mean(axis=0))] + [(f"p{q:02d}", r) for q, r in zip(PERCENTILES, pct)]:
                columns.setdefault(f"{name}_{stat}{suffix}", []).append(row)
    return pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})


def total_carbon_samples(arrays: DesignArrays, samples: dict[str, np.ndarray], Nwork: float, indices) -> np.ndarray:
    """Total carbon samples for the designs at `indices`, shape (n_samples, len(indices))."""
    return carbon_samples(arrays.subset(np.asarray(indices)), samples, Nwork)["Total"]
