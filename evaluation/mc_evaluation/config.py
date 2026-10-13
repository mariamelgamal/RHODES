"""Inputs for the Monte Carlo (MC) evaluation.

Edit the defaults in `MCConfig` and `CarbonParameters`, or override them from the
command line (see `python evaluation/run_mc_evaluation.py --help`).

Parameter names (CIfab, EPW_c, GPW_cIO, ...) mirror input parameters/input_parameters.jl
and the paper notation. Units: CI in kg CO2e/kWh, EPW in kWh/mm², GPW and MPW in
kg CO2e/mm², areas in mm², D0 in defects/cm², power in W.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HILP_DATA_DIR = REPO_ROOT / "input data" / "HILP_data"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

A_W = 70685.83  # 300 mm wafer area, mm²

# Area (mm²) and time (s) limits per workload
EXAMPLE_WORKLOAD_CONSTRAINTS = {
    "bfs":        {"time": 110, "area": 768},
    "heartwall":  {"time": 2,   "area": 768},
    "lud":        {"time": 14,  "area": 768},
    "myocyte":    {"time": 6,   "area": 768},
    "nn":         {"time": 10,  "area": 768},
    "pathfinder": {"time": 73,  "area": 768},
    "sc":         {"time": 3,   "area": 768},
    "lavaMD":     {"time": 4,   "area": 768},
}

_DESIGN_PATTERN = re.compile(r"c(\d+)_?g(\d+)(?:(?:_f|@)(\d+(?:\.\d+)?))?")


@dataclass(frozen=True)
class Design:
    """An SoC design: CPU cores plus GPU SMs at a GPU clock. CPU-only designs have gpu_sms = 0."""

    cpu_cores: int
    gpu_sms: int
    gpu_clock: float = 0.0

    @property
    def label(self) -> str:
        return f"c{self.cpu_cores}g{self.gpu_sms}"

    def __str__(self) -> str:
        return f"{self.label} @ {self.gpu_clock:g} MHz"

    @classmethod
    def parse(cls, text: str, gpu_clock: float | None = None) -> Design:
        """Parse 'c21g20', 'c21g20@765', or the optimization output format 'c21_g20_f765'."""
        m = _DESIGN_PATTERN.fullmatch(text.strip())
        if m is None:
            raise ValueError(f"Cannot parse design '{text}' (expected e.g. 'c21g20@765' or 'c21_g20_f765')")
        cores, sms, clock = int(m[1]), int(m[2]), m[3]
        if clock is None:
            clock = gpu_clock if gpu_clock is not None else 0.0
        if sms > 0 and float(clock) == 0.0:
            raise ValueError(f"Design '{text}' has GPU SMs but no GPU clock")
        return cls(cores, sms, float(clock))


@dataclass(frozen=True)
class CarbonParameters:
    """Nominal (mean) values of the carbon model parameters used by the MC evaluation."""

    CIfab: float = 0.495               # carbon intensity of fabrication
    CIuse: float = 0.2926              # carbon intensity of use
    EPW_c: float = 1234.32 / A_W       # energy per wafer, CPU cores (7 nm EUV)
    EPW_cIO: float = 896.2 / A_W       # energy per wafer, CPU IO (12 nm)
    EPW_g: float = 1234.32 / A_W       # energy per wafer, GPU SMs (7 nm EUV)
    GPW_c: float = 145.47 / A_W        # gases per wafer, CPU cores
    GPW_cIO: float = 96.12 / A_W       # gases per wafer, CPU IO
    GPW_g: float = 145.47 / A_W        # gases per wafer, GPU SMs
    MPW: float = 353.42915 / A_W       # materials per wafer (0.5 kg CO2e/cm²)
    D0: float = 0.15                   # defect density
    A_cpu_core: float = 10.12          # die area per CPU core (before yield)
    A_cpu_io: float = 6.5              # die area of CPU IO per core (before yield)
    A_gpu_sm: float = 7.65             # die area per GPU SM (before yield)
    P_cpu_core: float = 7.0            # power per CPU core


@dataclass
class MCConfig:
    """What to evaluate and how."""

    workload: str = "heartwall"
    Nwork: float = 5e4                 # number of workload runs before hardware disposal
    tC_threshold: float = 10.0         # total carbon threshold, kg CO2e
    nominal: Design = Design(21, 20, 765)
    robust: Design = Design(21, 3, 765)
    n_samples: int = 2000
    seed: int = 42
    distributions: tuple[str, ...] = ("Normal", "LogNormal", "Exponential", "Uniform", "Bernoulli", "Cauchy")
    summary_distribution: str = "Normal"  # distribution used for the per-design summaries, violations, and tCDP
    params: CarbonParameters = field(default_factory=CarbonParameters)

    @property
    def rhs(self) -> dict:
        return EXAMPLE_WORKLOAD_CONSTRAINTS[self.workload]


def load_designs_from_results(path: str | Path, workload: str, Nwork: float) -> tuple[Design, Design]:
    """Read the nominal and robust designs for `workload` at the Nwork closest to `Nwork`.

    Expects columns 'nwork', '<workload> - nominal' and '<workload> - robust', with designs
    in the optimization output format 'c3_g46_f765'.
    """
    results = pd.read_csv(path)
    results.columns = [c.strip().lower() for c in results.columns]
    nominal_col, robust_col = f"{workload.lower()} - nominal", f"{workload.lower()} - robust"
    missing = [c for c in ("nwork", nominal_col, robust_col) if c not in results.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(results.columns)}")
    nwork = pd.to_numeric(results["nwork"], errors="coerce")
    row = results.loc[(nwork - Nwork).abs().idxmin()]
    return Design.parse(row[nominal_col]), Design.parse(row[robust_col])
