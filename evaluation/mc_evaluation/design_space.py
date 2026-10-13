"""Build the CPU+GPU design space for one workload from the HILP profiling data.

Each row is one design: a CPU configuration (1-32 cores) paired with either a GPU
configuration (clock, SMs) or no GPU. With a GPU, the CPU runs setup and teardown and
the GPU runs the kernel; without one, the CPU runs all three phases.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HILP_DATA_DIR, CarbonParameters

# Same pinned source as input data/HILP_data/fetch_HILP_data.jl
HILP_COMMIT = "dbd3894fb8e9b5740cdc4ee6b92a6ed1fef83fdd"
HILP_URL = f"https://raw.githubusercontent.com/EECS-NTNU/hilp/{HILP_COMMIT}/rodinia_profiled_data"
HILP_FILES = {"rodinia_cpu_data.csv": "cpu_data.csv", "rodinia_gpu_data.csv": "gpu_data.csv"}

KWH_PER_WS = 1.0 / 3_600_000.0


def ensure_hilp_data(data_dir: Path = HILP_DATA_DIR) -> tuple[Path, Path]:
    """Return the CPU and GPU CSV paths, downloading them from the HILP repo if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for local_name, hilp_name in HILP_FILES.items():
        path = data_dir / local_name
        if not path.is_file():
            print(f"Downloading {hilp_name} from the HILP repo (commit {HILP_COMMIT[:7]})")
            urllib.request.urlretrieve(f"{HILP_URL}/{hilp_name}", path)
    return data_dir / "rodinia_cpu_data.csv", data_dir / "rodinia_gpu_data.csv"


def load_profiles(workload: str, data_dir: Path = HILP_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the HILP CPU and GPU profiles for one workload."""
    cpu_path, gpu_path = ensure_hilp_data(data_dir)

    cpu = pd.read_csv(cpu_path)
    cpu_parts = cpu["Config"].str.extract(r"^(?P<kernel>.+)_(?P<cores>\d+)cores$")
    cpu = cpu[cpu_parts["kernel"] == workload].copy()
    cpu["CPU_cores"] = cpu_parts.loc[cpu.index, "cores"].astype(int)

    gpu = pd.read_csv(gpu_path)
    # non-greedy kernel name so e.g. 'hotspot3D210clk_8sms' parses as kernel 'hotspot3D'
    gpu_parts = gpu["Config"].str.extract(r"^(?P<kernel>.+?)(?P<clock>\d+)clk_(?P<sms>\d+)sms$")
    gpu = gpu[gpu_parts["kernel"] == workload].copy()
    gpu["GPU_Clock"] = gpu_parts.loc[gpu.index, "clock"].astype(float)
    gpu["GPU_SMs"] = gpu_parts.loc[gpu.index, "sms"].astype(int)

    if cpu.empty or gpu.empty:
        kernels = sorted(cpu_parts["kernel"].dropna().unique())
        raise ValueError(f"No HILP data for workload '{workload}'. Available: {kernels}")

    cpu = cpu.rename(columns={
        "Config": "CPU_Config",
        "SetupTime (s)": "CPU_SetupTime (s)",
        "KernelTime (s)": "CPU_KernelTime (s)",
        "TeardownTime (s)": "CPU_TeardownTime (s)",
    })
    gpu = gpu.rename(columns={
        "Config": "GPU_Config",
        "KernelTime (s)": "GPU_KernelTime (s)",
        "Power Usage (W)": "GPU_Power Usage (W)",
        "Bandwidth (GB/s)": "GPUBandwidth (GB/s)",
    })
    return cpu.reset_index(drop=True), gpu.reset_index(drop=True)


def yield_adjusted_area(die_area, D0: float):
    """Area including yield loss: die_area / Y, with Poisson yield Y = exp(-die_area[cm²] * D0)."""
    return die_area / np.exp(-die_area / 100 * D0)


def build_design_space(
    workload: str,
    Nwork: float,
    params: CarbonParameters,
    data_dir: Path = HILP_DATA_DIR,
) -> pd.DataFrame:
    """All CPU+GPU and CPU-only designs with time, area, and deterministic carbon (at the nominal parameters)."""
    cpu, gpu = load_profiles(workload, data_dir)

    with_gpu = cpu.merge(gpu, how="cross")
    cpu_only = cpu.assign(**{
        "GPU_Config": 0, "GPU_KernelTime (s)": 0.0, "GPU_Clock": 0.0, "GPU_SMs": 0,
        "GPU_Power Usage (W)": 0.0, "GPUBandwidth (GB/s)": 0.0,
    })
    designs = pd.concat([with_gpu, cpu_only], ignore_index=True)
    has_gpu = designs["GPU_SMs"] > 0

    # With a GPU, the GPU runs the kernel instead of the CPU
    designs.loc[has_gpu, "CPU_KernelTime (s)"] = 0.0
    cpu_time = designs["CPU_SetupTime (s)"] + designs["CPU_KernelTime (s)"] + designs["CPU_TeardownTime (s)"]
    designs["TotalTime (s)"] = cpu_time + designs["GPU_KernelTime (s)"]
    designs["Total_bandwidth"] = (
        designs["SetupBandwidth (GB/s)"] + designs["TeardownBandwidth (GB/s)"]
        + np.where(has_gpu, designs["GPUBandwidth (GB/s)"], designs["KernelBandwidth (GB/s)"])
    )

    p = params
    cores, sms = designs["CPU_cores"], designs["GPU_SMs"]
    A_c = yield_adjusted_area(cores * p.A_cpu_core, p.D0)
    A_cIO = yield_adjusted_area(cores * p.A_cpu_io, p.D0)
    A_g = yield_adjusted_area(sms * p.A_gpu_sm, p.D0)
    P_c = cores * p.P_cpu_core

    designs["calc_power"] = P_c + designs["GPU_Power Usage (W)"]
    designs["Area_c"] = A_c + A_cIO
    designs["Area_g"] = A_g
    designs["Total_area"] = designs["Area_c"] + designs["Area_g"]
    designs["Embodied_c"] = (
        p.CIfab * (p.EPW_c * A_c) + p.CIfab * (p.EPW_cIO * A_cIO)
        + p.GPW_c * A_c + p.GPW_cIO * A_cIO + p.MPW * (A_c + A_cIO)
    )
    designs["Embodied_g"] = p.CIfab * (p.EPW_g * A_g) + p.GPW_g * A_g + p.MPW * A_g
    designs["Energy_c_perrun"] = P_c * cpu_time * KWH_PER_WS
    designs["Energy_g_perrun"] = designs["GPU_Power Usage (W)"] * designs["GPU_KernelTime (s)"] * KWH_PER_WS
    designs["Operational_c_perrun"] = p.CIuse * designs["Energy_c_perrun"]
    designs["Operational_g_perrun"] = p.CIuse * designs["Energy_g_perrun"]
    designs["Total_embodied"] = designs["Embodied_c"] + designs["Embodied_g"]
    designs["Total_operational_perrun"] = designs["Operational_c_perrun"] + designs["Operational_g_perrun"]
    designs["Total_carbon"] = designs["Total_embodied"] + designs["Total_operational_perrun"] * Nwork

    designs.insert(0, "total_config", "c" + cores.astype(str) + "g" + sms.astype(str))
    return designs
