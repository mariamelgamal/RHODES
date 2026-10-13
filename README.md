# RHODES

This is the repository for our ISCA 2026 paper ["RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems"](https://ieeexplore.ieee.org/abstract/document/11617864).

## Repository structure

```
models/                    
  main.jl                  runs the nominal and robust models
  opt_constraints.jl       defines design constraints
  nominal/                 nominal optimization models
  robust/                  robust optimization models and uncertainty sets
input parameters/          fabrication, carbon, and area parameters shared by all models
input data/                input profiling data
evaluation/                Monte Carlo evaluation of nominal vs. robust designs
```

## Setup

### Julia

1. Install Julia from [https://julialang.org/downloads/](https://julialang.org/downloads/).
2. Install the required packages from Julia REPL:
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "JuMP", "Gurobi", "StatsBase", "CategoricalArrays", "Plots", "StatsPlots", "Distributions"])
```

### Gurobi (free academic license)

1. Create an account with your academic email at [https://www.gurobi.com/](https://www.gurobi.com/).
2. Request a "Named-User Academic" license (or whichever license applies to you), then follow Gurobi's instructions to install it on your machine.

To use a different solver, replace `Gurobi.Optimizer` in `models/nominal/nominal_baseline_model.jl` and `models/robust/robust_baseline_model.jl`, and replace or remove the Gurobi-specific settings such as `MIPGap` and `InfUnbdInfo`.

### Python (Monte Carlo evaluation only)

```
pip install -r evaluation/requirements.txt
```

## Usage

```
julia models/main.jl
```

For one workload, this runs four models and prints the selected design (CPU cores, GPU SMs, GPU clock) with its area, execution time, and total carbon:
1. Nominal: minimize total carbon (performance-constrained)
2. Nominal: minimize execution time (carbon-constrained)
3. Robust: minimize total carbon (performance-constrained)
4. Robust: minimize execution time (carbon-constrained)

We provide a few examples, which you can also update to suit your modeling:
- **Workload**: set `workload` at the top of `models/main.jl` (`bfs`, `heartwall`, `lavaMD`, `lud`, `myocyte`, `nn`, `pathfinder`, or `sc`).
- **Constraints**: edit `t_max`, `tC_max`, and `a_max` in `models/opt_constraints.jl`.
- **Parameters**: edit `input parameters/input_parameters.jl`, for example `Nwork` (how many times the workload runs before the hardware is retired) or `Γ` (the robustness budget).

For the Monte Carlo evaluation, see [evaluation/README.md](evaluation/README.md).
## Citation

If you use this work or find it useful, please cite our ISCA paper:

```bibtex
@inproceedings{Elgamal2026ISCA,
author = {Elgamal, Mariam and Mahmoud, Abdulrahman and Wei, Gu-Yeon and Brooks, David and Hills, Gage},
title = {RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems},
year = {2026},
isbn = {},
publisher = {Association for Computing Machinery},
address = {USA},
url = {https://ieeexplore.ieee.org/abstract/document/11617864},
doi = {10.1109/ISCA66397.2026.00028},
booktitle = {2026 ACM/IEEE 53rd Annual International Symposium on Computer Architecture (ISCA)},
pages = {179-193},
location = {Raleigh, NC},
series = {ISCA '26}
}
```
