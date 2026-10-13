## Robust optimization models

| File | Contents |
|---|---|
| `robust_baseline_model.jl` | Decision variables, execution time, area, and the robust formulation of embodied and operational carbon shared by the robust models |
| `RO_opttC.jl` | Minimizes total carbon under performance and area constraints |
| `RO_optPerf_tCconstrained.jl` | Minimizes execution time under carbon and area constraints |
| `uncertainty sets/` | Data availability (DA) uncertainty sets for fab electricity, gases, and materials per wafer; data value (DV) uncertainty sets for operational carbon; and the code that loads the CAISO carbon intensity samples |

The models are run from [`models/main.jl`](../main.jl). Their parameters are in [`input parameters/input_parameters.jl`](../../input%20parameters/input_parameters.jl), including `Γ`, the robustness budget of the DV uncertainty sets.
