"""
The baseline model for all nominal optimization formulations.
Defines the decision variables and baseline calculations.
"""
function nominal_baseline_model(Nwork)
    n_cpu = 1:nrow(cpu_k)
    n_gpu = 1:nrow(gpu_k)

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # Ask for a near-exact optimum
    set_optimizer_attribute(model, "MIPGap", 0.0)        # or 1e-6

    # decision variables
    ## primary variables for number of CPU cores, GPU SMs, and whether the system has a GPU or not
    @variable(model, c[n_cpu], Bin)
    @variable(model, g[n_gpu], Bin)
    @variable(model, x, Bin)  # 1 = GPU selected, 0 = CPU selected for kernel execution               
    ## auxillary variables for linearization of quadratic products with x - relaxed
    @variable(model, 0 <= w[i in n_cpu] <= 1)
    @variable(model, 0 <= z[j in n_gpu] <= 1)

    # linearization constraints
    ## for CPU: w[i] = c[i] * x
    @constraint(model, [i in n_cpu], w[i] <= c[i])
    @constraint(model, [i in n_cpu], w[i] <= x)
    @constraint(model, [i in n_cpu], w[i] >= c[i] + x - 1)
    ## for GPU: z[j] = g[j] * x
    @constraint(model, [j in n_gpu], z[j] <= g[j])
    @constraint(model, [j in n_gpu], z[j] <= x)
    @constraint(model, [j in n_gpu], z[j] >= g[j] + x - 1)

    # selection constraints
    ## each SoC must have at least one CPU core, if x is 1 then select a GPU config
    @constraint(model, sum(c[i] for i in n_cpu) == 1)
    @constraint(model, sum(g[j] for j in n_gpu) == x)

    # SoC calculations
    ## execution time calculation (i.e. performance)
    exectime_c = (T_cs .+ T_ct) .* c .+ T_ck .* (c .- w)
    exectime_g = T_g .* z
    total_exectime = sum(exectime_c) + sum(exectime_g)

    ## area calculation
    area_c = (A_c .+ A_cIO) .* c
    area_g = A_g .* z
    total_area = sum(area_c) + sum(area_g)

    ## embodied carbon calculation
    embodied_c = (CIfab .* (EPW_c .* A_c) .+ CIfab .* (EPW_cIO .* A_cIO) .+ (GPW_c .* A_c) + (GPW_cIO .* A_cIO) .+ (MPW .* (A_c .+ A_cIO))) .* c
    embodied_g = (CIfab .* (EPW_g .* A_g) .+ (GPW_g .* A_g) .+ (MPW .* A_g)) .* z

    ## operational carbon calculation
    operational_c = ((CIuse .* P_c) ./ 3_600_000) .* ((T_cs .+ T_ct) .* c .+ T_ck .* (c .- w))
    operational_g = (CIuse .* P_g .* T_g .* z ./ 3_600_000)

    ## total carbon calculation
    total_carbon = sum(embodied_c) + sum(operational_c)*Nwork + sum(embodied_g) + sum(operational_g)*Nwork

    return model, c, g, x, w, z, total_exectime, total_area, total_carbon,
    embodied_c, embodied_g, operational_c, operational_g
end