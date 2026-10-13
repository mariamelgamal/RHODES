"""
The baseline model for all RO formulations.
Defines the decision variables and baseline calculations.
"""
function robust_baseline_model(Nwork)
    rng = MersenneTwister(1234)  
    
    n_cpu = 1:nrow(cpu_k)
    n_gpu = 1:nrow(gpu_k)

    ## determine which CI year data to account for in uncertainty set
    ci_row = all_results_matrix[1, :]
    valid_idx = findall(!isnan, ci_row)
    CIuse_2022 = ci_row[valid_idx]        # 1 × N_2022_clean
    
    # replicate for CPUs and GPUs
    n_cpus  = nrow(cpu_k)
    n_gpus  = nrow(gpu_k)
    CIuse_samples_c = repeat(reshape(CIuse_2022, 1, :), n_cpus, 1)
    CIuse_samples_g = repeat(reshape(CIuse_2022, 1, :), n_gpus, 1)

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

    # # Encode data availability uncertainties
    uFPW_c, model = DA_FPW(model, FPW_c, c, n_cpu, rho, FPW_c_dev_p, FPW_c_dev_n)
    uFPW_cIO, model = DA_FPW(model, FPW_cIO, c, n_cpu, rho, FPW_IO_dev_p, FPW_IO_dev_n)
    uFPW_g, model = DA_FPW(model, FPW_g, z, n_gpu, rho, FPW_g_dev_p, FPW_g_dev_n)
    uGPW_c, model = DA_GPW(model, GPW_c, c, n_cpu, rho, GPW_c_dev_p, GPW_c_dev_n)
    uGPW_cIO, model = DA_GPW(model, GPW_cIO, c, n_cpu, rho, GPW_IO_dev_p, GPW_IO_dev_n)
    uGPW_g, model = DA_GPW(model, GPW_g, z, n_gpu, rho, GPW_g_dev_p, GPW_g_dev_n)
    uMPW_c, model = DA_MPW(model, MPW, c, n_cpu, rho, MPW_dev_p, MPW_dev_n)
    uMPW_g, model = DA_MPW(model, MPW, z, n_gpu, rho, MPW_dev_p, MPW_dev_n)

    # # HBM embodied carbon (GB × kg CO2e/GB with uncertainty)
    # HBM_nominal_per_gb = (HBM_dev_p + HBM_dev_n) / 2  # Or use HBM_dev_n as nominal
    # uHBM_c, model = DA_HBM(model, HBM_nominal_per_gb, c, n_cpu, rho, HBM_dev_p, HBM_dev_n)
    # # GPU configs don't add extra HBM (already included in system)

    uCop_c_setup, model = DV_Cop_setup(model, c, CIuse_samples_c, Nwork)           
    uCop_c_kernel, model = DV_Cop_kernel(model, (c .- w), CIuse_samples_c, Nwork)  
    uCop_g, model = DV_Cop_g(model, z, CIuse_samples_g, Nwork)                    

    M = 1e13
    # CPU kernel:
    @constraint(model, uCop_c_kernel <= M * sum(c .- w))
    # GPU:
    @constraint(model, uCop_g        <= M * sum(z))

    # GPW, EPW, MPW uncertainties
    embodied_c = sum(uFPW_c .* A_c) + sum(uFPW_cIO .* A_cIO) + sum(uGPW_c .* A_c) + sum(uGPW_cIO .* A_cIO) + sum(uMPW_c .* (A_c .+ A_cIO))
    embodied_g = sum(uFPW_g .* A_g) + sum(uGPW_g .* A_g) + sum(uMPW_g .* A_g)
    # embodied_hbm = sum(uHBM_c) * HBM_GB  # ADD HBM

    ## operational carbon calculation
    operational_c = uCop_c_setup + uCop_c_kernel # include mem if accounting for HBM in the Cop calc
    operational_g = uCop_g                       # include mem if accounting for HBM in the Cop calc

    ## total carbon calculation
    total_carbon = sum(embodied_c) + sum(operational_c) + sum(embodied_g) + sum(operational_g) #+ sum(embodied_hbm) 

    return model, c, g, x, w, z, total_exectime, total_area, total_carbon, uFPW_c, uFPW_cIO, uFPW_g,
       uGPW_c, uGPW_cIO, uMPW_c,
       uGPW_g, uMPW_g, #uHBM_c,
       uCop_c_setup, uCop_g, uCop_c_kernel, embodied_c, embodied_g, operational_c, operational_g
end