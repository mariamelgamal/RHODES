# tight original envelope

"""
DV uncertainty set of CPU setup and teardown operational carbon. 
"""
function DV_Cop_setup(model, mask, CIuse_samples, Nwork)
    n, N_data = size(CIuse_samples)  # n is the number of designs, N_data is the number of scenarios
    @variable(model, Cop_setup >= 0)            # Objective variable t
    @variable(model, α_setup >= 0)    # Dual variable α for the probabilistic constraint
    @variable(model, β_setup[1:n] >= 0)     # Dual variables β for each design j

    for i in 1:n
        # Sort samples descending by loss magnitude
        sorted_CI = sort(CIuse_samples[i, :], rev = true)
        for s in 1:N_data
            log_term = -log((N_data + 1 - s) / N_data)
            Cop_expr = Nwork * sorted_CI[s] * P_c[i] * (T_cs[i] + T_ct[i]) / 3_600_000
            @constraint(model, α_setup * log_term + β_setup[i] >= Cop_expr * mask[i])
        end
    end

    # Constraint for robust objective (dual feasibility)
    @constraint(model, Γ * α_setup + sum(β_setup[i] for i in 1:n) <= Cop_setup)

    return Cop_setup, model
end


"""
DV uncertainty setof CPU kernel operational carbon. 
"""
function DV_Cop_kernel(model, mask, CIuse_samples, Nwork)
    n, N_data = size(CIuse_samples)  # n is the number of designs, N_data is the number of scenarios
    @variable(model, Cop_kernel >= 0)            # Objective variable t
    @variable(model, α_kernel >= 0)    # Dual variable α for the probabilistic constraint
    @variable(model, β_kernel[1:n] >= 0)     # Dual variables β for each design j

    for i in 1:n
        # Sort samples descending by loss magnitude
        sorted_CI = sort(CIuse_samples[i, :], rev = true)
        for s in 1:N_data
            log_term = -log((N_data + 1 - s) / N_data)
            Cop_expr = Nwork * sorted_CI[s] * P_c[i] * (T_ck[i]) / 3_600_000
            @constraint(model, α_kernel * log_term + β_kernel[i] >= Cop_expr * mask[i])
        end
    end

    # Constraint for robust objective (dual feasibility)
    @constraint(model, Γ * α_kernel + sum(β_kernel[i] for i in 1:n) <= Cop_kernel)

    return Cop_kernel, model
end


"""
DV uncertainty set of GPU operational carbon. 
"""
function DV_Cop_g(model, mask, CIuse_samples, Nwork)
    n, N_data = size(CIuse_samples)     # n is the number of designs, N_data is the number of scenarios
    @variable(model, Cop_g >= 0)        # Objective variable t
    @variable(model, α_g >= 0)          # Dual variable α for the probabilistic constraint
    @variable(model, β_g[1:n] >= 0)     # Dual variables β for each design j

    for i in 1:n
        # Sort samples descending by loss magnitude
        sorted_CI = sort(CIuse_samples[i, :], rev = true)
        for s in 1:N_data
            log_term = -log((N_data + 1 - s) / N_data)
            Cop_expr = Nwork * sorted_CI[s] * P_g[i] * (T_g[i]) / 3_600_000
            @constraint(model, α_g * log_term + β_g[i] >= Cop_expr * mask[i])
        end
    end

    # Constraint for robust objective (dual feasibility)
    @constraint(model, Γ * α_g + sum(β_g[i] for i in 1:n) <= Cop_g)

    return Cop_g, model
end

"""
DV uncertainty set of FPW.
Draws a random subset of scenarios for each i to keep model size manageable.
"""
function DV_FPW_MC(model, mask, A, samples; base = "FPW")
    n, N_data = size(samples)
    FPW_var = @variable(model, lower_bound = 0, base_name = "$(base)")
    α       = @variable(model, lower_bound = 0, base_name = "$(base)_alpha")
    β       = @variable(model, [i=1:n], lower_bound = 0, base_name = "$(base)_beta")

    log_terms     = [-log((N_data + 1 - s)/N_data) for s in 1:N_data]

    for i in 1:n
        FPW_i_sorted   = sort(samples[i, :], rev = true) / A_w
        for s in 1:N_data
            FPW_expr = FPW_i_sorted[s] * A[i] 
            @constraint(model, α * log_terms[s] + β[i] >= FPW_expr * mask[i])
        end
    end
    
    # Dual feasibility constraint
    @constraint(model, Γ * α + sum(β[i] for i in 1:n) <= FPW_var)
    
    return FPW_var, model
end
