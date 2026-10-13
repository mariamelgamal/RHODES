"""
Optimize performance under carbon-constrained robust optimization.
This code can be modified to optimize for other metrics such as power, area, energy, etc.
"""
function robust_carbon_constrained(; a_max, tC_max)
    # call baseline model
    model, c, g, x, w, z, total_exectime, total_area, total_carbon,
    uFPW_c, uFPW_cIO, uFPW_g,
    uGPW_c, uGPW_cIO, uMPW_c,
    uGPW_g, uMPW_g,
    uCop_c_setup, uCop_g, uCop_c_kernel, embodied_c, embodied_g, operational_c, operational_g = robust_baseline_model(Nwork)

    set_optimizer_attribute(model, "InfUnbdInfo", 1)

    # objective function
    @objective(model, Min, total_exectime)

    # constraints
    @constraint(model, total_carbon_constraint, total_carbon <= tC_max)
    @constraint(model, area_constraint, total_area <= a_max)

    # optimize model
    optimize!(model)

    # check solver status
    term_status = termination_status(model)

    if term_status != MOI.OPTIMAL && term_status != MOI.LOCALLY_SOLVED
        println("No feasible/optimal solution for Nwork = $Nwork (status: $term_status)")
        return (
            cpu_cores=missing, gpu_sms=missing, gpu_clk=missing,
            objective=missing, time=missing, power=missing, area=missing,
            bw=missing, total_carbon=missing, embodied_c=missing,
            embodied_g=missing, operational_c=missing, operational_g=missing
        )
    end

    ## guardrails
    @assert all(value.(uCop_c_setup) .>= 0)  # setup must be ≥ 0 
    @assert all(value.(uCop_c_kernel) .>= 0) # kernel must be ≥ 0
    @assert all(value.(uCop_g) .>= 0)        # gpu must be ≥ 0 kg

    # dimensions expected
    @assert length(uGPW_c) == length(A_c) == length(c)
    @assert length(uGPW_g) == length(A_g) == length(z)
    
    term = JuMP.termination_status(model)
    prim = JuMP.primal_status(model)
    
    println("Termination: ", term, " | Primal: ", prim)
    
    if !(prim in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT))
        error("No feasible solution. (termination=$term, primal=$prim)")
    end

    # get optimal results
    c_opt = value.(c); g_opt = value.(g); w_opt = value.(w); z_opt = value.(z)
    obj = objective_value(model)

    # convert to vectors for index search
    c_vec = collect(c_opt)    
    g_vec = collect(g_opt)
    
    i_sel = findfirst(x -> x > 0.5, c_vec)  # CPU may be 1 or greater (==1 picked)
    j_sel = findfirst(x -> x > 0.5, g_vec)  # GPU may be zero if x=0, or greater

    # find optimal CPU cores, GPU SMs, and clock frequencies
    cpu_cores = isnothing(i_sel) ? 0 : cpu_k[i_sel, 9]
    gpu_sms   = isnothing(j_sel) ? 0 : gpu_k[j_sel, 7]
    gpu_clk   = isnothing(j_sel) ? 0 : gpu_k[j_sel, 6]    

    # print out optimal design specifications results
    print_specifications(cpu_cores, gpu_sms, gpu_clk, obj, value(total_exectime), value(total_area), value(total_carbon))
    
    return (
        cpu_cores    = cpu_cores,
        gpu_sms      = gpu_sms,
        gpu_clk      = gpu_clk,
        objective    = obj,
        time         = value(total_exectime),
        area         = value(total_area),
        total_carbon = value(total_carbon),
        embodied_c   = sum(value.(embodied_c)),
        embodied_g   = sum(value.(embodied_g)),
        operational_c = sum(value.(operational_c)),
        operational_g = sum(value.(operational_g)),
    )
end