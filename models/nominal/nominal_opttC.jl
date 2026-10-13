"""
Minimize total carbon under performance-constrained nominal optimization (no uncertainty).
"""
function nominal_opt_tC(; t_max, a_max) # uses keyword arguments instead of positionals
    # baseline model
    model, c, g, x, w, z, total_exectime, total_area, total_carbon,
    embodied_c, embodied_g, operational_c, operational_g = nominal_baseline_model(Nwork)
    # println("Nwork here is: $Nwork")

    # objective function
    @objective(model, Min, total_carbon)

    # constraints
    @constraint(model, performance_constraint, total_exectime <= t_max)
    @constraint(model, area_constraint, total_area <= a_max)

    # optimize model
    optimize!(model)

    # check solver status
    term_status = termination_status(model)

    if term_status != MOI.OPTIMAL && term_status != MOI.LOCALLY_SOLVED
        println("No feasible/optimal solution for Nwork = $Nwork (status: $term_status)")
        return (
            cpu_cores=missing, gpu_sms=missing, gpu_clk=missing,
            objective=missing, time=missing, area=missing,
            total_carbon=missing, embodied_c=missing,
            embodied_g=missing, operational_c=missing, operational_g=missing
        )
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
    
    # return a named tuple
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