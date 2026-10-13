"""
DA uncertainty set of GPW in embodied carbon. 
"""
function DA_GPW(model, GPW, x, n, rho, dev_p, dev_n) # A, c)
    p = @variable(model, [n], base_name = "p_GPW", lower_bound = 0)
    q = @variable(model, [n], base_name = "q_GPW", lower_bound = 0)
    @constraint(model, [i in n], p[i] - q[i] == x[i])
    GPW_lr = ((p .* dev_p) .- (q .* dev_n))
    return GPW_lr, model
end
 
"""
DA uncertainty set of MPW in embodied carbon. 
"""
# accounting for the case where there is one datapoint across all process technology nodes
function DA_MPW(model, MPW, x, n, rho, dev_p, dev_n) # A, c)
    p = @variable(model, [n], base_name = "p_MPW", lower_bound = 0)
    q = @variable(model, [n], base_name = "q_MPW", lower_bound = 0)
    @constraint(model, [i in n], p[i] - q[i] == x[i])
    MPW_lr = ((p .* dev_p) .- (q .* dev_n))
    return MPW_lr, model
end

"""
FPW if accounted as a DA uncertainty set in embodied carbon. 
"""
function DA_FPW(model, FPW, x, n, rho, dev_p, dev_n) 
    p = @variable(model, [n], base_name = "p_FPW", lower_bound = 0)
    q = @variable(model, [n], base_name = "q_FPW", lower_bound = 0)
    @constraint(model, [i in n], p[i] - q[i] == x[i])
    FPW_lr = ((p .* dev_p) .- (q .* dev_n))
    return FPW_lr, model
end

## you can account for other uncertain parameters as DA uncertainties, depending on your data and modeling conditions