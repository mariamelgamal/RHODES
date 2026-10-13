"""
Print results of the optimal SoC design specifications of the optimization model.
"""
function print_specifications(opt_cpu_cores, opt_gpu_sms, opt_gpu_clk, opt_obj, total_exectime, total_area, total_carbon)
    println("===============================================================================")
    println("==========                 Optimal SoC Specifications                ==========")
    println("===============================================================================")
    println("Selected number of optimal CPU cores: ", opt_cpu_cores)
    println("Selected number of optimal GPU SMs: ", opt_gpu_sms)
    println("Clock frequency for the GPU SMs: ", opt_gpu_clk, " MHz (baseline frequency is 765 MHz)")
    println("-------------------------------------------------------------------------------")
    println("The corresponding optimal objective value: ", round(opt_obj, digits=3))
    println("Total area of SoC: ", total_area, " mm²")
    println("Total execution time: ", total_exectime, " seconds")
    println("Total carbon of SoC: ", total_carbon, " kg CO₂e per SoC")
    println("===============================================================================")
    return
end