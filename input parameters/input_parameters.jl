## Predefined settings shared by the nominal and robust optimization models

# Nwork: Amount of fixed work to be done before hardware disposal (e.g. number of cycles or tokens)
# Nwork can also be accounted as an uncertainty when computed with CIuse
Nwork = 1000 # number of times this workload will be run before hardware disposal

# setup_teardown_scaling: scales CPU setup and teardown times (0.2 = 5x faster)
setup_teardown_scaling = 0.2

# Γ: robustness budget (controls protection level) used by the DV uncertainty sets (robust models only)
Γ = 1.0

"""
Initialize all global parameters.
"""
function init_parameters(workload::String)
    println("──────────────────────────────────────────────────")
    println(" Initializing DSE fabrication & carbon parameters")
    println("──────────────────────────────────────────────────")

    # parameters need to be defined as global
    global A_w, N_c, N_g, EPW_c, EPW_cIO, EPW_g
    global GPW_c, GPW_cIO, GPW_g, MPW
    global D0, CIfab, CIuse
    global T_cs, T_ck, T_ct, T_g
    global P_c, P_g, B_cs, B_ck, B_ct, B_g
    global A_c, A_cIO, A_g
    global EPW_c_dev_p, EPW_c_dev_n, EPW_IO_dev_p, EPW_IO_dev_n, EPW_g_dev_p, EPW_g_dev_n
    global GPW_c_dev_p, GPW_c_dev_n, GPW_IO_dev_p, GPW_IO_dev_n, GPW_g_dev_p, GPW_g_dev_n
    global MPW_dev_p, MPW_dev_n
    global rho
    global FPW_c, FPW_cIO, FPW_g
    global FPW_c_dev_p, FPW_c_dev_n, FPW_IO_dev_p, FPW_IO_dev_n, FPW_g_dev_p, FPW_g_dev_n
    global HBM_GB, HBM_dev_extremep, HBM_dev_p, HBM_dev_n  # ← ADD THIS
    
    # 300 mm wafer area in mm2
    A_w = 70685.83
    
    # N_c: number of CPU cores
    N_c = cpu_k[:,9]
    
    # N_g: is number of GPU SMs
    N_g = gpu_k[:,7]
    
    ## accelerators are when modeling workload level parallelism by implementing HILP's methodology (not in this repo)
    # N_d: number of DSA accelerators
    # N_d = 10
    # accelerators: range of accelerators
    # accelerators = 1:N_d
    # PE_range = [2, 4, 16, 32, 64]
    # PEs = 1:length(PE_range)

    # CIfab: Carbon intensity of fabrication facility
    CIfab = 0.380 # 0.2926 # 0.8 # 0.096 # kgCO2e
    
    # CIuse: Carbon intensity of use
    CIuse = 0.380 # 0.2926 # 0.8 # 0.096 # kgCO2e

    CI = 0.380 # 0.527 # kgCO2e
    CI_p = 0.562
    CI_n = 0.474
    
    # EPW: Electrical energy consumption per wafer during fabrication
    EPW_c = (1234.32) / A_w
    EPW_cIO = (896.2) / A_w
    EPW_g = (1234.32) / A_w
    
    # GPW: Gases Per Wafer are direct gases emitted during IC fabrication (here with abatement)
    GPW_c = 114.33 / A_w 
    GPW_cIO = 91.09 / A_w
    GPW_g = 114.33 / A_w
    # without abatement
    # GPW_c = 604.47 / A_w
    # GPW_cIO = 511.05 / A_w
    # GPW_g = 604.47 / A_w
    
    # MPW: Materials Per Wafer are emissions due to materials procured for fabrication
    MPW = 353.42915 / A_w  # kgCO2e per wafer (0.5 * 706.8583), 0.5 kgCO2e per cm2
    
    # D0: defect density
    D0 = 0.15 #percm2
    
    # T: Execution time of workload k on compute unit n
    T_cs = cpu_k[:,2] * setup_teardown_scaling # setup
    T_ck = cpu_k[:,3] # kernel
    T_ct = cpu_k[:,4] * setup_teardown_scaling # teardown
    T_g = gpu_k[:,2] 
    
    # P: Power consumption of workload k on compute unit n
    P_c = N_c * 7 # 7W per core
    P_g = gpu_k[:,4]
    
    # B: Memory bandwidth required per workload k on compute unit n
    B_cs = cpu_k[:,5] # setup
    B_ck = cpu_k[:,6] # kernel
    B_ct = cpu_k[:,7] # teardown
    B_g = gpu_k[:,3]
    
    # A: Die area per compute unit n in mm2
    DA_c = N_c * 10.12 
    DA_cIO = N_c * 6.5
    DA_g = N_g * 7.65;

    Y_c = @. exp(-DA_c/100*D0)
    Y_cIO = @. exp(-DA_cIO/100*D0)
    Y_g = @. exp(-DA_g/100*D0)

    A_c = @. DA_c / Y_c
    A_cIO = @. DA_cIO / Y_cIO
    A_g = @. DA_g / Y_g

    ## EPW uncertainty
    # upper bound for EPW cpu core (7 nm EUV) uncertainty
    EPW_c_dev_p = EPW_c + ((345.87) / A_w)
    # lower bound for EPW cpu core (7 nm EUV) uncertainty
    EPW_c_dev_n = EPW_c - ((41.32) / A_w)
    # upper bound for EPW IO (12 nm) 
    EPW_IO_dev_p = EPW_cIO + ((148.54) / A_w)
    # lower bound for EPW IO (12 nm)
    EPW_IO_dev_n = EPW_cIO - ((93.20) / A_w)
    # upper bound for EPW GPU SM (7 nm EUV)
    EPW_g_dev_p = EPW_g + ((345.87) / A_w)
    # lower bound for EPW GPU SM (7 nm EUV)
    EPW_g_dev_n = EPW_g - ((41.32) / A_w)

    # EPW: Electrical energy consumption per wafer during fabrication
    FPW_c = CI * (1234.32) / A_w
    FPW_cIO = CI * (896.2) / A_w
    FPW_g = CI * (1234.32) / A_w
    
    ## FPW uncertainty
    # upper bound for EPW cpu core (7 nm EUV) uncertainty
    FPW_c_dev_p = CI_p * (EPW_c + ((345.87) / A_w)) #((374.34) / A_w)) #((345.87) / A_w)) # for DUV
    # lower bound for EPW cpu core (7 nm EUV) uncertainty
    FPW_c_dev_n = CI_n * (EPW_c - ((41.32) / A_w))
    # upper bound for EPW IO (12 nm) 
    FPW_IO_dev_p = CI_p * (EPW_cIO + ((148.54) / A_w))
    # lower bound for EPW IO (12 nm)
    FPW_IO_dev_n = CI_n * (EPW_cIO - ((93.20) / A_w))
    # upper bound for EPW GPU SM (7 nm EUV)
    FPW_g_dev_p = CI_p * (EPW_g + ((374.34) / A_w)) #((345.87) / A_w)) # for DUV
    # lower bound for EPW GPU SM (7 nm EUV)
    FPW_g_dev_n = CI_n * (EPW_g - ((41.32) / A_w))

    ## GPW uncertainty
    # upper bound for GPW cpu core (7 nm EUV) uncertainty
    GPW_c_dev_p = GPW_c + ((132.77) / A_w)
    GPW_c_dev_n = GPW_c - ((48.63) / A_w)
    GPW_IO_dev_p = GPW_cIO + ((50.11) / A_w)
    GPW_IO_dev_n = GPW_cIO - ((38.59) / A_w)
    GPW_g_dev_p = GPW_g + ((132.77) / A_w)
    GPW_g_dev_n = GPW_g - ((48.63) / A_w)

    ## MPW uncertainty    
    MPW_dev_p =  MPW + (((353.42915 * 1.1)-353.42915) / A_w)
    MPW_dev_n = MPW - ((353.42915-(353.42915 * 0.9))/ A_w)

    # ## HBM
    # HBM_GB = 64.0
    # ## HBM uncertainty per GB
    # HBM_dev_extremep = 0.71
    # HBM_dev_p = 0.33
    # HBM_dev_n = 0.24
    
    rho = 1.0

    println("Wafer area (A_w): ", A_w)
    println("Defect density (D0): ", D0, "-- With Yield")
    println("CI: ", CI, " kgCO₂e/kWh")
    println("EPW_c: ", EPW_c, ", EPW_g: ", EPW_g)
    println("GPW_c: ", GPW_c, ", GPW_g: ", GPW_g, " -- without abatement")
    println("MPW: ", MPW)
    # println("HBM: ", )
    println("Loaded CPU configs for workload: ", workload, " → ", nrow(cpu_k), " entries")
    println("Loaded GPU configs for workload: ", workload, " → ", nrow(gpu_k), " entries")
    println("──────────────────────────────────────────────────")
end