include(joinpath(@__DIR__, "..", "input data", "HILP_data", "fetch_HILP_data.jl"))

""" Reading in CPU data """
cpu = CSV.read(HILP_CPU_CSV, DataFrame)
# extract configs column
cpu = transform(cpu, :Config => ByRow(cfg -> begin
    m = match(r"^([a-zA-Z]+)_(\d+)cores$", cfg)
    isnothing(m) ? (missing, missing) :
        (m.captures[1], parse(Int, m.captures[2]))
end) => [:Workload, :cores])

cpu = dropmissing(cpu, [:Workload, :cores])

""" Reading in GPU data """
gpu = CSV.read(HILP_GPU_CSV, DataFrame)
# extract configs column
gpu = transform(gpu, :Config => ByRow(cfg -> begin
    m = match(r"^([a-zA-Z]+)(\d+)clk_(\d+)sms$", cfg)
    isnothing(m) ? (missing, missing, missing) :
        (m.captures[1], parse(Int, m.captures[2]), parse(Int, m.captures[3]))
end) => [:Workload, :ClkFrequency, :SMs])

""" Filter data for the workload of this DSE (`workload` is set in main.jl) """
## filter CPU data
cpu_k = cpu[coalesce.(cpu.Workload .== workload, false), :];
## filter GPU data
gpu_k = gpu[coalesce.(gpu.Workload .== workload, false), :];

# # If you want to limit to specific clock frequencies 
# gpu_k = gpu_k[coalesce.(gpu_k.ClkFrequency .== 360.0, false), :]

