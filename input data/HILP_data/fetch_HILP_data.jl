""" Fetch the Rodinia profiled CPU/GPU data from the HILP repo """
# Source: https://github.com/EECS-NTNU/hilp/tree/master/rodinia_profiled_data
# The data is downloaded on first use rather than redistributed in this repo.
# Pinned to a fixed commit so results are reproducible even if HILP changes upstream.
using Downloads

HILP_COMMIT = "dbd3894fb8e9b5740cdc4ee6b92a6ed1fef83fdd"
HILP_CPU_CSV = joinpath(@__DIR__, "rodinia_cpu_data.csv")
HILP_GPU_CSV = joinpath(@__DIR__, "rodinia_gpu_data.csv")

for (hilp_name, local_path) in [("cpu_data.csv", HILP_CPU_CSV), ("gpu_data.csv", HILP_GPU_CSV)]
    if !isfile(local_path)
        url = "https://raw.githubusercontent.com/EECS-NTNU/hilp/$HILP_COMMIT/rodinia_profiled_data/$hilp_name"
        @info "Downloading HILP data" url
        Downloads.download(url, local_path)
    end
end
