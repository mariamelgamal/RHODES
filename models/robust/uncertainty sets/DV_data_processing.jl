""" 
Read in DV uncertainty.
"""

CIuse_data = CSV.read(joinpath(@__DIR__, "..", "..", "..", "input data", "carbon_intensity_data", "CAISO_CI_MGHGperMWH_data.csv"), DataFrame);

# Parse the All Results strings to arrays
all_results = [
    try
        parse.(Float64, filter(x -> occursin(r"^\d", x), split(s, r"[^\d\.eE-]+")))
    catch
        Float64[]
    end for s in CIuse_data."All Results"
]

lengths = map(length, all_results)

maxlen = maximum(lengths)

all_results = filter(x -> any(!isnan, x), all_results)

# Pad with NaN to make all vectors the same length
padded_results = [
    vcat(x, fill(NaN, maxlen - length(x))) for x in all_results
]

all_results_matrix = reduce(hcat, padded_results)'  # Each row = processor

FPW_data = CSV.read(joinpath(@__DIR__, "..", "..", "..", "input data", "FPW_data.csv"), DataFrame);

FPW_2022_14nm = FPW_data[(FPW_data.Year .== 2022) .& (FPW_data.EPW .== "EPW_14nm"), :]

# Parse "Multiplied Results" strings into numeric arrays
all_results = [
    try
        parse.(Float64, filter(x -> occursin(r"^\d", x), split(s, r"[^\d\.eE-]+")))
    catch
        Float64[]
    end for s in FPW_2022_14nm."Multiplied Results"
]

# Compute lengths of parsed arrays
lengths = map(length, all_results)
maxlen = maximum(lengths)

# Filter out any empty / invalid arrays
results = filter(x -> any(!isnan, x), all_results)

# Pad arrays with NaN to match the maximum length
padded_results = [
    vcat(x, fill(NaN, maxlen - length(x))) for x in results
]

# Combine into a matrix (each row = one entry from FPW_2022_14nm)
FPW_2022_14nm_matrix = reduce(hcat, padded_results)'

FPW_2022_7nmEUV = FPW_data[(FPW_data.Year .== 2022) .& (FPW_data.EPW .== "EPW_7nmEUV"), :]

# Parse "Multiplied Results" strings into numeric arrays
all_results = [
    try
        parse.(Float64, filter(x -> occursin(r"^\d", x), split(s, r"[^\d\.eE-]+")))
    catch
        Float64[]
    end for s in FPW_2022_7nmEUV."Multiplied Results"
]

# Compute lengths of parsed arrays
lengths = map(length, all_results)
maxlen = maximum(lengths)

# Filter out any empty / invalid arrays
results = filter(x -> any(!isnan, x), all_results)

# Pad arrays with NaN to match the maximum length
padded_results = [
    vcat(x, fill(NaN, maxlen - length(x))) for x in results
]

# Combine into a matrix (each row = one entry from FPW_2022_14nm)
FPW_2022_7nmEUV_matrix = reduce(hcat, padded_results)'