#################################################################################
# RHODES main optimization script
#
# Runs the nominal (no uncertainty) and robust optimization models for one workload.
# Run:
#     julia models/main.jl
#
# Inputs:
#   input data/HILP_data/                 HILP profiling data (downloaded on first run)
#   input data/carbon_intensity_data/     CAISO carbon intensity data (robust model)
#   input data/FPW_data.csv               FPW data (robust model)
#   input parameters/input_parameters.jl  parameters shared by the nominal and robust models
#   models/opt_constraints.jl             optimization constraints
#################################################################################
# Load in libraries
using DataFrames, CSV
using JuMP, Gurobi
using LinearAlgebra, Random, Printf, StatsBase, CategoricalArrays
using Plots, StatsPlots
using Distributions

## Workload for this DSE
workload = "bfs" #"lavaMD" #"heartwall" #"pathfinder" #"myocyte" #"sc" #"bfs" #"lud" # "nn"

## Setup: helper functions, workload data, constraints, and parameters
include(joinpath(@__DIR__, "helper_functions.jl"))
include(joinpath(@__DIR__, "input_data_processing.jl"))  # cpu_k, gpu_k for `workload`
include(joinpath(@__DIR__, "opt_constraints.jl"))        # t_max, tC_max, a_max
include(joinpath(@__DIR__, "..", "input parameters", "input_parameters.jl"))  # Nwork, setup_teardown_scaling, Γ

init_parameters(workload)

#################################################################################
# Nominal optimization (no uncertainty)
#################################################################################
include(joinpath(@__DIR__, "nominal", "nominal_baseline_model.jl"))
include(joinpath(@__DIR__, "nominal", "nominal_opttC.jl"))
include(joinpath(@__DIR__, "nominal", "nominal_optPerf_tCconstrained.jl"))

println("\nNominal: minimize total carbon (performance-constrained)")
nominal_tC_results = nominal_opt_tC(t_max = t_max, a_max = a_max)

println("\nNominal: optimize performance (carbon-constrained)")
nominal_perf_results = nominal_carbon_constrained(a_max = a_max, tC_max = tC_max)

#################################################################################
# Robust optimization with RHODES
#################################################################################
include(joinpath(@__DIR__, "robust", "uncertainty sets", "DA_uncertainty_sets.jl"))
include(joinpath(@__DIR__, "robust", "uncertainty sets", "DV_uncertainty_sets.jl"))
include(joinpath(@__DIR__, "robust", "uncertainty sets", "DV_data_processing.jl"))
include(joinpath(@__DIR__, "robust", "robust_baseline_model.jl"))
include(joinpath(@__DIR__, "robust", "RO_opttC.jl"))
include(joinpath(@__DIR__, "robust", "RO_optPerf_tCconstrained.jl"))

println("\nRobust: minimize total carbon (performance-constrained)")
robust_tC_results = robust_opt_tC(t_max = t_max, a_max = a_max)

println("\nRobust: optimize performance (carbon-constrained)")
robust_perf_results = robust_carbon_constrained(a_max = a_max, tC_max = tC_max)
