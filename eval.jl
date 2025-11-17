using Pkg
# Pkg.add("NPZ")
# Pkg.add("Flux")
# Pkg.add("BSON")
# Pkg.add("Statistics")
using NPZ, LinearAlgebra, Statistics
using Flux: Chain, relu, Conv, BatchNorm, sigmoid, Dense

using BSON: @load

file = "HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy"
data = NPZ.npzread(file)[:, 1:3, :]
ground_truth = NPZ.npzread(file)[:, 4:6, :]

for j in 1:3
    mmin = minimum(data[:, j, :])
    mmax = maximum(data[:, j, :])
    data[:, j, :] .= ((data[:, j, :] .- mmin) .* 2.0) ./ (mmax - mmin) .- 1.0
end

@load "mhd_12288_3072.bson" model

solutions = Array(model(data))

for j in 1:3
    mmin = minimum(ground_truth[:, j, :])
    mmax = maximum(ground_truth[:, j, :])
    solutions[:, j, :] .= solutions[:, j, :] .* (mmax - mmin) .+ mmin
end

NPZ.npzwrite("solutions.npy", solutions)
NPZ.npzwrite("error.npy", mean((solutions .- ground_truth).^2, dims=2))