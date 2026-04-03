using Pkg
# Package installation commands (uncomment if needed)
# Pkg.add("NPZ")
# Pkg.add("Flux")
# Pkg.add("BSON")
# Pkg.add("Statistics")
# Pkg.add("BenchmarkTools")
using NPZ, LinearAlgebra, Statistics
using Flux: Chain, relu, Conv, BatchNorm, sigmoid, Dense, loadmodel!
using BenchmarkTools
using BSON: @load

start_time = time_ns()

include("../PointNetMHD.jl")
file = "Samples/aten_nsample_12288_nbdry_3072_low_discrepancy.npy"
data = NPZ.npzread(file)[:, 1:3, :]
ground_truth = NPZ.npzread(file)[:, 4:6, :]

global_in_min = minimum(data)
global_in_max = maximum(data)
data .= ((data .- global_in_min) .* 2.0) ./ (global_in_max - global_in_min) .- 1.0

model = create_pointnet_model(3, 3)
@load "mhd.bson" model_state
loadmodel!(model, model_state)
#@btime model($data)
solutions = Array(model(data))

global_out_min = minimum(ground_truth)
global_out_max = maximum(ground_truth)
solutions .= solutions .* (global_out_max - global_out_min) .+ global_out_min

sq_errors = (solutions .- ground_truth).^2
n_samples = length(vec(sq_errors))

sorted_total = sort(vec(sq_errors), rev=true)
total_avg_mse = mean(sq_errors)
n_top5 = max(1, Int(ceil(n_samples * 0.05)))
total_top5_avg = mean(sorted_total[1:n_top5])
n_top1 = max(1, Int(ceil(n_samples * 0.01)))
total_top1_avg = mean(sorted_total[1:n_top1])

println("    - Total Avg. MSE: $total_avg_mse")
println("    - Total Top 5% Avg. MSE: $total_top5_avg")
println("    - Total Top 1% Avg. MSE: $total_top1_avg")

println(file)
for ch in 1:3
    channel_errors = vec(sq_errors[:, ch:ch, :])
    n_samples = length(channel_errors)

    avg_mse = mean(channel_errors)

    sorted_errors = sort(channel_errors, rev=true)
    n_top5 = max(1, Int(ceil(n_samples * 0.05)))
    top5_avg = mean(sorted_errors[1:n_top5])

    n_top1 = max(1, Int(ceil(n_samples * 0.01)))
    top1_avg = mean(sorted_errors[1:n_top1])

    println("    - Channel $ch Avg. MSE: $avg_mse")
    println("    - Channel $ch Top 5% Avg. MSE: $top5_avg")
    println("    - Channel $ch Top 1% Avg. MSE: $top1_avg")
end

NPZ.npzwrite("Evaluation/solutions.npy", solutions)
NPZ.npzwrite("Evaluation/error.npy", mean((solutions .- ground_truth).^2, dims=2))

end_time = time_ns()

println("Evaluation took $((end_time - start_time) / 1_000_000) ms")