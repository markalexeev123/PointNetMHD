using Pkg
# Pkg.add("Random")
# Pkg.add("NPZ")
# Pkg.add("Optimisers")
# Pkg.add("LinearAlgebra")
# Pkg.add("Statistics")
# Pkg.add("Flux")
# Pkg.add("BSON")
# Pkg.add("Plots")

using Random, NPZ, Flux, Optimisers, LinearAlgebra, Statistics, Plots
using Flux: Chain, relu, Conv, BatchNorm, sigmoid, DataLoader
using BSON: @save

#Model and data Values
input_channels = 3
output_channels = 3
training_samples = 18
validation_files = 3
number = 1

total_epochs = 4000
validation_frequency = 25

#Helper functions
function load_and_preprocess_data(files)
    all_input_data = []
    all_output_data = []
    
    for file in files
        data = NPZ.npzread(file)
        
        input_sample = data[:, 1:input_channels, :]
        output_sample = data[:, input_channels+1:input_channels+output_channels, :]
        
        for j in 1:input_channels
            mmin = minimum(input_sample[:, j, :])
            mmax = maximum(input_sample[:, j, :])
            input_sample[:, j, :] .= ((input_sample[:, j, :] .- mmin) .* 2.0) ./ (mmax - mmin) .- 1.0
        end
        
        for j in 1:output_channels
            mmin = minimum(output_sample[:, j, :])
            mmax = maximum(output_sample[:, j, :])
            output_sample[:, j, :] .= (output_sample[:, j, :] .- mmin) ./ (mmax - mmin)
        end
        
        push!(all_input_data, input_sample)
        push!(all_output_data, output_sample)
    end
    
    return cat(all_input_data..., dims=3), cat(all_output_data..., dims=3)
end

function load_and_preprocess_single_file(file)
    data = NPZ.npzread(file)
    input_sample = data[:, 1:input_channels, :]
    output_sample = data[:, input_channels+1:input_channels+output_channels, :]
    
    for j in 1:input_channels
        mmin = minimum(input_sample[:, j, :])
        mmax = maximum(input_sample[:, j, :])
        input_sample[:, j, :] .= ((input_sample[:, j, :] .- mmin) .* 2.0) ./ (mmax - mmin) .- 1.0
    end
    
    for j in 1:output_channels
        mmin = minimum(output_sample[:, j, :])
        mmax = maximum(output_sample[:, j, :])
        output_sample[:, j, :] .= (output_sample[:, j, :] .- mmin) ./ (mmax - mmin)
    end
    
    return input_sample, output_sample
end

function read_hyperparameters(filepath::String)
    params = Number[]

    open(filepath, "r") do file
        for line in eachline(file)
    
        if isempty(strip(line))
            continue
            end

            parts = split(line, '=')
        
            value_str = strip(parts[2])
            value = parse(Float64, value_str)
        
            push!(params, value)
        end
    end

    return (Int(params[1]), params[2], params[3], params[4])
end

function train_pointnet(input_files::Vector{String}, output_model_file::String, input_channels::Int, output_channels::Int, training_files::Int, validation_files::Int, number::Int)
    shuffled_files = shuffle(input_files)
    train_files = shuffled_files[1:training_files]
    val_files = shuffled_files[training_files+1:training_files+validation_files]
    println("Validation files set to: ", val_files)

    #Global Values needed for plotting
    global history_loss = Float32[]
    global history_val_loss = Float32[]

    # New histories for per-channel validation loss
    global history_channel_avg_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top5_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top1_mse = [Float32[] for _ in 1:output_channels]

    global val_loss_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_5_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_1_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global overall_val_loss_accumulator = 0.0f0

    input_data_train, output_data_train = load_and_preprocess_data(train_files)
    input_data_val, output_data_val = load_and_preprocess_data(val_files)

    # Create mapping of validation files to their sample ranges
    val_file_map = Dict{String, UnitRange{Int}}()
    current_start = 1
    for file in val_files
        data = NPZ.npzread(file)
        num_samples = size(data, 3)
        val_file_map[file] = current_start:(current_start + num_samples - 1)
        current_start += num_samples
    end

    branch1 = Chain(
        Conv((1,), input_channels => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
    )

    branch2 = Chain(
        Conv((1,), input_channels => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 128, relu),
        BatchNorm(128, momentum=0.99, eps=0.001),
        Conv((1,), 128 => 1024, relu),
        BatchNorm(1024, momentum=0.99, eps=0.001),
    )

    model = Chain(
        x -> cat(branch1(x), repeat(maximum(branch2(x), dims=1), 12288, 1, 1); dims=2),
        Conv((1,), 1088 => 512, relu),
        BatchNorm(512, momentum=0.99, eps=0.001),
        Conv((1,), 512 => 256, relu),
        BatchNorm(256, momentum=0.99, eps=0.001),
        Conv((1,), 256 => 128, relu),
        BatchNorm(128, momentum=0.99, eps=0.001),
        Conv((1,), 128 => 128, relu),
        BatchNorm(128, momentum=0.99, eps=0.001),
        Conv((1,), 128 => output_channels, sigmoid)
    )

    #bs, lr, b1, b2 = read_hyperparameters("hyperparam_$number")
    bs = 6
    lr = 0.000075
    b1 = 0.925
    b2 = 0.995

    opt = Flux.setup(Optimisers.Adam(lr, (b1, b2)), model)
    loss(m, x, y) = Flux.Losses.mse(m(x), y)
    train_dataloader = DataLoader((input_data_train, output_data_train), batchsize=bs, shuffle=true)

    for epoch in 1:total_epochs
        Flux.train!(loss, model, train_dataloader, opt)

        # Validate every 25 epochs
        if epoch % validation_frequency == 0
            train_loss = loss(model, input_data_train, output_data_train)
            val_loss = loss(model, input_data_val, output_data_val)
            push!(history_loss, train_loss)
            push!(history_val_loss, val_loss)

            # Calculate per-channel metrics on the entire validation set for plotting
            y_pred_val = model(input_data_val)
            for c in 1:output_channels
                pointwise_sq_errors = (y_pred_val[:, c:c, :] .- output_data_val[:, c:c, :]).^2
                pointwise_sq_errors_vec = vec(pointwise_sq_errors)

                # Avg MSE for channel c
                push!(history_channel_avg_mse[c], mean(pointwise_sq_errors_vec))

                # Sort once and reuse for both top 5% and top 1%
                sorted_errors = sort(pointwise_sq_errors_vec, rev=true)
                num_points = length(sorted_errors)

                # Top 5% MSE for channel c
                num_top_5_percent = ceil(Int, 0.05 * num_points)
                push!(history_channel_top5_mse[c], mean(@view sorted_errors[1:num_top_5_percent]))

                # Top 1% MSE for channel c
                num_top_1_percent = ceil(Int, 0.01 * num_points)
                push!(history_channel_top1_mse[c], mean(@view sorted_errors[1:num_top_1_percent]))
            end

            # Per-file metrics calculation
            overall_val_loss_accumulator += val_loss
            for file in val_files
                sample_range = val_file_map[file]
                x_val_slice = input_data_val[:, :, sample_range]
                y_val_slice = output_data_val[:, :, sample_range]
                y_pred_slice = model(x_val_slice)

                for c in 1:output_channels
                    pointwise_sq_errors = (y_pred_slice[:, c:c, :] .- y_val_slice[:, c:c, :]).^2
                    pointwise_sq_errors_vec = vec(pointwise_sq_errors)
                    val_loss_accumulators[file][c] += mean(pointwise_sq_errors_vec)

                    # Sort once and reuse
                    sorted_errors = sort(pointwise_sq_errors_vec, rev=true)
                    num_points = length(sorted_errors)

                    num_top_5_percent = ceil(Int, 0.05 * num_points)
                    top_5_percent_mse_accumulators[file][c] += mean(@view sorted_errors[1:num_top_5_percent])

                    num_top_1_percent = ceil(Int, 0.01 * num_points)
                    top_1_percent_mse_accumulators[file][c] += mean(@view sorted_errors[1:num_top_1_percent])
                end
            end

            if epoch > 0 && epoch % 100 == 0
                avg_overall_val_loss = overall_val_loss_accumulator / (100 / validation_frequency)

                println("\n" * "="^60)
                println("Epoch: $epoch")
                println("Avg. Overall Validation Loss (last 100 epochs): $avg_overall_val_loss")
                println("-"^60)
                println("Per-File Channel MSE (Average over last 100 Epochs):")

                for file in val_files
                    println("  File: $(basename(file))")
                    for c in 1:output_channels
                        avg_channel_loss = val_loss_accumulators[file][c] / (100 / validation_frequency)
                        avg_top_5_mse = top_5_percent_mse_accumulators[file][c] / (100 / validation_frequency)
                        avg_top_1_mse = top_1_percent_mse_accumulators[file][c] / (100 / validation_frequency)
                        println("    - Channel $c Avg. MSE: $avg_channel_loss")
                        println("    - Channel $c Top 5% Avg. MSE: $avg_top_5_mse")
                        println("    - Channel $c Top 1% Avg. MSE: $avg_top_1_mse")
                    end
                end
                println("="^60 * "\n")

                overall_val_loss_accumulator = 0.0f0
                for file in val_files
                    val_loss_accumulators[file] .= 0.0f0
                    top_5_percent_mse_accumulators[file] .= 0.0f0
                    top_1_percent_mse_accumulators[file] .= 0.0f0
                end
            end
        end
    end

    @save output_model_file model
end

 function plot_losses()    
    # Define the x-axis epochs (runs every 25 epochs up to 4000)
    epochs = validation_frequency:validation_frequency:total_epochs

    function moving_average(data::Vector, window::Int)
        return [mean(data[max(1, i-window+1):i]) for i in 1:length(data)]
    end

    # --- Plot 1: Raw MSE vs. Epoch (Log Magnitude Y-axis, Correct X-axis) ---
    p1 = Plots.plot(
        epochs, history_loss,
        label="Training MSE", linewidth=2, color=:blue,
        yaxis=:log10, # Displaying magnitude: 10^-1, 10^-2, etc.
        title="PointNet Loss History (Raw Mean Squared Error)",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        legend=:topright,
        size=(800, 600)
    )
    Plots.plot!(p1, epochs, history_val_loss, label="Validation MSE", linewidth=2, color=:orange)
    Plots.savefig(p1, "MSE_History.png")

    # --- Plot 2: Smoothed MSE vs. Epoch (Log Magnitude Y-axis, Correct X-axis) ---
    # Note: Using the user's original window=50 for overall smoothing
    smoothed_train = moving_average(history_loss, 50)
    smoothed_val = moving_average(history_val_loss, 50)

    p2 = Plots.plot(
        epochs, smoothed_train,
        label="Smoothed Training MSE (Window=50)", linewidth=2, color=:blue,
        yaxis=:log10,
        title="PointNet Loss History (Smoothed Mean Squared Error)",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        legend=:topright,
        size=(800, 600)
    )
    Plots.plot!(p2, epochs, smoothed_val, label="Smoothed Validation MSE (Window=50)", linewidth=2, color=:orange)
    Plots.savefig(p2, "Smoothed_MSE_History.png")


    # --- Plot 3: Smoothed Per-Channel Validation Loss (NEW, Correct Axes) ---
    channel_plot = Plots.plot(
        title="Smoothed Per-Channel Validation MSE (Window=10)",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        yaxis=:log10, # Magnitude scale
        legend=:topright,
        size=(900, 700)
    )

    colors = palette(:default)
    smoothing_window = 10 

    for c in 1:output_channels
        # Apply smoothing to raw MSE history for the channel
        smoothed_avg = moving_average(history_channel_avg_mse[c], smoothing_window)
        smoothed_top5 = moving_average(history_channel_top5_mse[c], smoothing_window)
        smoothed_top1 = moving_average(history_channel_top1_mse[c], smoothing_window)

        # Average MSE with a solid line
        Plots.plot!(channel_plot, epochs, smoothed_avg,
            label="Ch $c Avg. MSE",
            linestyle=:solid,
            linewidth=2,
            color=colors[c % length(colors) + 1] # Cycle colors
        )

        # Top 5% MSE with a dashed line
        Plots.plot!(channel_plot, epochs, smoothed_top5,
            label="Ch $c Top 5%",
            linestyle=:dash,
            color=colors[c % length(colors) + 1]
        )

        # Top 1% MSE with a dotted line
        Plots.plot!(channel_plot, epochs, smoothed_top1,
            label="Ch $c Top 1%",
            linestyle=:dot,
            color=colors[c % length(colors) + 1]
        )
    end
    Plots.savefig(channel_plot, "Smoothed_Channel_MSE_History.png")

    # --- Plot 4: Per-Channel Validation Loss (Corrected) ---
    channel_plot_raw = Plots.plot(
        title="Per-Channel Validation MSE",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        yaxis=:log10, # Let the plot handle the log scaling
        legend=:topright,
        size=(900, 700)
    )

    colors = palette(:default)
    for c in 1:output_channels
        # Average MSE with a solid line
        Plots.plot!(channel_plot_raw, epochs, history_channel_avg_mse[c],
                    label="Channel $c Avg. MSE",
                    linestyle=:solid,
                    color=colors[c % length(colors) + 1])

        # Top 5% MSE with a dashed line
        Plots.plot!(channel_plot_raw, epochs, history_channel_top5_mse[c],
                    label="Channel $c Top 5% MSE",
                    linestyle=:dash,
                    color=colors[c % length(colors) + 1])

        # Top 1% MSE with a dotted line
        Plots.plot!(channel_plot_raw, epochs, history_channel_top1_mse[c],
                    label="Channel $c Top 1% MSE",
                    linestyle=:dot,
                    color=colors[c % length(colors) + 1])
    end
    Plots.savefig(channel_plot_raw, "Channel_Loss_History.png")
end

all_npy_files = ["aten_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_10p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_1p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_2p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_3p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_4p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_5p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_6p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_7p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_8p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_9p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill11p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill6p2_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well11p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well6p2_nsample_12288_nbdry_3072_low_discrepancy.npy", "QHS46_nsample_12288_nbdry_3072_low_discrepancy.npy"]

train_pointnet(all_npy_files, "mhd_12288_3072.bson", input_channels, output_channels, training_samples, validation_files, number);
plot_losses()
# ============================================================
# Epoch: 4000
# Avg. Overall Validation Loss (last 100 epochs): 0.0005192995972692928
# ------------------------------------------------------------
# Per-File Channel MSE (Average over last 100 Epochs):
#   File: HSX_Well6p2_nsample_12288_nbdry_3072_low_discrepancy.npy
#     - Channel 1 Avg. MSE: 0.0005890794564038515
#     - Channel 1 Top 5% Avg. MSE: 0.004081995692104101
#     - Channel 1 Top 1% Avg. MSE: 0.006692613009363413
#     - Channel 2 Avg. MSE: 0.00045051376218907535
#     - Channel 2 Top 5% Avg. MSE: 0.002404952421784401
#     - Channel 2 Top 1% Avg. MSE: 0.0036740140058100224
#     - Channel 3 Avg. MSE: 0.000738619128242135
#     - Channel 3 Top 5% Avg. MSE: 0.004198412876576185
#     - Channel 3 Top 1% Avg. MSE: 0.006261501926928759
#   File: HSX_Well4p4_nsample_12288_nbdry_3072_low_discrepancy.npy
#     - Channel 1 Avg. MSE: 0.00048658152809366584
#     - Channel 1 Top 5% Avg. MSE: 0.0031335344538092613
#     - Channel 1 Top 1% Avg. MSE: 0.005456052720546722
#     - Channel 2 Avg. MSE: 0.00045980288996361196
#     - Channel 2 Top 5% Avg. MSE: 0.0025956416502594948
#     - Channel 2 Top 1% Avg. MSE: 0.0038888449780642986
#     - Channel 3 Avg. MSE: 0.0008849312434904277
#     - Channel 3 Top 5% Avg. MSE: 0.005083579104393721
#     - Channel 3 Top 1% Avg. MSE: 0.007404148578643799
#   File: HSX_F14_4p_nsample_12288_nbdry_3072_low_discrepancy.npy
#     - Channel 1 Avg. MSE: 0.0002699119213502854
#     - Channel 1 Top 5% Avg. MSE: 0.001560272416099906
#     - Channel 1 Top 1% Avg. MSE: 0.0023920286912471056
#     - Channel 2 Avg. MSE: 0.00031864148331806064
#     - Channel 2 Top 5% Avg. MSE: 0.0017291363328695297
#     - Channel 2 Top 1% Avg. MSE: 0.002428060630336404
#     - Channel 3 Avg. MSE: 0.0004756149719469249
#     - Channel 3 Top 5% Avg. MSE: 0.002893676981329918
#     - Channel 3 Top 1% Avg. MSE: 0.004605326801538467
# ============================================================
