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

input_channels = 3
output_channels = 3
training_samples = 18
validation_files = 3

total_epochs = 4000
validation_frequency = 25

"""
Normalize data channel-wise to a specified range.
Normalizes each channel independently by computing min-max normalization. The data is
modified in-place to reduce memory allocation.
# Arguments
- `data`: 3D array with dimensions (points, channels, samples) containing the data to normalize
- `normalize_to_range::Symbol`: Target normalization range, either:
  - `:minus_one_to_one` - Normalize to [-1, 1] range
  - `:zero_to_one` - Normalize to [0, 1] range
# Details
For each channel j, the transformation is:
- `:minus_one_to_one`: `x_normalized = 2 * (x - min) / (max - min) - 1`
- `:zero_to_one`: `x_normalized = (x - min) / (max - min)`
"""
function normalize_data!(data, normalize_to_range::Symbol)
    num_channels = size(data, 2)

    for j in 1:num_channels
        mmin = minimum(data[:, j, :])
        mmax = maximum(data[:, j, :])

        if normalize_to_range == :minus_one_to_one
            data[:, j, :] .= ((data[:, j, :] .- mmin) .* 2.0) ./ (mmax - mmin) .- 1.0
        elseif normalize_to_range == :zero_to_one
            data[:, j, :] .= (data[:, j, :] .- mmin) ./ (mmax - mmin)
        end
    end
end

"""
Load and preprocess multiple NPY files containing MHD simulation data.

Reads multiple .npy files, splits each into input and output channels, normalizes them,
and concatenates all samples into unified training/validation tensors.

# Arguments
- `files`: Vector of file paths to .npy files to load

# Returns
- `input_data`: 3D array (points, input_channels, total_samples) normalized to [-1, 1]
- `output_data`: 3D array (points, output_channels, total_samples) normalized to [0, 1]

# Data Format
Each .npy file is expected to have shape (points, total_channels, samples) where:
- Columns 1:`input_channels` contain input features (e.g., spatial coordinates)
- Columns (`input_channels`+1):(`input_channels`+`output_channels`) contain output targets (e.g., field values)
"""
function load_and_preprocess_data(files)
    all_input_data = []
    all_output_data = []
    
    for file in files
        data = NPZ.npzread(file)
        
        input_sample = data[:, 1:input_channels, :]
        output_sample = data[:, input_channels+1:input_channels+output_channels, :]
        
        normalize_data!(input_sample, :minus_one_to_one)
        normalize_data!(output_sample, :zero_to_one)
        
        push!(all_input_data, input_sample)
        push!(all_output_data, output_sample)
    end
    
    return cat(all_input_data..., dims=3), cat(all_output_data..., dims=3)
end

"""
Train a PointNet model for MHD field prediction.

Implements a PointNet-style architecture with two parallel branches that process point cloud data
to predict magnetohydrodynamic (MHD) field values. The model uses a combination of local and global
features through a max-pooling aggregation layer.

# Arguments
- `input_files::Vector{String}`: Paths to all available .npy data files
- `output_model_file::String`: Path where the trained model will be saved (BSON format)
- `input_channels::Int`: Number of input feature channels per point (e.g., 3 for x,y,z coordinates)
- `output_channels::Int`: Number of output channels to predict (e.g., 3 for Bx,By,Bz field components)
- `training_files::Int`: Number of files to use for training
- `validation_files::Int`: Number of files to use for validation

# Architecture
The model consists of:
- **Branch 1**: Local feature extractor (2 conv layers, 64 channels each)
- **Branch 2**: Global feature extractor (5 conv layers, ending with 1024-dim global descriptor)
- **Aggregation**: Max-pooling over points in branch 2, then broadcast and concatenate with branch 1
- **Decoder**: 4 conv layers reducing from 1088 -> 512 -> 256 -> 128 -> output_channels

# Training Details
- Optimizer: Adam (lr=0.000075, β₁=0.925, β₂=0.995)
- Batch size: 6
- Loss: Mean Squared Error (MSE)
- Total epochs: Determined by global `total_epochs` variable
- Validation: Every `validation_frequency` epochs

# Metrics Tracked
Global variables are populated during training for use by `plot_losses()`:
- Overall training and validation loss
- Per-channel average MSE, top 5% MSE, and top 1% MSE on validation set
- Per-file metrics averaged over 100-epoch windows

# Side Effects
- Saves trained model to `output_model_file`
- Prints validation file assignments and periodic performance metrics
- Modifies global history variables for plotting
"""
function train_pointnet(input_files::Vector{String}, output_model_file::String, input_channels::Int, output_channels::Int, training_files::Int, validation_files::Int)
    shuffled_files = shuffle(input_files)
    train_files = shuffled_files[1:training_files]
    val_files = shuffled_files[training_files+1:training_files+validation_files]
    println("Validation files set to: ", val_files)

    global history_loss = Float32[]
    global history_val_loss = Float32[]

    global history_channel_avg_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top5_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top1_mse = [Float32[] for _ in 1:output_channels]

    global val_loss_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_5_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_1_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global overall_val_loss_accumulator = 0.0f0

    input_data_train, output_data_train = load_and_preprocess_data(train_files)
    input_data_val, output_data_val = load_and_preprocess_data(val_files)

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
        x -> cat(branch1(x), repeat(maximum(branch2(x), dims=1), size(x, 1), 1, 1); dims=2),
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

    bs = 6
    lr = 0.000075
    b1 = 0.925
    b2 = 0.995

    opt = Flux.setup(Optimisers.Adam(lr, (b1, b2)), model)
    loss(m, x, y) = Flux.Losses.mse(m(x), y)
    train_dataloader = DataLoader((input_data_train, output_data_train), batchsize=bs, shuffle=true)

    for epoch in 1:total_epochs
        Flux.train!(loss, model, train_dataloader, opt)

        if epoch % validation_frequency == 0
            train_loss = loss(model, input_data_train, output_data_train)
            val_loss = loss(model, input_data_val, output_data_val)
            push!(history_loss, train_loss)
            push!(history_val_loss, val_loss)

            y_pred_val = model(input_data_val)
            for c in 1:output_channels
                pointwise_sq_errors = (y_pred_val[:, c:c, :] .- output_data_val[:, c:c, :]).^2
                pointwise_sq_errors_vec = vec(pointwise_sq_errors)

                push!(history_channel_avg_mse[c], mean(pointwise_sq_errors_vec))

                sorted_errors = sort(pointwise_sq_errors_vec, rev=true)
                num_points = length(sorted_errors)

                num_top_5_percent = ceil(Int, 0.05 * num_points)
                push!(history_channel_top5_mse[c], mean(@view sorted_errors[1:num_top_5_percent]))

                num_top_1_percent = ceil(Int, 0.01 * num_points)
                push!(history_channel_top1_mse[c], mean(@view sorted_errors[1:num_top_1_percent]))
            end

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

"""
Generate comprehensive training history visualizations.

Creates four PNG plots showing training progress from the global history variables populated
during `train_pointnet()`. All plots use log-scale y-axes to better visualize error magnitudes
across multiple orders of magnitude.

# Plots Generated

1. **MSE_History.png**: Raw training and validation MSE per epoch
2. **Smoothed_MSE_History.png**: Smoothed overall losses (50-epoch moving average window)
3. **Smoothed_Channel_MSE_History.png**: Per-channel validation metrics with smoothing (10-epoch window)
   - Average MSE (solid lines)
   - Top 5% worst predictions MSE (dashed lines)
   - Top 1% worst predictions MSE (dotted lines)
4. **Channel_Loss_History.png**: Raw per-channel validation metrics without smoothing

# Dependencies
Requires the following global variables to be populated by `train_pointnet()`:
- `history_loss`, `history_val_loss`: Overall training/validation loss histories
- `history_channel_avg_mse`: Per-channel average MSE on validation set
- `history_channel_top5_mse`: Per-channel top 5% worst predictions MSE
- `history_channel_top1_mse`: Per-channel top 1% worst predictions MSE
- `total_epochs`, `validation_frequency`: Training configuration parameters
- `output_channels`: Number of output channels

# Notes
The moving average smoothing helps identify trends by reducing noise in the loss curves.
Top k% metrics highlight model performance on the most difficult predictions, which can
reveal failure modes not visible in average metrics.
"""
function plot_losses()
    """
        moving_average(data, window)

    Compute moving average with specified window size.

    For each position i, computes the mean of elements from max(1, i-window+1) to i,
    providing a smoothed version of the input data while preserving the original length.
    """
    function moving_average(data::Vector, window::Int)
        return [mean(data[max(1, i-window+1):i]) for i in 1:length(data)]
    end

    epochs = validation_frequency:validation_frequency:total_epochs

    p1 = Plots.plot(
        epochs, history_loss,
        label="Training MSE", linewidth=2, color=:blue,
        yaxis=:log10,
        title="PointNet Loss History (Raw Mean Squared Error)",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        legend=:topright,
        size=(800, 600)
    )
    Plots.plot!(p1, epochs, history_val_loss, label="Validation MSE", linewidth=2, color=:orange)
    Plots.savefig(p1, "MSE_History.png")

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

    channel_plot = Plots.plot(
        title="Smoothed Per-Channel Validation MSE (Window=10)",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        yaxis=:log10,
        legend=:topright,
        size=(900, 700)
    )

    colors = palette(:default)
    smoothing_window = 10

    for c in 1:output_channels
        smoothed_avg = moving_average(history_channel_avg_mse[c], smoothing_window)
        smoothed_top5 = moving_average(history_channel_top5_mse[c], smoothing_window)
        smoothed_top1 = moving_average(history_channel_top1_mse[c], smoothing_window)

        Plots.plot!(channel_plot, epochs, smoothed_avg,
            label="Ch $c Avg. MSE",
            linestyle=:solid,
            linewidth=2,
            color=colors[c % length(colors) + 1]
        )

        Plots.plot!(channel_plot, epochs, smoothed_top5,
            label="Ch $c Top 5%",
            linestyle=:dash,
            color=colors[c % length(colors) + 1]
        )

        Plots.plot!(channel_plot, epochs, smoothed_top1,
            label="Ch $c Top 1%",
            linestyle=:dot,
            color=colors[c % length(colors) + 1]
        )
    end
    Plots.savefig(channel_plot, "Smoothed_Channel_MSE_History.png")

    channel_plot_raw = Plots.plot(
        title="Per-Channel Validation MSE",
        xlabel="Epoch",
        ylabel="Mean Squared Error (Magnitude)",
        yaxis=:log10,
        legend=:topright,
        size=(900, 700)
    )

    colors = palette(:default)
    for c in 1:output_channels
        Plots.plot!(channel_plot_raw, epochs, history_channel_avg_mse[c],
                    label="Channel $c Avg. MSE",
                    linestyle=:solid,
                    color=colors[c % length(colors) + 1])

        Plots.plot!(channel_plot_raw, epochs, history_channel_top5_mse[c],
                    label="Channel $c Top 5% MSE",
                    linestyle=:dash,
                    color=colors[c % length(colors) + 1])

        Plots.plot!(channel_plot_raw, epochs, history_channel_top1_mse[c],
                    label="Channel $c Top 1% MSE",
                    linestyle=:dot,
                    color=colors[c % length(colors) + 1])
    end
    Plots.savefig(channel_plot_raw, "Channel_Loss_History.png")
end

all_npy_files = ["Samples/aten_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_10p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_1p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_2p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_3p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_4p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_5p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_6p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_7p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_8p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_F14_9p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Hill11p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Hill2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Hill4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Hill6p2_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Well11p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Well2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Well4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/HSX_Well6p2_nsample_12288_nbdry_3072_low_discrepancy.npy", 
"Samples/QHS46_nsample_12288_nbdry_3072_low_discrepancy.npy"]

train_pointnet(all_npy_files, "mhd_12288_3072.bson", input_channels, output_channels, training_samples, validation_files);
plot_losses()