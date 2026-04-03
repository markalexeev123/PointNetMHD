using Pkg
Pkg.add("Random")
Pkg.add("NPZ")
Pkg.add("Optimisers")
Pkg.add("Statistics")
Pkg.add("Flux")
Pkg.add("BSON")
Pkg.add("Plots")

using Random, NPZ, Flux, Optimisers, Statistics, Plots
using Flux: Chain, relu, Conv, BatchNorm, sigmoid, DataLoader
using BSON: @save

input_channels = 3
output_channels = 3
training_samples = 15
validation_files = 3

bs = 6
lr = 0.000075
b1 = 0.925
b2 = 0.995

total_epochs = 100
validation_frequency = 25

struct PointNetAggregation
    local_mlp
    global_mlp
end

Flux.@layer PointNetAggregation

function (m::PointNetAggregation)(x)
    local_feat = m.local_mlp(x)
    global_feat = m.global_mlp(local_feat)
    global_feat = repeat(maximum(global_feat, dims=1), size(x, 1), 1, 1)
    return cat(local_feat, global_feat; dims=2)
end

function create_pointnet_model(input_channels::Int, output_channels::Int)
    local_mlp = Chain(
        Conv((1,), input_channels => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
    )

    global_mlp = Chain(
        Conv((1,), 64 => 64, relu),
        BatchNorm(64, momentum=0.99, eps=0.001),
        Conv((1,), 64 => 128, relu),
        BatchNorm(128, momentum=0.99, eps=0.001),
        Conv((1,), 128 => 1024, relu),
        BatchNorm(1024, momentum=0.99, eps=0.001),
    )

    return Chain(
        PointNetAggregation(local_mlp, global_mlp),
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
end

function load_and_preprocess_data(files)
    all_input_data = []
    all_output_data = []
    all_raw_output_data = []
    all_out_min = Float32[]
    all_out_max = Float32[]
    file_sample_counts = Int[]

    for file in files
        data = NPZ.npzread(file)

        input_sample = data[:, 1:input_channels, :]
        output_sample = data[:, input_channels+1:input_channels+output_channels, :]

        raw_output = copy(output_sample)

        global_in_min = minimum(input_sample)
        global_in_max = maximum(input_sample)
        input_sample .= ((input_sample .- global_in_min) .* 2.0) ./ (global_in_max - global_in_min) .- 1.0

        global_out_min = minimum(output_sample)
        global_out_max = maximum(output_sample)
        output_sample .= (output_sample .- global_out_min) ./ (global_out_max - global_out_min)

        push!(all_input_data, input_sample)
        push!(all_output_data, output_sample)
        push!(all_raw_output_data, raw_output)

        n_samples = size(data, 3)
        append!(all_out_min, fill(global_out_min, n_samples))
        append!(all_out_max, fill(global_out_max, n_samples))
        push!(file_sample_counts, n_samples)
    end

    return cat(all_input_data..., dims=3), cat(all_output_data..., dims=3),
           cat(all_raw_output_data..., dims=3), all_out_min, all_out_max, file_sample_counts
end

function compute_error_stats(sq_errors_vec::AbstractVector)
    avg_mse = mean(sq_errors_vec)
    sorted_errors = sort(sq_errors_vec, rev=true)
    n = length(sorted_errors)
    top5_mse = mean(@view sorted_errors[1:ceil(Int, 0.05 * n)])
    top1_mse = mean(@view sorted_errors[1:ceil(Int, 0.01 * n)])
    return avg_mse, top5_mse, top1_mse
end

function denormalize_predictions(y_pred, out_mins, out_maxs)
    ranges = reshape(Float32.(out_maxs .- out_mins), 1, 1, :)
    mins = reshape(Float32.(out_mins), 1, 1, :)
    return y_pred .* ranges .+ mins
end

function print_file_eval(file, channel_stats, file_avg, file_top5, file_top1)
    println("  File: $(basename(file))")
    println("    - File Avg. MSE: $file_avg")
    println("    - File Top 5% MSE: $file_top5")
    println("    - File Top 1% MSE: $file_top1")
    for (c, (avg, top5, top1)) in enumerate(channel_stats)
        println("    - Channel $c Avg. MSE: $avg")
        println("    - Channel $c Top 5% Avg. MSE: $top5")
        println("    - Channel $c Top 1% Avg. MSE: $top1")
    end
end

function train_pointnet(input_files::Vector{String}, output_model_file::String, input_channels::Int, output_channels::Int, training_files::Int, validation_files::Int)
    shuffled_files = shuffle(input_files)
    train_files = shuffled_files[1:training_files]
    val_files = shuffled_files[training_files+1:training_files+validation_files]
#     train_files = ["HSX_F14_10p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_1p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_2p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_3p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_4p_nsample_12288_nbdry_3072_low_discrepancy.npy",
# "HSX_F14_5p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_6p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_7p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_F14_8p_nsample_12288_nbdry_3072_low_discrepancy.npy", 
# "HSX_F14_9p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill11p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Hill6p2_nsample_12288_nbdry_3072_low_discrepancy.npy",
# "HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy",	"HSX_Well11p_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well2p9_nsample_12288_nbdry_3072_low_discrepancy.npy", "HSX_Well4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", 
# "HSX_Well6p2_nsample_12288_nbdry_3072_low_discrepancy.npy", "QHS46_nsample_12288_nbdry_3072_low_discrepancy.npy"] 
#     val_files = ["HSX_Hill4p4_nsample_12288_nbdry_3072_low_discrepancy.npy", "aten_nsample_12288_nbdry_3072_low_discrepancy.npy"]
    println("Validation files set to: ", val_files)

    global history_loss = Float32[]
    global history_val_loss = Float32[]

    global history_channel_avg_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top5_mse = [Float32[] for _ in 1:output_channels]
    global history_channel_top1_mse = [Float32[] for _ in 1:output_channels]

    global val_loss_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_5_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global top_1_percent_mse_accumulators = Dict(file => zeros(Float32, output_channels) for file in val_files)
    global file_avg_mse_accumulators = Dict(file => 0.0f0 for file in val_files)
    global file_top5_mse_accumulators = Dict(file => 0.0f0 for file in val_files)
    global file_top1_mse_accumulators = Dict(file => 0.0f0 for file in val_files)
    global overall_val_loss_accumulator = 0.0f0
    global overall_top5_accumulator = 0.0f0
    global overall_top1_accumulator = 0.0f0

    input_data_train, output_data_train, raw_output_train, train_out_mins, train_out_maxs, _ = load_and_preprocess_data(train_files)
    input_data_val, _, raw_output_val, val_out_mins, val_out_maxs, val_file_counts = load_and_preprocess_data(val_files)

    val_file_map = Dict{String, UnitRange{Int}}()
    current_start = 1
    for (file, count) in zip(val_files, val_file_counts)
        val_file_map[file] = current_start:(current_start + count - 1)
        current_start += count
    end

    model = create_pointnet_model(input_channels, output_channels)

    opt = Flux.setup(Optimisers.Adam(lr, (b1, b2)), model)
    loss(m, x, y) = Flux.Losses.mse(m(x), y)
    train_dataloader = DataLoader((input_data_train, output_data_train), batchsize=bs, shuffle=true)

    for epoch in 1:total_epochs
        Flux.train!(loss, model, train_dataloader, opt)

        if epoch % validation_frequency == 0
            y_pred_train_denorm = denormalize_predictions(model(input_data_train), train_out_mins, train_out_maxs)
            train_loss = Flux.Losses.mse(y_pred_train_denorm, raw_output_train)
            y_pred_val_denorm = denormalize_predictions(model(input_data_val), val_out_mins, val_out_maxs)
            val_loss = Flux.Losses.mse(y_pred_val_denorm, raw_output_val)
            push!(history_loss, train_loss)
            push!(history_val_loss, val_loss)

            for c in 1:output_channels
                ch_avg, ch_top5, ch_top1 = compute_error_stats(vec((y_pred_val_denorm[:, c:c, :] .- raw_output_val[:, c:c, :]).^2))
                push!(history_channel_avg_mse[c], ch_avg)
                push!(history_channel_top5_mse[c], ch_top5)
                push!(history_channel_top1_mse[c], ch_top1)
            end

            overall_val_loss_accumulator += val_loss

            _, ov_top5, ov_top1 = compute_error_stats(vec((y_pred_val_denorm .- raw_output_val).^2))
            overall_top5_accumulator += ov_top5
            overall_top1_accumulator += ov_top1

            for file in val_files
                sample_range = val_file_map[file]
                y_raw_slice = raw_output_val[:, :, sample_range]
                y_pred_slice = denormalize_predictions(model(input_data_val[:, :, sample_range]), val_out_mins[sample_range], val_out_maxs[sample_range])

                for c in 1:output_channels
                    ch_avg, ch_top5, ch_top1 = compute_error_stats(vec((y_pred_slice[:, c:c, :] .- y_raw_slice[:, c:c, :]).^2))
                    val_loss_accumulators[file][c] += ch_avg
                    top_5_percent_mse_accumulators[file][c] += ch_top5
                    top_1_percent_mse_accumulators[file][c] += ch_top1
                end

                f_avg, f_top5, f_top1 = compute_error_stats(vec((y_pred_slice .- y_raw_slice).^2))
                file_avg_mse_accumulators[file] += f_avg
                file_top5_mse_accumulators[file] += f_top5
                file_top1_mse_accumulators[file] += f_top1
            end

            if epoch > 0 && epoch % 100 == 0
                num_validations = 100 / validation_frequency
                avg_overall_val_loss = overall_val_loss_accumulator / num_validations
                avg_overall_top5 = overall_top5_accumulator / num_validations
                avg_overall_top1 = overall_top1_accumulator / num_validations
                println("\n" * "="^60)
                println("Epoch: $epoch")
                println("Avg. Overall Validation MSE (last 100 epochs): $avg_overall_val_loss")
                println("Avg. Overall Top 5% MSE (last 100 epochs): $avg_overall_top5")
                println("Avg. Overall Top 1% MSE (last 100 epochs): $avg_overall_top1")
                println("-"^60)

                for file in val_files
                    channel_stats = [(val_loss_accumulators[file][c] / num_validations,
                                      top_5_percent_mse_accumulators[file][c] / num_validations,
                                      top_1_percent_mse_accumulators[file][c] / num_validations) for c in 1:output_channels]
                    print_file_eval(file, channel_stats,
                                    file_avg_mse_accumulators[file] / num_validations,
                                    file_top5_mse_accumulators[file] / num_validations,
                                    file_top1_mse_accumulators[file] / num_validations)
                end
                println("="^60 * "\n")

                overall_val_loss_accumulator = 0.0f0
                overall_top5_accumulator = 0.0f0
                overall_top1_accumulator = 0.0f0
                for file in val_files
                    val_loss_accumulators[file] .= 0.0f0
                    top_5_percent_mse_accumulators[file] .= 0.0f0
                    top_1_percent_mse_accumulators[file] .= 0.0f0
                    file_avg_mse_accumulators[file] = 0.0f0
                    file_top5_mse_accumulators[file] = 0.0f0
                    file_top1_mse_accumulators[file] = 0.0f0
                end
            end
        end
    end

    y_pred_val_denorm = denormalize_predictions(model(input_data_val), val_out_mins, val_out_maxs)
    overall_mse, overall_top5, overall_top1 = compute_error_stats(vec((y_pred_val_denorm .- raw_output_val).^2))

    println("\n" * "="^60)
    println("FINAL MODEL EVALUATION (Epoch: $total_epochs)")
    println("Overall Validation MSE: $overall_mse")
    println("Overall Top 5% MSE: $overall_top5")
    println("Overall Top 1% MSE: $overall_top1")
    println("-"^60)
    for file in val_files
        sample_range = val_file_map[file]
        y_raw_slice = raw_output_val[:, :, sample_range]
        y_pred_slice = denormalize_predictions(model(input_data_val[:, :, sample_range]), val_out_mins[sample_range], val_out_maxs[sample_range])

        channel_stats = [compute_error_stats(vec((y_pred_slice[:, c:c, :] .- y_raw_slice[:, c:c, :]).^2)) for c in 1:output_channels]
        f_avg, f_top5, f_top1 = compute_error_stats(vec((y_pred_slice .- y_raw_slice).^2))
        print_file_eval(file, channel_stats, f_avg, f_top5, f_top1)
    end
    println("="^60 * "\n")

    model_state = Flux.state(model)
    @save output_model_file model_state
end

function plot_losses()
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

all_npy_files = [# "Samples/aten_nsample_12288_nbdry_3072_low_discrepancy.npy",
# "Samples/QHS46_nsample_12288_nbdry_3072_low_discrepancy.npy"
# "Samples/HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_1p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_2p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_3p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_4p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_5p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_6p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_7p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_8p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_9p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_F14_10p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Hill11p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Hill2p9_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Hill4p4_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Hill6p2_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Well11p_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Well2p9_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Well4p4_nsample_12288_nbdry_3072_low_discrepancy.npy",
"Samples/HSX_Well6p2_nsample_12288_nbdry_3072_low_discrepancy.npy"]

train_pointnet(all_npy_files, "mhd.bson", input_channels, output_channels, training_samples, validation_files);
plot_losses()
