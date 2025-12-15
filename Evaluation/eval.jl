"""
# Model Evaluation Script for PointNet MHD

This script evaluates a trained PointNet model on MHD (Magnetohydrodynamics) simulation data.

## Overview
The script loads a pre-trained model, runs inference on test data, and computes predictions
for magnetic field components (Bx, By, Bz) from 3D point cloud coordinates (x, y, z).

## Data Format
- Input file: NPZ file with shape (num_samples, num_features, num_points)
  - Columns 1-3: x, y, z coordinates (input features)
  - Columns 4-6: Bx, By, Bz magnetic field components (ground truth)

## Output Files
- `solutions.npy`: Model predictions for Bx, By, Bz components
- `error.npy`: Mean squared error per sample

## Workflow
1. Load and normalize input coordinates (x, y, z)
2. Load pre-trained model from BSON file
3. Run model inference
4. Denormalize predictions to original scale
5. Compute and save errors
"""

using Pkg
# Package installation commands (uncomment if needed)
# Pkg.add("NPZ")
# Pkg.add("Flux")
# Pkg.add("BSON")
# Pkg.add("Statistics")
using NPZ, LinearAlgebra, Statistics
using Flux: Chain, relu, Conv, BatchNorm, sigmoid, Dense
using BSON: @load

# Load the PointNet model architecture and utilities
include("../PointNetMHD.jl")

# --- Load Data ---
# Load the test sample containing both input coordinates and ground truth values
file = "../Solution/HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy"

# Extract input features (x, y, z coordinates) from columns 1-3
data = NPZ.npzread(file)[:, 1:3, :]

# Extract ground truth magnetic field components (Bx, By, Bz) from columns 4-6
ground_truth = NPZ.npzread(file)[:, 4:6, :]

# --- Normalize Input Data ---
# Normalize coordinates to [-1, 1] range for better model performance
normalize_data!(data, :minus_one_to_one)

# --- Load Pre-trained Model ---
# Load the trained PointNet model from BSON file
@load "mhd_12288_3072.bson" model

# --- Run Inference ---
# Run the model to get predictions for Bx, By, Bz components
solutions = Array(model(data))

# --- Denormalize Predictions ---
# Scale predictions back to original physical units
# For each component (Bx, By, Bz):
for j in 1:3
    # Get min and max values from ground truth for this component
    mmin = minimum(ground_truth[:, j, :])
    mmax = maximum(ground_truth[:, j, :])

    # Denormalize: scaled_value = normalized_value * (max - min) + min
    solutions[:, j, :] .= solutions[:, j, :] .* (mmax - mmin) .+ mmin
end

# --- Save Results ---
# Save model predictions
NPZ.npzwrite("solutions.npy", solutions)

# Save mean squared error per sample (averaged over spatial dimensions)
NPZ.npzwrite("error.npy", mean((solutions .- ground_truth).^2, dims=2))