import numpy as np
import matplotlib.pyplot as plt

def plot_side_by_side(data_file_path, solutions_file_path, error_file_path, cut_fraction=0.2, zoom_factor=0.5):
    # --- 1. Load all data sources ---
    all_data = np.load(data_file_path)
    xyz_coords = all_data[:, 0:3]
    
    ground_truth_colors = all_data[:, 4] 
    prediction_colors = np.load(solutions_file_path)[:, 1]

    error_colors = np.load(error_file_path).flatten()

    # --- 2. Create the cutting mask (applied to all plots) ---
    angles = np.arctan2(xyz_coords[:, 1], xyz_coords[:, 0])
    angles[angles < 0] += 2 * np.pi 
    cut_mask = angles > cut_fraction * 2 * np.pi
    
    cut_xyz_coords = xyz_coords[cut_mask]
    cut_ground_truth_colors = ground_truth_colors[cut_mask]
    cut_prediction_colors = prediction_colors[cut_mask]
    cut_error_colors = error_colors[cut_mask]

    # --- 3. Create a figure with three 3D subplots ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Ground Truth vs. Prediction vs. Error of {data_file_path}', fontsize=20)

    # --- 4. Synchronize color scale for Ground Truth and Prediction ---
    vmin_gt_pred = min(cut_ground_truth_colors.min(), cut_prediction_colors.min())
    vmax_gt_pred = max(cut_ground_truth_colors.max(), cut_prediction_colors.max())

    # --- 5. Plot the Ground Truth (Left) ---
    scatter1 = ax1.scatter(
        cut_xyz_coords[:, 0], cut_xyz_coords[:, 1], cut_xyz_coords[:, 2], 
        s=5, alpha=0.5, c=cut_ground_truth_colors, cmap='inferno_r',
        vmin=vmin_gt_pred, vmax=vmax_gt_pred
    )
    fig.colorbar(scatter1, ax=ax1, label='P Component Value', shrink=0.6)
    ax1.set_title('Ground Truth')

    # --- 6. Plot the Prediction (Middle) ---
    scatter2 = ax2.scatter(
        cut_xyz_coords[:, 0], cut_xyz_coords[:, 1], cut_xyz_coords[:, 2], 
        s=5, alpha=0.5, c=cut_prediction_colors, cmap='inferno_r',
        vmin=vmin_gt_pred, vmax=vmax_gt_pred
    )
    fig.colorbar(scatter2, ax=ax2, label='P Component Value', shrink=0.6)
    ax2.set_title('Prediction')

    # --- 7. Plot the Error (Right) ---
    scatter3 = ax3.scatter(
        cut_xyz_coords[:, 0], cut_xyz_coords[:, 1], cut_xyz_coords[:, 2], 
        s=5, alpha=0.5, c=cut_error_colors, cmap='Reds',
        vmin=0, vmax=np.percentile(cut_error_colors, 100)
    )
    fig.colorbar(scatter3, ax=ax3, label='Mean Squared Error', shrink=0.6)
    ax3.set_title('Error')

    # --- 8. Apply zoom and labels to ALL plots ---
    x_min, x_max = cut_xyz_coords[:, 0].min(), cut_xyz_coords[:, 0].max()
    y_min, y_max = cut_xyz_coords[:, 1].min(), cut_xyz_coords[:, 1].max()
    z_min, z_max = cut_xyz_coords[:, 2].min(), cut_xyz_coords[:, 2].max()

    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
    adjusted_range = max_range * zoom_factor

    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(mid_x - adjusted_range / 2, mid_x + adjusted_range / 2)
        ax.set_ylim(mid_y - adjusted_range / 2, mid_y + adjusted_range / 2)
        ax.set_zlim(mid_z - adjusted_range / 2, mid_z + adjusted_range / 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.show()

plot_side_by_side("HSX_QHS_nsample_12288_nbdry_3072_low_discrepancy.npy", "solutions.npy", "error.npy", cut_fraction=0.0, zoom_factor=0.65)