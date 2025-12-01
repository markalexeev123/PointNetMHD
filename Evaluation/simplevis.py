import numpy as np
import matplotlib.pyplot as plt

def plot_and_cut_data(data_file_path, coord_file_path, zoom_factor=0.5):
    all_data = np.load(data_file_path)
    coords_data = np.load(coord_file_path)
    
    xyz_coords = coords_data[:, 0:3]
    color_values = all_data#[:, 2]

    angles = np.arctan2(xyz_coords[:, 1], xyz_coords[:, 0])
    angles [angles < 0] += 2 * np.pi

    abs_color_values = np.abs(color_values)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        xyz_coords[:, 0],
        xyz_coords[:, 1],
        xyz_coords[:, 2],
        s=70,
        alpha=0.5,
        c=abs_color_values,
        cmap='Reds',
        vmin=0,
        vmax=np.percentile(abs_color_values, 95)
    )

    cbar = fig.colorbar(scatter, ax=ax,  shrink=0.6)

    x_min, x_max = xyz_coords[:, 0].min(), xyz_coords[:, 0].max()
    y_min, y_max = xyz_coords[:, 1].min(), xyz_coords[:, 1].max()
    z_min, z_max = xyz_coords[:, 2].min(), xyz_coords[:, 2].max()

    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()

    adjusted_range = max_range * zoom_factor

    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - adjusted_range / 2, mid_x + adjusted_range / 2)
    ax.set_ylim(mid_y - adjusted_range / 2, mid_y + adjusted_range / 2)
    ax.set_zlim(mid_z - adjusted_range / 2, mid_z + adjusted_range / 2)

    plt.show()

plot_and_cut_data("error.npy", "HSX_F14_1p_samples.npy", zoom_factor=0.7)