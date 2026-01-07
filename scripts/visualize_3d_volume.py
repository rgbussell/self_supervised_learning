#!/usr/bin/env python3
"""
Create 3D visualization of reconstructed volume for README.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def create_3d_visualization(nifti_path, output_path):
    """Create multi-view 3D visualization."""

    # Load NIfTI
    img = nib.load(nifti_path)
    volume = img.get_fdata()

    print(f"Loaded volume: {volume.shape}")
    print(f"Intensity range: [{volume.min():.4f}, {volume.max():.4f}]")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # Get middle slices
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    # 1. Axial view (top-down)
    ax1 = plt.subplot(2, 4, 1)
    axial = volume[z_mid, :, :]
    ax1.imshow(axial, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax1.set_title(f'Axial View (Z={z_mid})', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Sagittal view (side)
    ax2 = plt.subplot(2, 4, 2)
    sagittal = volume[:, :, x_mid]
    ax2.imshow(sagittal.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax2.set_title(f'Sagittal View (X={x_mid})', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Coronal view (front)
    ax3 = plt.subplot(2, 4, 3)
    coronal = volume[:, y_mid, :]
    ax3.imshow(coronal.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax3.set_title(f'Coronal View (Y={y_mid})', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 4. 3D Surface rendering
    ax4 = plt.subplot(2, 4, 4, projection='3d')

    # Threshold for surface
    threshold = 0.3

    # Downsample for visualization
    step = 2
    vol_down = volume[::step, ::step, ::step]

    # Get coordinates where volume > threshold
    z, y, x = np.where(vol_down > threshold)

    # Sample points for visualization (too many points slow down rendering)
    if len(z) > 5000:
        indices = np.random.choice(len(z), 5000, replace=False)
        z, y, x = z[indices], y[indices], x[indices]

    # Color by intensity
    colors = vol_down[z, y, x]

    scatter = ax4.scatter(x, y, z, c=colors, cmap='hot', marker='.', s=1, alpha=0.6)
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_zlabel('Z (slices)', fontsize=10)
    ax4.set_title('3D Volume Rendering', fontsize=14, fontweight='bold')
    ax4.view_init(elev=20, azim=45)

    # 5-7. Multiple axial slices at different depths
    slice_positions = [10, 20, 30]
    for i, z_pos in enumerate(slice_positions):
        ax = plt.subplot(2, 4, 5 + i)
        if z_pos < volume.shape[0]:
            slice_img = volume[z_pos, :, :]
            ax.imshow(slice_img, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Axial Z={z_pos}', fontsize=12, fontweight='bold')
            ax.axis('off')

    # 8. Montage of multiple slices
    ax8 = plt.subplot(2, 4, 8)

    # Create montage of every 5th slice
    montage_slices = []
    for z in range(0, volume.shape[0], 5):
        montage_slices.append(volume[z, :, :])

    # Arrange in grid
    n_slices = len(montage_slices)
    n_cols = 4
    n_rows = (n_slices + n_cols - 1) // n_cols

    # Pad to fill grid
    while len(montage_slices) < n_rows * n_cols:
        montage_slices.append(np.zeros_like(montage_slices[0]))

    # Create montage
    rows = []
    for i in range(n_rows):
        row = np.concatenate(montage_slices[i*n_cols:(i+1)*n_cols], axis=1)
        rows.append(row)
    montage = np.concatenate(rows, axis=0)

    ax8.imshow(montage, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax8.set_title('Slice Montage (every 5th slice)', fontsize=12, fontweight='bold')
    ax8.axis('off')

    plt.suptitle('3D T1 Volume Reconstructed from PET→T1 Predictions',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Visualize sub-09 reconstruction
    nifti_path = 'outputs/reconstructed_volumes/sub-09_reconstructed_20260106_181621/sub-09_reconstructed.nii.gz'
    output_path = 'assets/3d_volume_reconstruction.png'

    # Create assets directory if needed
    Path('assets').mkdir(exist_ok=True)

    create_3d_visualization(nifti_path, output_path)
