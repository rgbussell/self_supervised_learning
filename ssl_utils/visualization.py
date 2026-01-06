"""
Visualization utilities for PET-to-T1 inference results.

Creates comparison images showing:
- Input PET image
- Predicted T1 image
- Ground truth T1 image
"""

import matplotlib.pyplot as plt
import numpy as np


def save_comparison_image(pet_input, pred_t1, gt_t1, save_path, metrics):
    """
    Save a 3-panel comparison image showing input PET, predicted T1, and ground truth T1.

    Args:
        pet_input: Input PET image (256x256, normalized to [0, 1])
        pred_t1: Predicted T1 image (256x256, normalized to [0, 1])
        gt_t1: Ground truth T1 image (256x256, normalized to [0, 1])
        save_path: Path where to save the PNG file
        metrics: Dictionary with 'psnr', 'ssim', 'nmse' values

    Returns:
        None (saves image to disk)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input PET
    axes[0].imshow(pet_input, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input PET', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Predicted T1 with metrics
    axes[1].imshow(pred_t1, cmap='gray', vmin=0, vmax=1)
    title_text = f'Predicted T1\nPSNR: {metrics["psnr"]:.2f} dB | SSIM: {metrics["ssim"]:.4f}'
    axes[1].set_title(title_text, fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Ground Truth T1
    axes[2].imshow(gt_t1, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth T1', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
