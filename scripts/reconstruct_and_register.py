#!/usr/bin/env python3
"""
3D Volume Reconstruction and Registration Preparation

Reconstructs 3D T1 volumes from 2D PET->T1 predictions and prepares them
for registration to standard T1 atlas (e.g., MNI152).

Usage:
    python reconstruct_and_register.py \\
        --predictions_dir ./outputs/pet_t1_inference/run_20260106_153750/predictions/ \\
        --subject_id sub-09 \\
        --reference_t1 /path/to/sub-09_T1w_1mm.nii.gz \\
        --output_dir ./outputs/reconstructed_volumes/

Output:
    - Reconstructed 3D NIfTI volumes
    - Quality check visualizations
    - Processing logs
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ssl_utils.volume_processing import (
    stack_slices_to_3d,
    normalize_intensity_histogram_matching,
    apply_n4_bias_correction,
    smooth_volume_anisotropic,
    remove_outliers_median_filter,
    normalize_intensity_global,
    create_nifti_from_reference,
    check_intensity_continuity,
    full_reconstruction_pipeline
)


# Subject metadata: slice ranges for test subjects
SUBJECT_INFO = {
    'sub-09': {
        'slice_range': range(441, 481),
        'num_slices': 40,
        'description': 'Test subject 1'
    },
    'sub-10': {
        'slice_range': range(481, 521),
        'num_slices': 40,
        'description': 'Test subject 2'
    }
}


class ReconstructionConfig:
    """Configuration for 3D reconstruction pipeline."""

    def __init__(self, args):
        """Initialize from command-line arguments."""

        # Validate subject ID
        if args.subject_id not in SUBJECT_INFO:
            raise ValueError(
                f"Unknown subject: {args.subject_id}. "
                f"Valid subjects: {list(SUBJECT_INFO.keys())}"
            )

        # Paths
        self.predictions_dir = Path(args.predictions_dir)
        self.subject_id = args.subject_id
        self.reference_t1 = args.reference_t1
        self.output_dir = Path(args.output_dir)

        # Processing options
        self.apply_n4 = args.apply_n4
        self.smooth_sigma_z = args.smooth_sigma_z
        self.smooth_sigma_xy = args.smooth_sigma_xy
        self.save_intermediates = args.save_intermediates

        # Subject-specific info
        self.slice_range = SUBJECT_INFO[args.subject_id]['slice_range']
        self.num_slices = SUBJECT_INFO[args.subject_id]['num_slices']

        # Validate paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required paths exist."""
        if not self.predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {self.predictions_dir}")

        if not Path(self.reference_t1).exists():
            raise FileNotFoundError(f"Reference T1 not found: {self.reference_t1}")

        # Check that at least some prediction files exist
        first_slice = list(self.slice_range)[0]
        first_pred = self.predictions_dir / f"{first_slice:06d}_prediction.npy"
        if not first_pred.exists():
            raise FileNotFoundError(
                f"No prediction files found. Expected: {first_pred}"
            )

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': datetime.now().isoformat(),
            'subject_id': self.subject_id,
            'predictions_dir': str(self.predictions_dir),
            'reference_t1': str(self.reference_t1),
            'output_dir': str(self.output_dir),
            'num_slices': self.num_slices,
            'slice_range': f"{self.slice_range.start}-{self.slice_range.stop}",
            'processing_options': {
                'apply_n4': self.apply_n4,
                'smooth_sigma_z': self.smooth_sigma_z,
                'smooth_sigma_xy': self.smooth_sigma_xy,
                'save_intermediates': self.save_intermediates
            }
        }


def create_output_directory(base_path: Path, subject_id: str) -> Path:
    """
    Create output directory structure.

    Args:
        base_path: Base output directory
        subject_id: Subject identifier

    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"{subject_id}_reconstructed_{timestamp}"

    # Create subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'intermediates').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    return output_dir


def save_config(config: ReconstructionConfig, output_dir: Path):
    """Save run configuration."""
    config_path = output_dir / 'reconstruction_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved: {config_path}")


def visualize_intensity_profile(
    volume_3d: np.ndarray,
    output_path: Path,
    title: str = "Intensity Profile"
):
    """
    Create visualization of mean intensity across slices.

    Args:
        volume_3d: 3D volume (D, H, W)
        output_path: Path to save figure
        title: Plot title
    """
    mean_intensities, max_discontinuity = check_intensity_continuity(volume_3d)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_intensities, linewidth=2)
    ax.set_xlabel('Slice Index', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title(f'{title}\nMax Discontinuity: {max_discontinuity:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved intensity profile: {output_path}")


def visualize_orthogonal_slices(
    volume_3d: np.ndarray,
    output_path: Path,
    title: str = "Orthogonal Views"
):
    """
    Create orthogonal slice visualization (axial, sagittal, coronal).

    Args:
        volume_3d: 3D volume (D, H, W)
        output_path: Path to save figure
        title: Plot title
    """
    # Get middle slices
    z_mid = volume_3d.shape[0] // 2
    y_mid = volume_3d.shape[1] // 2
    x_mid = volume_3d.shape[2] // 2

    # Extract slices
    axial = volume_3d[z_mid, :, :]
    sagittal = volume_3d[:, :, x_mid]
    coronal = volume_3d[:, y_mid, :]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'Axial (Z={z_mid})', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sagittal.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f'Sagittal (X={x_mid})', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(coronal.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title(f'Coronal (Y={y_mid})', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved orthogonal views: {output_path}")


def run_step_by_step_reconstruction(config: ReconstructionConfig, output_dir: Path):
    """
    Run reconstruction with intermediate outputs saved.

    Args:
        config: Reconstruction configuration
        output_dir: Output directory

    Returns:
        Final reconstructed NIfTI image
    """
    print("="*80)
    print("3D Volume Reconstruction Pipeline (Step-by-Step)")
    print("="*80)
    print(f"Subject: {config.subject_id}")
    print(f"Slices: {config.num_slices} ({config.slice_range.start}-{config.slice_range.stop})")
    print(f"Reference T1: {config.reference_t1}")
    print("="*80)

    intermediates_dir = output_dir / 'intermediates'
    viz_dir = output_dir / 'visualizations'

    # Step 1: Stack slices
    print("\n[1/7] Stacking 2D slices...")
    volume_3d = stack_slices_to_3d(config.predictions_dir, config.slice_range)

    if config.save_intermediates:
        np.save(intermediates_dir / '01_stacked.npy', volume_3d)
        visualize_intensity_profile(
            volume_3d,
            viz_dir / '01_intensity_profile_stacked.png',
            "After Stacking"
        )

    # Step 2: Remove outliers
    print("\n[2/7] Removing outliers...")
    volume_3d = remove_outliers_median_filter(volume_3d, kernel_size=3)

    if config.save_intermediates:
        np.save(intermediates_dir / '02_outliers_removed.npy', volume_3d)

    # Step 3: Histogram matching
    print("\n[3/7] Normalizing intensity (histogram matching)...")
    volume_3d = normalize_intensity_histogram_matching(volume_3d)

    if config.save_intermediates:
        np.save(intermediates_dir / '03_histogram_matched.npy', volume_3d)
        visualize_intensity_profile(
            volume_3d,
            viz_dir / '02_intensity_profile_histogram_matched.png',
            "After Histogram Matching"
        )

    # Step 4: N4 bias correction
    if config.apply_n4:
        print("\n[4/7] Applying N4 bias correction...")
        volume_3d = apply_n4_bias_correction(volume_3d)

        if config.save_intermediates:
            np.save(intermediates_dir / '04_n4_corrected.npy', volume_3d)
            visualize_intensity_profile(
                volume_3d,
                viz_dir / '03_intensity_profile_n4_corrected.png',
                "After N4 Bias Correction"
            )
    else:
        print("\n[4/7] Skipping N4 bias correction")

    # Step 5: Anisotropic smoothing
    print("\n[5/7] Applying anisotropic smoothing...")
    volume_3d = smooth_volume_anisotropic(
        volume_3d,
        config.smooth_sigma_z,
        config.smooth_sigma_xy
    )

    if config.save_intermediates:
        np.save(intermediates_dir / '05_smoothed.npy', volume_3d)
        visualize_intensity_profile(
            volume_3d,
            viz_dir / '04_intensity_profile_smoothed.png',
            "After Smoothing"
        )

    # Step 6: Global normalization
    print("\n[6/7] Global intensity normalization...")
    volume_3d = normalize_intensity_global(volume_3d)

    if config.save_intermediates:
        np.save(intermediates_dir / '06_normalized.npy', volume_3d)

    # Step 7: Create NIfTI
    print("\n[7/7] Creating NIfTI with spatial metadata...")
    nifti_img = create_nifti_from_reference(volume_3d, config.reference_t1)

    # Final quality check
    print("\nFinal quality check:")
    check_intensity_continuity(volume_3d)

    # Create final visualizations
    print("\nCreating visualizations...")
    visualize_intensity_profile(
        volume_3d,
        viz_dir / '05_intensity_profile_final.png',
        "Final Reconstruction"
    )
    visualize_orthogonal_slices(
        volume_3d,
        viz_dir / '06_orthogonal_views.png',
        f"{config.subject_id} - Reconstructed Volume"
    )

    print("\n" + "="*80)
    print("Reconstruction complete!")
    print("="*80)

    return nifti_img


def main():
    """Main reconstruction pipeline."""

    parser = argparse.ArgumentParser(
        description='3D Volume Reconstruction from 2D Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction for sub-09
  python reconstruct_and_register.py \\
      --predictions_dir ./outputs/pet_t1_inference/run_20260106_153750/predictions/ \\
      --subject_id sub-09 \\
      --reference_t1 /home/rbussell/data/pet_mri/resampled/sub-09/sub-09_T1w_1mm.nii.gz

  # With N4 bias correction and intermediate outputs
  python reconstruct_and_register.py \\
      --predictions_dir ./outputs/pet_t1_inference/run_20260106_153750/predictions/ \\
      --subject_id sub-10 \\
      --reference_t1 /home/rbussell/data/pet_mri/resampled/sub-10/sub-10_T1w_1mm.nii.gz \\
      --apply_n4 \\
      --save_intermediates
        """
    )

    parser.add_argument(
        '--predictions_dir',
        type=str,
        required=True,
        help='Directory containing prediction .npy files'
    )
    parser.add_argument(
        '--subject_id',
        type=str,
        required=True,
        choices=list(SUBJECT_INFO.keys()),
        help='Subject identifier (sub-09 or sub-10)'
    )
    parser.add_argument(
        '--reference_t1',
        type=str,
        required=True,
        help='Path to reference T1 NIfTI for spatial metadata'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/reconstructed_volumes/',
        help='Base output directory'
    )
    parser.add_argument(
        '--apply_n4',
        action='store_true',
        help='Apply N4 bias field correction (slower but better quality)'
    )
    parser.add_argument(
        '--smooth_sigma_z',
        type=float,
        default=0.5,
        help='Smoothing sigma along Z-axis (default: 0.5)'
    )
    parser.add_argument(
        '--smooth_sigma_xy',
        type=float,
        default=0.3,
        help='Smoothing sigma in XY plane (default: 0.3)'
    )
    parser.add_argument(
        '--save_intermediates',
        action='store_true',
        help='Save intermediate processing steps'
    )

    args = parser.parse_args()

    # Initialize configuration
    print("="*80)
    print("3D Volume Reconstruction Pipeline")
    print("="*80)
    config = ReconstructionConfig(args)

    # Create output directory
    output_dir = create_output_directory(config.output_dir, config.subject_id)
    save_config(config, output_dir)

    # Run reconstruction
    nifti_img = run_step_by_step_reconstruction(config, output_dir)

    # Save final NIfTI
    output_path = output_dir / f'{config.subject_id}_reconstructed.nii.gz'
    nib.save(nifti_img, output_path)
    print(f"\nFinal NIfTI saved: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("RECONSTRUCTION SUMMARY")
    print("="*80)
    print(f"Subject: {config.subject_id}")
    print(f"Input slices: {config.num_slices}")
    print(f"Output shape: {nifti_img.shape}")
    print(f"Output file: {output_path}")
    print(f"Visualizations: {output_dir / 'visualizations'}")
    if config.save_intermediates:
        print(f"Intermediates: {output_dir / 'intermediates'}")
    print("="*80)


if __name__ == '__main__':
    main()
