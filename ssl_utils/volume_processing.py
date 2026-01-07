#!/usr/bin/env python3
"""
Volume Processing Utilities for 3D Reconstruction

Functions for reconstructing 3D volumes from 2D slice predictions,
including intensity normalization, smoothing, and NIfTI creation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter
from skimage.exposure import match_histograms


def stack_slices_to_3d(prediction_dir: Path, slice_range: range) -> np.ndarray:
    """
    Stack 2D prediction slices into a 3D volume.

    Args:
        prediction_dir: Directory containing prediction .npy files
        slice_range: Range of slice indices to load (e.g., range(441, 481))

    Returns:
        3D numpy array with shape (D, H, W) where D is number of slices

    Raises:
        FileNotFoundError: If prediction files are not found
        ValueError: If slice dimensions are inconsistent
    """
    slices = []

    for idx in slice_range:
        # Look for prediction file
        pred_file = prediction_dir / f"{idx:06d}_prediction.npy"

        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        # Load prediction
        pred_slice = np.load(pred_file)

        # Validate shape
        if len(pred_slice.shape) != 2:
            raise ValueError(f"Expected 2D slice, got shape {pred_slice.shape}")

        # Validate consistency
        if slices and pred_slice.shape != slices[0].shape:
            raise ValueError(
                f"Inconsistent slice dimensions: {pred_slice.shape} vs {slices[0].shape}"
            )

        slices.append(pred_slice)

    # Stack along depth dimension
    volume_3d = np.stack(slices, axis=0)  # Shape: (D, H, W)

    print(f"Stacked {len(slices)} slices into volume: {volume_3d.shape}")

    return volume_3d


def normalize_intensity_histogram_matching(
    volume_3d: np.ndarray,
    reference_slice_idx: Optional[int] = None
) -> np.ndarray:
    """
    Normalize intensity across slices using histogram matching.

    Each slice is matched to a reference slice (default: middle slice).

    Args:
        volume_3d: Input 3D volume (D, H, W)
        reference_slice_idx: Index of reference slice (default: middle)

    Returns:
        Intensity-normalized 3D volume
    """
    # Use middle slice as reference if not specified
    if reference_slice_idx is None:
        reference_slice_idx = volume_3d.shape[0] // 2

    reference_slice = volume_3d[reference_slice_idx, :, :]
    normalized = np.zeros_like(volume_3d)

    for z in range(volume_3d.shape[0]):
        normalized[z, :, :] = match_histograms(
            volume_3d[z, :, :],
            reference_slice,
            channel_axis=None
        )

    print(f"Applied histogram matching to {volume_3d.shape[0]} slices")

    return normalized


def apply_n4_bias_correction(volume_3d: np.ndarray) -> np.ndarray:
    """
    Apply N4 bias field correction to remove intensity non-uniformity.

    Uses SimpleITK's N4BiasFieldCorrectionImageFilter.

    Args:
        volume_3d: Input 3D volume (D, H, W)

    Returns:
        Bias-corrected 3D volume
    """
    # Convert to SimpleITK image
    img_sitk = sitk.GetImageFromArray(volume_3d.astype(np.float32))

    # Apply N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])

    print("Applying N4 bias field correction...")
    corrected = corrector.Execute(img_sitk)

    # Convert back to numpy
    corrected_np = sitk.GetArrayFromImage(corrected)

    print("N4 bias correction complete")

    return corrected_np


def smooth_volume_anisotropic(
    volume_3d: np.ndarray,
    sigma_z: float = 0.5,
    sigma_xy: float = 0.3
) -> np.ndarray:
    """
    Apply anisotropic Gaussian smoothing.

    More smoothing along Z-axis (inter-slice), less in-plane (XY).

    Args:
        volume_3d: Input 3D volume (D, H, W)
        sigma_z: Standard deviation along Z (depth) axis
        sigma_xy: Standard deviation along X and Y (in-plane) axes

    Returns:
        Smoothed 3D volume
    """
    # sigma order: (Z, Y, X)
    smoothed = gaussian_filter(volume_3d, sigma=(sigma_z, sigma_xy, sigma_xy))

    print(f"Applied anisotropic smoothing: sigma=(Z:{sigma_z}, Y:{sigma_xy}, X:{sigma_xy})")

    return smoothed


def remove_outliers_median_filter(
    volume_3d: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Remove outlier slices using median filter along Z-axis.

    Args:
        volume_3d: Input 3D volume (D, H, W)
        kernel_size: Kernel size for median filter along Z

    Returns:
        Filtered 3D volume
    """
    # Apply median filter only along Z direction
    filtered = median_filter(volume_3d, size=(kernel_size, 1, 1))

    print(f"Applied median filter along Z-axis: kernel_size={kernel_size}")

    return filtered


def normalize_intensity_global(volume_3d: np.ndarray) -> np.ndarray:
    """
    Normalize intensity using global volume statistics.

    Uses 0.2 and 99.8 percentiles to clip extreme values.

    Args:
        volume_3d: Input 3D volume (D, H, W)

    Returns:
        Normalized 3D volume in [0, 1] range
    """
    p2, p98 = np.percentile(volume_3d, [0.2, 99.8])
    normalized = np.clip((volume_3d - p2) / (p98 - p2 + 1e-8), 0, 1)

    print(f"Global normalization: [{p2:.4f}, {p98:.4f}] -> [0, 1]")

    return normalized


def create_nifti_from_reference(
    volume_3d: np.ndarray,
    reference_t1_path: str,
    slice_offset: Optional[int] = None
) -> nib.Nifti1Image:
    """
    Create NIfTI image with proper spatial metadata from reference T1.

    Adjusts affine matrix to account for slice selection from original volume.

    Args:
        volume_3d: 3D volume array (D, H, W)
        reference_t1_path: Path to original T1 NIfTI for affine matrix
        slice_offset: Number of slices from bottom that were skipped
                     (auto-calculated if None, assuming centered extraction)

    Returns:
        nibabel.Nifti1Image with correct affine matrix
    """
    # Load reference T1
    ref_img = nib.load(reference_t1_path)
    ref_affine = ref_img.affine.copy()

    # Calculate slice offset if not provided
    if slice_offset is None:
        original_depth = ref_img.shape[2]
        extracted_depth = volume_3d.shape[0]
        slice_offset = (original_depth - extracted_depth) // 2

    # Create new affine for the slice subset
    new_affine = ref_affine.copy()

    # Adjust Z translation in affine matrix
    # affine[2, 3] is the Z-origin, affine[2, 2] is the Z-spacing
    new_affine[2, 3] = ref_affine[2, 3] + slice_offset * ref_affine[2, 2]

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume_3d, new_affine)

    print(f"Created NIfTI with shape {volume_3d.shape}")
    print(f"  Slice offset: {slice_offset}")
    print(f"  Z-origin adjusted: {ref_affine[2, 3]:.2f} -> {new_affine[2, 3]:.2f}")

    return nifti_img


def check_intensity_continuity(volume_3d: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Check for intensity discontinuities between slices.

    Args:
        volume_3d: Input 3D volume (D, H, W)

    Returns:
        Tuple of (mean_intensities_per_slice, max_discontinuity)
    """
    mean_intensities = np.array([
        np.mean(volume_3d[z, :, :])
        for z in range(volume_3d.shape[0])
    ])

    # Calculate maximum discontinuity between adjacent slices
    if len(mean_intensities) > 1:
        diffs = np.abs(np.diff(mean_intensities))
        max_discontinuity = np.max(diffs)
    else:
        max_discontinuity = 0.0

    print(f"Intensity continuity check:")
    print(f"  Mean intensity range: [{mean_intensities.min():.4f}, {mean_intensities.max():.4f}]")
    print(f"  Max discontinuity: {max_discontinuity:.4f}")

    return mean_intensities, max_discontinuity


def pad_volume_to_depth(
    volume_3d: np.ndarray,
    target_depth: int,
    mode: str = 'edge'
) -> np.ndarray:
    """
    Pad volume to target depth.

    Args:
        volume_3d: Input 3D volume (D, H, W)
        target_depth: Desired number of slices
        mode: Padding mode ('edge', 'constant', 'reflect', etc.)

    Returns:
        Padded 3D volume
    """
    current_depth = volume_3d.shape[0]
    pad_needed = target_depth - current_depth

    if pad_needed <= 0:
        print(f"No padding needed: current depth {current_depth} >= target {target_depth}")
        return volume_3d

    # Pad symmetrically
    pad_top = pad_needed // 2
    pad_bottom = pad_needed - pad_top

    padded = np.pad(
        volume_3d,
        ((pad_top, pad_bottom), (0, 0), (0, 0)),
        mode=mode
    )

    print(f"Padded volume: {volume_3d.shape} -> {padded.shape} (mode={mode})")

    return padded


def full_reconstruction_pipeline(
    prediction_dir: Path,
    slice_range: range,
    reference_t1_path: str,
    apply_n4: bool = True,
    smooth_sigma_z: float = 0.5,
    smooth_sigma_xy: float = 0.3
) -> nib.Nifti1Image:
    """
    Complete reconstruction pipeline from 2D predictions to 3D NIfTI.

    Pipeline steps:
    1. Stack slices
    2. Remove outliers (median filter)
    3. Histogram matching
    4. N4 bias correction (optional)
    5. Anisotropic smoothing
    6. Global normalization
    7. Create NIfTI with proper affine

    Args:
        prediction_dir: Directory with prediction .npy files
        slice_range: Range of slice indices
        reference_t1_path: Path to reference T1 for affine matrix
        apply_n4: Whether to apply N4 bias correction (slower)
        smooth_sigma_z: Smoothing parameter along Z-axis
        smooth_sigma_xy: Smoothing parameter in XY plane

    Returns:
        Reconstructed NIfTI image
    """
    print("="*80)
    print("3D Volume Reconstruction Pipeline")
    print("="*80)

    # Step 1: Stack slices
    print("\n[1/7] Stacking 2D slices...")
    volume_3d = stack_slices_to_3d(prediction_dir, slice_range)

    # Step 2: Remove outliers
    print("\n[2/7] Removing outliers...")
    volume_3d = remove_outliers_median_filter(volume_3d, kernel_size=3)

    # Step 3: Histogram matching
    print("\n[3/7] Normalizing intensity (histogram matching)...")
    volume_3d = normalize_intensity_histogram_matching(volume_3d)

    # Step 4: N4 bias correction (optional)
    if apply_n4:
        print("\n[4/7] Applying N4 bias correction...")
        volume_3d = apply_n4_bias_correction(volume_3d)
    else:
        print("\n[4/7] Skipping N4 bias correction")

    # Step 5: Anisotropic smoothing
    print("\n[5/7] Applying anisotropic smoothing...")
    volume_3d = smooth_volume_anisotropic(volume_3d, smooth_sigma_z, smooth_sigma_xy)

    # Step 6: Global normalization
    print("\n[6/7] Global intensity normalization...")
    volume_3d = normalize_intensity_global(volume_3d)

    # Step 7: Create NIfTI
    print("\n[7/7] Creating NIfTI with spatial metadata...")
    nifti_img = create_nifti_from_reference(volume_3d, reference_t1_path)

    # Check continuity
    print("\nFinal quality check:")
    check_intensity_continuity(volume_3d)

    print("\n" + "="*80)
    print("Reconstruction complete!")
    print("="*80)

    return nifti_img
