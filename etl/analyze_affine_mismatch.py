#!/usr/bin/env python
"""
Investigate why T1 and PET have different affines after resampling to 1mm isotropic.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')
from etl_pipeline import DataPipelineConfig, resample_to_isotropic

config = DataPipelineConfig()

# Get paths for first subject
resampled_dir = Path(config.output_dir) / "resampled"
pet_single_tp_dir = Path(config.output_dir) / "pet_single_timepoint"
subject = "sub-01"

# Load ORIGINAL images (before resampling)
t1_original_path = Path(config.input_dir) / subject / "anat" / f"{subject}_T1w.nii.gz"
pet_single_path = pet_single_tp_dir / f"{subject}_pet_t200.nii.gz"

t1_orig = nib.load(t1_original_path)
pet_orig = nib.load(pet_single_path)

print("=" * 80)
print("INVESTIGATING AFFINE MISMATCH AFTER RESAMPLING")
print("=" * 80)

print(f"\n[1] ORIGINAL IMAGES AFFINES")
print(f"{'='*80}")
print(f"\nT1 original:")
print(f"  Shape: {t1_orig.shape}")
print(f"  Affine:\n{t1_orig.affine}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(t1_orig.affine[:3, i]) for i in range(3)])}")

print(f"\nPET original:")
print(f"  Shape: {pet_orig.shape}")
print(f"  Affine:\n{pet_orig.affine}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(pet_orig.affine[:3, i]) for i in range(3)])}")

# Now resample both
print(f"\n[2] RESAMPLING TO 1MM ISOTROPIC")
print(f"{'='*80}")

t1_resampled = resample_to_isotropic(t1_orig, 1.0)
pet_resampled = resample_to_isotropic(pet_orig, 1.0)

print(f"\nT1 resampled:")
print(f"  Shape: {t1_resampled.shape}")
print(f"  Affine:\n{t1_resampled.affine}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(t1_resampled.affine[:3, i]) for i in range(3)])}")

print(f"\nPET resampled:")
print(f"  Shape: {pet_resampled.shape}")
print(f"  Affine:\n{pet_resampled.affine}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(pet_resampled.affine[:3, i]) for i in range(3)])}")

print(f"\n[3] AFFINE DIFFERENCES")
print(f"{'='*80}")
affines_equal = np.allclose(t1_resampled.affine, pet_resampled.affine)
print(f"Affines equal: {affines_equal}")

if not affines_equal:
    print(f"\nDifference matrix:")
    diff = t1_resampled.affine - pet_resampled.affine
    print(diff)
    
    print(f"\nT1 diagonal (should be ±1 for isotropic):")
    print(np.diag(t1_resampled.affine))
    
    print(f"\nPET diagonal (should be ±1 for isotropic):")
    print(np.diag(pet_resampled.affine))
    
    print(f"\nT1 origin:")
    print(t1_resampled.affine[:3, 3])
    
    print(f"\nPET origin:")
    print(pet_resampled.affine[:3, 3])

# Check the voxel size preservation
print(f"\n[4] VOXEL SIZE ANALYSIS")
print(f"{'='*80}")

t1_orig_voxel_sizes = np.array([np.linalg.norm(t1_orig.affine[:3, i]) for i in range(3)])
pet_orig_voxel_sizes = np.array([np.linalg.norm(pet_orig.affine[:3, i]) for i in range(3)])

print(f"\nOriginal voxel sizes:")
print(f"  T1:  {t1_orig_voxel_sizes}")
print(f"  PET: {pet_orig_voxel_sizes}")

print(f"\nExpected new shape (based on voxel sizes):")
print(f"  T1 should scale by: {t1_orig_voxel_sizes}")
print(f"  T1 new shape: {np.round(np.array(t1_orig.shape) * t1_orig_voxel_sizes)}")
print(f"  PET should scale by: {pet_orig_voxel_sizes}")
print(f"  PET new shape: {np.round(np.array(pet_orig.shape) * pet_orig_voxel_sizes)}")

print(f"\nActual resampled shapes:")
print(f"  T1:  {t1_resampled.shape}")
print(f"  PET: {pet_resampled.shape}")

print(f"\n[5] PHYSICAL SPACE CONSISTENCY")
print(f"{'='*80}")

# Map same physical point in both original images
test_voxel = np.array([0, 0, 0, 1])

t1_orig_phys = t1_orig.affine @ test_voxel
pet_orig_phys = pet_orig.affine @ test_voxel

print(f"\nOrigin (0,0,0,1) maps to physical:")
print(f"  T1 original: {t1_orig_phys[:3]}")
print(f"  PET original: {pet_orig_phys[:3]}")

t1_resampled_phys = t1_resampled.affine @ test_voxel
pet_resampled_phys = pet_resampled.affine @ test_voxel

print(f"\nAfter resampling, origin (0,0,0,1) maps to:")
print(f"  T1 resampled: {t1_resampled_phys[:3]}")
print(f"  PET resampled: {pet_resampled_phys[:3]}")

# Check center points
t1_center_orig = t1_orig.affine @ np.array([t1_orig.shape[0]/2, t1_orig.shape[1]/2, t1_orig.shape[2]/2, 1])
pet_center_orig = pet_orig.affine @ np.array([pet_orig.shape[0]/2, pet_orig.shape[1]/2, pet_orig.shape[2]/2, 1])

t1_center_resampled = t1_resampled.affine @ np.array([t1_resampled.shape[0]/2, t1_resampled.shape[1]/2, t1_resampled.shape[2]/2, 1])
pet_center_resampled = pet_resampled.affine @ np.array([pet_resampled.shape[0]/2, pet_resampled.shape[1]/2, pet_resampled.shape[2]/2, 1])

print(f"\nCenter points:")
print(f"  T1 original center (phys): {t1_center_orig[:3]}")
print(f"  PET original center (phys): {pet_center_orig[:3]}")
print(f"  T1 resampled center (phys): {t1_center_resampled[:3]}")
print(f"  PET resampled center (phys): {pet_center_resampled[:3]}")

print(f"\n[6] ROOT CAUSE ANALYSIS")
print(f"{'='*80}")
print(f"""
The issue is that T1 and PET preserve their ORIGINAL affine origins,
but scale only the voxel sizes to 1mm isotropic.

This means:
- T1's new affine origin = T1's original affine origin
- PET's new affine origin = PET's original affine origin
- These are in DIFFERENT physical coordinate systems!

The two images describe different regions of physical space with 
different orientations/origins. To properly align them, we need to:

1. Map them to a COMMON physical space before or after resampling, OR
2. Use physical-space correspondence when extracting slices

CURRENT BUG:
The code assumes T1 and PET are in the same space and just extracts
sequential slices by index, which is wrong.

SOLUTION:
For each PET slice at voxel position Z_pet:
1. Convert to physical coordinates using PET affine
2. Convert physical coordinates to T1 voxel space using T1 inverse affine
3. Extract T1 slice at resulting voxel position

This ensures anatomical correspondence!
""")

print("=" * 80)
