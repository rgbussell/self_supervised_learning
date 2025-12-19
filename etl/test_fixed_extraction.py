#!/usr/bin/env python
"""
Test the fixed slice extraction with proper anatomical correspondence.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')
from etl_pipeline import DataPipelineConfig

config = DataPipelineConfig()

resampled_dir = Path(config.output_dir) / "resampled"
subject = "sub-01"

t1_path = resampled_dir / subject / f"{subject}_T1w_1mm.nii.gz"
pet_path = resampled_dir / subject / f"{subject}_pet_1mm.nii.gz"

t1_img = nib.load(t1_path)
pet_img = nib.load(pet_path)

t1_data = t1_img.get_fdata()
pet_data = pet_img.get_fdata()

print("=" * 80)
print("TESTING FIXED SLICE EXTRACTION WITH PROPER ANATOMICAL CORRESPONDENCE")
print("=" * 80)

NUM_SLICES = 40

# Calculate slice ranges (center-based)
t1_shape = t1_data.shape
t1_center_voxel = np.array([t1_shape[0]/2, t1_shape[1]/2, t1_shape[2]/2, 1])
t1_center_physical = t1_img.affine @ t1_center_voxel

pet_center_voxel = np.linalg.inv(pet_img.affine) @ t1_center_physical
pet_z_center = int(np.round(pet_center_voxel[2]))

t1_center_voxel_mapped = np.linalg.inv(t1_img.affine) @ t1_center_physical
t1_z_center = int(np.round(t1_center_voxel_mapped[2]))

pet_start = max(0, pet_z_center - NUM_SLICES // 2)
pet_end = min(pet_data.shape[2], pet_start + NUM_SLICES)
pet_start = max(0, pet_end - NUM_SLICES)

t1_start = max(0, t1_z_center - NUM_SLICES // 2)
t1_end = min(t1_data.shape[2], t1_start + NUM_SLICES)
t1_start = max(0, t1_end - NUM_SLICES)

print(f"\n[1] SLICE RANGES")
print(f"{'='*80}")
print(f"PET center Z: {pet_z_center}, range: [{pet_start}, {pet_end}]")
print(f"T1 center Z: {t1_z_center}, range: [{t1_start}, {t1_end}]")

# Test the fixed extraction logic
print(f"\n[2] TESTING FIXED EXTRACTION LOGIC")
print(f"{'='*80}")
print(f"\nFor each PET slice, finding anatomically corresponding T1 slice:\n")

t1_z_list = []
for slice_idx, pet_z in enumerate(range(pet_start, min(pet_start + 5, pet_end))):
    # Get physical coordinates at center of PET slice
    pet_center_xy = np.array([pet_data.shape[0]//2, pet_data.shape[1]//2, pet_z, 1])
    pet_physical = pet_img.affine @ pet_center_xy
    
    # Convert physical coordinates to T1 voxel space
    t1_voxel = np.linalg.inv(t1_img.affine) @ pet_physical
    t1_z = int(np.round(t1_voxel[2]))
    t1_z = np.clip(t1_z, 0, t1_data.shape[2] - 1)
    t1_z_list.append(t1_z)
    
    # Get slices
    pet_2d = pet_data[..., pet_z]
    t1_2d = t1_data[..., t1_z]
    
    print(f"Slice {slice_idx}:")
    print(f"  PET Z={pet_z}, center physical Z={pet_physical[2]:.2f}")
    print(f"  → T1 Z={t1_z}")
    print(f"  PET stats: min={pet_2d.min():.1f}, max={pet_2d.max():.1f}, mean={pet_2d.mean():.1f}")
    print(f"  T1 stats:  min={t1_2d.min():.1f}, max={t1_2d.max():.1f}, mean={t1_2d.mean():.1f}")

print(f"\n[3] ANATOMICAL CORRESPONDENCE VERIFICATION")
print(f"{'='*80}")

# Check if T1 Z values make sense (should be monotonically increasing)
t1_z_all = []
for pet_z in range(pet_start, pet_end):
    pet_center_xy = np.array([pet_data.shape[0]//2, pet_data.shape[1]//2, pet_z, 1])
    pet_physical = pet_img.affine @ pet_center_xy
    t1_voxel = np.linalg.inv(t1_img.affine) @ pet_physical
    t1_z = np.clip(int(np.round(t1_voxel[2])), 0, t1_data.shape[2] - 1)
    t1_z_all.append(t1_z)

print(f"\nT1 Z indices for all {NUM_SLICES} PET slices:")
print(f"  {t1_z_all}")

# Check monotonicity
is_monotonic = all(t1_z_all[i] <= t1_z_all[i+1] for i in range(len(t1_z_all)-1))
print(f"\nMonotonically increasing: {is_monotonic}")

if is_monotonic:
    print(f"✓ Correspondence is consistent (monotonic)")
else:
    print(f"✗ Correspondence has issues (non-monotonic)")
    print(f"  This could indicate misalignment or different slice orientations")

# Check for reasonable correspondence (should be roughly linear)
z_diff = np.diff(t1_z_all)
print(f"\nDifferences between consecutive T1 Z indices: {z_diff}")
print(f"  Mean diff: {np.mean(z_diff):.2f}")
print(f"  Std diff: {np.std(z_diff):.2f}")

if np.std(z_diff) < 1.0:
    print(f"✓ Correspondence is smooth (low variation)")
else:
    print(f"⚠ Correspondence has jumps (high variation)")

print(f"\n[4] CONCLUSION")
print(f"{'='*80}")
if is_monotonic and np.std(z_diff) < 1.0:
    print(f"✓ FIXED: T1 and PET slices are now anatomically corresponding!")
    print(f"  Each PET slice is paired with the T1 slice at the same physical location.")
else:
    print(f"⚠ Correspondence may still have issues")

print("=" * 80)
