#!/usr/bin/env python
"""
Diagnostic script to test the resampling fix.
Compares single-timepoint PET data before and after resampling.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import sys
sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')
from etl_pipeline import resample_to_isotropic

# Paths
DATA_DIR = Path("/home/rbussell/data/pet_mri")
PET_SINGLE_TP_DIR = DATA_DIR / "pet_single_timepoint"
RESAMPLED_DIR = DATA_DIR / "resampled"

# Find first subject to test
pet_files = sorted(PET_SINGLE_TP_DIR.glob("*_pet_t200.nii.gz"))

if not pet_files:
    print("Error: No PET single timepoint files found!")
    sys.exit(1)

test_file = pet_files[0]
subject = test_file.stem.split('_')[0]
print(f"Testing with: {test_file.name}")
print(f"Subject: {subject}\n")

# Load original single-timepoint PET
pet_img = nib.load(test_file)
pet_data = pet_img.get_fdata()

print(f"Original PET single timepoint:")
print(f"  Shape: {pet_data.shape}")
print(f"  Affine:\n{pet_img.affine}")
print(f"  Data stats: min={pet_data.min():.4f}, max={pet_data.max():.4f}, mean={pet_data.mean():.4f}")
print(f"  Non-zero voxels: {np.count_nonzero(pet_data)} / {pet_data.size}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(pet_img.affine[:3, i]) for i in range(3)])}")

# Resample to isotropic
print(f"\nResampling to 1mm isotropic...")
pet_resampled = resample_to_isotropic(pet_img, target_voxel_size=1.0)
pet_resampled_data = pet_resampled.get_fdata()

print(f"Resampled PET:")
print(f"  Shape: {pet_resampled_data.shape}")
print(f"  Affine:\n{pet_resampled.affine}")
print(f"  Data stats: min={pet_resampled_data.min():.4f}, max={pet_resampled_data.max():.4f}, mean={pet_resampled_data.mean():.4f}")
print(f"  Non-zero voxels: {np.count_nonzero(pet_resampled_data)} / {pet_resampled_data.size}")
print(f"  Voxel sizes: {np.array([np.linalg.norm(pet_resampled.affine[:3, i]) for i in range(3)])}")

# Check for zeroing issue
if pet_resampled_data.max() < 0.001:
    print(f"\n⚠️  WARNING: Resampled data is essentially all zeros!")
    print(f"     This indicates the resampling is still broken.")
else:
    print(f"\n✓ Resampled data contains non-zero values.")
    print(f"   Data preservation: {(pet_resampled_data.mean() / pet_data.mean()):.2%}")

# Save test resampled file for inspection
test_output = RESAMPLED_DIR / subject / f"{subject}_pet_1mm_test.nii.gz"
test_output.parent.mkdir(parents=True, exist_ok=True)
nib.save(pet_resampled, test_output)
print(f"\nTest resampled file saved to: {test_output}")
print("You can now view this in ITK-SNAP to verify the fix worked.")
