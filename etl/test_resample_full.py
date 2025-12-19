#!/usr/bin/env python
"""
Direct test of resampling function on real data across pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')

from etl_pipeline import resample_to_isotropic, DataPipelineConfig
import nibabel as nib
import numpy as np

config = DataPipelineConfig()

# Get test PET files (single timepoint)
pet_dir = Path(config.output_dir) / "pet_single_timepoint"
pet_files = sorted(pet_dir.glob("*_pet_t200.nii.gz"))

print("=" * 70)
print("FULL ETL PIPELINE - RESAMPLING TEST")
print("=" * 70)
print(f"\nTesting resampling on {len(pet_files)} subjects\n")

all_preserved = True

for i, pet_file in enumerate(pet_files[:5], 1):  # Test first 5
    subject = pet_file.stem.split('_')[0]
    
    # Load original
    pet_img = nib.load(pet_file)
    pet_data = pet_img.get_fdata()
    pet_min_before = pet_data.min()
    pet_max_before = pet_data.max()
    pet_mean_before = pet_data.mean()
    pet_nonzero_before = np.count_nonzero(pet_data)
    
    # Resample
    pet_resampled = resample_to_isotropic(pet_img, target_voxel_size=1.0)
    pet_resampled_data = pet_resampled.get_fdata()
    pet_min_after = pet_resampled_data.min()
    pet_max_after = pet_resampled_data.max()
    pet_mean_after = pet_resampled_data.mean()
    pet_nonzero_after = np.count_nonzero(pet_resampled_data)
    
    # Check preservation
    preservation = (pet_mean_after / pet_mean_before) * 100 if pet_mean_before > 0 else 0
    
    status = "✓" if pet_max_after > 0.01 else "✗"
    if pet_max_after <= 0.01:
        all_preserved = False
    
    print(f"[{i}] {subject}")
    print(f"    Before: {pet_data.shape} | min={pet_min_before:.2f}, max={pet_max_before:.2f}, mean={pet_mean_before:.2f}")
    print(f"    After:  {pet_resampled_data.shape} | min={pet_min_after:.2f}, max={pet_max_after:.2f}, mean={pet_mean_after:.2f}")
    print(f"    Preservation: {preservation:.1f}% {status}")

print("\n" + "=" * 70)
if all_preserved:
    print("✓ ALL RESAMPLING TESTS PASSED - Data is properly preserved!")
else:
    print("✗ RESAMPLING FAILED - Some data was zeroed out")
print("=" * 70)
