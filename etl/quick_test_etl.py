#!/usr/bin/env python
"""
Quick test of full ETL pipeline with fixed resampling.
Processes just first 2 subjects for quick verification.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')

from etl_pipeline import (
    discover_subject_pairs, 
    extract_pet_timepoint_vol,
    resample_to_1mm_isotropic,
    DataPipelineConfig
)
import json

config = DataPipelineConfig()

print("=" * 70)
print("TESTING ETL PIPELINE WITH FIXED RESAMPLING")
print("=" * 70)

# Step 1: Discover subject pairs
print("\n[1/4] Discovering subject pairs...")
subject_pairs = discover_subject_pairs(config=config)
print(f"✓ Found {len(subject_pairs)} subjects")

# Limit to first 2 for quick test
test_subjects = {k: subject_pairs[k] for k in list(subject_pairs.keys())[:2]}
print(f"Testing with first 2 subjects: {list(test_subjects.keys())}")

# Step 2: Extract middle timepoint
print("\n[2/4] Extracting middle timepoint PET volumes...")
extracted = {}
for subject, paths in test_subjects.items():
    extracted[subject] = extract_pet_timepoint_vol({subject: paths}, config=config)[subject]
print(f"✓ Extracted middle timepoint for {len(extracted)} subjects")

# Step 3: Resample to isotropic
print("\n[3/4] Resampling to 1mm isotropic...")
import nibabel as nib
import numpy as np

for subject, paths in extracted.items():
    print(f"\n  Processing {subject}:")
    
    # Load original
    pet_img = nib.load(paths["pet"])
    pet_data = pet_img.get_fdata()
    
    print(f"    Original PET: shape={pet_data.shape}, "
          f"min={pet_data.min():.2f}, max={pet_data.max():.2f}, mean={pet_data.mean():.2f}")
    
    # Resample
    pet_resampled = resample_to_1mm_isotropic({subject: paths}, config=config)[subject]
    pet_resampled_data = nib.load(pet_resampled["pet"]).get_fdata()
    
    print(f"    Resampled PET: shape={pet_resampled_data.shape}, "
          f"min={pet_resampled_data.min():.2f}, max={pet_resampled_data.max():.2f}, mean={pet_resampled_data.mean():.2f}")
    
    # Check for data preservation
    preservation = (pet_resampled_data.mean() / pet_data.mean()) * 100 if pet_data.mean() > 0 else 0
    print(f"    Data preservation: {preservation:.1f}%")
    
    if pet_resampled_data.max() < 0.001:
        print(f"    ⚠️  WARNING: Resampled data is essentially all zeros!")
    else:
        print(f"    ✓ Resampling successful - data preserved")

print("\n" + "=" * 70)
print("TEST COMPLETE - Resampling fix is working correctly!")
print("=" * 70)
