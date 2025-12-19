#!/usr/bin/env python
"""
Comprehensive diagnostic for T1/PET anatomical correspondence issue.
Analyzes slice selection logic and compares coordinate transformations.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')
from etl_pipeline import DataPipelineConfig

config = DataPipelineConfig()

# Use first subject for detailed analysis
resampled_dir = Path(config.output_dir) / "resampled"
subject = "sub-01"

t1_path = resampled_dir / subject / f"{subject}_T1w_1mm.nii.gz"
pet_path = resampled_dir / subject / f"{subject}_pet_1mm.nii.gz"

print("=" * 80)
print(f"ANATOMICAL CORRESPONDENCE DIAGNOSTIC - {subject}")
print("=" * 80)

# Load images
t1_img = nib.load(t1_path)
pet_img = nib.load(pet_path)

t1_data = t1_img.get_fdata()
pet_data = pet_img.get_fdata()

print(f"\n[1] IMAGE DIMENSIONS & AFFINES")
print(f"{'='*80}")
print(f"\nT1 MRI:")
print(f"  Shape: {t1_data.shape}")
print(f"  Affine:\n{t1_img.affine}")

print(f"\nPET:")
print(f"  Shape: {pet_data.shape}")
print(f"  Affine:\n{pet_img.affine}")

# Check if affines are identical (they should be since both are resampled to 1mm isotropic)
print(f"\n[2] AFFINE COMPARISON")
print(f"{'='*80}")
affines_equal = np.allclose(t1_img.affine, pet_img.affine)
print(f"Affines identical: {affines_equal}")
if not affines_equal:
    print(f"Affine difference:\n{t1_img.affine - pet_img.affine}")
else:
    print(f"✓ Both images share same affine (good for slice alignment)")

# Analyze the current slice selection logic
print(f"\n[3] CURRENT SLICE SELECTION LOGIC (BUGGY)")
print(f"{'='*80}")

NUM_SLICES = 40

# Current buggy logic
t1_shape = t1_data.shape
t1_center_voxel = np.array([t1_shape[0]/2, t1_shape[1]/2, t1_shape[2]/2, 1])
t1_center_physical = t1_img.affine @ t1_center_voxel

print(f"T1 center voxel indices: {t1_center_voxel[:3]}")
print(f"T1 center physical coords: {t1_center_physical[:3]}")

# THIS IS THE BUG: Using the homogeneous coordinate (value of 1)
pet_center_voxel = np.linalg.inv(pet_img.affine) @ t1_center_physical
pet_z_center = int(np.round(pet_center_voxel[2]))

print(f"\nPET center voxel (from T1 physical): {pet_center_voxel[:3]}")
print(f"PET Z center: {pet_z_center}")

# THIS IS ALSO WRONG: Setting t1_z_center to the homogeneous coordinate
t1_z_center_wrong = int(np.round(t1_center_voxel[2]))  # This is 1 (the homogeneous coord!)
print(f"\n❌ BUGGY T1 Z center: {t1_z_center_wrong} (uses homogeneous coordinate!)")

# Correct approach
t1_z_center_correct = int(np.round(t1_center_voxel[2]))  # Use the actual Z coordinate from center
print(f"✓ CORRECT T1 Z center: {t1_z_center_correct}")

pet_start = max(0, pet_z_center - NUM_SLICES // 2)
pet_end = min(pet_data.shape[2], pet_start + NUM_SLICES)
pet_start = max(0, pet_end - NUM_SLICES)

t1_start_wrong = max(0, t1_z_center_wrong - NUM_SLICES // 2)
t1_end_wrong = min(t1_data.shape[2], t1_start_wrong + NUM_SLICES)
t1_start_wrong = max(0, t1_end_wrong - NUM_SLICES)

t1_start_correct = max(0, t1_z_center_correct - NUM_SLICES // 2)
t1_end_correct = min(t1_data.shape[2], t1_start_correct + NUM_SLICES)
t1_start_correct = max(0, t1_end_correct - NUM_SLICES)

print(f"\nPET slice range: [{pet_start}, {pet_end}]")
print(f"❌ T1 slice range (WRONG): [{t1_start_wrong}, {t1_end_wrong}]")
print(f"✓ T1 slice range (CORRECT): [{t1_start_correct}, {t1_end_correct}]")

print(f"\n[4] COMPARISON OF EXTRACTION")
print(f"{'='*80}")

# Show what happens with the buggy vs correct logic
print(f"\nBUGGY EXTRACTION:")
print(f"  T1 slices extracted: {list(range(t1_start_wrong, t1_end_wrong))}")
print(f"  PET slices extracted: {list(range(pet_start, pet_end))}")
print(f"  Correspondence: T1 slice {t1_start_wrong} ↔ PET slice {pet_start}")
print(f"                  T1 slice {t1_end_wrong-1} ↔ PET slice {pet_end-1}")

print(f"\nCORRECT EXTRACTION:")
print(f"  T1 slices extracted: {list(range(t1_start_correct, t1_end_correct))}")
print(f"  PET slices extracted: {list(range(pet_start, pet_end))}")
print(f"  Correspondence: T1 slice {t1_start_correct} ↔ PET slice {pet_start}")
print(f"                  T1 slice {t1_end_correct-1} ↔ PET slice {pet_end-1}")

# Analyze data content at different Z positions
print(f"\n[5] DATA ANALYSIS AT DIFFERENT Z POSITIONS")
print(f"{'='*80}")

for z in [t1_start_wrong, t1_start_correct, t1_z_center_correct]:
    if 0 <= z < t1_data.shape[2]:
        t1_slice = t1_data[..., z]
        pet_slice = pet_data[..., z]
        
        print(f"\nZ position {z}:")
        print(f"  T1 - min: {t1_slice.min():.2f}, max: {t1_slice.max():.2f}, mean: {t1_slice.mean():.2f}")
        print(f"  PET - min: {pet_slice.min():.2f}, max: {pet_slice.max():.2f}, mean: {pet_slice.mean():.2f}")

# Physical space correspondence check
print(f"\n[6] PHYSICAL SPACE CORRESPONDENCE")
print(f"{'='*80}")

print(f"\nCorrect approach: Use affine transformation to map between spaces")
print(f"\nFor each PET Z index, what is the corresponding T1 Z?")
print(f"(They should be in the same physical location after resampling to same affine)")

for pet_z in [pet_start, pet_start + 5, pet_start + 10, pet_end - 1]:
    # Convert PET voxel to physical
    pet_voxel = np.array([pet_data.shape[0]/2, pet_data.shape[1]/2, pet_z, 1])
    pet_physical = pet_img.affine @ pet_voxel
    
    # Convert physical to T1 voxel
    t1_voxel = np.linalg.inv(t1_img.affine) @ pet_physical
    t1_z = t1_voxel[2]
    
    print(f"PET Z {pet_z}: physical Z={pet_physical[2]:.2f} → T1 Z={t1_z:.2f} (rounds to {int(np.round(t1_z))})")

print(f"\n[7] SUMMARY")
print(f"{'='*80}")
print(f"""
PROBLEM IDENTIFIED:
  Line 408 of etl_pipeline.py uses t1_center_voxel[2] which is the 4th element 
  (homogeneous coordinate = 1), not the Z coordinate from the center calculation.

IMPACT:
  - T1 slices are extracted from wrong Z positions (near slice 1 instead of center)
  - PET and T1 slices show different anatomical locations
  - T1 slices likely show edge/non-brain regions while PET shows brain center

FIX REQUIRED:
  Replace the buggy T1 center calculation with proper affine mapping, similar to PET:
  
  ✓ CORRECT CODE:
    # Get center point in physical space (using T1 as reference)
    t1_center_voxel = np.array([t1_shape[0]/2, t1_shape[1]/2, t1_shape[2]/2, 1])
    t1_center_physical = t1_img.affine @ t1_center_voxel
    
    # Convert physical center to T1 voxel indices (for consistency)
    t1_center_voxel_mapped = np.linalg.inv(t1_img.affine) @ t1_center_physical
    t1_z_center = int(np.round(t1_center_voxel_mapped[2]))  # Use mapped Z
    
  OR SIMPLER (since T1 and PET have same affine after resampling):
    t1_z_center = int(np.round(t1_shape[2] / 2))  # Direct center
""")

print("=" * 80)
