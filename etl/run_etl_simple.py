#!/usr/bin/env python
"""
Simple ETL pipeline runner (sequential, no Dagster overhead).
"""

import sys
from pathlib import Path
sys.path.insert(0, '/home/rbussell/repos/self_supervised_learning/etl')

from etl_pipeline import (
    find_subject_folders,
    find_t1_file, 
    find_pet_file,
    resample_to_isotropic,
    DataPipelineConfig,
    INPUT_DATA_DIR, OUTPUT_DATA_DIR,
)
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from pathlib import Path

config = DataPipelineConfig()

print("=" * 70)
print("PET/MRI ETL PIPELINE - FULL RUN")
print("=" * 70)

# Step 1: Discover and pair subjects
print("\n[STEP 1] Discovering subject pairs...")
input_dir = Path(config.input_dir)
subjects = find_subject_folders(input_dir)
print(f"Found {len(subjects)} subjects")

subject_pairs = {}
for subject in subjects:
    try:
        subject_dir = input_dir / subject
        t1_path = find_t1_file(subject_dir)
        pet_path = find_pet_file(subject_dir)
        subject_pairs[subject] = {"t1": str(t1_path), "pet": str(pet_path)}
    except FileNotFoundError as e:
        print(f"  Skipping {subject}: {e}")

print(f"✓ Paired {len(subject_pairs)} subjects")

# Step 2: Extract middle timepoint PET
print("\n[STEP 2] Extracting middle timepoint PET volumes...")
PET_TIMEPOINT_INDEX = 200
pet_single_tp_dir = OUTPUT_DATA_DIR / "pet_single_timepoint"
pet_single_tp_dir.mkdir(parents=True, exist_ok=True)

for subject, paths in subject_pairs.items():
    try:
        extracted_pet_path = pet_single_tp_dir / f"{subject}_pet_t{PET_TIMEPOINT_INDEX}.nii.gz"
        if extracted_pet_path.exists():
            print(f"  {subject}: using cached")
            continue
        
        pet_img = nib.load(paths["pet"])
        vol = pet_img.slicer[:, :, :, PET_TIMEPOINT_INDEX].get_fdata()
        extracted_img = nib.Nifti1Image(vol, pet_img.affine, pet_img.header)
        nib.save(extracted_img, extracted_pet_path)
        print(f"  {subject}: extracted")
    except Exception as e:
        print(f"  {subject}: ERROR - {e}")

print("✓ Extracted single timepoints")

# Step 3: Resample to 1mm isotropic
print("\n[STEP 3] Resampling to 1mm isotropic resolution...")
resampled_dir = OUTPUT_DATA_DIR / "resampled"
resampled_dir.mkdir(parents=True, exist_ok=True)

resampled_pairs = {}
for subject in subject_pairs.keys():
    try:
        subject_resampled_dir = resampled_dir / subject
        t1_resampled_path = subject_resampled_dir / f"{subject}_T1w_1mm.nii.gz"
        pet_resampled_path = subject_resampled_dir / f"{subject}_pet_1mm.nii.gz"
        
        if t1_resampled_path.exists() and pet_resampled_path.exists():
            print(f"  {subject}: using cached")
            resampled_pairs[subject] = {
                "t1": str(t1_resampled_path),
                "pet": str(pet_resampled_path),
            }
            continue
        
        subject_resampled_dir.mkdir(exist_ok=True)
        
        # Resample T1
        t1_img = nib.load(subject_pairs[subject]["t1"])
        t1_resampled = resample_to_isotropic(t1_img, 1.0)
        nib.save(t1_resampled, t1_resampled_path)
        
        # Resample PET
        pet_img = nib.load(pet_single_tp_dir / f"{subject}_pet_t{PET_TIMEPOINT_INDEX}.nii.gz")
        pet_resampled = resample_to_isotropic(pet_img, 1.0)
        nib.save(pet_resampled, pet_resampled_path)
        
        resampled_pairs[subject] = {
            "t1": str(t1_resampled_path),
            "pet": str(pet_resampled_path),
        }
        print(f"  {subject}: resampled")
    except Exception as e:
        print(f"  {subject}: ERROR - {e}")

print(f"✓ Resampled {len(resampled_pairs)} subject pairs")

# Step 4: Create train/val/test split
print("\n[STEP 4] Creating train/val/test split...")
subjects_list = list(resampled_pairs.keys())
train_subjects, temp_subjects = train_test_split(
    subjects_list, test_size=0.3, random_state=42
)
val_subjects, test_subjects = train_test_split(
    temp_subjects, test_size=0.5, random_state=42
)

split_assignments = {
    "train": train_subjects,
    "val": val_subjects,
    "test": test_subjects,
}

split_info = {
    split_name: {"subjects": split_subjects, "count": len(split_subjects)}
    for split_name, split_subjects in split_assignments.items()
}

with open(OUTPUT_DATA_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)

print(f"  Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
print("✓ Created split assignments")

# Step 5: Extract 2D slices
print("\n[STEP 5] Extracting 2D slices and creating combined .npy files...")
NUM_SLICES = 20
TARGET_SIZE = 256
viz_dir = OUTPUT_DATA_DIR / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

def crop_or_pad_to_target(img_2d, target_size=256):
    """Crop or pad 2D image to target size."""
    h, w = img_2d.shape
    center_h, center_w = h // 2, w // 2
    half_h, half_w = target_size // 2, target_size // 2
    
    start_h = max(0, center_h - half_h)
    end_h = min(h, start_h + target_size)
    start_w = max(0, center_w - half_w)
    end_w = min(w, start_w + target_size)
    
    cropped = img_2d[start_h:end_h, start_w:end_w]
    
    pad_h_top = max(0, half_h - (center_h - start_h))
    pad_h_bottom = target_size - cropped.shape[0] - pad_h_top
    pad_w_left = max(0, half_w - (center_w - start_w))
    pad_w_right = target_size - cropped.shape[1] - pad_w_left
    
    padded = np.pad(cropped, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                   mode='constant', constant_values=0)
    return padded

total_slices = 0

for split_name, split_subjects in split_assignments.items():
    split_dir = OUTPUT_DATA_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for subject in split_subjects:
        try:
            t1_img = nib.load(resampled_pairs[subject]["t1"])
            pet_img = nib.load(resampled_pairs[subject]["pet"])
            
            t1_data = t1_img.get_fdata()
            pet_data = pet_img.get_fdata()
            
            # Get anatomically corresponding slices
            t1_shape = t1_data.shape
            t1_center_voxel = np.array([t1_shape[0]/2, t1_shape[1]/2, t1_shape[2]/2, 1])
            t1_center_physical = t1_img.affine @ t1_center_voxel
            
            pet_center_voxel = np.linalg.inv(pet_img.affine) @ t1_center_physical
            pet_z_center = int(np.round(pet_center_voxel[2]))
            t1_z_center = int(np.round(t1_center_physical[2]))
            
            pet_z_center = np.clip(pet_z_center, 0, pet_data.shape[2] - 1)
            t1_z_center = np.clip(t1_z_center, 0, t1_data.shape[2] - 1)
            
            pet_start = max(0, pet_z_center - NUM_SLICES // 2)
            pet_end = min(pet_data.shape[2], pet_start + NUM_SLICES)
            pet_start = max(0, pet_end - NUM_SLICES)
            
            t1_start = max(0, t1_z_center - NUM_SLICES // 2)
            t1_end = min(t1_data.shape[2], t1_start + NUM_SLICES)
            t1_start = max(0, t1_end - NUM_SLICES)
            
            # Extract and save slices
            for slice_idx, (pet_z, t1_z) in enumerate(zip(range(pet_start, pet_end), 
                                                           range(t1_start, t1_end))):
                pet_2d = pet_data[..., pet_z]
                t1_2d = t1_data[..., t1_z]
                
                pet_2d_resized = crop_or_pad_to_target(pet_2d, TARGET_SIZE)
                t1_2d_resized = crop_or_pad_to_target(t1_2d, TARGET_SIZE)
                
                # Normalize using 0.2-99.8 percentiles
                pet_p002, pet_p998 = np.percentile(pet_2d_resized, 0.2), np.percentile(pet_2d_resized, 99.8)
                pet_2d_norm = (pet_2d_resized - pet_p002) / (pet_p998 - pet_p002 + 1e-8)
                pet_2d_norm = np.clip(pet_2d_norm, 0, 1)
                
                t1_p002, t1_p998 = np.percentile(t1_2d_resized, 0.2), np.percentile(t1_2d_resized, 99.8)
                t1_2d_norm = (t1_2d_resized - t1_p002) / (t1_p998 - t1_p002 + 1e-8)
                t1_2d_norm = np.clip(t1_2d_norm, 0, 1)
                
                # Save visualization
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                fig.suptitle(f"{subject} - Slice {slice_idx}", fontsize=14)
                
                axes[0, 0].imshow(pet_2d_resized, cmap='hot')
                axes[0, 0].set_title(f"PET Before")
                axes[0, 1].imshow(pet_2d_norm, cmap='hot', vmin=0, vmax=1)
                axes[0, 1].set_title(f"PET After")
                axes[1, 0].imshow(t1_2d_resized, cmap='gray')
                axes[1, 0].set_title(f"T1 Before")
                axes[1, 1].imshow(t1_2d_norm, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title(f"T1 After")
                
                plt.savefig(viz_dir / f"{subject}_slice_{slice_idx:03d}.png", dpi=100, bbox_inches='tight')
                plt.close()
                
                # Save combined .npy
                combined = np.array([t1_2d_norm, pet_2d_norm], dtype=np.float32)
                total_slices += 1
                filename = f"{total_slices:06d}.npy"
                np.save(split_dir / filename, combined)
            
            print(f"  {subject} ({split_name}): extracted {NUM_SLICES} slices")
        except Exception as e:
            print(f"  {subject}: ERROR - {e}")

print(f"✓ Extracted and saved {total_slices} total 2D slices")

print("\n" + "=" * 70)
print("✓ ETL PIPELINE COMPLETE!")
print("=" * 70)
print(f"Output directory: {OUTPUT_DATA_DIR}")
print(f"Visualizations: {viz_dir}")
