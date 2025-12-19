#!/usr/bin/env python
"""
Visual demonstration of the anatomical correspondence fix.
Creates side-by-side comparison showing before/after alignment.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

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

NUM_SLICES = 40

# Calculate slice ranges
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

# Create visualization comparing old vs new extraction
fig, axes = plt.subplots(5, 4, figsize=(16, 15))
fig.suptitle('Anatomical Correspondence Fix Demonstration\nLeft: Old (Wrong) | Right: New (Fixed)', 
             fontsize=16, fontweight='bold')

selected_indices = [0, 5, 10, 15, 19]  # Show 5 slices

for row_idx, slice_idx in enumerate(selected_indices):
    pet_z = pet_start + slice_idx
    
    # OLD (BUGGY) EXTRACTION: Sequential T1 index
    t1_start = max(0, t1_z_center - NUM_SLICES // 2)
    t1_z_old = t1_start + slice_idx
    
    # NEW (FIXED) EXTRACTION: Affine-mapped T1 index
    pet_center_xy = np.array([pet_data.shape[0]//2, pet_data.shape[1]//2, pet_z, 1])
    pet_physical = pet_img.affine @ pet_center_xy
    t1_voxel = np.linalg.inv(t1_img.affine) @ pet_physical
    t1_z_new = int(np.round(t1_voxel[2]))
    t1_z_new = np.clip(t1_z_new, 0, t1_data.shape[2] - 1)
    
    # Get slices
    pet_slice = pet_data[..., pet_z]
    t1_old = t1_data[..., t1_z_old]
    t1_new = t1_data[..., t1_z_new]
    
    # Plot PET (same for both)
    ax = axes[row_idx, 0]
    im = ax.imshow(pet_slice, cmap='hot')
    ax.set_title(f'PET Z={pet_z}\n(Base slice {slice_idx})')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Plot OLD T1 (incorrect)
    ax = axes[row_idx, 1]
    im = ax.imshow(t1_old, cmap='gray')
    color = 'red' if t1_z_old != t1_z_new else 'green'
    ax.set_title(f'T1 Z={t1_z_old} [OLD]\n❌ Wrong', color=color, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Plot NEW T1 (correct)
    ax = axes[row_idx, 2]
    im = ax.imshow(t1_new, cmap='gray')
    ax.set_title(f'T1 Z={t1_z_new} [NEW]\n✓ Correct', color='green', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Plot difference (NEW - OLD) to show correction
    ax = axes[row_idx, 3]
    if t1_z_old != t1_z_new:
        # Show blend of old and new to visualize difference
        diff = np.abs(t1_new.astype(float) - t1_old.astype(float))
        im = ax.imshow(diff, cmap='Reds')
        ax.set_title(f'Difference\nShift: {t1_z_new - t1_z_old:+d}', 
                     color='darkred', fontweight='bold')
    else:
        ax.imshow(np.zeros_like(t1_new), cmap='gray')
        ax.set_title('No difference\n(Same slice)', color='green')
    ax.axis('off')
    if t1_z_old != t1_z_new:
        plt.colorbar(im, ax=ax)

plt.tight_layout()
output_path = Path(config.output_dir) / "ANATOMICAL_FIX_DEMONSTRATION.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved demonstration to: {output_path}")
plt.close()

# Create a summary plot showing correspondence mapping
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Old (wrong) mapping
t1_z_old_list = []
for slice_idx in range(NUM_SLICES):
    t1_start = max(0, t1_z_center - NUM_SLICES // 2)
    t1_z_old_list.append(t1_start + slice_idx)

ax1.plot(range(NUM_SLICES), t1_z_old_list, 'r-o', linewidth=2, markersize=6, label='Old (wrong)')
ax1.set_xlabel('PET Slice Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('T1 Voxel Z Index', fontsize=12, fontweight='bold')
ax1.set_title('OLD (BUGGY): Sequential Index Extraction\n❌ Misaligned', 
              fontsize=13, fontweight='bold', color='red')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: New (correct) mapping
t1_z_new_list = []
for slice_idx in range(NUM_SLICES):
    pet_z = pet_start + slice_idx
    pet_center_xy = np.array([pet_data.shape[0]//2, pet_data.shape[1]//2, pet_z, 1])
    pet_physical = pet_img.affine @ pet_center_xy
    t1_voxel = np.linalg.inv(t1_img.affine) @ pet_physical
    t1_z = int(np.round(t1_voxel[2]))
    t1_z = np.clip(t1_z, 0, t1_data.shape[2] - 1)
    t1_z_new_list.append(t1_z)

ax2.plot(range(NUM_SLICES), t1_z_new_list, 'g-o', linewidth=2, markersize=6, label='New (correct)')
ax2.set_xlabel('PET Slice Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('T1 Voxel Z Index', fontsize=12, fontweight='bold')
ax2.set_title('NEW (FIXED): Affine-Based Anatomical Mapping\n✓ Aligned', 
              fontsize=13, fontweight='bold', color='green')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Add statistics
old_std = np.std(np.diff(t1_z_old_list))
new_std = np.std(np.diff(t1_z_new_list))

fig.text(0.25, 0.02, f'Consistency (std of diffs): {old_std:.2f}', 
         ha='center', fontsize=11, color='red', fontweight='bold')
fig.text(0.75, 0.02, f'Consistency (std of diffs): {new_std:.2f}', 
         ha='center', fontsize=11, color='green', fontweight='bold')

plt.tight_layout(rect=[0, 0.04, 1, 1])
output_path2 = Path(config.output_dir) / "CORRESPONDENCE_MAPPING_COMPARISON.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved correspondence mapping to: {output_path2}")

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"\nGenerated 2 demonstration plots:")
print(f"1. ANATOMICAL_FIX_DEMONSTRATION.png")
print(f"   Shows side-by-side comparison of old vs new T1 slices paired with PET")
print(f"   Red highlights: incorrect pairing (old method)")
print(f"   Green indicates: correct pairing (fixed method)")
print(f"\n2. CORRESPONDENCE_MAPPING_COMPARISON.png")
print(f"   Shows the mapping function for all 20 slices")
print(f"   Old method: sequential (often wrong)")
print(f"   New method: affine-based (physically accurate)")
