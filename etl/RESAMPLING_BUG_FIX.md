# Resampling Bug Fix - Analysis Report

## Problem Found
The PET data was being zeroed out during the resampling step due to **incorrect affine coordinate transformation**.

## Root Cause

### Original Buggy Code (Lines 208-218)
```python
# Calculate coordinates in original space
coords_new = np.meshgrid(np.arange(new_shape[0]), np.arange(new_shape[1]), 
                         np.arange(new_shape[2]), indexing='ij')
coords_new_homogeneous = np.stack([coords_new[0], coords_new[1], coords_new[2], 
                                   np.ones_like(coords_new[0])], axis=-1)

# Transform to original space
coords_old_homogeneous = coords_new_homogeneous @ np.linalg.inv(new_affine).T @ affine.T
coords_old = coords_old_homogeneous[..., :3]
```

### Why It Failed
The transformation: `coords_old_homogeneous = coords_new_homogeneous @ np.linalg.inv(new_affine).T @ affine.T`

1. **Incorrect matrix order**: Homogeneous coordinates should be multiplied from the LEFT, not batched on the right with matrix transposes
2. **Wrong affine chain**: The chain `inv(new_affine).T @ affine.T` doesn't properly map physical space coordinates back to original voxel space
3. **Result**: Most coordinates fell outside valid bounds → `map_coordinates()` returned the `cval=0.0` padding for nearly all voxels, zeroing out the entire volume

## The Fix

### New Correct Code (Lines 176-212)
The fix uses proper homogeneous coordinate transformation:

```python
# For each voxel in the NEW resampled space, find its coordinates in the OLD space
new_indices = np.meshgrid(np.arange(new_shape[0]), np.arange(new_shape[1]), 
                          np.arange(new_shape[2]), indexing='ij')

# Convert voxel indices to homogeneous physical coordinates in new space
voxel_coords_new = np.array([new_indices[0].ravel(), 
                              new_indices[1].ravel(),
                              new_indices[2].ravel(),
                              np.ones(new_indices[0].size)])

# Convert to physical coordinates in new space
phys_coords = new_affine @ voxel_coords_new

# Convert physical coordinates back to OLD voxel space
old_voxel_coords = np.linalg.inv(affine) @ phys_coords

# Extract x, y, z coordinates (drop homogeneous coordinate)
coords_old = old_voxel_coords[:3, :]
```

### Why This Works
1. **Correct transformation chain**: 
   - New voxel indices → new affine → physical coordinates
   - Physical coordinates → inverse of old affine → old voxel indices
2. **Proper homogeneous coordinate handling**: Matrices are multiplied from the LEFT (standard affine transformation)
3. **Result**: All coordinates are correctly mapped to the original data space, preserving the data

## Verification Results

### Test on sub-01 PET Data
- **Original (1.39×1.39×2.03 mm voxels)**:
  - Shape: 344×344×127
  - Max: 40968.17
  - Mean: 523.20
  - Non-zero voxels: 1,999,840

- **Resampled (1×1×1 mm isotropic)**:
  - Shape: 478×478×258 ✓ (correctly scaled up)
  - Max: 40275.21 ✓ (data preserved, slight interpolation smoothing)
  - Mean: 524.13 ✓ (nearly identical: 100.18% preservation)
  - Non-zero voxels: 8,674,276 ✓ (properly interpolated)

### Conclusion
✓ **Fix confirmed working** - Resampled PET data now contains valid non-zero values
✓ **Data integrity preserved** - Mean values almost identical before/after
✓ **Proper scaling** - Volume expands correctly for finer 1mm isotropic resolution

## Files Modified
- `/home/rbussell/repos/self_supervised_learning/etl/etl_pipeline.py` - Lines 176-232: `resample_to_isotropic()` function

## Next Steps
You can now:
1. Delete the cached resampled data if it exists and is corrupted:
   ```bash
   rm -rf /home/rbussell/data/pet_mri/resampled/
   ```
2. Re-run the full ETL pipeline to generate correct resampled PET and T1 data
3. View the resampled PET files in ITK-SNAP - they should now display properly with visible anatomy

Test file location (for verification in ITK-SNAP):
- `/home/rbussell/data/pet_mri/resampled/sub-01/sub-01_pet_1mm_test.nii.gz`
