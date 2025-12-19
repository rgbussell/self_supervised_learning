# PET/MRI ETL Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA SOURCE                             │
│        /home/rbussell/data/openneuro_pet_mri/ds002898-download   │
│                  (BIDS Format: sub-01, sub-02, ...)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      SETUP_MTNET_ENVIRONMENT (Runs First)                       │
│  • Check if mtnet conda environment exists                      │
│  • If not: Run /mtnet/setup/setup_mtnet_env.bash                │
│  • Build environment with all dependencies                      │
│  Output: Status message                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           DISCOVER_SUBJECT_PAIRS                                │
│  • Depends on setup_mtnet_environment                           │
│  • Scan for subject folders (sub-XX)                            │
│  • Find T1w files in anat/ subdirectories                       │
│  • Find PET files in pet/ subdirectories                        │
│  • Pair matching T1 and PET files                               │
│  Output: Dict[subject_id -> {t1_path, pet_path}]                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│        EXTRACT_MIDDLE_PET_VOLUME                                │
│  • Load PET data (multi-timepoint 4D volumes)                   │
│  • Extract middle timepoint (3D)                                │
│  • Save to /temp_pet/                                           │
│  • Cache check: Skip if files already exist                     │
│  Output: Dict[subject_id -> {t1_path, pet_3d_path}]             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│        RESAMPLE_TO_1MM_ISOTROPIC                                │
│  • Load T1 and PET images                                       │
│  • Resample both to 1mm isotropic voxel resolution              │
│  • Apply proper affine transformations                          │
│  • Save to /resampled/                                          │
│  • Cache check: Skip if files already exist                     │
│  Output: Dict[subject_id -> {t1_1mm_path, pet_1mm_path}]        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      CREATE_TRAIN_TEST_VAL_SPLIT                                │
│  • Assign subjects to train (70%), val (15%), test (15%)        │
│  • Create split directories: /train/, /val/, /test/             │
│  • Save split_info.json with subject assignments                │
│  • Does NOT copy data yet (deferred to next step)               │
│  Output: Dict[split_name -> List[subject_ids]]                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      EXTRACT_2D_PET_SLICES_TO_SPLITS                            │
│  • For each subject in each split:                              │
│    - Load resampled T1 and PET 3D volumes                       │
│    - Find anatomical center using T1 affine matrix              │
│    - Map center to PET coordinates using inverse affine         │
│    - Extract 20 consecutive 2D slices around center             │
│    - Slices at corresponding anatomical locations               │
│  • Save both 2D slice pairs to subject folders:                 │
│    {subject}_T1w_slice_000.nii.gz, {subject}_pet_slice_000...  │
│    {subject}_T1w_slice_001.nii.gz, {subject}_pet_slice_001...  │
│    ... (20 T1/PET slice pairs total)                            │
│  Output: "2D slice extraction complete"                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      CLEANUP_TEMP_FILES                                         │
│  • Remove /temp_pet/ directory                                  │
│  • Keep /resampled/ for future reuse                            │
│  Output: "Cleanup complete"                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT DATA STRUCTURE                           │
│              /home/rbussell/data/pet_mri/                        │
│                                                                  │
│  ├── train/                                                     │
│  │   ├── sub-01/                                               │
│  │   │   ├── sub-01_T1w_slice_000.nii.gz (2D)                 │
│  │   │   ├── sub-01_pet_slice_000.nii.gz (2D)                 │
│  │   │   ├── sub-01_T1w_slice_001.nii.gz (2D)                 │
│  │   │   ├── sub-01_pet_slice_001.nii.gz (2D)                 │
│  │   │   ├── ...                                              │
│  │   │   ├── sub-01_T1w_slice_019.nii.gz (2D)                 │
│  │   │   └── sub-01_pet_slice_019.nii.gz (2D)                 │
│  │   ├── sub-02/                                               │
│  │   │   ├── sub-02_T1w_slice_000.nii.gz                      │
│  │   │   ├── sub-02_pet_slice_000.nii.gz                      │
│  │   │   └── ... (20 T1/PET slice pairs)                      │
│  │   └── ...                                                   │
│  │                                                             │
│  ├── val/                                                       │
│  │   ├── sub-XX/                                               │
│  │   │   ├── sub-XX_T1w_slice_000.nii.gz                      │
│  │   │   ├── sub-XX_pet_slice_000.nii.gz                      │
│  │   │   └── ... (20 T1/PET slice pairs)                      │
│  │   └── ...                                                   │
│  │                                                             │
│  ├── test/                                                      │
│  │   ├── sub-YY/                                               │
│  │   │   ├── sub-YY_T1w_slice_000.nii.gz                      │
│  │   │   ├── sub-YY_pet_slice_000.nii.gz                      │
│  │   │   └── ... (20 T1/PET slice pairs)                      │
│  │   └── ...                                                   │
│  │                                                             │
│  ├── resampled/                          (Cached - not deleted) │
│  │   ├── sub-01/                                               │
│  │   │   ├── sub-01_T1w_1mm.nii.gz      (Reference for coords)│
│  │   │   └── sub-01_pet_1mm.nii.gz      (Source for 2D slices)│
│  │   └── ...                                                   │
│  │                                                             │
│  └── split_info.json                     (Subject assignments) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Asset Dependencies

```
discover_subject_pairs (depends on setup_mtnet_environment)
        │
        ▼
extract_middle_pet_volume
        │
        ▼
resample_to_1mm_isotropic
        │
        ├────────────────┬──────────────────────┐
        │                │                      │
        ▼                ▼                      ▼
create_train_    extract_2d_pet_   (setup_mtnet already ran)
test_val_split   slices_to_splits
        │                │
        └─────────────────┘
                 │
                 ▼
        cleanup_temp_files
                 │
                 ▼
            DONE
```

## Key Features

- **Caching**: `extract_middle_pet_volume` and `resample_to_1mm_isotropic` check for existing output files and skip processing
- **Affine-Aware**: 2D slice positions determined by T1 anatomical center, mapped to PET using inverse affine
- **Isotropic Resampling**: Both modalities resampled to 1mm³ voxels using proper interpolation
- **2D Paired Slices**: Each training input includes both T1 and PET 2D slices at matching anatomical locations
- **20 Slice Pairs per Subject**: 20 consecutive T1/PET slice pairs extracted per subject (~8-20mm depth range)
- **BIDS Compliant**: Discovers and processes BIDS-formatted PET/MRI data automatically
- **Data Organization**: Clean train/val/test split with individual 2D slice pairs as inputs
- **Reproducible**: Uses random_state=42 for consistent splits across runs

## Execution

Run the pipeline with:
```bash
conda run -n pet_to_mri_translation python run_etl_pipeline.py
```

Or in VS Code with the launch configuration: **"Run PET/MRI ETL Pipeline"**
