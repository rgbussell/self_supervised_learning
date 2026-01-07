## Objective: 
Develop an image-to-image translation method for PET to T1
(positron emission tomography to T1 magnetic resonance iamges)

## Constraints
T1 should be of sufficient image quality for accurate gray/white/ventricle segmentation using standard procesing protocols.<br>
Use only open source, publicly available data sources<br>
Trainable in < 48 hours single mid-grade GPU (specs below)<br>
Inference including preprocessing pipeline < 60 seconds<br>

![alt text](assets/image-3.png)

## Project plan
Source data from public challenge data sets ISLES, BRaTS, etc.<br>
Set up ETL pipeline using pipeline orchestrator (dagster)<br>
Use self-supervision paradigm to learn a multi-modal latent space representation<br>
Minimize dependence on difficult to train architectures<br>

## Planned Architecture
Three networks are planned

1. Vision transformer as encoder network
Stores latent multi-model representation
Patch-wise reconstruction loss with masked auto-encoder setup

2. Decoder for ViT in the pre-training stage (patch-wise reconstruction)

3. SwinUNET fine-tuned on T1|PET pairs for the primary image-to-image translation task

## Progress
Pre-trained transformer results

Example validation data set from the pre-training (<24 hours training)
![alt text](assets/image-1.png)

Example Fine-tuned on T1 to FLAIR data task (using pretrained ViT encoder)
Training time < 12 hours, 16 GB 
left-to-right: T1 image (input), FLAIR ground truth, network prediction<br>
![alt text](assets/image-2.png)

## PET image sourcing
Openneuro Monash data set available with PET and MRI data in same sessions
Preprocessing pipeline required to set up fine-tuning

Here is one slice from the Monash resting state fMRI-PET data set from open neuro.
With some preprocessing and registration to the corresponding MRI this could
serve as a fine-tuning data set for the PET->T1 task

![alt text](assets/image.png)

## Image preparation fmr PET-T1 image-to-image translation
cd ./self_supervised_learning/etl
python run_etl_pipeline.py

## PET-T1 image-to-image translation initial results
* 20 paired slices selected from one of the PET timepoints
* 150 epochs of fine-tuning
* pre-training on the mae task with multi-modal MRI (no PET in training)
* nuumber of subjects: Train: 9, Val: 2, Test: 2 

Result of PET to T1 image-to-image translation after 150 epochs
![alt text](assets/image_pet_t1_i2i_initial_finetuning_150.png)


## Additional Data for Fine Tuning PET-T1
Increase the number of paired slices in finetuning
Distribution:
Train: 9 subjects (360 slices)
Val: 2 subjects (80 slices)
Test: 2 subjects (80 slices)

![alt text](assets/pet_t1_i2i_moreslices.png)


## Inference on Test Data

### Running PET-to-T1 Inference

After training is complete, use the standalone inference script to apply the trained model to test data:

```bash
# Activate the mtnet conda environment
eval "$(conda shell.bash hook)" && conda activate mtnet

# Run inference on all test samples (default)
python scripts/pet_to_t1_inference.py

# Or run on a specific number of samples
python scripts/pet_to_t1_inference.py --num_samples 5
```

### Command-Line Options

```bash
python scripts/pet_to_t1_inference.py \
    --test_data_dir /home/rbussell/data/pet_mri/test/ \
    --encoder_ckpt /home/rbussell/repos/mtnet/weight/pet_mri_finetune/E.pth \
    --generator_ckpt /home/rbussell/repos/mtnet/weight/pet_mri_finetune/G.pth \
    --output_dir ./outputs/pet_t1_inference/ \
    --num_samples 80
```

**Options:**
- `--test_data_dir`: Directory containing test .npy files (default: /home/rbussell/data/pet_mri/test/)
- `--encoder_ckpt`: Path to encoder checkpoint E.pth (default: /home/rbussell/repos/mtnet/weight/pet_mri_finetune/E.pth)
- `--generator_ckpt`: Path to generator checkpoint G.pth (default: /home/rbussell/repos/mtnet/weight/pet_mri_finetune/G.pth)
- `--output_dir`: Base output directory (default: ./outputs/pet_t1_inference/)
- `--num_samples`: Number of samples to process (default: all samples in test directory)

### Inference Outputs

The script generates timestamped output directories with the following structure:

```
outputs/pet_t1_inference/run_YYYYMMDD_HHMMSS/
├── visualizations/           # PNG comparison images
│   ├── 000441.png           # Input PET | Predicted T1 | Ground Truth T1
│   ├── 000442.png
│   └── ...
├── metrics.csv              # Quantitative evaluation metrics
└── config.json              # Run configuration for reproducibility
```

**Visual Outputs:**
- 3-panel comparison images showing:
  - Left: Input PET image
  - Center: Predicted T1 image (with PSNR and SSIM metrics overlaid)
  - Right: Ground truth T1 image

**Metrics CSV:**
- Per-sample metrics: PSNR, SSIM, NMSE
- Summary statistics: mean and standard deviation across all samples

### Test Set Performance

**Latest results on 80 test samples (2 subjects, 40 slices each):**
- **PSNR**: 15.29 ± 0.54 dB (Peak Signal-to-Noise Ratio)
- **SSIM**: 0.56 ± 0.04 (Structural Similarity Index)
- **NMSE**: 0.33 ± 0.06 (Normalized Mean Squared Error)
- **Inference Time**: ~24 seconds for 80 slices (~3.3 samples/second on RTX 5070 Ti)
- **Memory**: Processes one sample at a time to minimize GPU memory usage

### Implementation Details

The inference pipeline:
1. Loads the trained MAE encoder (E.pth) and MTNet generator (G.pth)
2. Processes test .npy files containing paired (T1, PET) data
3. Extracts features using the encoder
4. Generates T1 predictions using the generator
5. Calculates quality metrics (PSNR, SSIM, NMSE)
6. Saves visual comparisons and quantitative results

**Code Location:**
- Main script: `scripts/pet_to_t1_inference.py`
- Metrics: `ssl_utils/metrics.py`
- Visualization: `ssl_utils/visualization.py`

## 3D Volume Reconstruction for Atlas Registration

After generating 2D slice predictions, reconstruct 3D T1 volumes suitable for registration to standard atlases (e.g., MNI152).

### Overview

The reconstruction pipeline transforms independent 2D predictions into smooth, spatially-consistent 3D volumes by:
- Stacking slices with proper spatial metadata
- Normalizing intensity across slices
- Removing inter-slice discontinuities
- Creating NIfTI files with correct affine matrices

![3D Volume Reconstruction](assets/3d_volume_reconstruction.png?v=2)

### Running 3D Reconstruction

```bash
# Step 1: Run inference with predictions saved
eval "$(conda shell.bash hook)" && conda activate mtnet
python scripts/pet_to_t1_inference.py --save_predictions

# Step 2: Reconstruct 3D volume for each subject
python scripts/reconstruct_and_register.py \
    --predictions_dir ./outputs/pet_t1_inference/run_YYYYMMDD_HHMMSS/predictions/ \
    --subject_id sub-09 \
    --reference_t1 /home/rbussell/data/pet_mri/resampled/sub-09/sub-09_T1w_1mm.nii.gz \
    --save_intermediates

# Repeat for sub-10
python scripts/reconstruct_and_register.py \
    --predictions_dir ./outputs/pet_t1_inference/run_YYYYMMDD_HHMMSS/predictions/ \
    --subject_id sub-10 \
    --reference_t1 /home/rbussell/data/pet_mri/resampled/sub-10/sub-10_T1w_1mm.nii.gz \
    --save_intermediates
```

### Command-Line Options

```bash
python scripts/reconstruct_and_register.py \
    --predictions_dir <path_to_predictions> \
    --subject_id sub-09 \
    --reference_t1 <path_to_reference_T1> \
    --apply_n4 \
    --smooth_sigma_z 0.5 \
    --smooth_sigma_xy 0.3 \
    --save_intermediates \
    --output_dir ./outputs/reconstructed_volumes/
```

**Options:**
- `--predictions_dir`: Directory containing prediction .npy files (required)
- `--subject_id`: Subject identifier: `sub-09` or `sub-10` (required)
- `--reference_t1`: Path to reference T1 NIfTI for spatial metadata (required)
- `--output_dir`: Base output directory (default: `./outputs/reconstructed_volumes/`)
- `--apply_n4`: Apply N4 bias field correction (slower but better quality)
- `--smooth_sigma_z`: Smoothing sigma along Z-axis (default: 0.5)
- `--smooth_sigma_xy`: Smoothing sigma in XY plane (default: 0.3)
- `--save_intermediates`: Save intermediate processing steps

### Reconstruction Pipeline

The 7-stage pipeline ensures smooth, registration-ready volumes:

**Stage 1: Stack Slices**
- Assembles 40 consecutive 2D slices into 3D volume
- Each subject: 40 slices × 256 × 256 pixels

**Stage 2: Remove Outliers**
- Median filter along Z-axis (kernel size: 3)
- Removes extreme outlier slices

**Stage 3: Histogram Matching**
- Normalizes intensity distribution across slices
- Uses middle slice as reference
- Eliminates inter-slice intensity jumps

**Stage 4: N4 Bias Correction (Optional)**
- Corrects intensity non-uniformity
- Uses SimpleITK N4BiasFieldCorrectionImageFilter
- Recommended for better registration quality

**Stage 5: Anisotropic Smoothing**
- Gaussian smoothing: σ_z=0.5 (inter-slice), σ_xy=0.3 (in-plane)
- More smoothing along Z to reduce discontinuities
- Less in-plane to preserve anatomical detail

**Stage 6: Global Normalization**
- Clips to 0.2-99.8 percentile range
- Maps to [0, 1] intensity range

**Stage 7: NIfTI Creation**
- Recovers spatial metadata from reference T1
- Adjusts affine matrix for slice subset
- Creates registration-ready NIfTI file

### Reconstruction Outputs

Timestamped output directories contain:

```
outputs/reconstructed_volumes/sub-09_reconstructed_YYYYMMDD_HHMMSS/
├── sub-09_reconstructed.nii.gz              # Final 3D volume
├── reconstruction_config.json               # Run configuration
├── visualizations/
│   ├── 01_intensity_profile_stacked.png    # After stacking
│   ├── 02_intensity_profile_histogram_matched.png
│   ├── 04_intensity_profile_smoothed.png   # After smoothing
│   ├── 05_intensity_profile_final.png      # Final result
│   └── 06_orthogonal_views.png             # Axial/sagittal/coronal
└── intermediates/                           # Intermediate .npy files
    ├── 01_stacked.npy
    ├── 02_outliers_removed.npy
    ├── 03_histogram_matched.npy
    ├── 04_n4_corrected.npy
    ├── 05_smoothed.npy
    └── 06_normalized.npy
```

### Quality Metrics

**Intensity Continuity (Inter-Slice Discontinuity):**
- **sub-09**: 0.0067 → 0.0000 (after processing)
- **sub-10**: 0.0030 → 0.0000 (after processing)

**Output Volumes:**
- Shape: (40, 256, 256) per subject
- Spacing: 1mm isotropic (inherited from reference T1)
- Intensity range: [0, 1] normalized
- Format: NIfTI (.nii.gz) with proper affine matrix

### Registration to Atlas

The reconstructed volumes are ready for registration to standard atlases:

**Using ANTs:**
```bash
# Affine registration
antsRegistrationSyN.sh -d 3 \
    -f /path/to/MNI152_T1_1mm_brain.nii.gz \
    -m sub-09_reconstructed.nii.gz \
    -o sub-09_to_MNI_ \
    -t a

# Non-linear registration (SyN)
antsRegistrationSyN.sh -d 3 \
    -f /path/to/MNI152_T1_1mm_brain.nii.gz \
    -m sub-09_reconstructed.nii.gz \
    -o sub-09_to_MNI_ \
    -t s
```

**Using FSL:**
```bash
# Affine registration (FLIRT)
flirt -in sub-09_reconstructed.nii.gz \
      -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz \
      -out sub-09_to_MNI_affine.nii.gz \
      -omat affine_transform.mat \
      -dof 12

# Non-linear registration (FNIRT)
fnirt --in=sub-09_reconstructed.nii.gz \
      --ref=${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz \
      --aff=affine_transform.mat \
      --cout=sub-09_to_MNI_warp \
      --iout=sub-09_to_MNI_nonlinear.nii.gz
```

### Implementation Details

**Code Location:**
- Main script: `scripts/reconstruct_and_register.py`
- Volume processing utilities: `ssl_utils/volume_processing.py`
- 3D visualization: `scripts/visualize_3d_volume.py`

**Key Functions:**
- `stack_slices_to_3d()`: Assembles 2D slices into 3D volume
- `normalize_intensity_histogram_matching()`: Inter-slice intensity normalization
- `apply_n4_bias_correction()`: N4 bias field correction
- `smooth_volume_anisotropic()`: Anisotropic Gaussian smoothing
- `create_nifti_from_reference()`: NIfTI creation with proper affine matrix
- `full_reconstruction_pipeline()`: End-to-end reconstruction

**Dependencies:**
- nibabel: NIfTI file I/O
- SimpleITK: N4 bias correction
- scipy: Gaussian and median filtering
- scikit-image: Histogram matching
- matplotlib: Visualization

## Citation
we thank these sources for code and ideas
Tao 2023: arXiv:2212.01108v3
Liu 2023: arXiv:2311.04049v1