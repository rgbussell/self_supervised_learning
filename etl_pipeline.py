"""
Dagster ETL pipeline for PET/MRI data processing.
Handles data discovery, pairing, middle timepoint extraction, and train/test/validation split.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

from dagster import (
    asset,
    define_asset_job,
    in_process_executor,
    AssetSelection,
    Config,
)


# Configuration
INPUT_DATA_DIR = Path("/home/rbussell/data/openneuro_pet_mri/ds002898-download")
OUTPUT_DATA_DIR = Path("/home/rbussell/data/pet_mri")
TRAIN_DIR = Path("/home/rbussell/data/pet_mri/train")
VAL_DIR = Path("/home/rbussell/data/pet_mri/validate")
TEST_DIR = Path("/home/rbussell/data/pet_mri/test")

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


class DataPipelineConfig(Config):
    """Configuration for the data pipeline."""
    input_dir: str = str(INPUT_DATA_DIR)
    output_dir: str = str(OUTPUT_DATA_DIR)
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT


def find_subject_folders(input_dir: Path) -> List[str]:
    """Find all subject folders in BIDS format (sub-XX)."""
    subjects = []
    for folder in sorted(input_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("sub-"):
            subjects.append(folder.name)
    return subjects


def find_t1_file(subject_dir: Path) -> Path:
    """Find T1w MRI file in anat subdirectory."""
    anat_dir = subject_dir / "anat"
    if not anat_dir.exists():
        raise FileNotFoundError(f"Anat directory not found: {anat_dir}")
    
    t1_files = list(anat_dir.glob("*_T1w.nii.gz"))
    if not t1_files:
        raise FileNotFoundError(f"No T1w file found in {anat_dir}")
    
    return t1_files[0]


def find_pet_file(subject_dir: Path) -> Path:
    """Find PET file in pet subdirectory."""
    pet_dir = subject_dir / "pet"
    if not pet_dir.exists():
        raise FileNotFoundError(f"Pet directory not found: {pet_dir}")
    
    pet_files = list(pet_dir.glob("*_task-rest*trc-18F*pet.nii.gz"))
    if not pet_files:
        raise FileNotFoundError(f"No PET file found in {pet_dir}")
    
    return pet_files[0]


@asset
def discover_subject_pairs(config: DataPipelineConfig) -> Dict[str, Dict[str, str]]:
    """
    Discover and pair T1 MRI and PET data for each subject.
    
    Returns:
        Dictionary mapping subject ID to file paths for T1 and PET data.
    """
    input_dir = Path(config.input_dir)
    subject_pairs = {}
    
    subjects = find_subject_folders(input_dir)
    
    for subject in subjects:
        subject_dir = input_dir / subject
        try:
            t1_path = find_t1_file(subject_dir)
            pet_path = find_pet_file(subject_dir)
            
            subject_pairs[subject] = {
                "t1": str(t1_path),
                "pet": str(pet_path),
            }
        except FileNotFoundError as e:
            print(f"Skipping {subject}: {e}")
            continue
    
    print(f"Discovered {len(subject_pairs)} subject pairs")
    return subject_pairs


@asset
def extract_middle_pet_volume(
    discover_subject_pairs: Dict[str, Dict[str, str]], 
    config: DataPipelineConfig
) -> Dict[str, Dict[str, str]]:
    """
    Extract middle timepoint from multi-timepoint PET volumes.
    Save extracted volumes to temporary directory.
    
    Returns:
        Updated dictionary with paths to extracted single-volume PET files.
    """
    temp_pet_dir = Path(config.output_dir) / "temp_pet"
    temp_pet_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_pairs = {}
    
    for subject, file_paths in discover_subject_pairs.items():
        # Check if already extracted
        extracted_pet_path = temp_pet_dir / f"{subject}_pet_middle.nii.gz"
        if extracted_pet_path.exists():
            print(f"Using cached extracted middle timepoint for {subject}")
            extracted_pairs[subject] = {
                "t1": file_paths["t1"],
                "pet": str(extracted_pet_path),
            }
            continue
        
        try:
            # Load PET data
            pet_img = nib.load(file_paths["pet"])
            pet_data = pet_img.get_fdata()
            
            # Handle multi-timepoint data: extract middle timepoint
            if len(pet_data.shape) == 4:  # 3D volume + time dimension
                middle_idx = pet_data.shape[3] // 2
                pet_middle = pet_data[..., middle_idx]
            else:  # Already 3D
                pet_middle = pet_data
            
            # Save extracted volume
            extracted_img = nib.Nifti1Image(pet_middle, pet_img.affine, pet_img.header)
            nib.save(extracted_img, extracted_pet_path)
            
            extracted_pairs[subject] = {
                "t1": file_paths["t1"],
                "pet": str(extracted_pet_path),
            }
            
            print(f"Extracted middle timepoint for {subject}: shape {pet_middle.shape}")
            
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            continue
    
    print(f"Extracted middle timepoint for {len(extracted_pairs)} subjects")
    return extracted_pairs


def resample_to_isotropic(img: nib.Nifti1Image, target_voxel_size: float = 1.0) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to isotropic voxel size.
    
    Args:
        img: Input NIfTI image
        target_voxel_size: Target voxel size in mm (default: 1 mm)
    
    Returns:
        Resampled NIfTI image
    """
    from scipy.ndimage import map_coordinates
    
    data = img.get_fdata()
    affine = img.affine
    
    # Get current voxel size
    current_voxel_size = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    
    # Calculate new shape
    new_shape = np.round(data.shape * current_voxel_size / target_voxel_size).astype(int)
    
    # Create new affine with isotropic voxels
    new_affine = affine.copy()
    new_affine[0, 0] = target_voxel_size if affine[0, 0] > 0 else -target_voxel_size
    new_affine[1, 1] = target_voxel_size if affine[1, 1] > 0 else -target_voxel_size
    new_affine[2, 2] = target_voxel_size if affine[2, 2] > 0 else -target_voxel_size
    
    # Calculate coordinates in original space
    coords_new = np.meshgrid(np.arange(new_shape[0]), np.arange(new_shape[1]), 
                             np.arange(new_shape[2]), indexing='ij')
    coords_new_homogeneous = np.stack([coords_new[0], coords_new[1], coords_new[2], 
                                       np.ones_like(coords_new[0])], axis=-1)
    
    # Transform to original space
    coords_old_homogeneous = coords_new_homogeneous @ np.linalg.inv(new_affine).T @ affine.T
    coords_old = coords_old_homogeneous[..., :3]
    
    # Interpolate using map_coordinates
    resampled_data = map_coordinates(data, [coords_old[..., i] for i in range(3)], 
                                     order=1, mode='constant', cval=0.0)
    
    # Create new resampled image
    resampled_img = nib.Nifti1Image(resampled_data, new_affine, img.header)
    
    return resampled_img


@asset
def resample_to_1mm_isotropic(
    extract_middle_pet_volume: Dict[str, Dict[str, str]],
    config: DataPipelineConfig
) -> Dict[str, Dict[str, str]]:
    """
    Resample PET and T1 MRI data to 1 mm isotropic resolution.
    
    Returns:
        Updated dictionary with paths to resampled files.
    """
    resampled_dir = Path(config.output_dir) / "resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    
    resampled_pairs = {}
    TARGET_VOXEL_SIZE = 1.0  # 1 mm isotropic
    
    for subject, file_paths in extract_middle_pet_volume.items():
        # Check if already resampled
        subject_resampled_dir = resampled_dir / subject
        t1_resampled_path = subject_resampled_dir / f"{subject}_T1w_1mm.nii.gz"
        pet_resampled_path = subject_resampled_dir / f"{subject}_pet_1mm.nii.gz"
        
        if t1_resampled_path.exists() and pet_resampled_path.exists():
            print(f"Using cached resampled data for {subject}")
            resampled_pairs[subject] = {
                "t1": str(t1_resampled_path),
                "pet": str(pet_resampled_path),
            }
            continue
        
        try:
            # Create subject resampled directory
            subject_resampled_dir.mkdir(exist_ok=True)
            
            # Resample T1
            t1_img = nib.load(file_paths["t1"])
            t1_resampled = resample_to_isotropic(t1_img, TARGET_VOXEL_SIZE)
            nib.save(t1_resampled, t1_resampled_path)
            print(f"Resampled T1 for {subject}: {t1_img.shape} -> {t1_resampled.shape}")
            
            # Resample PET
            pet_img = nib.load(file_paths["pet"])
            pet_resampled = resample_to_isotropic(pet_img, TARGET_VOXEL_SIZE)
            nib.save(pet_resampled, pet_resampled_path)
            print(f"Resampled PET for {subject}: {pet_img.shape} -> {pet_resampled.shape}")
            
            resampled_pairs[subject] = {
                "t1": str(t1_resampled_path),
                "pet": str(pet_resampled_path),
            }
            
        except Exception as e:
            print(f"Error resampling {subject}: {e}")
            continue
    
    print(f"Resampled {len(resampled_pairs)} subject pairs to 1 mm isotropic")
    return resampled_pairs


@asset
def create_train_test_val_split(
    resample_to_1mm_isotropic: Dict[str, Dict[str, str]],
    config: DataPipelineConfig
) -> Dict[str, List[str]]:
    """
    Create train/test/validation splits and organize data.
    
    Returns:
        Dictionary mapping split names to list of subject IDs.
    """
    output_dir = Path(config.output_dir)
    subjects = list(resample_to_1mm_isotropic.keys())
    
    # Create split directories
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        splits[split_name] = {"dir": split_dir, "subjects": []}
    
    # First split: train + temp (val+test)
    train_subjects, temp_subjects = train_test_split(
        subjects,
        test_size=(1 - config.train_split),
        random_state=42,
    )
    
    # Second split: val and test from temp
    val_size = config.val_split / (config.val_split + config.test_split)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=(1 - val_size),
        random_state=42,
    )
    
    split_assignments = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    
    # Copy data to split directories
    for split_name, split_subjects in split_assignments.items():
        split_dir = splits[split_name]["dir"]
        splits[split_name]["subjects"] = split_subjects
        
        for subject in split_subjects:
            file_paths = resample_to_1mm_isotropic[subject]
            
            # Create subject subdirectory
            subject_split_dir = split_dir / subject
            subject_split_dir.mkdir(exist_ok=True)
            
            # Copy T1 file
            t1_src = Path(file_paths["t1"])
            t1_dst = subject_split_dir / f"{subject}_T1w.nii.gz"
            shutil.copy2(t1_src, t1_dst)
            
            # Copy extracted PET file
            pet_src = Path(file_paths["pet"])
            pet_dst = subject_split_dir / f"{subject}_pet.nii.gz"
            shutil.copy2(pet_src, pet_dst)
            
            print(f"Copied {subject} to {split_name} split")
    
    # Save split assignments to JSON
    split_info = {
        split_name: {
            "subjects": split_subjects,
            "count": len(split_subjects),
        }
        for split_name, split_subjects in split_assignments.items()
    }
    
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_subjects)} subjects")
    print(f"  Val:   {len(val_subjects)} subjects")
    print(f"  Test:  {len(test_subjects)} subjects")
    print(f"  Total: {len(subjects)} subjects")
    
    return split_assignments


@asset
def setup_mtnet_environment(config: DataPipelineConfig) -> str:
    """
    Check if mtnet conda environment exists, and build it if not.
    """
    import subprocess
    import sys
    
    # Check if mtnet env exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if "mtnet" in result.stdout:
            print("mtnet environment already exists.")
            return "mtnet environment already exists"
    except subprocess.CalledProcessError:
        print("mtnet environment not found. Building it...")
    
    # Build the environment
    mtnet_setup_script = Path("/home/rbussell/repos/mtnet/setup/setup_mtnet_env.bash")
    if not mtnet_setup_script.exists():
        raise FileNotFoundError(f"Setup script not found: {mtnet_setup_script}")
    
    try:
        result = subprocess.run(
            ["/bin/bash", str(mtnet_setup_script)],
            capture_output=True,
            text=True,
            check=True,
            cwd=mtnet_setup_script.parent
        )
        print("mtnet environment built successfully.")
        print(result.stdout)
        return "mtnet environment built successfully"
    except subprocess.CalledProcessError as e:
        print(f"Failed to build mtnet environment: {e}")
        print(e.stderr)
        raise


@asset
def cleanup_temp_files(
    create_train_test_val_split: Dict[str, List[str]],
    config: DataPipelineConfig
) -> str:
    """Clean up temporary PET files."""
    temp_pet_dir = Path(config.output_dir) / "temp_pet"
    
    if temp_pet_dir.exists():
        shutil.rmtree(temp_pet_dir)
        print(f"Cleaned up temporary directory: {temp_pet_dir}")
    
    return "Cleanup complete"


# Define the job
etl_job = define_asset_job(
    name="pet_mri_etl_job",
    selection=AssetSelection.all(),
    executor_def=in_process_executor,
)
