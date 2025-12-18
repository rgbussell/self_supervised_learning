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
from pet_extraction import extract_vol_from_multitmp_nifti

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

PET_TIMEPOINT_INDEX = 200

class DataPipelineConfig(Config):
    """Configuration for the data pipeline."""
    input_dir: str = str(INPUT_DATA_DIR)
    output_dir: str = str(OUTPUT_DATA_DIR)
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT
    pet_tp: int = PET_TIMEPOINT_INDEX


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
def extract_single_pet_volume(
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
        try:
            # Load PET data
            pet_img = nib.load(file_paths["pet"])

            # Extract the specified plane (default is the first plane along z-axis)
            extracted_pet_path = temp_pet_dir / f"{subject}_pet.nii.gz"

            vol = pet_img.slicer[:, :, :, PET_TIMEPOINT_INDEX].get_fdata()
            extracted_img = nib.Nifti1Image(vol, pet_img.affine)

            # Save extracted volume
            nib.save(extracted_img, extracted_pet_path)
            print(f"Extracted plane saved to: {extracted_pet_path}")
            
            extracted_pairs[subject] = {
                "t1": file_paths["t1"],
                "pet": str(extracted_pet_path),
            }
            
            print(f"Extracted middle timepoint for {subject}: shape {vol.shape}")
            
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            continue
    
    print(f"Extracted middle timepoint for {len(extracted_pairs)} subjects")
    return extracted_pairs


@asset
def create_train_test_val_split(
    extract_single_pet_volume : Dict[str, Dict[str, str]],
    config: DataPipelineConfig
) -> Dict[str, List[str]]:
    """
    Create train/test/validation splits and organize data.
    
    Returns:
        Dictionary mapping split names to list of subject IDs.
    """
    output_dir = Path(config.output_dir)
    subjects = list(extract_single_pet_volume.keys())
    
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
            file_paths = extract_single_pet_volume[subject]
            
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
