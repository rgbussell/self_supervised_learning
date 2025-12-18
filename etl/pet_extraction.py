import nibabel as nib
import numpy as np
import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt


def extract_vol_from_multitmp_nifti(nifti_path: str, vol_index: int = 0) -> None:
    """Extract and display a specific volume from a multi-timepoint NIfTI file."""
    # Load the NIfTI file
    try:
        img = nib.load(nifti_path)
    except FileNotFoundError:
        print(f"Error: File '{nifti_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Get the data and shape
    shape = img.dataobj.shape
    print(f"NIfTI file shape: {shape}")

    # Extract the specified plane (default is the first plane along z-axis)
    vol = img.slicer[:, :, :, plane_index].get_fdata()
    shape = vol.shape
    extracted_img = nib.Nifti1Image(vol, img.affine)
    nib.save(extracted_img, args.output_path)
    print(f"Extracted plane saved to: {args.output_path}")

    return img
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display a plane from a NIfTI file.")
    parser.add_argument("--input_path", help="Path to the NIfTI file")
    parser.add_argument("--plane_index", type=int, default=0, help="Plane index to extract (default: 0)")
    parser.add_argument("--output_path", help="Path to save the extracted plane as NIfTI file")
    
    args = parser.parse_args()
    
    extract_vol_from_multitmp_nifti(args.input_path, args.plane_index)