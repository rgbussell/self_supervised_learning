import nibabel as nib
import numpy as np
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python view_nifti.py <path_to_nifti_file>")
        sys.exit(1)
    
    nifti_path = sys.argv[1]
    
    # Load the NIfTI file
    try:
        img = nib.load(nifti_path)
    except FileNotFoundError:
        print(f"Error: File '{nifti_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Get the data and shape
    shape = img.dataobj.shape
    print(f"NIfTI file shape: {shape}")




    for t in np.arange(0, 350, 50):
        vol = img.slicer[:, :, :, t].get_fdata()
        shape = vol.shape

        

        print(f"NIfTI file shape: {shape}")
        print(f"Data type: {vol.dtype}")
        print(f"Min value: {np.min(vol):.4f}, Max value: {np.max(vol):.4f}")
        
        # Extract and save the first plane
        first_plane = vol[:, :, 64]

        plt.figure(figsize=(8, 8))
        plt.imshow(first_plane, cmap='gray')
        plt.colorbar()
        plt.title(f"First Plane (z=0)")
        plt.axis('off')
        
        plt.show()
        plt.close()

        #output_path = Path(nifti_path).stem + "_first_plane.png"
        #plt.savefig(output_path, bbox_inches='tight', dpi=150)
        #print(f"First plane saved to: {output_path}")
        #plt.close()


if __name__ == "__main__":
    main()