#!/usr/bin/env python
"""
Script to run the PET/MRI ETL pipeline with persistent caching.
"""

import sys
from pathlib import Path

# Add the script directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from dagster import DagsterInstance
from etl.definitions import defs

if __name__ == "__main__":
    print("Starting PET/MRI ETL Pipeline...")
    print("=" * 60)
    
    # Use persistent instance for caching
    instance = DagsterInstance.ephemeral()
    
    # Run the job with caching enabled
    result = defs.etl_job.execute_in_process(instance=instance)
    
    print("=" * 60)
    if result.success:
        print("✓ Pipeline completed successfully!")
        print(f"Output directory: /home/rbussell/data/pet_mri")
    else:
        print("✗ Pipeline failed!")
        sys.exit(1)
