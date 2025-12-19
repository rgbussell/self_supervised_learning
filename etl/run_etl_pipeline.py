#!/usr/bin/env python
"""
Script to run the PET/MRI ETL pipeline.
"""

import sys
from pathlib import Path

# Add the script directory to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dagster import materialize_to_memory
from definitions import defs

if __name__ == "__main__":
    print("Starting PET/MRI ETL Pipeline...")
    print("=" * 60)
    
    # Run the job
    from dagster import materialize_to_memory
    
    result = materialize_to_memory(
        defs.assets,
        raise_on_error=True,
    )
    
    print("=" * 60)
    if result.success:
        print("✓ Pipeline completed successfully!")
    else:
        print("✗ Pipeline failed!")
        sys.exit(1)
