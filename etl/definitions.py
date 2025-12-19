"""
Dagster definitions for PET/MRI ETL pipeline.
"""

from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    in_process_executor,
    load_assets_from_modules,
)

import etl_pipeline

# Load all assets from the pipeline module
all_assets = load_assets_from_modules([etl_pipeline])

# Define the job to run all assets
etl_job = define_asset_job(
    name="pet_mri_etl_job",
    selection=AssetSelection.all(),
    executor_def=in_process_executor,
)

# Create definitions
defs = Definitions(
    assets=all_assets,
    jobs=[etl_job],
)
