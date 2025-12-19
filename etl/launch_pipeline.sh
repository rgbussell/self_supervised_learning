#!/bin/bash
# Launch script for PET/MRI ETL + MTNet fine-tuning
#
# Usage:
#   bash launch_pipeline.sh [--etl-only] [--finetune-only]

set -e

# Configuration
REPO_ROOT="${HOME}/repos/self_supervised_learning"
MTNET_REPO="${HOME}/repos/mtnet"
ETL_DIR="${REPO_ROOT}/etl"
DATA_DIR="${HOME}/data/pet_mri"
INPUT_DATA_DIR="${HOME}/data/openneuro_pet_mri/ds002898-download"
CONDA_ENV="pet_to_mri_translation"

RUN_ETL=true
RUN_FINETUNE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --etl-only) RUN_FINETUNE=false; shift ;;
        --finetune-only) RUN_ETL=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Step 1: ETL Pipeline
if [ "$RUN_ETL" = true ]; then
    echo "=========================================="
    echo "Running ETL Pipeline..."
    echo "=========================================="
    cd ${ETL_DIR}
    python run_etl_pipeline.py
    echo "✓ ETL complete"
    echo ""
fi

# Step 2: Fine-tune MTNet
if [ "$RUN_FINETUNE" = true ]; then
    echo "=========================================="
    echo "Fine-tuning MTNet..."
    echo "=========================================="
    cd ${MTNET_REPO}
    python finetune_etl.py \
        --data_root ${DATA_DIR}/train \
        --source_modal t1 \
        --target_modal pet \
        --epoch 150 \
        --batch_size 8 \
        --weight_save_path ./weight/pet_mri_finetune/ \
        --img_save_path ./snapshot/pet_mri_finetune/ \
        --log_path ./log/finetune_etl.log
    echo "✓ Fine-tuning complete"
    echo ""
fi

echo "Done! Models saved to: ${MTNET_REPO}/weight/pet_mri_finetune/"
