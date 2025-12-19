# PET/MRI ETL Pipeline + MTNet Fine-tuning

## Overview

This pipeline prepares PET/MRI data and fine-tunes the MTNet model in two steps:

1. **ETL Pipeline**: Extracts paired 2D T1/PET slices as combined `.npy` files
2. **Fine-tuning**: Fine-tunes MTNet on the prepared data using the existing MTNet codebase

## Files

### ETL Module
- `etl_pipeline.py` - Dagster assets for data preparation (outputs `.npy` files)
- `definitions.py` - Dagster definitions
- `run_etl_pipeline.py` - Execute ETL pipeline
- `launch_pipeline.sh` - One-command launcher

### MTNet Fine-tuning
- `mtnet/utils/etl_dataloader.py` - Dataset class compatible with existing MTNet interface
- `mtnet/finetune_etl.py` - Minimal wrapper around existing `finetune.py`

## Data Format

### ETL Output
The ETL pipeline saves `.npy` files with shape `(2, 256, 256)` where:
- Index 0: T1 weighted (normalized to [0, 1])
- Index 1: PET (normalized to [0, 1])

Files are saved in splits:
```
/home/rbussell/data/pet_mri/
├── train/
│   ├── 000001.npy
│   ├── 000002.npy
│   └── ... (num_subjects * num_slices)
├── val/
└── test/
```

This format is compatible with the existing MTNet `BraTS_Train_Dataset` interface.

## Quick Start

### Run Complete Pipeline

```bash
cd /home/rbussell/repos/self_supervised_learning/etl
bash launch_pipeline.sh
```

### Run Only ETL

```bash
bash launch_pipeline.sh --etl-only
```

### Run Only Fine-tuning

```bash
bash launch_pipeline.sh --finetune-only
```

## Manual Execution

### Step 1: ETL Pipeline

```bash
conda activate pet_to_mri_translation
cd /home/rbussell/repos/self_supervised_learning/etl
python run_etl_pipeline.py
```

Output: `.npy` files in `/home/rbussell/data/pet_mri/{train,val,test}/`

### Step 2: Fine-tune MTNet

```bash
conda activate pet_to_mri_translation
cd /home/rbussell/repos/mtnet
python finetune_etl.py \
    --data_root /home/rbussell/data/pet_mri/train \
    --source_modal t1 \
    --target_modal pet
```

Output: Model weights in `/home/rbussell/repos/mtnet/weight/pet_mri_finetune/`

## How It Works

### ETL Pipeline
1. Discovers BIDS-formatted T1/PET pairs
2. Extracts single timepoint from 4D PET volumes (index 200)
3. Resamples to 1mm³ isotropic resolution
4. Extracts 20 paired 2D slices per subject at corresponding anatomical locations
5. Crops/pads to 256×256 resolution
6. Normalizes to [0, 1] range
7. Saves as combined `.npy` files: `(2, 256, 256)` for [t1, pet]
8. Creates 70/15/15 train/val/test splits
9. Sequential numbering: `000001.npy`, `000002.npy`, ...

### MTNet Fine-tuning
- Uses `ETL_PET_MRI_Dataset` class in `etl_dataloader.py`
- Compatible with existing MTNet training framework
- Loads pre-trained EdgeMAE encoder
- Fine-tunes on T1→PET translation task
- Saves checkpoints every 20 epochs
- Uses cosine annealing with warmup

## Design Philosophy

Minimal implementation that reuses existing MTNet code:

✅ Uses existing `finetune.py` training loop unchanged
✅ Uses existing option parsing system  
✅ Uses existing loss computation and scheduling
✅ Creates only one dataset class adapter
✅ ETL outputs in format compatible with existing dataloader interface

## Configuration

### ETL Pipeline
Edit `etl_pipeline.py`:
```python
INPUT_DATA_DIR = "/home/rbussell/data/openneuro_pet_mri/ds002898-download"
OUTPUT_DATA_DIR = "/home/rbussell/data/pet_mri"
PET_TIMEPOINT_INDEX = 200  # Middle timepoint
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

### MTNet Fine-tuning
Options in `finetune_etl.py` (uses `Finetune_Options` defaults):
```python
--epoch 150              # Number of epochs
--batch_size 8          # Batch size
--lr 1e-3              # Learning rate
--source_modal t1      # Source modality
--target_modal pet     # Target modality
```

## Outputs

### After ETL
```
/home/rbussell/data/pet_mri/
├── train/
│   ├── 000001.npy  # shape: (2, 256, 256) [t1, pet]
│   ├── 000002.npy
│   └── ...
├── val/
│   ├── 100000.npy
│   └── ...
└── test/
    ├── 200000.npy
    └── ...
```

### After Fine-tuning
```
/home/rbussell/repos/mtnet/weight/pet_mri_finetune/
├── 020E.pth, 020G.pth  # Checkpoint at epoch 20
├── 040E.pth, 040G.pth  # Checkpoint at epoch 40
├── ...
├── E.pth, G.pth        # Final weights
├── snapshots/
│   └── epoch_1_batch_0.png  # Training visualizations
└── finetune_etl.log    # Training log
```

## Troubleshooting

### No data found in img_root
- Verify ETL pipeline completed successfully
- Check `/home/rbussell/data/pet_mri/train/` contains `.npy` files

### Index out of range in dataloader
- ETL outputs `.npy` files with indices [0]=t1, [1]=pet
- Use `--source_modal t1 --target_modal pet` in finetune_etl.py

### CUDA out of memory
- Reduce batch size: `--batch_size 4` in finetune_options.py
- Or modify finetune.py img_size to 128

## Compatibility

The solution maintains full compatibility with existing MTNet code:
- `BraTS_Train_Dataset` expects `.npy` files with modality indexing
- `ETL_PET_MRI_Dataset` implements the same interface
- Both work with existing `get_loader()` function signature
- Fine-tuning script is minimal wrapper of `finetune.py`
