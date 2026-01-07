#!/usr/bin/env python3
"""
PET-to-T1 MRI Image Translation Inference Script

This script applies a trained MTNet model to perform PET-to-T1 image-to-image translation
on test data. It generates visual comparisons and calculates quantitative metrics.

Usage:
    python pet_to_t1_inference.py [--options]

Output:
    - Visual comparison PNGs (Input PET | Predicted T1 | Ground Truth T1)
    - Quantitative metrics CSV (PSNR, SSIM, NMSE per sample and averages)
    - Run configuration JSON for reproducibility
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Import local utilities FIRST (before MTNet to avoid naming conflicts)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local ssl_utils before MTNet sys.path modification
from ssl_utils.metrics import psnr, ssim, nmse
from ssl_utils.visualization import save_comparison_image

# Add MTNet repository to Python path for model imports
MTNET_REPO_PATH = '/home/rbussell/repos/mtnet'
if MTNET_REPO_PATH not in sys.path:
    sys.path.insert(0, MTNET_REPO_PATH)

# Import MTNet models
try:
    from model.EdgeMAE import MAE_finetune
    from model.MTNet import MTNet
except ImportError as e:
    print(f"Error: Failed to import MTNet models: {e}")
    print(f"Please ensure MTNet repository is available at: {MTNET_REPO_PATH}")
    sys.exit(1)


class PETMRITestDataset:
    """
    Dataset class for loading PET-T1 paired test data.

    Each .npy file contains a (2, 256, 256) array where:
    - Index 0: T1 ground truth image
    - Index 1: PET input image
    """

    def __init__(self, data_dir):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing .npy test files
        """
        self.data_dir = Path(data_dir)
        self.data_files = sorted(self.data_dir.glob('*.npy'))

        if len(self.data_files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        print(f"Found {len(self.data_files)} test samples")

        # Validate first file
        self._validate_data_format(self.data_files[0])

    def _validate_data_format(self, file_path):
        """Verify data format matches expected structure."""
        data = np.load(file_path)

        assert data.shape == (2, 256, 256), \
            f"Expected shape (2, 256, 256), got {data.shape}"
        assert data.dtype in [np.float32, np.float64], \
            f"Expected float dtype, got {data.dtype}"
        assert 0 <= data.min() and data.max() <= 1, \
            f"Expected [0,1] range, got [{data.min()}, {data.max()}]"

        print(f"Data validation passed: {file_path.name}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - 'pet': PET input tensor (1, 1, 256, 256)
            - 't1': T1 ground truth tensor (1, 1, 256, 256)
            - 'filename': Original filename
        """
        data = np.load(self.data_files[idx])

        t1_gt = data[0]  # Ground truth T1
        pet_input = data[1]  # Input PET

        # Convert to torch tensors with shape (1, 1, 256, 256)
        pet_tensor = torch.from_numpy(pet_input).float().unsqueeze(0).unsqueeze(0)
        t1_tensor = torch.from_numpy(t1_gt).float().unsqueeze(0).unsqueeze(0)

        return {
            'pet': pet_tensor,
            't1': t1_tensor,
            'filename': self.data_files[idx].name
        }


class InferenceConfig:
    """Configuration for inference pipeline."""

    def __init__(self, args):
        """Initialize configuration from command-line arguments."""

        # Paths
        self.test_data_dir = args.test_data_dir
        self.encoder_checkpoint = args.encoder_ckpt
        self.generator_checkpoint = args.generator_ckpt
        self.output_dir = args.output_dir
        self.mtnet_repo_path = args.mtnet_repo_path

        # Model hyperparameters (must match training configuration)
        self.img_size = 256
        self.mae_patch_size = 8
        self.patch_size = 4
        self.encoder_dim = 128
        self.depth = 12
        self.num_heads = 16
        self.vit_dim = 128
        self.window_size = 8
        self.mlp_ratio = 4

        # Runtime options
        self.num_samples = args.num_samples
        self.save_predictions = args.save_predictions
        self.device = self._get_device()

    def _get_device(self):
        """Auto-detect CUDA availability."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU (inference will be slower)")
        return device

    def to_dict(self):
        """Convert configuration to dictionary for JSON serialization."""
        return {
            'timestamp': datetime.now().isoformat(),
            'test_data_dir': str(self.test_data_dir),
            'encoder_checkpoint': str(self.encoder_checkpoint),
            'generator_checkpoint': str(self.generator_checkpoint),
            'output_dir': str(self.output_dir),
            'device': str(self.device),
            'num_samples': self.num_samples,
            'model_hyperparameters': {
                'img_size': self.img_size,
                'mae_patch_size': self.mae_patch_size,
                'patch_size': self.patch_size,
                'encoder_dim': self.encoder_dim,
                'depth': self.depth,
                'num_heads': self.num_heads,
                'vit_dim': self.vit_dim,
                'window_size': self.window_size,
                'mlp_ratio': self.mlp_ratio
            }
        }


def initialize_models(config):
    """
    Initialize encoder and generator models.

    Args:
        config: InferenceConfig instance

    Returns:
        Tuple of (encoder, generator) models
    """
    print("\nInitializing models...")

    # Initialize encoder (MAE_finetune)
    encoder = MAE_finetune(
        img_size=config.img_size,
        patch_size=config.mae_patch_size,
        in_chans=1,
        embed_dim=config.encoder_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        norm_layer=nn.LayerNorm
    )

    # Initialize generator (MTNet)
    generator = MTNet(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=1,
        num_classes=1,
        embed_dim=config.vit_dim,
        depths=[2, 2, 2, 2],
        depths_decoder=[2, 2, 2, 2],
        num_heads=[8, 8, 16, 32],
        window_size=config.window_size,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
        fine_tune=True
    )

    print("Models initialized successfully")
    return encoder, generator


def load_checkpoints(encoder, generator, config):
    """
    Load trained weights from checkpoint files.

    Args:
        encoder: MAE_finetune model
        generator: MTNet model
        config: InferenceConfig instance

    Returns:
        Tuple of (encoder, generator) with loaded weights
    """
    print("\nLoading checkpoints...")

    # Move models to device
    encoder = encoder.to(config.device)
    generator = generator.to(config.device)

    # Load encoder checkpoint
    print(f"Loading encoder from: {config.encoder_checkpoint}")
    encoder.load_state_dict(
        torch.load(config.encoder_checkpoint, map_location=config.device),
        strict=False
    )

    # Load generator checkpoint
    print(f"Loading generator from: {config.generator_checkpoint}")
    generator.load_state_dict(
        torch.load(config.generator_checkpoint, map_location=config.device),
        strict=False
    )

    # Set to evaluation mode
    encoder.eval()
    generator.eval()

    print("Checkpoints loaded successfully")
    return encoder, generator


def run_inference(encoder, generator, pet_image, device):
    """
    Run inference on a single PET image.

    Args:
        encoder: MAE_finetune model
        generator: MTNet model
        pet_image: Input PET tensor (1, 1, 256, 256)
        device: torch device

    Returns:
        Predicted T1 tensor (1, 1, 256, 256)
    """
    with torch.no_grad():
        pet_image = pet_image.to(device, dtype=torch.float)

        # Extract features using encoder
        features = encoder(pet_image)

        # Clone final feature for dual input to generator
        f1 = features[-1].clone()
        f2 = features[-1].clone()

        # Generate prediction
        pred = generator(f1, f2)

    return pred


def create_output_directory(base_path):
    """
    Create timestamped output directory structure.

    Args:
        base_path: Base output directory path

    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"run_{timestamp}"

    # Create subdirectories
    (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    return output_dir


def save_run_config(config, output_dir):
    """
    Save run configuration for reproducibility.

    Args:
        config: InferenceConfig instance
        output_dir: Output directory path
    """
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to: {config_path}")


def main():
    """Main inference pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='PET-to-T1 Image Translation Inference'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='/home/rbussell/data/pet_mri/test/',
        help='Directory containing test .npy files'
    )
    parser.add_argument(
        '--encoder_ckpt',
        type=str,
        default='/home/rbussell/repos/mtnet/weight/pet_mri_finetune/E.pth',
        help='Path to encoder checkpoint (E.pth)'
    )
    parser.add_argument(
        '--generator_ckpt',
        type=str,
        default='/home/rbussell/repos/mtnet/weight/pet_mri_finetune/G.pth',
        help='Path to generator checkpoint (G.pth)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/pet_t1_inference/',
        help='Base output directory'
    )
    parser.add_argument(
        '--mtnet_repo_path',
        type=str,
        default='/home/rbussell/repos/mtnet',
        help='Path to MTNet repository'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save raw prediction arrays as .npy files'
    )

    args = parser.parse_args()

    # Initialize configuration
    print("="*80)
    print("PET-to-T1 Image Translation Inference")
    print("="*80)
    config = InferenceConfig(args)

    # Create output directory
    output_dir = create_output_directory(config.output_dir)
    save_run_config(config, output_dir)

    # Create predictions subdirectory if saving raw predictions
    if config.save_predictions:
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        print(f"Raw predictions will be saved to: {pred_dir}")

    # Initialize models
    encoder, generator = initialize_models(config)

    # Load checkpoints
    encoder, generator = load_checkpoints(encoder, generator, config)

    # Load test dataset
    print(f"\nLoading test data from: {config.test_data_dir}")
    dataset = PETMRITestDataset(config.test_data_dir)

    # Determine number of samples to process
    num_samples = config.num_samples if config.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"Processing {num_samples} samples")

    # Run inference on all samples
    print("\nRunning inference...")
    results = []

    for idx in tqdm(range(num_samples), desc="Processing samples"):
        sample = dataset[idx]

        # Run inference
        pred = run_inference(encoder, generator, sample['pet'], config.device)

        # Convert to numpy for metrics and visualization
        pet_np = sample['pet'].cpu().numpy().squeeze()
        t1_np = sample['t1'].cpu().numpy().squeeze()
        pred_np = pred.cpu().numpy().squeeze()

        # Calculate metrics
        metrics = {
            'filename': sample['filename'],
            'sample_id': idx,
            'psnr': psnr(pred_np, t1_np),
            'ssim': ssim(pred_np, t1_np),
            'nmse': nmse(pred_np, t1_np)
        }
        results.append(metrics)

        # Save visualization
        vis_path = output_dir / 'visualizations' / sample['filename'].replace('.npy', '.png')
        save_comparison_image(pet_np, pred_np, t1_np, vis_path, metrics)

        # Save raw prediction if requested
        if config.save_predictions:
            pred_dir = output_dir / 'predictions'
            pred_path = pred_dir / sample['filename'].replace('.npy', '_prediction.npy')
            np.save(pred_path, pred_np)

    # Calculate summary statistics
    df = pd.DataFrame(results)
    summary = {
        'filename': 'MEAN',
        'sample_id': -1,
        'psnr': df['psnr'].mean(),
        'ssim': df['ssim'].mean(),
        'nmse': df['nmse'].mean()
    }
    std_summary = {
        'filename': 'STD',
        'sample_id': -1,
        'psnr': df['psnr'].std(),
        'ssim': df['ssim'].std(),
        'nmse': df['nmse'].std()
    }

    # Add summary rows
    df = pd.concat([df, pd.DataFrame([summary, std_summary])], ignore_index=True)

    # Save metrics to CSV
    metrics_path = output_dir / 'metrics.csv'
    df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Print summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Samples processed: {num_samples}")
    print(f"\nAverage Metrics:")
    print(f"  PSNR: {summary['psnr']:.4f} ± {std_summary['psnr']:.4f} dB")
    print(f"  SSIM: {summary['ssim']:.4f} ± {std_summary['ssim']:.4f}")
    print(f"  NMSE: {summary['nmse']:.4f} ± {std_summary['nmse']:.4f}")
    print(f"\nOutputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
