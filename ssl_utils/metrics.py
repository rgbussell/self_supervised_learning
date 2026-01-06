"""
Evaluation metrics for image quality assessment.

Metrics:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- NMSE: Normalized Mean Squared Error

All metrics assume input images are normalized to [0, 1] range.
"""

import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim_func


def psnr(pred, gt):
    """
    Calculate Peak Signal-to-Noise Ratio between predicted and ground truth images.

    Args:
        pred: Predicted image array (normalized to [0, 1])
        gt: Ground truth image array (normalized to [0, 1])

    Returns:
        PSNR value in dB
    """
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse))
    return psnr_value


def nmse(pred, gt):
    """
    Calculate Normalized Mean Squared Error between predicted and ground truth images.

    Args:
        pred: Predicted image array (normalized to [0, 1])
        gt: Ground truth image array (normalized to [0, 1])

    Returns:
        NMSE value
    """
    norm = np.linalg.norm(gt * gt, ord=2)
    if np.all(norm == 0):
        return 0
    else:
        nmse_value = np.linalg.norm((pred - gt) * (pred - gt), ord=2) / norm
    return nmse_value


def ssim(pred, gt):
    """
    Calculate Structural Similarity Index between predicted and ground truth images.

    Args:
        pred: Predicted image array (normalized to [0, 1])
        gt: Ground truth image array (normalized to [0, 1])

    Returns:
        SSIM value in range [-1, 1] (typically [0, 1] for similar images)
    """
    return ssim_func(pred, gt, data_range=1.0)
