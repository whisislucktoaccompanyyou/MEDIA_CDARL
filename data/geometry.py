"""Geometry of the *target* mask; positive signed distance is inside vessels."""
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize


def geometry_from_mask(mask, radius=None):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('expected a 2D binary target mask')
    radius = float(16 * max(mask.shape) / 384 if radius is None else radius)
    if radius <= 0:
        raise ValueError('distance radius must be positive')
    boundary = binary_dilation(mask) ^ binary_erosion(mask, border_value=0)
    if not mask.any():
        distance = np.full(mask.shape, -1, np.float32)
    elif mask.all():
        distance = np.ones(mask.shape, np.float32)
    else:
        distance = np.clip((distance_transform_edt(mask) - distance_transform_edt(~mask)) / radius, -1, 1)
    return np.stack([mask, boundary, skeletonize(mask), distance]).astype(np.float32)


def batch_geometry(target):
    """Target uses [-1,1]; call after all geometric augmentation."""
    arrays = target.detach().cpu().numpy()[:, 0] > 0
    return torch.from_numpy(np.stack([geometry_from_mask(m) for m in arrays])).to(target)
