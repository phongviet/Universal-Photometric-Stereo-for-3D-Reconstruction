import torch
import torch.nn as nn
import numpy as np

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple([x] * 2)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

def ind2coords(array_shape, ind):
    row = torch.div(ind, array_shape[1], rounding_mode='floor')
    col = ind % array_shape[1]
    coords = torch.zeros((1, 1, len(ind), 2), dtype=torch.float32)
    # Normalize to [-1, 1] for grid_sample
    coords[:, :, :, 1] = 2 * row.to(torch.float32) / array_shape[0] - 1
    coords[:, :, :, 0] = 2 * col.to(torch.float32) / array_shape[1] - 1
    return coords

def angular_error(x1, x2, mask=None):
    """
    Compute angular error between two normal maps in degrees.

    Calculates the angle between predicted and ground truth surface normals.
    This is the standard metric for evaluating normal map quality.

    Mathematical formulation:
        For two unit normal vectors n₁, n₂ ∈ ℝ³:

        1. Compute dot product:
           cos(θ) = n₁ · n₂ = Σᵢ n₁ᵢ × n₂ᵢ

        2. Clamp to valid range:
           cos(θ) ∈ [-1 + ε, 1 - ε]  (avoid numerical issues)

        3. Compute angle in degrees:
           θ = arccos(cos(θ)) × (180/π)

        4. Mean Angular Error (MAE):
           MAE = (1/|P|) Σ_{p∈P} |θ_p|
           where P is the set of valid pixels (mask > 0)

    Args:
        x1 (torch.Tensor): First normal map of shape [B, 3, H, W]
        x2 (torch.Tensor): Second normal map of shape [B, 3, H, W]
        mask (torch.Tensor, optional): Valid pixel mask of shape [B, 1, H, W].
                                       If None, all pixels are considered valid.

    Returns:
        torch.Tensor: If mask is provided, returns scalar MAE (mean angular error).
                      If mask is None, returns per-pixel angular error map [B, 1, H, W].

    Note:
        - Input normals should be L2-normalized: ||n||₂ = 1
        - ε = 1e-6 is used to avoid numerical instability in arccos
        - Output is in degrees (not radians)

    Example:
        >>> pred_normals = torch.randn(1, 3, 256, 256)
        >>> pred_normals = F.normalize(pred_normals, dim=1, p=2)
        >>> gt_normals = torch.randn(1, 3, 256, 256)
        >>> gt_normals = F.normalize(gt_normals, dim=1, p=2)
        >>> mask = torch.ones(1, 1, 256, 256)
        >>> mae = angular_error(pred_normals, gt_normals, mask)
    """
    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
        emap = torch.abs(180 * torch.acos(dot) / np.pi) * mask
        mae = torch.sum(emap) / (torch.sum(mask) + 1e-6)
        return mae
    else:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.abs(180 * torch.acos(dot) / np.pi)