"""SPACE's rank selection and right-subspace calibration."""
from __future__ import annotations

import math

import torch


def _validate_parameters(rho: float, alpha: float) -> None:
    for name, value in (("rho", rho), ("alpha", alpha)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")


@torch.no_grad()
def _calibrate_weight(w_pre, w_post, *, rho=0.5, alpha=1.0):
    _validate_parameters(rho, alpha)
    if w_pre.ndim != 2 or w_post.ndim != 2 or w_pre.shape != w_post.shape:
        raise ValueError("Expected two 2D tensors of identical shape")
    if not w_pre.is_floating_point() or not w_post.is_floating_point():
        raise ValueError("SPACE requires floating-point, non-quantized weights")
    if not torch.isfinite(w_pre).all() or not torch.isfinite(w_post).all():
        raise ValueError("Weights must contain only finite values")
    stats = {"selected_rank": 0, "captured_energy_ratio": 0.0}
    if rho == 0 or alpha == 0 or w_pre.numel() == 0:
        return w_post.detach().clone(), stats
    base = w_pre.detach().to(device=w_post.device, dtype=torch.float32)
    if not torch.count_nonzero(base):
        return w_post.detach().clone(), stats
    current = w_post.detach().float()
    _, singular_values, vh = torch.linalg.svd(base, full_matrices=False)
    energy = singular_values.square()
    cumulative = energy.cumsum(0)

    if rho == 1:
        k = int(torch.count_nonzero(singular_values).item())
    else:
        k = min(int(torch.searchsorted(cumulative, rho * cumulative[-1]).item()) + 1, len(energy))
    directions = vh[:k].T
    delta = current - base
    result = current - alpha * ((delta @ directions) @ directions.T)
    stats = {"selected_rank": k, "captured_energy_ratio": float((cumulative[k - 1] / cumulative[-1]).item())}
    return result.to(dtype=w_post.dtype).contiguous(), stats


def calibrate_weight(w_pre: torch.Tensor, w_post: torch.Tensor, *, rho: float = 0.5, alpha: float = 1.0) -> torch.Tensor:
    """Remove SFT updates along the pre-SFT leading right singular subspace.

    ``rho`` selects the smallest rank capturing that fraction of squared
    singular-value energy. ``alpha`` controls how much of its update is removed.
    SVD and projection use FP32; the result retains w_post's dtype and device.
    Inputs are not modified. Zero rho/alpha or a zero reference is a no-op.
    """
    return _calibrate_weight(w_pre, w_post, rho=rho, alpha=alpha)[0]
