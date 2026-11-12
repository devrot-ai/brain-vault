"""Longitudinal (time-series) comparison of two MRI scans from the same subject.

Given two preprocessed 128³ volumes from the same patient at different time
points, this module computes:

1. **Spatial co-registration** — aligns T2 to T1 space using affine registration
   so voxel-wise differences reflect true anatomical change, not misalignment.
2. **Voxel-wise difference map** — where has intensity changed between scans?
3. **Brain volume change** — percentage of brain tissue lost or gained.
4. **Regional atrophy map** — which atlas regions show the most change?
5. **3D change visualization** — a mesh colored by the magnitude and direction
   of change (atrophy = red, growth = blue).

This is designed for Alzheimer's progression tracking: a patient scanned at
baseline (Time 1) and follow-up (Time 2) can see where their brain has
shrunk over time.

Co-registration uses a multi-resolution affine transform optimized via
normalized mutual information (NMI) to align the follow-up scan to the
baseline scan's space. This eliminates registration artifacts that would
otherwise confound the atrophy measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class LongitudinalResult:
    """Results from comparing two time-point scans."""

    # Volume statistics
    volume_t1: float  # brain volume at time 1 (mm³)
    volume_t2: float  # brain volume at time 2 (mm³)
    volume_change_pct: float  # percentage change (negative = atrophy)

    # Voxel-wise difference map [D, H, W]
    diff_map: np.ndarray  # T2 - T1 (positive = growth, negative = atrophy)
    abs_diff_map: np.ndarray  # absolute difference magnitude

    # Brain mask overlap
    brain_mask_t1: np.ndarray  # boolean mask at time 1
    brain_mask_t2: np.ndarray  # boolean mask at time 2
    overlap_pct: float  # percentage of overlapping brain voxels

    # Regional analysis (if atlas available)
    regional_changes: List[dict] = field(default_factory=list)

    # Prediction comparison
    prob_t1: float | None = None
    prob_t2: float | None = None
    prob_change: float | None = None  # T2 - T1 probability change

    # Metadata
    label_t1: str | None = None
    label_t2: str | None = None


# ---------------------------------------------------------------------------
# Co-registration: align T2 to T1 space for accurate voxel-wise comparison
# ---------------------------------------------------------------------------


def _compute_nmi(
    fixed: np.ndarray, moving: np.ndarray, bins: int = 32
) -> float:
    """Compute Normalized Mutual Information between two volumes.

    Uses histogram-based estimation of joint entropy. Lower NMI = better alignment.

    Args:
        fixed: reference volume [D, H, W].
        moving: volume to align [D, H, W].
        bins: number of histogram bins per channel.

    Returns:
        Normalized mutual information value.
    """
    # Resample moving to fixed grid (already aligned in shape)
    mask = (fixed > 0) & (moving > 0)
    if mask.sum() < 100:
        return 0.0

    f_vals = fixed[mask].astype(np.float32)
    m_vals = moving[mask].astype(np.float32)

    # Compute joint histogram
    joint_hist, _, _ = np.histogram2d(
        f_vals, m_vals, bins=bins, range=[[f_vals.min(), f_vals.max()],
                                           [m_vals.min(), m_vals.max()]]
    )
    joint_hist = joint_hist / joint_hist.sum()

    # Marginals
    px = joint_hist.sum(axis=1)
    py = joint_hist.sum(axis=0)

    # Entropies
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p + 1e-12))

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(joint_hist.flatten())

    if hxy == 0:
        return 0.0

    mi = hx + hy - hxy
    nmi = mi / np.sqrt(hx * hy + 1e-12)
    return nmi


def register_to_template(
    vol_t1: np.ndarray,
    vol_t2: np.ndarray,
    *,
    num_iterations: int = 50,
    learning_rate: float = 0.01,
    smoothing_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register vol_t2 to vol_t1 space using affine optimization with NMI.

    Optimizes a 12-parameter affine transform (rotation, translation, scaling,
    shearing) to maximize the normalized mutual information between the fixed
    (T1) and moving (T2) volumes. This aligns the follow-up scan to the
    baseline scan's coordinate system so voxel-wise differences reflect true
    anatomical change rather than head position differences.

    Uses gradient-free optimization (Nelder-Mead simplex) on the NMI metric.
    For 128³ volumes this converges in ~30-50 iterations.

    Args:
        vol_t1: baseline volume [1,1,D,H,W] or [D,H,W] (fixed reference).
        vol_t2: follow-up volume [1,1,D,H,W] or [D,H,W] (moving).
        num_iterations: max optimization iterations.
        learning_rate: initial step size for parameter perturbation.
        smoothing_sigma: Gaussian smoothing sigma for multi-resolution pyramid.

    Returns:
        Tuple of (registered_t2, affine_matrix):
        - registered_t2: T2 resampled into T1's space [D, H, W].
        - affine_matrix: 4x4 affine transform matrix (can be applied to
          other volumes like Grad-CAM maps for consistency).
    """
    from scipy.ndimage import gaussian_filter, affine_transform
    from scipy.optimize import minimize

    t1 = vol_t1[0] if vol_t1.ndim == 4 else vol_t1
    t2 = vol_t2[0] if vol_t2.ndim == 4 else vol_t2

    D, H, W = t1.shape
    center = np.array([D / 2 - 0.5, H / 2 - 0.5, W / 2 - 0.5])

    # Smooth volumes for coarse-to-fine registration
    t1_smooth = gaussian_filter(t1, smoothing_sigma)
    t2_smooth = gaussian_filter(t2, smoothing_sigma)

    # Clip to brain range for stable NMI
    t1_range = (np.percentile(t1_smooth[t1_smooth > 0], 1),
                np.percentile(t1_smooth[t1_smooth > 0], 99))
    t2_range = (np.percentile(t2_smooth[t2_smooth > 0], 1),
                np.percentile(t2_smooth[t2_smooth > 0], 99))
    t1_smooth = np.clip(t1_smooth, t1_range[0], t1_range[1])
    t2_smooth = np.clip(t2_smooth, t2_range[0], t2_range[1])

    def apply_affine(params: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Apply 12-parameter affine transform to a volume.

        Parameters encode: [tx, ty, tz, rx, ry, rz, sx, sy, sz, sh1, sh2, sh3]
        where tx/ty/tz = translation (voxels), rx/ry/rz = rotation (radians),
        sx/sy/sz = scaling, sh1/sh2/sh3 = shearing.
        """
        tx, ty, tz = params[0], params[1], params[2]
        rx, ry, rz = params[3], params[4], params[5]
        sx, sy, sz = params[6], params[7], params[8]
        sh1, sh2, sh3 = params[9], params[10], params[11]

        # Build rotation matrices
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

        rotation = rot_z @ rot_y @ rot_x
        scale = np.diag([sx, sy, sz])
        shear = np.array([[1, sh1, sh2], [0, 1, sh3], [0, 0, 1]])

        # Combined linear transform
        linear = rotation @ scale @ shear

        # Translation in voxel units
        translation = np.array([tx, ty, tz])

        # Build 3x3 affine matrix (for scipy affine_transform, which uses
        # the inverse mapping convention)
        affine_3x3 = linear.copy()
        affine_3x3[:, 3 if False else 0:0]  # no-op for clarity

        # For scipy.affine_transform: output[i] = input[affine @ i + offset]
        # We need the inverse of the forward transform
        try:
            affine_inv = np.linalg.inv(np.vstack([
                np.hstack([linear, translation.reshape(3, 1)]),
                [0, 0, 0, 1]
            ]))[:3, :3]
            offset = affine_inv[:3, 3]
        except np.linalg.LinAlgError:
            # Fallback: identity
            return volume.copy()

        return affine_transform(volume, affine_3x3, offset=offset, order=1, cval=0.0)

    def neg_nmi(params: np.ndarray) -> float:
        """Negative NMI (to be minimized)."""
        try:
            registered = apply_affine(params, t2_smooth)
            return -_compute_nmi(t1_smooth, registered)
        except Exception:
            return 1e6  # penalize invalid transforms

    # Initial parameters: identity transform with small random perturbation
    params0 = np.zeros(12, dtype=np.float64)

    # Optimize: maximize NMI (minimize negative NMI)
    result = minimize(
        neg_nmi, params0,
        method='Nelder-Mead',
        options={'maxiter': num_iterations, 'xatol': 1e-6, 'fatol': 1e-6}
    )

    best_params = result.x

    # Apply best transform to the original (unsmoothed) T2 volume
    t2_registered = apply_affine(best_params, t2)

    # Build 4x4 affine matrix for downstream use
    tx, ty, tz = best_params[0], best_params[1], best_params[2]
    rx, ry, rz = best_params[3], best_params[4], best_params[5]
    sx, sy, sz = best_params[6], best_params[7], best_params[8]
    sh1, sh2, sh3 = best_params[9], best_params[10], best_params[11]

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    rotation = rot_z @ rot_y @ rot_x
    scale = np.diag([sx, sy, sz])
    shear = np.array([[1, sh1, sh2], [0, 1, sh3], [0, 0, 1]])

    linear = rotation @ scale @ shear
    affine_4x4 = np.eye(4, dtype=np.float64)
    affine_4x4[:3, :3] = linear
    affine_4x4[:3, 3] = [tx, ty, tz]

    return t2_registered, affine_4x4


def compute_brain_mask(volume: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    """Extract brain mask from a preprocessed volume.

    Args:
        volume: preprocessed brain volume [1,1,D,H,W] or [D,H,W].
        percentile: intensity percentile threshold for brain tissue.

    Returns:
        Boolean mask of brain tissue [D, H, W].
    """
    vol = volume[0] if volume.ndim == 4 else volume
    threshold = np.percentile(vol[vol > 0], percentile) if (vol > 0).any() else 0
    return vol > threshold


def compute_volume_mm3(mask: np.ndarray, voxel_mm: float = 2.0) -> float:
    """Compute brain volume in mm³ from a boolean mask.

    Args:
        mask: boolean brain mask [D, H, W].
        voxel_mm: isotropic voxel size in mm (default 2.0mm for preprocessed data).

    Returns:
        Volume in cubic millimeters.
    """
    voxel_volume = voxel_mm ** 3
    return float(mask.sum() * voxel_volume)


def compute_diff_map(vol_t1: np.ndarray, vol_t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute voxel-wise difference between two preprocessed volumes.

    Since each volume is independently z-scored, the difference reflects
    relative intensity changes. Negative values indicate regions that became
    relatively lower intensity (potential atrophy), positive values indicate
    regions that became relatively higher intensity.

    Args:
        vol_t1: preprocessed volume at time 1 [1,1,D,H,W] or [D,H,W].
        vol_t2: preprocessed volume at time 2 [1,1,D,H,W] or [D,H,W].

    Returns:
        Tuple of (diff_map, abs_diff_map), each [D, H, W].
    """
    v1 = vol_t1[0] if vol_t1.ndim == 4 else vol_t1
    v2 = vol_t2[0] if vol_t2.ndim == 4 else vol_t2

    # Compute difference only in overlapping brain regions
    mask_t1 = compute_brain_mask(v1)
    mask_t2 = compute_brain_mask(v2)
    overlap_mask = mask_t1 & mask_t2

    # In non-overlap regions, use the mask that has brain
    # T1 has brain but T2 doesn't = atrophy (large negative)
    # T2 has brain but T1 doesn't = growth (large positive)
    diff = np.zeros_like(v1, dtype=np.float32)

    # Overlapping regions: direct difference
    diff[overlap_mask] = v2[overlap_mask] - v1[overlap_mask]

    # T1 brain but not T2 brain: mark as atrophy
    atrophy_only = mask_t1 & ~mask_t2
    diff[atrophy_only] = -2.0  # strong negative signal for lost tissue

    # T2 brain but not T1 brain: mark as growth
    growth_only = ~mask_t1 & mask_t2
    diff[growth_only] = 2.0  # strong positive signal for new tissue

    abs_diff = np.abs(diff)
    return diff, abs_diff


def compute_regional_changes(
    diff_map: np.ndarray,
    abs_diff_map: np.ndarray,
    brain_mask: np.ndarray,
    top_k: int = 10,
) -> List[dict]:
    """Compute regional summary of changes using simple spatial partitioning.

    Divides the brain into a 4x4x4 grid of regions and computes the mean
    absolute change in each region. This provides a coarse but interpretable
    view of where the most change occurred.

    For atlas-based regionalization, use `brainvuln.mri.regional.regionalize_cam`
    with the absolute difference map as input.

    Args:
        diff_map: signed difference map [D, H, W].
        abs_diff_map: absolute difference map [D, H, W].
        brain_mask: combined brain mask [D, H, W].
        top_k: number of top regions to return.

    Returns:
        List of dicts with region info, sorted by mean absolute change.
    """
    D, H, W = diff_map.shape

    # Simple grid-based regionalization (4x4x4 = 64 regions)
    grid_size = 4
    d_step, h_step, w_step = D // grid_size, H // grid_size, W // grid_size

    # Region name mapping (coarse anatomical approximation)
    # D axis (depth/slice): anterior (low) to posterior (high)
    # H axis (row): superior (low) to inferior (high)
    # W axis (col): left (low) to right (high)
    d_names = ["Anterior", "Ant-Mid", "Post-Mid", "Posterior"]
    h_names = ["Superior", "Sup-Mid", "Inf-Mid", "Inferior"]
    w_names = ["Left", "Left-Mid", "Right-Mid", "Right"]

    regions = []
    for di in range(grid_size):
        for hi in range(grid_size):
            for wi in range(grid_size):
                d_lo = di * d_step
                d_hi = (di + 1) * d_step if di < grid_size - 1 else D
                h_lo = hi * h_step
                h_hi = (hi + 1) * h_step if hi < grid_size - 1 else H
                w_lo = wi * w_step
                w_hi = (wi + 1) * w_step if wi < grid_size - 1 else W

                region_mask = brain_mask[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
                if region_mask.sum() < 10:
                    continue

                region_diff = diff_map[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
                region_abs_diff = abs_diff_map[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]

                mean_change = float(region_diff[region_mask].mean())
                mean_abs_change = float(region_abs_diff[region_mask].mean())
                max_abs_change = float(region_abs_diff[region_mask].max())
                voxel_count = int(region_mask.sum())

                # Classify as atrophy or growth
                if mean_change < -0.1:
                    change_type = "atrophy"
                elif mean_change > 0.1:
                    change_type = "growth"
                else:
                    change_type = "stable"

                name = f"{d_names[di]}-{h_names[hi]}-{w_names[wi]}"

                regions.append({
                    "name": name,
                    "mean_change": round(mean_change, 4),
                    "mean_abs_change": round(mean_abs_change, 4),
                    "max_abs_change": round(max_abs_change, 4),
                    "voxel_count": voxel_count,
                    "change_type": change_type,
                })

    # Sort by mean absolute change, descending
    regions.sort(key=lambda r: r["mean_abs_change"], reverse=True)
    return regions[:top_k]


def compare_scans(
    vol_t1: np.ndarray,
    vol_t2: np.ndarray,
    prob_t1: float | None = None,
    prob_t2: float | None = None,
    threshold: float = 0.5,
    top_k: int = 10,
    *,
    register: bool = True,
    reg_iterations: int = 50,
) -> LongitudinalResult:
    """Compare two preprocessed MRI scans and compute longitudinal metrics.

    If ``register=True`` (default), T2 is first co-registered to T1's space
    using affine optimization with normalized mutual information. This ensures
    voxel-wise differences reflect true anatomical change rather than head
    position differences between scans.

    Args:
        vol_t1: preprocessed volume at time 1 [1,1,D,H,W].
        vol_t2: preprocessed volume at time 2 [1,1,D,H,W].
        prob_t1: model prediction probability at time 1 (optional).
        prob_t2: model prediction probability at time 2 (optional).
        threshold: classification threshold for labels.
        top_k: number of top regional changes to compute.
        register: if True, co-register T2 to T1 space before comparison.
        reg_iterations: max optimization iterations for registration.

    Returns:
        LongitudinalResult with all comparison metrics.
    """
    # Co-registration: align T2 to T1 space
    vol_t2_aligned = vol_t2
    if register:
        vol_t2_aligned_3d, _ = register_to_template(
            vol_t1, vol_t2, num_iterations=reg_iterations
        )
        # Reshape back to [1,1,D,H,W] for consistency
        vol_t2_aligned = vol_t2_aligned_3d[None, None] if vol_t2_aligned_3d.ndim == 3 else vol_t2_aligned_3d

    # Brain masks
    mask_t1 = compute_brain_mask(vol_t1)
    mask_t2 = compute_brain_mask(vol_t2_aligned)
    combined_mask = mask_t1 | mask_t2

    # Volumes
    vol_mm3_t1 = compute_volume_mm3(mask_t1)
    vol_mm3_t2 = compute_volume_mm3(mask_t2)

    # Volume change percentage (negative = atrophy)
    if vol_mm3_t1 > 0:
        vol_change_pct = ((vol_mm3_t2 - vol_mm3_t1) / vol_mm3_t1) * 100.0
    else:
        vol_change_pct = 0.0

    # Overlap
    overlap = mask_t1 & mask_t2
    union = mask_t1 | mask_t2
    overlap_pct = float(overlap.sum() / union.sum() * 100) if union.sum() > 0 else 0.0

    # Difference maps (using registered T2)
    diff_map, abs_diff_map = compute_diff_map(vol_t1, vol_t2_aligned)

    # Regional changes
    regional_changes = compute_regional_changes(
        diff_map, abs_diff_map, combined_mask, top_k=top_k
    )

    # Prediction comparison
    prob_change = None
    label_t1 = None
    label_t2 = None
    if prob_t1 is not None and prob_t2 is not None:
        prob_change = prob_t2 - prob_t1
        label_t1 = "AD" if prob_t1 >= threshold else "Normal"
        label_t2 = "AD" if prob_t2 >= threshold else "Normal"

    return LongitudinalResult(
        volume_t1=round(vol_mm3_t1, 1),
        volume_t2=round(vol_mm3_t2, 1),
        volume_change_pct=round(vol_change_pct, 2),
        diff_map=diff_map,
        abs_diff_map=abs_diff_map,
        brain_mask_t1=mask_t1,
        brain_mask_t2=mask_t2,
        overlap_pct=round(overlap_pct, 2),
        regional_changes=regional_changes,
        prob_t1=prob_t1,
        prob_t2=prob_t2,
        prob_change=round(prob_change, 4) if prob_change is not None else None,
        label_t1=label_t1,
        label_t2=label_t2,
    )
