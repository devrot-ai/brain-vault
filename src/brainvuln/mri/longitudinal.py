"""Longitudinal (time-series) comparison of two MRI scans from the same subject.

Given two preprocessed 128³ volumes from the same patient at different time
points, this module computes:

1. **Voxel-wise difference map** — where has intensity changed between scans?
2. **Brain volume change** — percentage of brain tissue lost or gained.
3. **Regional atrophy map** — which atlas regions show the most change?
4. **3D change visualization** — a mesh colored by the magnitude and direction
   of change (atrophy = red, growth = blue).

This is designed for Alzheimer's progression tracking: a patient scanned at
baseline (Time 1) and follow-up (Time 2) can see where their brain has
shrunk over time.

Note: For accurate comparison, both scans should go through the SAME
preprocessing pipeline (RAS reorient → 2mm → crop/pad 128³ → brain z-score).
Since both scans are already z-scored independently, the difference map
reflects relative intensity changes within each scan's own brain distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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
) -> LongitudinalResult:
    """Compare two preprocessed MRI scans and compute longitudinal metrics.

    Args:
        vol_t1: preprocessed volume at time 1 [1,1,D,H,W].
        vol_t2: preprocessed volume at time 2 [1,1,D,H,W].
        prob_t1: model prediction probability at time 1 (optional).
        prob_t2: model prediction probability at time 2 (optional).
        threshold: classification threshold for labels.
        top_k: number of top regional changes to compute.

    Returns:
        LongitudinalResult with all comparison metrics.
    """
    # Brain masks
    mask_t1 = compute_brain_mask(vol_t1)
    mask_t2 = compute_brain_mask(vol_t2)
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

    # Difference maps
    diff_map, abs_diff_map = compute_diff_map(vol_t1, vol_t2)

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
