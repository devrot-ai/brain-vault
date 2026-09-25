"""Regional quantification of model attributions (Part L).

Maps per-subject relevance volumes onto the SAME predefined 87-parcel atlas
used for gene expression (DK68 ∪ Tian S1) — no region selection, the full
atlas always — producing:

* per-subject regional means (results/regional_relevance/<subject>.csv)
* the population map ``M_CNN(region)`` = mean across test subjects
* between-subject variability (SD / IQR per region)

Parcellation reuses ``brainvuln.disease_maps.parcellate_nifti`` (trilinear
resample into the atlas grid, parcel means, honest NaN) so CNN relevance and
gene expression live on exactly the same regional grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from brainvuln.disease_maps import parcellate_nifti


def regionalize_cam(
    cam_nifti: str | Path,
    atlas_img: str | Path,
    atlas_info: str | Path,
) -> pd.Series:
    """Per-parcel mean CAM for one subject (NaN = parcel outside CAM FOV)."""
    series, _qc, _notes = parcellate_nifti(
        cam_nifti, atlas_img, atlas_info_path=atlas_info, mask_zero=False
    )
    return series


def aggregate_regional(
    per_subject: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    """Population map M_CNN and its between-subject variability.

    ``per_subject`` maps subject id → regional series (label-indexed).
    Returns ``(table indexed by label with mean/sd/iqr/n columns, M_CNN
    series)``. Regions missing from a subject are ignored per-region
    (nanmean); regions missing from every subject stay NaN.
    """
    df = pd.DataFrame(per_subject).T  # subjects × parcels
    mean = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, ddof=1, skipna=True)
    q1 = df.quantile(0.25, axis=0)
    q3 = df.quantile(0.75, axis=0)
    table = pd.DataFrame({
        "M_CNN": mean,
        "sd": sd,
        "iqr": q3 - q1,
        "n_subjects": df.notna().sum(axis=0).astype(int),
    })
    return table, mean.rename("M_CNN")


def method_agreement(
    cam_regional: pd.Series, occlusion_regional: pd.Series
) -> dict:
    """Part W: spatial agreement between Grad-CAM and occlusion maps."""
    joined = pd.concat([cam_regional.rename("cam"),
                        occlusion_regional.rename("occ")], axis=1).dropna()
    if len(joined) < 5 or joined["cam"].std() == 0 or joined["occ"].std() == 0:
        return {"spearman": float("nan"), "n_regions": len(joined)}
    rho = joined["cam"].corr(joined["occ"], method="spearman")
    return {"spearman": float(rho), "n_regions": int(len(joined))}
