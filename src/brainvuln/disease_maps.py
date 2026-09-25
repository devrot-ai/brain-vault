"""External disease-vulnerability maps: import, parcellate, and document.

Public neuroimaging maps (PET, atrophy, group statistical maps) are almost
never distributed on a custom parcellation grid. This module converts an
arbitrary MNI-space NIfTI map (or an already-parcellated CSV) into the
BrainVuln parcellated-map format:

    <out_dir>/<name>.csv                 one row per parcel: label, value
    <out_dir>/<name>.provenance.json     full machine-readable provenance

Design rules (mirroring the abagen philosophy of this project):

* Never invent values. Parcels with no in-map voxels stay NaN and are
  reported, never filled.
* Masks are explicit. Many group maps store exact 0.0 outside the brain /
  gray-matter mask; treating those voxels as data would bias parcel means.
  ``mask_zero=True`` (the default) excludes them, and this choice is
  recorded in the provenance.
* Resampling is one-directional and documented: the continuous map is
  trilinearly resampled INTO the atlas grid; the parcellation itself is
  never interpolated.
* Sign conventions are recorded, never assumed.

Verified public sources (see data/external/disease_maps/README.md):

* Alzheimer — La Joie et al. 2020 (Sci Transl Med), NeuroVault collection 3829
* Parkinson — Zeighami et al. 2015 (eLife), NeuroVault collection 860, IC 0007
* Huntington — no public map exists (all cohorts controlled access); the
  CSV path supports literature-derived region tables with per-source
  citation in the provenance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "DiseaseMapImportError",
    "PROVENANCE_REQUIRED",
    "make_provenance",
    "save_provenance",
    "load_provenance",
    "load_atlas_parcellation",
    "parcellate_nifti",
    "load_parcellated_csv",
    "validate_disease_map",
    "import_disease_map",
]


class DiseaseMapImportError(RuntimeError):
    """Raised when a disease map cannot be imported or validated."""


# Fields that every imported map's provenance must carry. ``citation`` is
# non-negotiable: an unattributable map must not enter the analysis.
PROVENANCE_REQUIRED = (
    "name",
    "disease",
    "modality",
    "space",
    "sign_convention",
    "citation",
    "source_url",
    "license_note",
    "download_date",
    "raw_file",
)

_DEFAULT_LICENSE_NOTE = (
    "No explicit license set by the uploader on NeuroVault; use for research "
    "with full citation of the paper DOI and the NeuroVault persistent "
    "identifier, and contact the corresponding author before redistribution."
)


def make_provenance(
    *,
    name: str,
    disease: str,
    modality: str,
    sign_convention: str,
    citation: str,
    source_url: str = "",
    space: str = "MNI152",
    license_note: str = _DEFAULT_LICENSE_NOTE,
    download_date: Optional[str] = None,
    raw_file: str = "",
    n_subjects: Optional[int] = None,
    description: str = "",
    persistent_id: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> dict:
    """Build a provenance record, enforcing the required fields."""
    prov = {
        "name": name,
        "disease": disease,
        "modality": modality,
        "space": space,
        "sign_convention": sign_convention,
        "citation": citation,
        "source_url": source_url,
        "license_note": license_note,
        "download_date": download_date
        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "raw_file": raw_file,
        "n_subjects": n_subjects,
        "description": description,
        "persistent_id": persistent_id,
    }
    if extra:
        prov.update(extra)
    missing = [f for f in PROVENANCE_REQUIRED if not prov.get(f)]
    if missing:
        raise DiseaseMapImportError(
            f"provenance missing required fields: {missing}"
        )
    return prov


def save_provenance(prov: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    return path


def load_provenance(path: str | Path) -> dict:
    prov = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [f for f in PROVENANCE_REQUIRED if f not in prov]
    if missing:
        raise DiseaseMapImportError(
            f"{path}: provenance missing required fields {missing}"
        )
    return prov


# ---------------------------------------------------------------------------
# atlas + parcellation
# ---------------------------------------------------------------------------


def load_atlas_parcellation(
    atlas_img_path: str | Path, atlas_info_path: str | Path
) -> tuple[Any, np.ndarray, list[str]]:
    """Load the parcellation NIfTI and its info table.

    Returns ``(nibabel image, integer parcel ids array, label strings)``;
    the ids and labels are aligned by row with ``atlas_info``.
    """
    import nibabel as nib

    atlas_img = nib.load(str(atlas_img_path))
    info = pd.read_csv(atlas_info_path)
    adata = np.asanyarray(atlas_img.dataobj).astype(int)
    present = set(np.unique(adata).tolist()) - {0}
    ids = info["id"].to_numpy()
    labels = info["label"].astype(str).tolist()
    absent = [int(i) for i in ids if int(i) not in present]
    if absent:
        raise DiseaseMapImportError(
            f"atlas info ids absent from image {atlas_img_path}: {absent[:8]}"
        )
    return atlas_img, ids, labels


def parcellate_nifti(
    map_path: str | Path,
    atlas_img_path: str | Path,
    atlas_info_path: str | Path,
    *,
    mask_zero: bool = True,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Resample a continuous map onto the atlas grid and average per parcel.

    The map is trilinearly resampled into the atlas grid (order=1, NaN
    outside the map's field of view); the parcellation itself is never
    interpolated. Parcel value = mean of the (optionally zero-masked) map
    over the parcel's voxels. Parcels with no usable voxels stay NaN.

    Returns ``(values indexed by label, per-parcel voxel QC table,
    import-notes dict)``.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    map_img = nib.load(str(map_path))
    atlas_img, ids, labels = load_atlas_parcellation(atlas_img_path,
                                                     atlas_info_path)
    mdata_r = resample_from_to(
        map_img, (atlas_img.shape, atlas_img.affine), order=1, cval=np.nan
    )
    mdata = np.asanyarray(mdata_r.dataobj, dtype=float)
    adata = np.asanyarray(atlas_img.dataobj).astype(int)

    brain = np.isfinite(mdata)
    n_masked_zero = 0
    if mask_zero:
        zero_voxels = brain & (mdata == 0)
        n_masked_zero = int(zero_voxels.sum())
        brain &= ~zero_voxels

    values = np.full(len(ids), np.nan)
    n_used = np.zeros(len(ids), dtype=int)
    for i, pid in enumerate(ids):
        mask = (adata == int(pid)) & brain
        n = int(mask.sum())
        if n:
            values[i] = float(np.nanmean(mdata[mask]))
            n_used[i] = n

    qc = pd.DataFrame(
        {
            "label": labels,
            "id": ids,
            "n_voxels_used": n_used,
            "value": values,
        }
    )
    notes = {
        "resampling": "trilinear (order=1) into atlas grid, cval=NaN",
        "mask_zero": bool(mask_zero),
        "n_voxels_masked_zero": n_masked_zero,
        "n_voxels_outside_map_fov": int((~np.isfinite(mdata)).sum()),
        "map_grid": list(map_img.shape),
        "atlas_grid": list(atlas_img.shape),
    }
    return pd.Series(values, index=labels, name=Path(map_path).stem), qc, notes


# ---------------------------------------------------------------------------
# already-parcellated input (e.g. literature-derived region tables)
# ---------------------------------------------------------------------------


def load_parcellated_csv(
    path: str | Path,
    *,
    value_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.Series:
    """Load a parcellated map CSV with flexible column names.

    ``label_col`` defaults to the first of (label, region, parcel, structure)
    found; ``value_col`` defaults to the explicitly named column or the only
    remaining numeric column.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if label_col is None:
        for cand in ("label", "region", "parcel", "structure", "name"):
            if cand in df.columns:
                label_col = cand
                break
    if label_col is None:
        raise DiseaseMapImportError(
            f"{path}: no label column (tried label/region/parcel/structure/name)"
        )
    if value_col is not None:
        value_col = value_col.strip().lower()
        if value_col not in df.columns:
            raise DiseaseMapImportError(
                f"{path}: value column {value_col!r} not found"
            )
    else:
        numeric = [
            c for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric:
            raise DiseaseMapImportError(f"{path}: no numeric value column")
        if len(numeric) > 1:
            raise DiseaseMapImportError(
                f"{path}: ambiguous value columns {numeric}; pass value_col="
            )
        value_col = numeric[0]

    labels = df[label_col].astype(str).str.strip()
    values = pd.to_numeric(df[value_col], errors="coerce")
    if labels.duplicated().any():
        dupes = sorted(set(labels[labels.duplicated()]))
        raise DiseaseMapImportError(f"{path}: duplicate parcel labels {dupes}")
    return pd.Series(values.to_numpy(), index=labels, name=Path(path).stem)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_disease_map(
    series: pd.Series, atlas_info: pd.DataFrame, *, name: str = "map"
) -> dict:
    """Coverage report for a parcellated map against the atlas.

    Reports missing parcels per structure class. Missing parcels are
    legitimate (e.g. a cortical-only map on a cortex+subcortex atlas) but
    must be visible, because they change the correlation's effective n and
    which spatial nulls are appropriate.
    """
    info = atlas_info.copy()
    info["label"] = info["label"].astype(str)
    merged = info.set_index("label").reindex(series.index)
    present = series.notna()
    report = {
        "map": name,
        "n_parcels_atlas": int(len(merged)),
        "n_parcels_with_value": int(present.sum()),
        "coverage_fraction": float(present.mean()) if len(merged) else np.nan,
        "missing_by_structure": (
            merged.loc[~present.values, "structure"].value_counts().to_dict()
            if (~present).any()
            else {}
        ),
        "missing_labels": sorted(series.index[~present].tolist()),
    }
    sub = merged["structure"].eq("subcortex/brainstem") & present.values
    report["n_subcortical_with_value"] = int(sub.sum())
    return report


# ---------------------------------------------------------------------------
# high-level import
# ---------------------------------------------------------------------------


def import_disease_map(
    map_path: str | Path,
    *,
    provenance: dict,
    atlas_img_path: str | Path,
    atlas_info_path: str | Path,
    out_dir: str | Path,
    source_format: str = "nifti",
    mask_zero: bool = True,
    value_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> tuple[pd.Series, dict, dict]:
    """Import one external map into the BrainVuln parcellated format.

    Writes ``<out_dir>/<name>.csv`` and ``<out_dir>/<name>.provenance.json``.
    Returns ``(parcellated series, provenance dict incl. import/QC blocks,
    coverage report)``.
    """
    map_path = Path(map_path)
    if not map_path.exists():
        raise DiseaseMapImportError(f"map file not found: {map_path}")
    prov = dict(provenance)  # do not mutate the caller's record

    if source_format == "nifti":
        series, qc, notes = parcellate_nifti(
            map_path, atlas_img_path, atlas_info_path, mask_zero=mask_zero
        )
        prov["import"] = {
            "format": "nifti",
            "mask_zero": bool(mask_zero),
            **notes,
        }
    elif source_format == "csv":
        series = load_parcellated_csv(map_path, value_col=value_col,
                                      label_col=label_col)
        prov["import"] = {"format": "csv"}
    else:
        raise DiseaseMapImportError(
            f"unknown source_format {source_format!r} (nifti|csv)"
        )

    atlas_info = pd.read_csv(atlas_info_path)
    coverage = validate_disease_map(series, atlas_info, name=prov["name"])
    prov["coverage"] = {
        k: v for k, v in coverage.items() if k != "missing_labels"
    }
    prov["import"]["map_file"] = str(map_path.resolve())
    prov["imported_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{prov['name']}.csv"
    out_csv.write_text(
        "label,value\n"
        + "\n".join(f"{lbl},{val}" for lbl, val in series.items())
        + "\n",
        encoding="utf-8",
    )
    save_provenance(prov, out_dir / f"{prov['name']}.provenance.json")
    return series, prov, coverage
