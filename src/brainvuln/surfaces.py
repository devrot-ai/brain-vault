"""Cortical-surface assets for spin-permutation nulls.

Loads abagen's shipped Desikan-Killiany surface parcellation and fsaverage5
spheres, and emits per-vertex parcel labels keyed to the union-atlas cortical
labels. The union-atlas info table carries ``surface_dk_id`` (the DK parcel id
per cortical parcel); DK label GIFTIs carry that id per vertex, giving the
vertex -> parcel map the spin test needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SurfaceAssetError(RuntimeError):
    """Raised when surface assets cannot be prepared."""


def load_fsaverage_spin_assets(parcel_table: pd.DataFrame) -> dict:
    """Load DK surface labels + fsaverage5 sphere vertices for spin tests.

    Returns ``{"verts_lh", "verts_rh", "labels_lh", "labels_rh"}`` where
    labels are union-atlas cortical parcel names (or "" for vertices outside
    the parcellation, e.g. medial wall) and vertices are fsaverage5 sphere
    coordinates in the same row order.
    """
    try:
        import abagen
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SurfaceAssetError(
            "abagen + nibabel are required for surface spin nulls "
            "(pip install brainvuln[atlas])"
        ) from exc

    cortex = parcel_table[(parcel_table["structure"] == "cortex")
                          & (parcel_table["surface_dk_id"] > 0)].copy()
    if cortex.empty:
        raise SurfaceAssetError("atlas has no cortical parcels with surface_dk_id")
    # union label -> DK region name (strip the '_L'/'_R' hemisphere suffix)
    name_to_label_by_hemi: dict[str, dict[str, str]] = {"L": {}, "R": {}}
    for _, row in cortex.iterrows():
        hemi_letter = "L" if str(row["hemisphere"]).upper().startswith("L") else "R"
        name = str(row["label"]).rsplit("_", 1)[0]
        name_to_label_by_hemi[hemi_letter][name] = str(row["label"])

    dk = abagen.datasets.fetch_desikan_killiany(surface=True)
    fs = abagen.datasets.fetch_fsaverage5(load=True)

    assets: dict = {}
    for hemi_idx, hemi in enumerate(("lh", "rh")):
        letter = "L" if hemi == "lh" else "R"
        lab_gii = nib.load(dk["image"][hemi_idx])
        vertex_ids = np.asarray(lab_gii.darrays[0].data).astype(int)
        # id -> union label: prefer the GIFTI labeltable region names, which
        # stay correct even when the rh uses global FreeSurfer ids (42-75)
        # while the lh uses per-hemisphere ids (1-34).
        labeltable = (lab_gii.labeltable.get_labels_as_dict()
                      if lab_gii.labeltable else {})
        id_to_label: dict[int, str] = {}
        for vid, vname in labeltable.items():
            if int(vid) == 0 or str(vname).lower() in ("unknown", "??"):
                continue
            union = name_to_label_by_hemi[letter].get(str(vname))
            if union is not None:
                id_to_label[int(vid)] = union
        if not id_to_label:
            # numeric fallback: per-hemisphere DK ids (lh convention)
            hemi_rows = cortex[cortex["hemisphere"].str.upper().str.startswith(letter)]
            id_to_label = {int(i): str(lbl) for i, lbl in
                           zip(hemi_rows["surface_dk_id"], hemi_rows["label"])}
        labels = np.array([id_to_label.get(int(v), "") for v in vertex_ids],
                          dtype=object)
        n_mapped = len(set(labels[labels != ""]))
        if not 20 <= n_mapped <= 40:
            raise SurfaceAssetError(
                f"unexpected DK parcel mapping for {hemi}: {n_mapped} parcels "
                "(id convention mismatch between label.gii and atlas info?)"
            )
        assets[f"verts_{hemi}"] = np.asarray(
            getattr(fs, hemi).vertices, dtype=float)
        assets[f"labels_{hemi}"] = labels
    return assets
