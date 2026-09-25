"""Tests for surface spin-permutation nulls and the fsaverage asset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.spatial import surface_spin_pvalues
from brainvuln.surfaces import SurfaceAssetError, load_fsaverage_spin_assets


def _sphere_vertices(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _parcellated_inputs(n_verts: int = 400, n_parcels: int = 20, seed: int = 1):
    """Synthetic sphere parcellated into contiguous angular patches."""
    verts = _sphere_vertices(n_verts, seed)
    labels = np.empty(n_verts, dtype=object)
    # contiguous patches: assign by first principal angular direction
    theta = np.arctan2(verts[:, 1], verts[:, 0])
    bins = np.linspace(-np.pi, np.pi, n_parcels + 1)
    for i in range(n_parcels):
        m = (theta >= bins[i]) & (theta < bins[i + 1])
        labels[m] = f"parcel_{i:02d}"
    return verts, labels


def test_spin_pvalues_detects_planted_signal():
    verts, labels = _parcellated_inputs()
    parcels = sorted(set(labels))
    # score is high in a spatially coherent half of parcels -> spin-permuting
    # the map should rarely reproduce this alignment
    high = {p for i, p in enumerate(parcels) if i < len(parcels) // 2}
    score = pd.Series([5.0 if p in high else -5.0 for p in parcels], index=parcels)
    disease = pd.Series([2.0 if p in high else -2.0 for p in parcels], index=parcels)

    res = surface_spin_pvalues(score, disease, verts[:, :3], verts[:, :3],
                               labels, labels, n_perm=200, seed=0)
    assert res["method"] == "spin"
    assert res["p_spatial"] <= 0.05
    assert res["n_valid_parcels"] == len(parcels)


def test_spin_survives_unmapped_vertices():
    """Regression: production maps exclude some parcels, so many vertices carry
    labels absent from the score map. Spin indices must live in the FILTERED
    vertex space (this used to raise IndexError when they did not)."""
    verts, labels = _parcellated_inputs()
    # mark a third of vertices as belonging to parcels NOT in the score map
    rng = np.random.default_rng(11)
    unmapped = rng.choice(labels.size, size=labels.size // 3, replace=False)
    labels = labels.copy()
    labels[unmapped] = "ABSENT_PARCEL"
    parcels = sorted(set(labels) - {"ABSENT_PARCEL"})
    score = pd.Series(rng.normal(size=len(parcels)), index=parcels)
    disease = pd.Series(rng.normal(size=len(parcels)), index=parcels)

    res = surface_spin_pvalues(score, disease, verts[:, :3], verts[:, :3],
                               labels, labels, n_perm=50, seed=0)
    assert res["n_valid_parcels"] == len(parcels)
    assert np.isfinite(res["p_spatial"])
    assert len(res["null_r"]) == 50


def test_spin_pvalues_null_map_gives_uniform_p():
    verts, labels = _parcellated_inputs(seed=3)
    parcels = sorted(set(labels))
    rng = np.random.default_rng(7)
    # spatially random score: p should not be systematically tiny
    score = pd.Series(rng.normal(size=len(parcels)), index=parcels)
    disease = pd.Series(rng.normal(size=len(parcels)), index=parcels)
    res = surface_spin_pvalues(score, disease, verts[:, :3], verts[:, :3],
                               labels, labels, n_perm=200, seed=0)
    assert res["p_spatial"] > 0.01  # not "significant" by construction


def test_spin_requires_enough_parcels():
    verts, labels = _parcellated_inputs(n_verts=40, n_parcels=5)
    parcels = sorted(set(labels))
    s = pd.Series(np.arange(len(parcels), dtype=float), index=parcels)
    with pytest.raises(Exception):
        surface_spin_pvalues(s, s, verts[:, :3], verts[:, :3],
                             labels, labels, n_perm=10, seed=0)


def _atlas_info_frame(n_cortical: int = 30) -> pd.DataFrame:
    # use real DK region names so the loader's name-based matching engages;
    # names come from the shipped rh labeltable (abagen is a test dependency
    # via importorskip in the tests that use this fixture)
    import abagen
    import nibabel as nib
    dk = abagen.datasets.fetch_desikan_killiany(surface=True)
    lt = nib.load(dk["image"][1]).labeltable.get_labels_as_dict()
    names = [str(nm) for vid, nm in sorted(lt.items())
             if int(vid) != 0 and str(nm).lower() != "unknown"][:n_cortical]
    assert len(names) == n_cortical
    rows = []
    for i, name in enumerate(names, start=1):
        rows.append({"id": len(rows) + 1, "label": f"{name}_L",
                     "hemisphere": "L", "structure": "cortex",
                     "surface_dk_id": i})
        rows.append({"id": len(rows) + 1, "label": f"{name}_R",
                     "hemisphere": "R", "structure": "cortex",
                     "surface_dk_id": i})
    rows.append({"id": len(rows) + 1, "label": "Tian_1", "hemisphere": "R",
                 "structure": "subcortex/brainstem", "surface_dk_id": -1})
    rows.append({"id": len(rows) + 2, "label": "Tian_2", "hemisphere": "L",
                 "structure": "subcortex/brainstem", "surface_dk_id": -1})
    return pd.DataFrame(rows)


def test_loader_maps_vertices_to_union_labels():
    pytest.importorskip("abagen")
    info = _atlas_info_frame()
    assets = load_fsaverage_spin_assets(info)
    lh = assets["labels_lh"]
    verts = assets["verts_lh"]
    assert lh.shape == verts.shape[:1] == (10242,)
    cortex = info[info.structure == "cortex"]
    n_cortical = cortex.hemisphere.value_counts().min()
    assert len(set(lh) - {""}) == n_cortical
    assert set(assets["labels_rh"]) - {""} == \
        {lbl[:-2] + "_R" for lbl in set(lh) - {""}}
    mapped = lh != ""
    assert mapped.sum() > 0
    # vertices carrying one label are spatially clustered on the sphere:
    # mean resultant length of the patch should be high
    first = sorted(set(lh) - {""})[0]
    pts = verts[lh == first]
    res_len = (np.linalg.norm(pts.mean(axis=0))
               / np.linalg.norm(pts, axis=1).mean())
    assert res_len > 0.5, f"parcel '{first}' not clustered (R={res_len:.2f})"


def test_loader_requires_cortex_column():
    info = _atlas_info_frame()
    info.loc[info.structure == "cortex", "surface_dk_id"] = -1
    with pytest.raises(SurfaceAssetError):
        load_fsaverage_spin_assets(info)
