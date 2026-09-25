"""Tests for the external disease-map import layer (offline, synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.disease_maps import (
    DiseaseMapImportError,
    import_disease_map,
    load_parcellated_csv,
    load_provenance,
    make_provenance,
    parcellate_nifti,
    save_provenance,
    validate_disease_map,
)

nib = pytest.importorskip("nibabel")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_atlas(tmp_path):
    """4 parcels of 2x2x2 voxels in a 4x4x4 grid; ids 1..4."""
    atlas = np.zeros((4, 4, 4), dtype=np.int16)
    order = [(slice(0, 2), slice(0, 2), slice(0, 2)),
             (slice(0, 2), slice(2, 4), slice(0, 2)),
             (slice(2, 4), slice(0, 2), slice(0, 2)),
             (slice(2, 4), slice(2, 4), slice(2, 4))]
    for i, sl in enumerate(order, start=1):
        atlas[sl] = i
    affine = np.eye(4)
    img = nib.Nifti1Image(atlas, affine)
    img_path = tmp_path / "atlas.nii.gz"
    nib.save(img, str(img_path))
    info = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "label": ["P1", "P2", "P3", "P4"],
        "structure": ["cortex", "cortex", "subcortex/brainstem",
                      "subcortex/brainstem"],
    })
    info_path = tmp_path / "atlas_info.csv"
    info.to_csv(info_path, index=False)
    return img_path, info_path, order


def _write_map(tmp_path, data, affine=None, name="map.nii.gz"):
    affine = np.eye(4) if affine is None else affine
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    path = tmp_path / name
    nib.save(img, str(path))
    return path


_PROV = dict(
    name="fixture_map",
    disease="alzheimer",
    modality="test",
    sign_convention="higher = more",
    citation="Fixture et al. 2026",
    source_url="https://example.org/map",
    raw_file="raw.nii.gz",
)


# ---------------------------------------------------------------------------
# parcellation
# ---------------------------------------------------------------------------


def test_parcellate_same_grid_recovers_planted_means(tmp_path):
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    data = np.zeros((4, 4, 4))
    for pid, val in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
        data[atlas_view := np.asarray(nib.load(str(atlas_path)).dataobj)
             == pid] = val
    map_path = _write_map(tmp_path, data)
    series, qc, _ = parcellate_nifti(map_path, atlas_path, info_path)
    assert series["P1"] == pytest.approx(10.0)
    assert series["P2"] == pytest.approx(20.0)
    assert series["P3"] == pytest.approx(30.0)
    assert series["P4"] == pytest.approx(40.0)
    assert (qc["n_voxels_used"] == 8).all()


def test_mask_zero_excludes_outside_brain(tmp_path):
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    atlas = np.asarray(nib.load(str(atlas_path)).dataobj)
    data = np.full((4, 4, 4), 100.0)
    # half of parcel 2's voxels are exact 0 (outside the "brain mask")
    p2 = atlas == 2
    idx = np.argwhere(p2)[:4]
    data[tuple(idx.T)] = 0.0
    map_path = _write_map(tmp_path, data)
    series, _, notes = parcellate_nifti(map_path, atlas_path, info_path,
                                        mask_zero=True)
    assert series["P2"] == pytest.approx(100.0)  # zeros excluded
    assert notes["n_voxels_masked_zero"] == 4
    series_raw, _, _ = parcellate_nifti(map_path, atlas_path, info_path,
                                        mask_zero=False)
    assert series_raw["P2"] == pytest.approx(50.0)  # zeros included


def test_resamples_foreign_grid_constant_map(tmp_path):
    """A constant map on a 2x finer grid must parcellate to that constant."""
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    atlas_img = nib.load(str(atlas_path))
    fine = nib.Nifti1Image(np.full((8, 8, 8), 7.0, dtype=np.float32),
                           atlas_img.affine @ np.diag(
                               [0.5, 0.5, 0.5, 1.0]))
    map_path = _write_map(tmp_path, np.asanyarray(fine.dataobj),
                          affine=fine.affine, name="fine.nii.gz")
    series, _, _ = parcellate_nifti(map_path, atlas_path, info_path)
    assert series.notna().all()
    assert np.allclose(series.to_numpy(dtype=float), 7.0)


def test_out_of_fov_parcels_stay_nan(tmp_path):
    """Parcels outside the map's FOV must stay NaN (never filled)."""
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    atlas_img = nib.load(str(atlas_path))
    # map grid covers only half of the atlas in y (shifted affine)
    data = np.full((4, 2, 4), 5.0, dtype=np.float32)
    affine = atlas_img.affine.copy()
    affine[1, 3] += 2.0  # start two voxels further along y
    map_path = _write_map(tmp_path, data, affine=affine)
    series, _, _ = parcellate_nifti(map_path, atlas_path, info_path)
    # map covers world y in [2, 4): parcels P2/P4 (atlas y 2..3) get values;
    # P1/P3 (atlas y 0..1) are outside the FOV and stay NaN
    assert np.isnan(series["P1"]) and np.isnan(series["P3"])
    assert series["P2"] == pytest.approx(5.0)
    assert series["P4"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def test_csv_loader_flexible_columns(tmp_path):
    p = tmp_path / "regions.csv"
    p.write_text("Region,Effect\nP1,1.5\nP2,-2\n")
    s = load_parcellated_csv(p)
    assert s["P1"] == pytest.approx(1.5)
    assert s["P2"] == pytest.approx(-2.0)


def test_csv_loader_ambiguous_numeric_raises(tmp_path):
    p = tmp_path / "regions.csv"
    p.write_text("label,a,b\nP1,1,2\n")
    with pytest.raises(DiseaseMapImportError, match="ambiguous"):
        load_parcellated_csv(p)


def test_csv_loader_duplicate_labels_raise(tmp_path):
    p = tmp_path / "regions.csv"
    p.write_text("label,value\nP1,1\nP1,2\n")
    with pytest.raises(DiseaseMapImportError, match="duplicate"):
        load_parcellated_csv(p)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_provenance_requires_citation_and_sign():
    bad = dict(_PROV, citation="")
    with pytest.raises(DiseaseMapImportError, match="citation"):
        make_provenance(**bad)
    bad2 = dict(_PROV, sign_convention="")
    with pytest.raises(DiseaseMapImportError, match="sign_convention"):
        make_provenance(**bad2)


def test_provenance_round_trip(tmp_path):
    prov = make_provenance(**_PROV)
    path = save_provenance(prov, tmp_path / "p.json")
    back = load_provenance(path)
    assert back["name"] == "fixture_map"
    assert back["citation"] == "Fixture et al. 2026"


# ---------------------------------------------------------------------------
# validation + end-to-end import
# ---------------------------------------------------------------------------


def test_validate_disease_map_reports_missing_by_structure(tmp_path):
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    series = pd.Series({"P1": 1.0, "P2": np.nan, "P3": 3.0, "P4": np.nan})
    info = pd.read_csv(info_path)
    rep = validate_disease_map(series, info, name="t")
    assert rep["n_parcels_with_value"] == 2
    assert rep["missing_by_structure"] == {"cortex": 1,
                                           "subcortex/brainstem": 1}
    assert rep["n_subcortical_with_value"] == 1


def test_import_disease_map_writes_outputs(tmp_path):
    atlas_path, info_path, _ = _make_atlas(tmp_path)
    data = np.zeros((4, 4, 4))
    atlas = np.asarray(nib.load(str(atlas_path)).dataobj)
    for pid, val in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        data[atlas == pid] = val
    map_path = _write_map(tmp_path, data)
    out_dir = tmp_path / "imported"
    series, prov, coverage = import_disease_map(
        map_path, provenance=make_provenance(**_PROV),
        atlas_img_path=atlas_path, atlas_info_path=info_path,
        out_dir=out_dir,
    )
    out_csv = out_dir / "fixture_map.csv"
    out_json = out_dir / "fixture_map.provenance.json"
    assert out_csv.exists() and out_json.exists()
    reloaded = load_parcellated_csv(out_csv)
    assert reloaded["P4"] == pytest.approx(4.0)
    assert prov["coverage"]["n_parcels_with_value"] == 4
    assert coverage["coverage_fraction"] == 1.0
