"""Tests for disease-map loading, synthetic maps, and donor LOOCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.validation import (
    DiseaseMapError,
    donor_loocv,
    generate_synthetic_map,
    load_disease_map_csv,
    score_method_sensitivity,
)


def test_load_csv_matches_by_label_case_insensitive(tmp_path, parcel_table):
    labels = parcel_table["label"].tolist()
    df = pd.DataFrame({
        "label": [l.lower() for l in labels],
        "vulnerability": np.arange(len(labels), dtype=float),
    })
    p = tmp_path / "map.csv"
    df.to_csv(p, index=False)
    series, prov = load_disease_map_csv(p, labels)
    assert prov["synthetic"] is False
    assert prov["n_matched_parcels"] == len(labels)
    assert series.notna().all()
    assert series.iloc[0] < series.iloc[-1]


def test_load_csv_partial_match_reindexes_with_nan(tmp_path, parcel_table):
    labels = parcel_table["label"].tolist()
    df = pd.DataFrame({"label": labels[:10], "value": np.arange(10.0)})
    p = tmp_path / "partial.csv"
    df.to_csv(p, index=False)
    series, prov = load_disease_map_csv(p, labels)
    assert series.notna().sum() == 10
    assert prov["n_matched_parcels"] == 10


def test_load_csv_no_match_raises(tmp_path, parcel_table):
    df = pd.DataFrame({"label": ["NoSuchRegion1", "NoSuchRegion2"], "value": [1, 2]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(DiseaseMapError, match="labels matched"):
        load_disease_map_csv(p, parcel_table["label"].tolist())


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(DiseaseMapError, match="not found"):
        load_disease_map_csv(tmp_path / "nope.csv", ["A"])


def test_synthetic_map_deterministic_and_normalized(parcel_table):
    pt = parcel_table.copy()
    s1, prov1 = generate_synthetic_map(pt, "anterior_medial_temporal", seed=42)
    s2, prov2 = generate_synthetic_map(pt, "anterior_medial_temporal", seed=42)
    pd.testing.assert_series_equal(s1, s2)
    assert prov1["synthetic"] is True
    assert "SYNTHETIC" in prov1["warning"]
    assert s1.max() == pytest.approx(1.0)
    assert (s1 >= 0).all()


def test_synthetic_map_profiles_differ(parcel_table):
    a, _ = generate_synthetic_map(parcel_table, "anterior_medial_temporal")
    b, _ = generate_synthetic_map(parcel_table, "striatal")
    assert np.corrcoef(a, b)[0, 1] < 0.9


def test_donor_loocv_recovers_signal(parcel_table, expr):
    """A map built from gene expression is recovered with high r in every fold."""
    rng = np.random.default_rng(0)
    genes = sorted(expr.columns[:15])
    # plant the disease map as a weighted mean of the gene set + small noise
    signal = expr[genes].mean(axis=1)
    disease_map = signal + rng.normal(0, 0.01, len(signal))
    rows = []
    base = expr.to_numpy()
    for d in range(6):
        noisy = base + rng.normal(0, 0.02, size=base.shape)
        wide = pd.DataFrame(noisy, index=expr.index, columns=expr.columns)
        for label in wide.index:
            for g in genes:
                rows.append({"donor": f"D{d}", "label": label, "gene": g,
                             "value": float(wide.loc[label, g])})
    donor_long = pd.DataFrame(rows)
    table, summary = donor_loocv(
        donor_long, genes, disease_map,
        lambda df: df.sub(df.mean()).div(df.std(ddof=1)).mean(axis=1),
    )
    assert len(table) == 6
    assert summary["n_donors"] == 6
    assert summary["median_r"] > 0.8  # same underlying signal in every donor


def test_donor_loocv_needs_three_donors(parcel_table, expr, synthetic_disease_map):
    rows = [{"donor": "D0", "label": l, "gene": g, "value": 1.0}
            for l in expr.index for g in expr.columns[:2]]
    with pytest.raises(DiseaseMapError, match=">=3 donors"):
        donor_loocv(pd.DataFrame(rows), list(expr.columns[:2]),
                    synthetic_disease_map, lambda df: df.mean(axis=1))


def test_score_method_sensitivity_returns_all_methods(parcel_table, expr,
                                                      synthetic_disease_map):
    sens = score_method_sensitivity(
        expr, sorted(expr.columns[:10]), ["mean_z", "mean_rank"],
        synthetic_disease_map, "genes", None,
    )
    assert set(sens["method"]) == {"mean_z", "mean_rank"}
    assert sens["r"].notna().all()
    assert (sens["error"] == "").all()
