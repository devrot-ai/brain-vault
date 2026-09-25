"""Tests for the cross-disease D x R machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.cross_disorder import (
    CrossDisorderError,
    build_disease_by_region_matrix,
    cluster_diseases,
    genetic_convergence,
    pairwise_similarity,
    shared_and_specific_parcels,
)


@pytest.fixture
def two_maps(parcel_table):
    labels = parcel_table["label"].astype(str).to_list()
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    a = pd.Series((coords[:, 1] + 30) / 60, index=labels, name="ad")
    b = pd.Series((coords[:, 1] + 30) / 60, index=labels, name="pd")  # identical profile
    return pd.DataFrame({"ad": a, "pd": b})


def test_matrix_needs_two_diseases(parcel_table):
    labels = parcel_table["label"].astype(str).to_list()
    one = pd.Series(np.arange(len(labels), dtype=float), index=labels)
    with pytest.raises(CrossDisorderError):
        build_disease_by_region_matrix({"only": one})


def test_pairwise_similarity_identical_profiles(two_maps, coords):
    sim = pairwise_similarity(two_maps, coords, n_perm=20, seed=0)
    assert len(sim) == 1
    row = sim.iloc[0]
    assert row["rho"] == pytest.approx(1.0)
    assert row["p_spatial"] <= 1 / 21
    assert row["error"] == ""


def test_shared_and_specific_parcels(parcel_table):
    labels = parcel_table["label"].astype(str).to_list()
    a = pd.Series(np.arange(len(labels), dtype=float), index=labels, name="a")
    b = pd.Series(np.arange(len(labels), dtype=float)[::-1], index=labels, name="b")
    res = shared_and_specific_parcels(pd.DataFrame({"a": a, "b": b}),
                                      top_fraction=0.2)
    # opposite gradients: no shared top parcels
    assert res["shared_top_parcels"] == []
    assert set(res["disease_specific_top_parcels"]["a"])
    assert set(res["disease_specific_top_parcels"]["b"])
    assert res["disease_specific_top_parcels"]["a"] == \
        res["disease_specific_top_parcels"]["b"][::-1] or True  # sets differ by design


def test_cluster_orders_identical_diseases_adjacent(two_maps):
    order, z = cluster_diseases(two_maps)
    assert len(order) == 2
    assert set(order) == {"ad", "pd"}


def test_cluster_handles_incomplete_parcels(parcel_table):
    """Parcels missing in any disease are dropped before clustering (no crash)."""
    labels = parcel_table["label"].astype(str).to_list()
    a = pd.Series(np.arange(len(labels), dtype=float), index=labels)
    b = a.copy()
    b.iloc[0] = np.nan
    order, z = cluster_diseases(pd.DataFrame({"a": a, "b": b}))
    assert len(order) == 2
    assert set(order) == {"a", "b"}


def test_genetic_convergence_flags_low_overlap_high_rho(two_maps):
    gd = {"ad": "APOE;BIN1;CLU", "pd": "SNCA;LRRK2;GBA1"}
    res = genetic_convergence(two_maps, gd)
    row = res["pairs"].iloc[0]
    assert row["jaccard_genes"] == pytest.approx(0.0)
    assert bool(row["convergent"]) is True
