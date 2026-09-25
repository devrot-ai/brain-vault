"""Tests for spatial nulls, FDR, and concordance (offline, on synthetic maps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.spatial import (
    _null_correlations,
    benjamini_hochberg,
    concordance,
    morans_i,
    run_spatial_null,
    spatial_corr,
    spatial_null_pvalue,
)


def test_spatial_corr_basic(parcel_table, synthetic_disease_map):
    y = synthetic_disease_map.to_numpy()
    assert spatial_corr(y, y) == pytest.approx(1.0)
    assert spatial_corr(y, -y) == pytest.approx(-1.0)


def test_spatial_corr_nan_handling(parcel_table, synthetic_disease_map):
    y = synthetic_disease_map.to_numpy().copy()
    y[:5] = np.nan
    assert np.isfinite(spatial_corr(y, y))


def test_spatial_null_pvalue_formula():
    null = np.array([-0.4, -0.2, 0.1, 0.3, 0.5])
    assert spatial_null_pvalue(0.5, null) == pytest.approx(2 / 6)
    assert spatial_null_pvalue(-0.5, null) == pytest.approx(2 / 6)
    assert spatial_null_pvalue(0.9, null) == pytest.approx(1 / 6)
    assert spatial_null_pvalue(0.0, null) == pytest.approx(1.0)


def test_bh_fdr_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.9])
    q = benjamini_hochberg(p)
    # manual BH: sorted p [0.005,0.01,0.03,0.04,0.9] -> q [0.025,0.025,0.05,0.05,0.9]
    assert q[3] == pytest.approx(0.025)
    assert q[0] == pytest.approx(0.025)
    assert q[2] == pytest.approx(0.05)
    assert q[1] == pytest.approx(0.05)
    assert q[4] == pytest.approx(0.9)
    q2 = benjamini_hochberg(np.array([0.5, 0.5, 0.5]))
    assert np.allclose(q2, 0.5)


def test_morans_i_positive_for_smooth_field(parcel_table, coords):
    rng = np.random.default_rng(3)
    n = len(parcel_table)
    # smooth field: sum of a few broad bumps
    cc = coords.to_numpy()
    field = np.zeros(n)
    for _ in range(5):
        c = rng.uniform(-30, 30, size=3)
        field += np.exp(-((cc - c) ** 2).sum(axis=1) / (2 * 25.0**2))
    noisy = field + rng.normal(0, 0.3, size=n)  # noise ~ field sd
    i_smooth = morans_i(field, cc)
    i_noisy = morans_i(noisy, cc)
    # inverse-distance W is diffuse by design (aligned with the Moran null),
    # so I is diluted; the meaningful assertion is the ordering.
    assert i_smooth > 0.1
    assert i_noisy < i_smooth  # noise destroys spatial autocorrelation


def test_moran_surrogates_preserve_autocorrelation(parcel_table, coords):
    """Moran surrogates keep I close to observed, measured on the SAME weights
    matrix the null model fits (inverse distance, self-weight 1; as in
    neuromaps.nulls.moran -> MoranRandomization)."""
    from scipy.spatial.distance import cdist

    from brainvuln.spatial import _moran_surrogates

    cc = coords.to_numpy()
    rng = np.random.default_rng(11)
    n = len(parcel_table)
    field = np.zeros(n)
    for _ in range(4):
        c = rng.uniform(-30, 30, size=3)
        field += np.exp(-((cc - c) ** 2).sum(axis=1) / (2 * 30.0**2))

    w = cdist(cc, cc).astype("float64")
    np.fill_diagonal(w, 1)
    w **= -1

    def i_on_surrogate_w(xv):
        z = xv - xv.mean()
        return float((z @ w @ z) / (z @ z))

    i_obs = i_on_surrogate_w(field)
    surrogates = _moran_surrogates(field, cc, n_perm=20, seed=11)
    i_null = np.array([i_on_surrogate_w(surrogates[i]) for i in range(20)])
    assert np.median(i_null) == pytest.approx(i_obs, abs=0.2 * max(abs(i_obs), 0.1))


def test_null_correlations_vectorized_matches_per_row():
    rng = np.random.default_rng(5)
    y = rng.normal(size=50)
    nulls = rng.normal(size=(40, 50))
    vec = _null_correlations(y, nulls)
    from scipy import stats

    per_row = np.array([stats.spearmanr(y, nulls[i]).statistic for i in range(40)])
    assert np.allclose(vec, per_row, atol=1e-8)


def test_run_spatial_null_identical_maps_significant(parcel_table, coords,
                                                     synthetic_disease_map):
    res = run_spatial_null(synthetic_disease_map, synthetic_disease_map, coords,
                           method="moran", n_perm=100, seed=1)
    assert res["r_obs"] == pytest.approx(1.0)
    assert res["p_spatial"] <= 1 / 101


def test_run_spatial_null_uncorrelated_not_significant(parcel_table, coords):
    rng = np.random.default_rng(9)
    labels = parcel_table["label"].astype(str).to_list()
    a = pd.Series(rng.normal(size=len(labels)), index=labels)
    b = pd.Series(rng.normal(size=len(labels)), index=labels)
    res = run_spatial_null(a, b, coords, method="moran", n_perm=100, seed=2)
    assert res["p_spatial"] > 0.05


def test_concordance_logic():
    robust = {"moran": {"p_spatial": 0.01}, "burt2020": {"p_spatial": 0.001}}
    c = concordance(robust, required=2)
    assert c["robust"] and c["n_methods_supporting"] == 2
    partial = {"moran": {"p_spatial": 0.01}, "burt2020": {"p_spatial": 0.4}}
    c2 = concordance(partial, required=2)
    assert not c2["robust"] and c2["n_methods_supporting"] == 1
    none = {"moran": {"p_spatial": 0.6}, "burt2020": {"p_spatial": 0.7}}
    c3 = concordance(none, required=2)
    assert not c3["robust"]


def test_burt2020_smoke(parcel_table, coords, synthetic_disease_map):
    res = run_spatial_null(synthetic_disease_map, synthetic_disease_map, coords,
                           method="burt2020", n_perm=50, seed=3)
    assert np.isfinite(res["p_spatial"])
