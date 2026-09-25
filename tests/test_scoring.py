"""Tests for gene-level spatial association and enrichment (secondary analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from brainvuln.scoring import (
    compute_score,
    enrichment_among_top_genes,
    gene_level_correlations,
    zscore_genes,
)


def test_gene_level_correlations_recovers_planted_signal(parcel_table, expr):
    """Genes overwritten with the map's own profile must top the rho table."""
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    rng = np.random.default_rng(4)
    planted = np.exp(-((coords - coords[7]) ** 2).sum(axis=1) / (2 * 20.0**2))
    planted += rng.normal(0, 0.01, size=len(coords))
    expr2 = expr.copy()
    for g in [f"GENE{i:03d}" for i in range(6)]:
        expr2[g] = planted + rng.normal(0, 0.01, size=len(coords))
    dm = pd.Series(planted, index=expr.index, name="map")
    table = gene_level_correlations(expr2, dm)
    assert table.iloc[0]["rho"] > 0.9
    top5 = set(table.head(5)["gene"])
    assert top5 & {f"GENE{i:03d}" for i in range(6)}


def test_gene_level_correlations_sorted_by_abs_rho(parcel_table, expr):
    dm = pd.Series(np.arange(len(expr), dtype=float), index=expr.index)
    table = gene_level_correlations(expr, dm)
    absrho = table["rho"].abs().to_numpy()
    assert (np.diff(absrho) <= 1e-12).all()


def test_enrichment_finds_planted_set(parcel_table, expr):
    """Genes sharing the map's spatial profile rank top; the set is 'enriched'."""
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    target = np.exp(-((coords - coords[30]) ** 2).sum(axis=1) / (2 * 20.0**2))
    dm = pd.Series(target, index=expr.index)
    expr2 = expr.copy()
    for g in ["GENE000", "GENE001", "GENE002", "GENE003"]:
        expr2[g] = target + np.random.default_rng(1).normal(0, 0.02, len(target))
    rhos = gene_level_correlations(expr2, dm)
    res = enrichment_among_top_genes(rhos, ["GENE000", "GENE001", "GENE002", "GENE003"],
                                     n_perm=2000, seed=0)
    assert res["n_in_universe"] == 4
    assert res["p_empirical"] < 0.05
    assert res["mean_rank_percentile"] > 0.75


def test_enrichment_null_set_not_enriched(parcel_table, expr):
    rng = np.random.default_rng(2)
    dm = pd.Series(rng.normal(size=len(expr)), index=expr.index)
    rhos = gene_level_correlations(expr, dm)
    res = enrichment_among_top_genes(rhos, sorted(expr.columns[:5]),
                                     n_perm=2000, seed=0)
    assert res["p_empirical"] > 0.05


def test_zscore_genes_dimensions(expr):
    zg = zscore_genes(expr, "genes")
    assert np.allclose(zg.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(zg.std(axis=0, ddof=1), 1, atol=1e-10)
    zp = zscore_genes(expr, "parcels")
    assert np.allclose(zp.mean(axis=1), 0, atol=1e-10)


def test_first_pc_aligned_to_mean_z(parcel_table, expr):
    """With a dominant shared component, PC1 must be sign-aligned with mean_z."""
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    shared = np.exp(-((coords - coords[10]) ** 2).sum(axis=1) / (2 * 25.0**2))
    rng = np.random.default_rng(6)
    expr2 = expr.copy()
    for g in sorted(expr.columns[:10]):
        expr2[g] = shared + rng.normal(0, 0.05, size=len(coords))
    genes = sorted(expr.columns[:10])
    pc = compute_score(expr2, genes, "first_pc")
    mz = compute_score(expr2, genes, "mean_z")
    r = stats.pearsonr(pc, mz)[0]
    assert r > 0.9  # sign-aligned and capturing the same shared mode
