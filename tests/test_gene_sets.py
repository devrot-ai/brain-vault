"""Tests for covariate-matched gene nulls (H1 machinery)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainvuln.gene_sets import (
    compute_gene_covariates,
    empirical_p_greater,
    match_genes_1to1,
)
from brainvuln.scoring import compute_score, zscore_genes


def test_compute_covariates_columns(expr):
    cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
    assert list(cov.columns) == ["mean_expression", "expression_variance"]
    assert len(cov) == expr.shape[1]


def test_matched_nulls_sizes_and_disjointness(expr):
    cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
    gene_set = sorted(expr.columns[:20])
    rng = np.random.default_rng(42)
    res = match_genes_1to1(gene_set, list(expr.columns), cov, n_nulls=10, rng=rng)
    assert res["n_nulls"] == 10
    for ms in res["matched_sets"]:
        assert len(ms) == 20
        assert not (set(ms) & set(gene_set))
        assert len(set(ms)) == 20  # 1:1, without replacement


def test_matching_is_reproducible_with_seed(expr):
    cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
    gene_set = sorted(expr.columns[:10])
    r1 = match_genes_1to1(gene_set, list(expr.columns), cov, 5,
                          np.random.default_rng(123))
    r2 = match_genes_1to1(gene_set, list(expr.columns), cov, 5,
                          np.random.default_rng(123))
    assert r1["matched_sets"] == r2["matched_sets"]


def test_matching_improves_covariate_balance(expr):
    """Matched nulls should resemble the true set better than the raw background."""
    cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
    # make the true set extreme: highest-variance genes
    gene_set = list(cov["expression_variance"].nlargest(15).index)
    res = match_genes_1to1(gene_set, list(expr.columns), cov, 20,
                           np.random.default_rng(0))
    z = (cov - cov.mean()) / cov.std(ddof=1)
    true_dist = z.loc[gene_set].to_numpy()
    bg_dist = z.loc[[g for g in expr.columns if g not in set(gene_set)]].to_numpy()
    matched_dist = np.vstack([z.loc[ms].to_numpy() for ms in res["matched_sets"]])
    raw_gap = np.abs(true_dist.mean(axis=0) - bg_dist.mean(axis=0)).max()
    matched_gap = np.abs(true_dist.mean(axis=0) - matched_dist.mean(axis=0)).max()
    assert matched_gap < raw_gap
    assert res["diagnostics"]["match_distance_mean"] < 3.0


def test_background_pool_too_small_raises(expr):
    cov = compute_gene_covariates(expr, ["mean_expression"])
    gene_set = sorted(expr.columns)
    with pytest.raises(Exception, match="smaller than gene set"):
        match_genes_1to1(gene_set, list(expr.columns), cov, 5,
                         np.random.default_rng(0))


def test_empirical_p_greater_toy():
    nulls = np.array([0.0, 0.2, 0.4, 0.6])
    assert empirical_p_greater(0.6, nulls) == pytest.approx(2 / 5)
    assert empirical_p_greater(0.5, nulls) == pytest.approx(2 / 5)
    assert empirical_p_greater(0.9, nulls) == pytest.approx(1 / 5)
    assert empirical_p_greater(-0.1, nulls) == pytest.approx(5 / 5)
    assert empirical_p_greater(0.4, nulls) == pytest.approx(3 / 5)


def test_mean_z_score_definition(expr):
    """S_r is exactly the mean of the parcel-wise gene z-scores."""
    genes = sorted(expr.columns[:5])
    manual = zscore_genes(expr[genes]).mean(axis=1)
    score = compute_score(expr, genes, "mean_z")
    pd.testing.assert_series_equal(score, manual, check_names=False)


def test_score_missing_gene_raises(expr):
    with pytest.raises(Exception, match="absent from expression matrix"):
        compute_score(expr, ["NOT_PRESENT"], "mean_z")
