"""Molecular scoring of brain parcels.

Primary metric: the regional gene-set score

    S_r = (1/|G|) * sum_{g in G} Z[r, g]

where Z is the parcel-wise z-scored expression of gene g across parcels.
Sensitivity methods (rank-based, first principal component, p-value-weighted)
re-run the same analysis with one knob changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class ScoringError(ValueError):
    """Raised when a score cannot be computed from the requested genes."""


def zscore_genes(expr: pd.DataFrame, dimension: str = "genes") -> pd.DataFrame:
    """Z-score expression across parcels (per gene) or across genes (per parcel)."""
    if dimension not in ("genes", "parcels"):
        raise ScoringError(f"Unknown z-score dimension {dimension!r}")
    axis = 0 if dimension == "genes" else 1
    mu = expr.mean(axis=axis)
    sd = expr.std(axis=axis, ddof=1)
    sd = sd.replace(0, np.nan)
    z = expr.sub(mu, axis=1 - axis).div(sd, axis=1 - axis)
    return z


def _resolve_genes(expr: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    missing = [g for g in genes if g not in expr.columns]
    if missing:
        raise ScoringError(
            f"{len(missing)} genes absent from expression matrix (e.g. {missing[:5]}); "
            "gene-set and expression universes are out of sync"
        )
    return expr.loc[:, genes]


def compute_score(
    expr: pd.DataFrame,
    genes: list[str],
    method: str = "mean_z",
    gene_z_dimension: str = "genes",
    gene_pvalues: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Compute the molecular score S_r for one gene set under a given method.

    Methods
    -------
    mean_z            : mean over genes of parcel-wise gene z-scores (primary).
    mean_rank         : mean over genes of parcel-wise expression ranks.
    first_pc          : first principal component of the genes x parcels matrix,
                        sign-aligned to mean_z.
    p_weighted_mean_z : mean_z weighted by -log10(association p-value) per gene
                        (uniform weight when no p-value is available).
    """
    sub = _resolve_genes(expr, genes)
    if method == "mean_z":
        z = zscore_genes(sub, gene_z_dimension)
        return z.mean(axis=1 if gene_z_dimension == "genes" else 0)
    if method == "mean_rank":
        ranks = sub.rank(axis=0 if gene_z_dimension == "genes" else 1)
        return ranks.mean(axis=1 if gene_z_dimension == "genes" else 0)
    if method == "first_pc":
        x = sub.to_numpy(dtype=float)  # parcels x genes
        x = np.nan_to_num(x, nan=0.0)
        x = (x - x.mean(axis=0)) / np.where(x.std(axis=0, ddof=1) == 0, 1, x.std(axis=0, ddof=1))
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        pc1 = pd.Series(u[:, 0] * s[0], index=expr.index)
        ref = compute_score(expr, genes, "mean_z", gene_z_dimension)
        if stats.pearsonr(pc1, ref)[0] < 0:
            pc1 = -pc1
        return pc1
    if method == "p_weighted_mean_z":
        if not gene_pvalues:
            raise ScoringError("p_weighted_mean_z requires gene p-values from the gene-set JSON")
        weights = np.array(
            [-np.log10(max(float(gene_pvalues.get(g, 1.0)), 1e-300)) for g in genes]
        )
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(genes))
        z = zscore_genes(sub, gene_z_dimension).to_numpy()
        return pd.Series(np.nanmean(z * weights[None, :], axis=1), index=expr.index)
    raise ScoringError(f"Unknown scoring method {method!r}")


def score_methods_matrix(
    expr: pd.DataFrame,
    genes: list[str],
    methods: list[str],
    gene_z_dimension: str = "genes",
    gene_pvalues: Optional[dict[str, float]] = None,
) -> dict[str, pd.Series]:
    """Compute S_r under several methods (sensitivity axis)."""
    return {
        m: compute_score(expr, genes, m, gene_z_dimension, gene_pvalues)
        for m in methods
    }


# ---------------------------------------------------------------------------
# gene-level spatial association
# ---------------------------------------------------------------------------


def gene_level_correlations(
    expr: pd.DataFrame,
    disease_map: pd.Series,
    method: str = "spearman",
) -> pd.DataFrame:
    """Per-gene spatial correlation with the disease vulnerability map.

    Returns a table sorted by |rho| descending with columns
    ``gene, rho, p_spatial_naive``. These p-values are NAIVE (non-spatial);
    inference uses the spatial nulls in :mod:`brainvuln.spatial`.
    """
    rows = []
    dm = disease_map.to_numpy(dtype=float)
    for gene in expr.columns:
        x = expr[gene].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(dm)
        if ok.sum() < 3 or np.std(x[ok]) == 0:
            rows.append((str(gene), np.nan, np.nan))
            continue
        if method == "spearman":
            r, p = stats.spearmanr(x[ok], dm[ok])
        else:
            r, p = stats.pearsonr(x[ok], dm[ok])
        rows.append((str(gene), float(r), float(p)))
    table = pd.DataFrame(rows, columns=["gene", "rho", "p_naive"]).dropna()
    return table.reindex(table["rho"].abs().sort_values(ascending=False).index)


def enrichment_among_top_genes(
    gene_rhos: pd.DataFrame,
    gene_set: list[str],
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """Are disease genes over-represented among genes with strong spatial rho?

    Statistic: mean |rho| of the disease gene set. Null: random gene samples
    of the same size from the expressed universe. Returns the observed
    statistic, its empirical one-sided p, and the set's mean rank percentile.
    """
    universe = gene_rhos["gene"].tolist()
    absrho = gene_rhos.set_index("gene")["rho"].abs()
    in_set = [g for g in gene_set if g in absrho.index]
    if not in_set:
        return {"observed_mean_abs_rho": np.nan, "p_empirical": np.nan,
                "mean_rank_percentile": np.nan, "n_in_universe": 0}
    rng = np.random.default_rng(seed)
    obs = float(absrho.loc[in_set].mean())
    pool = absrho.to_numpy()
    null = np.array([
        float(rng.choice(pool, size=len(in_set), replace=False).mean())
        for _ in range(n_perm)
    ])
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    ranks = stats.rankdata(pool) / len(pool)
    set_percentile = float(ranks[[list(absrho.index).index(g) for g in in_set]].mean())
    return {
        "observed_mean_abs_rho": obs,
        "p_empirical": float(p),
        "mean_rank_percentile": set_percentile,
        "n_in_universe": len(in_set),
    }
