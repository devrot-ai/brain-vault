"""Cross-disease analysis: the D (diseases) x R (parcels) molecular matrix.

Shared vs disease-specific vulnerability, disease-disease similarity, and
genetic convergence are all read off this matrix. Similarity p-values use the
same spatially constrained nulls as the single-disease analysis — clusters are
discovered in the data, then tested, never assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .spatial import run_spatial_null, spatial_corr


class CrossDisorderError(RuntimeError):
    """Raised when the cross-disease matrix cannot be assembled."""


def build_disease_by_region_matrix(
    score_maps: dict[str, pd.Series],
) -> pd.DataFrame:
    """Stack per-disease score maps into the D x R matrix (NaN-aware union)."""
    if len(score_maps) < 2:
        raise CrossDisorderError(
            "Cross-disease analysis needs score maps for >=2 diseases; "
            f"got {len(score_maps)}"
        )
    mat = pd.DataFrame(score_maps)
    return mat


def pairwise_similarity(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    n_perm: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Disease x disease Spearman similarity with spatial-null p-values.

    For each pair, Moran surrogates of one map are correlated against the
    other (both directions give the same two-sided p by symmetry); the
    reported p is the Moran-based spatial p.
    """
    diseases = list(matrix.columns)
    rows = []
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            r = spatial_corr(matrix[a].to_numpy(), matrix[b].to_numpy())
            try:
                res = run_spatial_null(matrix[a], matrix[b], coords,
                                       method="moran", n_perm=n_perm, seed=seed)
                p = res["p_spatial"]
            except Exception as exc:  # noqa: BLE001 - record, never crash the matrix
                p, err = float("nan"), f"{type(exc).__name__}: {exc}"
            else:
                err = ""
            rows.append({"disease_a": a, "disease_b": b, "rho": r,
                         "p_spatial": p, "error": err})
    return pd.DataFrame(rows)


def cluster_diseases(matrix: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Ward clustering on correlation distance; returns ordered labels + linkage."""
    complete = matrix.dropna(axis=0, how="any")  # parcels complete across diseases
    if complete.shape[0] < 10:
        raise CrossDisorderError(
            f"Only {complete.shape[0]} parcels shared across all diseases; "
            "clustering needs more complete overlap"
        )
    # pairwise Spearman between disease rows (robust, NaN-free by construction)
    from scipy.stats import spearmanr

    n_diseases = complete.shape[1]
    corr = np.eye(n_diseases)
    for i in range(n_diseases):
        for j in range(i + 1, n_diseases):
            r = spearmanr(complete.iloc[:, i], complete.iloc[:, j]).statistic
            corr[i, j] = corr[j, i] = float(r) if np.isfinite(r) else 0.0
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))  # correlation distance
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="ward")
    order = leaves_list(z)
    labels = [str(matrix.columns[i]) for i in order]
    return labels, z


def shared_and_specific_parcels(
    matrix: pd.DataFrame,
    top_fraction: float = 0.1,
) -> dict:
    """Top-decile vulnerable parcels per disease, shared set, and specific set."""
    result: dict[str, dict] = {}
    top_sets: dict[str, set[str]] = {}
    for disease in matrix.columns:
        col = matrix[disease].dropna()
        k = max(1, int(round(top_fraction * len(col))))
        top = col.nlargest(k).index.astype(str)
        top_sets[str(disease)] = set(top)
        result[str(disease)] = {"top_parcels": sorted(top), "n_top": k}

    diseases = list(top_sets)
    shared: set[str] = set.intersection(*top_sets.values()) if diseases else set()
    specific: dict[str, list[str]] = {}
    for d in diseases:
        others: set[str] = set()
        for other in diseases:
            if other != d:
                others |= top_sets[other]
        specific[d] = sorted(top_sets[d] - others)
    return {
        "per_disease": result,
        "shared_top_parcels": sorted(shared),
        "disease_specific_top_parcels": specific,
        "top_fraction": top_fraction,
    }


def genetic_convergence(
    matrix: pd.DataFrame,
    gene_definitions: dict[str, str],
) -> dict:
    """Do genetically different disorders converge on similar neuroanatomy?

    Computes, per disease, the correlation of its molecular map with every
    other disease's map alongside the Jaccard overlap of their gene sets —
    high map similarity with low gene-set overlap is evidence of convergence
    onto shared neuroanarchitectural axes from distinct genetic starting points.
    """
    diseases = list(matrix.columns)
    rows = []
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            genes_a = set(gene_definitions.get(a, "").split(";")) - {""}
            genes_b = set(gene_definitions.get(b, "").split(";")) - {""}
            jac = len(genes_a & genes_b) / len(genes_a | genes_b) if (genes_a or genes_b) else np.nan
            r = spatial_corr(matrix[a].to_numpy(), matrix[b].to_numpy())
            rows.append({
                "disease_a": a, "disease_b": b,
                "rho_molecular": r,
                "jaccard_genes": jac,
                "convergent": bool(np.isfinite(r) and np.isfinite(jac) and r > 0.3 and jac < 0.1),
            })
    return {"pairs": pd.DataFrame(rows),
            "note": "convergence thresholds are heuristic; report, do not gate"}
