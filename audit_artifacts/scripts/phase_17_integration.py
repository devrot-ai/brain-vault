#!/usr/bin/env python
"""Phase 17: CNN<->gene integration with EXPLICIT parcel-ID alignment."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.scoring import compute_score  # noqa: E402
from brainvuln.spatial import spatial_corr  # noqa: E402

# --- load both maps ---------------------------------------------------------
m_cnn_tbl = pd.read_csv(
    "results/ml/resnet_seed42/regional_relevance/M_CNN_regional.csv", index_col=0)
m_cnn = m_cnn_tbl["M_CNN"]
expr, _dl, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]
m_gene = compute_score(expr, genes, "mean_z")

# --- explicit alignment on parcel identifiers (never on array order) --------
atlas_info = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv")
label_set = set(atlas_info["label"])
print("M_CNN labels in atlas:", m_cnn.index.isin(label_set).all(),
      "| M_GENE labels in atlas:", m_gene.index.isin(label_set).all())
print("M_CNN index unique:", m_cnn.index.is_unique,
      "| M_GENE index unique:", m_gene.index.is_unique)

common = sorted(set(m_cnn.dropna().index) & set(m_gene.dropna().index))
a = m_cnn.reindex(common).to_numpy()
b = m_gene.reindex(common).to_numpy()
print(f"common parcels (by ID): {len(common)}")

rho_p = float(np.corrcoef(a, b)[0, 1])
rho_s = spatial_corr(a, b)

# matched-gene null comparison (100 nulls, production path)
from brainvuln.gene_sets import (  # noqa: E402
    compute_gene_covariates, get_background_genes, match_genes_1to1)
from brainvuln.scoring import zscore_genes  # noqa: E402
from brainvuln.spatial import spatial_null_pvalue  # noqa: E402

background, bg_prov = get_background_genes(expr, "bing_scale", "data/raw")
cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
matched = match_genes_1to1(genes, background, cov, 100,
                           np.random.default_rng(42))
Z = zscore_genes(expr, "genes")
null_r = np.array([
    spatial_corr(a, Z[s].mean(axis=1).reindex(common).to_numpy())
    for s in matched["matched_sets"]
])
p_matched = spatial_null_pvalue(rho_s, null_r)

out = pd.DataFrame([{
    "n_common_parcels": len(common),
    "alignment": "explicit parcel ID intersection (sorted)",
    "pearson_r": round(rho_p, 4),
    "spearman_rho": round(float(rho_s), 4),
    "matched_null_mean": round(float(null_r.mean()), 4),
    "matched_null_sd": round(float(null_r.std()), 4),
    "matched_null_p_100": round(float(p_matched), 4),
    "n_matched_nulls": 100,
}])
out.to_csv("audit_artifacts/phase_17_cnn_gene.tsv", sep="\t", index=False)
print(out.to_string(index=False))
print("PHASE17_VERDICT: PASS")
