#!/usr/bin/env python
"""Phase 15: matched-gene null correctness (100 null sets, production path)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.gene_sets import (  # noqa: E402
    compute_gene_covariates, get_background_genes, match_genes_1to1)
from brainvuln.scoring import compute_score, zscore_genes  # noqa: E402
from brainvuln.spatial import spatial_corr, spatial_null_pvalue  # noqa: E402

expr, _dl, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
import json
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]

background, bg_prov = get_background_genes(expr, "bing_scale", "data/raw")
cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])
rng = np.random.default_rng(42)
matched = match_genes_1to1(genes, background, cov, 100, rng)

print("background:", bg_prov)
sets = matched["matched_sets"]
print("n nulls:", len(sets), "set size:", {len(s) for s in sets})

# correctness checks
target = set(genes)
gene_pool = set(expr.columns)
no_target = all(not (set(s) & target) for s in sets)
no_dups = all(len(s) == len(set(s)) for s in sets)
all_known = all(set(s) <= gene_pool for s in sets)
print("target genes excluded from nulls:", no_target)
print("no within-set duplicates:", no_dups)
print("all null genes measured in AHBA:", all_known)

# matching quality: KS on covariates (from the production diagnostics)
diag = matched["diagnostics"]
print("KS mean_expression:", round(diag["ks_mean_expression"], 4),
      "p:", round(diag["ks_p_mean_expression"], 3))
print("KS expression_variance:", round(diag["ks_expression_variance"], 4),
      "p:", round(diag["ks_expression_variance"], 3))

# null map correlations against M_GENE (independent target for this test)
Z = zscore_genes(expr, "genes")
m_gene = compute_score(expr, genes, "mean_z")
obs = spatial_corr(m_gene.reindex(m_gene.index).to_numpy(),
                   m_gene.reindex(m_gene.index).to_numpy())
null_r = np.array([
    spatial_corr(Z[s].mean(axis=1).reindex(m_gene.index).to_numpy(),
                 m_gene.to_numpy())
    for s in sets
])
p = spatial_null_pvalue(float(np.corrcoef(m_gene.to_numpy(),
                                          m_gene.to_numpy())[0, 1]) * 0 + 1.0,
                        null_r)  # placeholder replaced below

# real observed statistic: corr(M_GENE, mean of a held-out gene slice)? No -
# the correct observed statistic for THIS test is corr(M_GENE with itself),
# trivially 1. Instead we report the null distribution vs the CNN map.
m_cnn = pd.read_csv(
    "results/ml/resnet_seed42/regional_relevance/M_CNN_regional.csv",
    index_col=0)["M_CNN"]
j = pd.concat([m_cnn.rename("cnn"), m_gene.rename("gene")], axis=1).dropna()
obs_cg = float(j["cnn"].corr(j["gene"], method="spearman"))
null_cg = np.array([
    float(pd.concat([
        m_cnn.rename("cnn"),
        Z[s].mean(axis=1).rename("g")], axis=1).dropna()
        .pipe(lambda d: d["cnn"].corr(d["g"], method="spearman")))
    for s in sets
])
p_cg = spatial_null_pvalue(obs_cg, null_cg)

out = pd.DataFrame({"null_idx": range(len(sets)),
                    "set_size": [len(s) for s in sets],
                    "corr_vs_M_CNN_spearman": null_cg})
out.to_csv("audit_artifacts/phase_15_null_test.tsv", sep="\t", index=False)
print(f"\nobs rho_CNN_GENE = {obs_cg:.4f}")
print(f"null mean/sd = {null_cg.mean():.4f} / {null_cg.std():.4f}")
print(f"empirical p (100 nulls) = {p_cg:.4f}")

ok = (no_target and no_dups and all_known and len(sets) == 100
      and diag["ks_p_mean_expression"] > 0.01
      and diag["ks_p_expression_variance"] > 0.01)
print("PHASE15_VERDICT:", "PASS" if ok else "FAIL")
