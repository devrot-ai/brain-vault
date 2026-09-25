#!/usr/bin/env python
"""Phase 19: leave-one-donor-out robustness of the CNN<->gene correlation."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.scoring import compute_score  # noqa: E402
from brainvuln.spatial import spatial_corr  # noqa: E402

expr, donor_long, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]
m_cnn = pd.read_csv(
    "results/ml/resnet_seed42/regional_relevance/M_CNN_regional.csv",
    index_col=0)["M_CNN"]

donors = sorted(donor_long["donor"].unique())
print("donors:", donors)
rows = []
for drop in [None] + donors:
    if drop is None:
        e = expr
        tag = "all_donors"
    else:
        e = donor_long[donor_long.donor != drop] \
            .pivot_table(index="label", columns="gene", values="value",
                         aggfunc="mean") \
            .reindex(index=expr.index, columns=expr.columns)
        tag = f"drop_{drop}"
    e = e.drop(index=[i for i in dropped], errors="ignore")
    mg = compute_score(e, genes, "mean_z")
    common = sorted(set(m_cnn.dropna().index) & set(mg.dropna().index))
    rho = spatial_corr(m_cnn.reindex(common).to_numpy(),
                       mg.reindex(common).to_numpy())
    rows.append({"configuration": tag, "n_parcels": len(common),
                 "spearman_rho_CNN_GENE": round(float(rho), 4)})
    print(f"{tag}: rho = {rho:.4f} on {len(common)} parcels")

out = pd.DataFrame(rows)
vals = out[out.configuration != "all_donors"]["spearman_rho_CNN_GENE"]
print(f"\nLOO median {vals.median():.4f}  IQR [{vals.quantile(.25):.4f}, "
      f"{vals.quantile(.75):.4f}]  min {vals.min():.4f}  max {vals.max():.4f}")
out.to_csv("audit_artifacts/phase_19_donor_robustness.tsv", sep="\t", index=False)
print("PHASE19_VERDICT:", "PASS" if len(rows) == 7 else "FAIL")
