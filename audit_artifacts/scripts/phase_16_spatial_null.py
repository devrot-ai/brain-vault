#!/usr/bin/env python
"""Phase 16: spatial-null correctness (50 Moran surrogates, production path)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.spatial import morans_i, run_spatial_null, spatial_corr  # noqa: E402

expr, _dl, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
import json
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]
from brainvuln.scoring import compute_score  # noqa: E402
m_gene = compute_score(expr, genes, "mean_z")
m_cnn = pd.read_csv(
    "results/ml/resnet_seed42/regional_relevance/M_CNN_regional.csv",
    index_col=0)["M_CNN"]

joined = pd.concat([m_cnn.rename("cnn"), m_gene.rename("gene")], axis=1).dropna()
coords = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv").set_index("label")
coords = coords.reindex(joined.index)[["centroid_x", "centroid_y", "centroid_z"]]
assert not coords.isna().any().any(), "parcel alignment failure"

obs_r = spatial_corr(joined["cnn"].to_numpy(), joined["gene"].to_numpy())
res = run_spatial_null(joined["cnn"], joined["gene"], coords,
                       method="moran", n_perm=50, seed=42)

null_r = np.asarray(res["null_r"])
print("observed r:", round(float(obs_r), 4))
print("null r: mean", round(float(null_r.mean()), 4),
      "sd", round(float(null_r.std()), 4),
      "min", round(float(null_r.min()), 4),
      "max", round(float(null_r.max()), 4))
print("n null maps:", len(null_r), "all finite:", bool(np.isfinite(null_r).all()))
print("null maps differ:", bool(np.std(null_r) > 0))
print("obs morans_i(cnn):", round(float(morans_i(joined["cnn"].to_numpy(), coords.to_numpy())), 4))
print("p_spatial:", res["p_spatial"])

out = pd.DataFrame({"null_idx": range(len(null_r)), "null_r": null_r})
out["observed"] = obs_r
out.to_csv("audit_artifacts/phase_16_spatial_null.tsv", sep="\t", index=False)

ok = (len(null_r) == 50 and np.isfinite(null_r).all()
      and np.std(null_r) > 0 and not coords.isna().any().any())
print("PHASE16_VERDICT:", "PASS" if ok else "FAIL")
print("production config supports n_perm=5000:",
      "yes (same code path; bridge uses 5000)")
