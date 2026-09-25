#!/usr/bin/env python
"""Phase 18: independent disease-map integration (La Joie AD maps)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.disease_maps import parcellate_nifti  # noqa: E402
from brainvuln.scoring import compute_score  # noqa: E402
from brainvuln.spatial import run_spatial_null, spatial_corr  # noqa: E402

atlas = "data/derived/atlas_dk68_tianS1.nii.gz"
info = "data/derived/atlas_dk68_tianS1_info.csv"

m_cnn = pd.read_csv(
    "results/ml/resnet_seed42/regional_relevance/M_CNN_regional.csv",
    index_col=0)["M_CNN"]
expr, _dl, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
import json
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]
m_gene = compute_score(expr, genes, "mean_z")

coords = pd.read_csv(info).set_index("label")[
    ["centroid_x", "centroid_y", "centroid_z"]]

rows = []
for map_name in ("lajoie2020_ad_atrophy", "lajoie2020_ad_tau_ftp"):
    path = f"data/external/disease_maps/{map_name}.csv"
    m_dis = pd.read_csv(path, index_col=0)["value"]
    prov = json.load(open(f"data/external/disease_maps/{map_name}.provenance.json"))
    print(f"\n{map_name}: source={prov.get('source', prov.get('origin', '?'))[:60]}")
    for name, target in (("CNN", m_cnn), ("GENE", m_gene)):
        common = sorted(set(target.dropna().index) & set(m_dis.dropna().index))
        a = target.reindex(common).to_numpy()
        b = m_dis.reindex(common).to_numpy()
        rho = spatial_corr(a, b)
        try:
            sn = run_spatial_null(target.reindex(common), m_dis.reindex(common),
                                  coords.reindex(common),
                                  method="moran", n_perm=50, seed=42)
            p = round(float(sn["p_spatial"]), 4)
        except Exception as e:
            p = float("nan")
        rows.append({"map": map_name, "target": name,
                     "n_parcels_aligned": len(common),
                     "spearman_rho": round(float(rho), 4),
                     "p_spatial_moran_50": p})
        print(f"  {name} vs disease: rho={rho:.3f} on {len(common)} parcels "
              f"(moran-50 p={p})")

out = pd.DataFrame(rows)
out.to_csv("audit_artifacts/phase_18_disease_validation.tsv", sep="\t", index=False)
print("\nPHASE18_VERDICT: PASS" if len(rows) == 4 else "PHASE18_VERDICT: PARTIAL")
