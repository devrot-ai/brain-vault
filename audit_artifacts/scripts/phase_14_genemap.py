#!/usr/bin/env python
"""Phase 14: gene map M_GENE(region) from the measured GWAS genes."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.scoring import compute_score  # noqa: E402

expr, _donor_long, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns]

m_gene = compute_score(expr, genes, method="mean_z")
tab = m_gene.rename("M_GENE").to_frame()
tab.index.name = "region"
tab.to_csv("audit_artifacts/phase_14_gene_map.tsv", sep="\t")

print("measured genes:", len(genes), "of", len(meta["genes"]),
      "(rest not in the AHBA matrix)")
print("parcels:", len(m_gene), "dropped unsampled:", len(dropped))
print("finite:", bool(np.isfinite(m_gene.values).all()))
print("mean/std:", round(float(m_gene.mean()), 4), round(float(m_gene.std()), 4))
print("top-5 M_GENE:", dict(m_gene.nlargest(5).round(3)))
print("bottom-3 M_GENE:", dict(m_gene.nsmallest(3).round(3)))

# identifier alignment: every score index must be an atlas label
atlas_info = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv")
missing = set(m_gene.index) - set(atlas_info["label"])
print("parcel labels not in atlas info:", len(missing))

ok = (len(genes) > 100 and len(m_gene) == 86
      and np.isfinite(m_gene.values).all() and not missing)
print("PHASE14_VERDICT:", "PASS" if ok else "FAIL")
