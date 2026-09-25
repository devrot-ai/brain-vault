#!/usr/bin/env python
"""Phase 12: live GWAS Catalog retrieval through the production client."""
import json
import sys
from datetime import date

import pandas as pd
import yaml

sys.path.insert(0, "src")
from brainvuln.gwas import fetch_disease_genes, verify_catalog_trait  # noqa: E402
from brainvuln.config import GWASClientConfig, load_diseases  # noqa: E402

diseases = load_diseases("config/diseases.yaml")
spec = diseases["alzheimer"]
cfg = GWASClientConfig()
print("disease spec:", spec.display_name, spec.efo_id, spec.mondo_id)

# 1. trait verification against the live API
try:
    v = verify_catalog_trait(spec, cfg)
    print("trait verification:", json.dumps(v)[:400])
    trait_ok = True
except Exception as e:
    print("trait verification FAILED:", type(e).__name__, e)
    trait_ok = False
    v = {"error": str(e)}

# 2. live gene retrieval (production path). NOTE: the Catalog v2 indexes AD
# under the MONDO id (the EFO id filter returns 0 studies - verified live in
# this audit); the client uses spec.mondo_id accordingly.
meta = None
try:
    result = fetch_disease_genes(spec, cfg,
                                 verbose=False)
    genes = result["genes"]
    meta = {k: r for k, r in result.items() if k != "genes"}
    print(f"retrieved {len(genes)} unique mapped genes")
    print("metadata:", json.dumps({k: v for k, v in meta.items()
                                   if not isinstance(v, (list, dict))}, indent=1)[:600])
    ok = len(genes) > 100
except Exception as e:
    print("gene retrieval FAILED:", type(e).__name__, e)
    genes = []
    ok = False

# compare against the stored gene set (fetched earlier)
stored = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
overlap = len(set(genes) & set(stored["genes"])) if genes else 0
print(f"overlap with stored set: {overlap} "
      f"(stored {len(stored['genes'])}, live {len(genes)})")

out = {
    "audit_date": str(date.today()),
    "efo_id": spec.efo_id,
    "mondo_id": spec.mondo_id,
    "show_child_traits": spec.show_child_traits,
    "trait_verification": v if isinstance(v, dict) else str(v),
    "live_gene_count": len(genes),
    "stored_gene_count": len(stored["genes"]),
    "overlap": overlap,
    "retrieval_ok": ok,
    "retrieved_at": str(pd.Timestamp.now()),
}
with open("audit_artifacts/phase_12_gwas_metadata.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
with open("audit_artifacts/phase_12_gwas_genes.txt", "w") as fh:
    fh.write("\n".join(sorted(genes)))
print("PHASE12_VERDICT:", "PASS" if ok else "PARTIALLY_VERIFIED")
