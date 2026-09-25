#!/usr/bin/env python
"""Phase 20: reproducibility - same seed reproduces, different seed differs."""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from brainvuln import ahba  # noqa: E402
from brainvuln.gene_sets import (  # noqa: E402
    compute_gene_covariates, get_background_genes, match_genes_1to1)
from brainvuln.scoring import compute_score, zscore_genes  # noqa: E402
from brainvuln.spatial import run_spatial_null, spatial_corr  # noqa: E402

results = {}

# --- 1. same-seed determinism: matched nulls --------------------------------
expr, _dl, coverage = ahba.load_expression("data/derived", "dk68_tianS1")
expr, _dropped = ahba.drop_unsampled_parcels(expr, coverage)
meta = json.load(open("results/gene_sets/alzheimer_gwascat.json"))
genes = [g for g in meta["genes"] if g in expr.columns][:100]  # small for speed
background, _ = get_background_genes(expr, "bing_scale", "data/raw")
cov = compute_gene_covariates(expr, ["mean_expression", "expression_variance"])

def run_match(seed):
    m = match_genes_1to1(genes, background, cov, 20, np.random.default_rng(seed))
    return ["|".join(s) for s in m["matched_sets"]]

a1, a2, a3 = run_match(42), run_match(42), run_match(43)
results["matched_null_same_seed_identical"] = a1 == a2
results["matched_null_diff_seed_differs"] = a1 != a3
print("matched nulls: same-seed identical:", a1 == a2,
      "| diff-seed differs:", a1 != a3)

# --- 2. same-seed determinism: Moran surrogates -----------------------------
m_gene = compute_score(expr, genes, "mean_z")
coords = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv").set_index("label")[
    ["centroid_x", "centroid_y", "centroid_z"]].reindex(m_gene.index)
s1 = run_spatial_null(m_gene, m_gene * 0 + np.arange(len(m_gene)) / len(m_gene),
                      coords, method="moran", n_perm=20, seed=42)["null_r"]
s2 = run_spatial_null(m_gene, m_gene * 0 + np.arange(len(m_gene)) / len(m_gene),
                      coords, method="moran", n_perm=20, seed=42)["null_r"]
s3 = run_spatial_null(m_gene, m_gene * 0 + np.arange(len(m_gene)) / len(m_gene),
                      coords, method="moran", n_perm=20, seed=7)["null_r"]
results["moran_same_seed_identical"] = bool(np.array_equal(np.asarray(s1),
                                                           np.asarray(s2)))
results["moran_diff_seed_differs"] = not np.array_equal(np.asarray(s1),
                                                        np.asarray(s3))
print("moran: same-seed identical:", results["moran_same_seed_identical"],
      "| diff-seed differs:", results["moran_diff_seed_differs"])

# --- 3. same-seed determinism: model forward on identical weights -----------
torch.manual_seed(0)
from brainvuln.mri.models import ResNet18Binary  # noqa: E402
model = ResNet18Binary().eval()
x = torch.randn(1, 1, 64, 64, 64)
with torch.no_grad():
    p1 = model(x)
    p2 = model(x)
results["model_forward_deterministic"] = bool(torch.equal(p1, p2))
print("model forward deterministic:", results["model_forward_deterministic"])

# --- 4. same-seed training determinism (2 steps, tiny model) ----------------
from brainvuln.mri.models import SimpleCNN3D  # noqa: E402

def tiny_train():
    torch.manual_seed(123)
    np.random.seed(123)
    m = SimpleCNN3D()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lossfn = torch.nn.BCEWithLogitsLoss()
    torch.manual_seed(5)
    xs = torch.randn(4, 1, 32, 32, 32)
    ys = torch.tensor([1., 0., 1., 0.])
    for _ in range(3):
        opt.zero_grad()
        loss = lossfn(m(xs), ys)
        loss.backward()
        opt.step()
    h = hashlib.sha256()
    for p in m.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()

w1, w2 = tiny_train(), tiny_train()
results["tiny_train_same_seed_bitwise_identical"] = w1 == w2
print("tiny training same-seed weights identical:", w1 == w2)

ok = all(results.values())
with open("audit_artifacts/phase_20_reproducibility.md", "w") as fh:
    fh.write("# Phase 20 - Reproducibility\n\n")
    for k, v in results.items():
        fh.write(f"* {k}: {'PASS' if v else 'FAIL'}\n")
    fh.write(f"\nverdict: {'PASS' if ok else 'FAIL'}\n")
print("PHASE20_VERDICT:", "PASS" if ok else "FAIL")
