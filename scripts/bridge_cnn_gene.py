#!/usr/bin/env python
"""Three-way bridge analysis (Parts P, Q, R, S, T).

Core statistic — the central hypothesis of the project:

    rho_CNN_GENE = Spearman(M_CNN, M_GENE)

where M_CNN is the model-derived regional relevance map (Part L) and
M_GENE the regional expression score of Alzheimer's GWAS-associated genes
(Part O). Inference:

* **matched-gene null** (Part Q): >=5000 random gene sets matched 1:1 on
  mean AHBA expression and expression variance; does the real gene set beat
  comparable genes at aligning with M_CNN?
* **spatial nulls** (Part R): >=5000 Moran spectral + Burt-2020 surrogates
  (M_CNN, M_GENE), plus surface spins for the cortical subset; is the
  spatial correspondence beyond spatial autocorrelation?
* **three-way test** (Part S/T): CNN<->Gene, CNN<->Disease, Gene<->Disease
  against the independent imported La Joie AD maps — same spatial machinery.

Run AFTER scripts/train_oasis1.py has produced
results/ml/<model>_seed<seed>/regional_relevance/M_CNN_regional.csv.

    python scripts/bridge_cnn_gene.py --ml-dir results/ml/resnet_seed42 \
        [--n-matched 5000] [--n-spatial 5000] [--quick]

All three stages checkpoint into <out-dir>/ckpt keyed by a config
fingerprint (model map mtime + null counts + seed), so a killed run resumes
instead of recomputing hours of nulls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from brainvuln import ahba
from brainvuln.config import ensure_output_tree
from brainvuln.gene_sets import (
    compute_gene_covariates,
    get_background_genes,
    match_genes_1to1,
)
from brainvuln.scoring import compute_score, zscore_genes
from brainvuln.spatial import (
    concordance,
    run_spatial_null,
    spatial_corr,
    spatial_null_pvalue,
)


def load_m_cnn(ml_dir: Path) -> pd.Series:
    df = pd.read_csv(ml_dir / "regional_relevance" / "M_CNN_regional.csv",
                     index_col=0)
    return df["M_CNN"]


def load_m_gene(disease_key: str, expr: pd.DataFrame) -> tuple[pd.Series, list[str], dict]:
    gs_dir = Path("results/gene_sets")
    meta = json.loads((gs_dir / f"{disease_key}_gwascat.json").read_text())
    genes = [g for g in meta["genes"] if g in expr.columns]
    m_gene = compute_score(expr, genes, method="mean_z")
    return m_gene, genes, meta


def align_maps(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return joined["a"], joined["b"]


def _jsonable(obj):
    """JSON encoder for numpy scalars/arrays (checkpoint serialization)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _fingerprint(args) -> str:
    """Config fingerprint: checkpoints are only reused when this matches."""
    m_cnn_csv = Path(args.ml_dir) / "regional_relevance" / "M_CNN_regional.csv"
    h = hashlib.sha256()
    h.update(str(m_cnn_csv.resolve()).encode())
    h.update(str(int(m_cnn_csv.stat().st_mtime)).encode())
    h.update(f"{args.disease}|{args.n_matched}|{args.n_spatial}|{args.seed}".encode())
    return h.hexdigest()[:12]


def _encode_matched(matched: dict) -> dict:
    """Compact encoding of the matched dict for npz storage.

    matched_sets (n_nulls x n_genes gene names) become '|'-joined strings so
    the checkpoint stays ~100 MB instead of ~1 GB of pickled lists.
    """
    enc = dict(matched)
    enc["matched_sets"] = np.array(["|".join(ms) for ms in matched["matched_sets"]])
    return enc


def _decode_matched(enc: dict) -> dict:
    dec = dict(enc)
    dec["matched_sets"] = [s.split("|") for s in enc["matched_sets"]]
    return dec


def _cnn_scorer(m_cnn: pd.Series, expr: pd.DataFrame):
    """Return (score_null_set, corr_fn, obs_for) closures over M_CNN.

    mean-z scoring = per-gene z across parcels, then parcel-wise mean over
    the set — so with a one-time precomputed z matrix each null set is a
    column mean, numerically identical to ``compute_score(..., "mean_z")``.
    """
    Z_full = zscore_genes(expr, "genes")  # parcels x genes, z once
    common = m_cnn.index.intersection(expr.index)
    common = common[m_cnn.reindex(common).notna()]
    m_cnn_vec = m_cnn.reindex(common).to_numpy()

    def corr(scored: pd.Series) -> float:
        s = scored.reindex(common)
        keep = s.notna().to_numpy()
        return spatial_corr(m_cnn_vec[keep], s.to_numpy()[keep])

    return Z_full, corr


# ---------------------------------------------------------------------------
# stage 1: matched-gene null (Part Q)
# ---------------------------------------------------------------------------

def stage1_matched_null(
    expr: pd.DataFrame,
    genes: list[str],
    m_cnn: pd.Series,
    n_sets: int,
    rng: np.random.Generator,
    dirs: dict,
    ckpt: Path,
    use_ckpt: bool,
    fingerprint: str,
) -> tuple[float, np.ndarray, dict, dict]:
    """Matched-gene null against M_CNN, with two checkpoint layers.

    Layer 1: the 5000 matched gene sets themselves (the expensive matching
    step) — saved immediately after matching so a restart never redoes it.
    Layer 2: incremental null scores (every 250) so a kill costs <=250 nulls.
    """
    c1 = ckpt / f"stage1_{fingerprint}.npz"
    c1m = ckpt / f"stage1_matched_{fingerprint}.npz"
    p1 = ckpt / f"stage1_partial_{fingerprint}.npz"

    if use_ckpt and c1.exists():
        with np.load(c1, allow_pickle=True) as z:
            obs = float(z["obs"])
            null_r = z["null_r"]
            matched = _decode_matched(z["matched"].item())
        print(f"  [resume] stage 1 restored from checkpoint "
              f"({len(null_r)} nulls)", flush=True)
        bg_prov = {"background": "restored_from_checkpoint"}
        return obs, null_r, matched, bg_prov

    background, bg_prov = get_background_genes(expr, "bing_scale",
                                               dirs["data_raw"])
    covariates = compute_gene_covariates(
        expr, ["mean_expression", "expression_variance"])
    print(f"  background pool: {len(background)} genes "
          f"({bg_prov['background']})", flush=True)

    if use_ckpt and c1m.exists():
        with np.load(c1m, allow_pickle=True) as z:
            matched = _decode_matched(z["matched"].item())
        print(f"  [resume] {matched['n_nulls']} matched sets restored",
              flush=True)
    else:
        matched = match_genes_1to1(genes, background, covariates, n_sets,
                                   rng, show_progress=True)
        if use_ckpt:
            np.savez(c1m, matched=_encode_matched(matched))
            print(f"  [ckpt] {n_sets} matched sets saved", flush=True)

    Z_full, corr = _cnn_scorer(m_cnn, expr)

    m_gene_vec = Z_full[genes].mean(axis=1)
    obs = corr(m_gene_vec)

    n = int(matched["n_nulls"])
    null_r = np.full(n, np.nan)
    done = 0
    if use_ckpt and p1.exists():
        with np.load(p1) as z:
            done = int(z["done"])
            null_r[:done] = z["null_r"]
        print(f"  [resume] {done}/{n} null scores restored", flush=True)
    chunk = 250
    while done < n:
        hi = min(done + chunk, n)
        for i in range(done, hi):
            null_r[i] = corr(Z_full[matched["matched_sets"][i]].mean(axis=1))
        done = hi
        if use_ckpt:
            np.savez(p1, null_r=null_r[:done], done=done)
        print(f"  [matched-null] scored {done}/{n}", flush=True)
    if use_ckpt:
        np.savez(c1, obs=obs, null_r=null_r, matched=_encode_matched(matched))
        p1.unlink(missing_ok=True)
        print("  [ckpt] stage 1 complete", flush=True)
    return obs, null_r, matched, bg_prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-dir", default="results/ml/resnet_seed42")
    ap.add_argument("--disease", default="alzheimer")
    ap.add_argument("--n-matched", type=int, default=5000)
    ap.add_argument("--n-spatial", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="reduced null counts for a fast wiring check")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable stage checkpointing (always recompute)")
    ap.add_argument("--out-dir", default="results/bridge",
                    help="output directory (use a scratch dir for tests)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_matched = 200
        args.n_spatial = 200

    dirs = ensure_output_tree()
    ml_dir = Path(args.ml_dir)
    m_cnn = load_m_cnn(ml_dir)
    print(f"M_CNN: {m_cnn.notna().sum()} parcels "
          f"(range {np.nanmin(m_cnn):.3g}..{np.nanmax(m_cnn):.3g})", flush=True)

    # expression + gene set -------------------------------------------------
    expr, _donor_long, coverage = ahba.load_expression(dirs["data_derived"],
                                                       "dk68_tianS1")
    expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)
    m_gene, genes, meta = load_m_gene(args.disease, expr)
    print(f"M_GENE: {len(genes)}/{len(meta['genes'])} measured GWAS genes "
          f"({len(dropped)} unsampled parcels dropped)", flush=True)
    if len(genes) < 10:
        raise SystemExit("too few measured genes for a meaningful bridge")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    use_ckpt = not args.no_ckpt
    fp = _fingerprint(args)
    rng = np.random.default_rng(args.seed)
    results: dict = {"n_genes_measured": len(genes),
                     "gene_definition": "gwascat",
                     "unsampled_parcels_dropped": dropped}

    a, b = align_maps(m_cnn, m_gene)

    # ---------------- 1. matched-gene null (Part Q) -----------------------
    print(f"[1/3] matched-gene nulls ({args.n_matched})", flush=True)
    obs_cg, null_r, matched, bg_prov = stage1_matched_null(
        expr, genes, m_cnn, args.n_matched, rng, dirs, ckpt, use_ckpt, fp)
    p_matched = spatial_null_pvalue(obs_cg, null_r)
    results["cnn_gene"] = {
        "rho_observed": obs_cg,
        "background": bg_prov,
        "matched_null": {
            "p": p_matched,
            "n": int(len(null_r)),
            "null_mean": float(null_r.mean()),
            "null_sd": float(null_r.std()),
            "ci95": [float(np.percentile(null_r, 2.5)),
                     float(np.percentile(null_r, 97.5))],
            "covariates": matched["covariates"],
            "diagnostics": matched["diagnostics"],
        },
    }
    print(f"  rho_CNN_GENE = {obs_cg:.3f}; matched-null p = {p_matched:.4f}",
          flush=True)

    # ---------------- 2. spatial nulls (Part R) ---------------------------
    print(f"[2/3] spatial nulls ({args.n_spatial} per family)", flush=True)
    parcel_table = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv")
    parcel_table = parcel_table.set_index("label")
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]]
    c2 = ckpt / f"stage2_{fp}.json"
    if use_ckpt and c2.exists():
        spatial_results = json.loads(c2.read_text())
        for fam in spatial_results.values():
            if "null_r" in fam:
                fam["null_r"] = np.asarray(fam["null_r"])
        print("  [resume] stage 2 restored from checkpoint", flush=True)
    else:
        spatial_results = {}
        for method in ("moran", "burt2020"):
            try:
                spatial_results[method] = run_spatial_null(
                    a, b, coords.reindex(a.index),
                    method=method, n_perm=args.n_spatial, seed=args.seed)
                print(f"  {method}: p = {spatial_results[method]['p_spatial']:.4f}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                spatial_results[method] = {"method": method, "error": str(exc)}
                print(f"  {method}: FAILED ({exc})", flush=True)
        try:
            from brainvuln.surfaces import load_fsaverage_spin_assets
            from brainvuln.spatial import surface_spin_pvalues
            assets = load_fsaverage_spin_assets(parcel_table.reset_index())
            spin = surface_spin_pvalues(
                a, b,
                assets["verts_lh"], assets["verts_rh"],
                assets["labels_lh"], assets["labels_rh"],
                n_perm=min(args.n_spatial, 5000), seed=args.seed)
            spatial_results["spin"] = spin
            print(f"  spin: p = {spin['p_spatial']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            spatial_results["spin"] = {"method": "spin", "error": str(exc)}
            print(f"  spin: FAILED ({exc})", flush=True)
        if use_ckpt:
            c2.write_text(json.dumps(spatial_results, default=_jsonable))
    results["cnn_gene"]["spatial_nulls"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "null_r"}
        for k, v in spatial_results.items()}
    conc = concordance(spatial_results, required=2)
    results["cnn_gene"]["concordance"] = conc
    print(f"  concordance robust = {conc['robust']}", flush=True)

    # ---------------- 3. three-way test (Part T) --------------------------
    print("[3/3] three-way test with independent disease maps", flush=True)
    c3 = ckpt / f"stage3_{fp}.json"
    if use_ckpt and c3.exists():
        tri = json.loads(c3.read_text())
        print("  [resume] stage 3 restored from checkpoint", flush=True)
    else:
        tri = {}
        maps_dir = Path("data/external/disease_maps")
        for map_name in ("lajoie2020_ad_atrophy", "lajoie2020_ad_tau_ftp"):
            csv = maps_dir / f"{map_name}.csv"
            if not csv.exists():
                print(f"  disease map missing: {csv}", flush=True)
                continue
            m_dis = pd.read_csv(csv, index_col=0)["value"]
            for target_name, target in (("CNN", m_cnn), ("GENE", m_gene)):
                t1, t2 = align_maps(target, m_dis)
                r = spatial_corr(t1.to_numpy(), t2.to_numpy())
                try:
                    sn = run_spatial_null(
                        t1, t2, coords.reindex(t1.index),
                        method="moran", n_perm=args.n_spatial, seed=args.seed)
                    p_sp = sn["p_spatial"]
                except Exception as exc:  # noqa: BLE001
                    print(f"  moran failed for {target_name} x {map_name}: {exc}",
                          flush=True)
                    p_sp = float("nan")
                tri[f"{target_name}_vs_{map_name}"] = {
                    "rho": float(r), "p_spatial_moran": p_sp,
                    "n_parcels": int(t1.notna().sum())}
                print(f"  {target_name} vs {map_name}: rho = {r:.3f} "
                      f"(moran p = {p_sp:.4f})", flush=True)
        if use_ckpt:
            c3.write_text(json.dumps(tri))
    results["three_way"] = tri

    (out_dir / "bridge_results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "cnn_gene_matched_null_r.npy", null_r)
    with open(out_dir / "cnn_gene_spatial_null_r.json", "w") as fh:
        json.dump({k: v.get("null_r", []) for k, v in spatial_results.items()
                   if "null_r" in v}, fh)
    print(f"\nwrote {out_dir}/bridge_results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
