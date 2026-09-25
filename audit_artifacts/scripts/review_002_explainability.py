#!/usr/bin/env python
"""AUDIT-REVIEW-002 — PART 1: explainability robustness (read-only on results/).

Gates everything on the canonical checkpoint (hard abort on SHA mismatch),
then runs real-data Grad-CAM checks, sanity checks, regional mapping and
method-agreement distributions. Writes outputs under audit_artifacts/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from brainvuln.config import checkpoint_identity, resolve_checkpoint  # noqa: E402

CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
OUT = PROJECT_ROOT / "audit_artifacts" / "explainability"
OUT.mkdir(parents=True, exist_ok=True)
ATLAS = PROJECT_ROOT / "data" / "derived" / "atlas_dk68_tianS1.nii.gz"
ATLAS_INFO = PROJECT_ROOT / "data" / "derived" / "atlas_dk68_tianS1_info.csv"

records: dict = {"subjects": {}, "checks": {}}


def find_volume(subject: str) -> Path:
    """Canonical pipeline input: the OASIS T88 masked_gfc derivative of the
    subject's first session — the same file class training consumed."""
    from brainvuln.mri.preprocess import find_scan_file
    sessions = sorted((PROJECT_ROOT / "data" / "raw" / "oasis1" / "extracted").glob(
        f"disc*/disc*/{subject}_MR*"))
    sessions = [s for s in sessions if s.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no extracted session dir for {subject}")
    return find_scan_file(sessions[0])


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cam_similarity(a: np.ndarray, b: np.ndarray, brain: np.ndarray) -> dict:
    """Spearman/Pearson on the union of positive CAM support (fallback: brain)."""
    from scipy.stats import pearsonr, spearmanr
    mask = (a > 0) | (b > 0)
    if mask.sum() < 1000:
        mask = brain
    x, y = a[mask].astype(float), b[mask].astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.std() == 0 or y.std() == 0 or len(x) < 100:
        return {"spearman": float("nan"), "pearson": float("nan"), "n_voxels": int(len(x))}
    return {"spearman": float(spearmanr(x, y).statistic),
            "pearson": float(pearsonr(x, y).statistic),
            "n_voxels": int(len(x))}


def main() -> int:
    # ---------------- 1.1 canonical model gate ----------------
    ident = checkpoint_identity()
    resolved = resolve_checkpoint()
    print("=== 1.1 canonical checkpoint ===")
    print(f"path      : {ident['path']}")
    print(f"filename  : {ident['filename']}")
    print(f"sha256    : {ident['sha256']}")
    print(f"size      : {ident['size_bytes']}")
    if ident["sha256"] != CANONICAL_SHA:
        print(f"FATAL: checkpoint sha256 {ident['sha256']} != canonical {CANONICAL_SHA}; "
              "refusing to run explainability against a different checkpoint.")
        return 2
    records["checkpoint"] = {**ident, "resolved": str(resolved)}

    from brainvuln.mri.models import ResNet18Binary
    from brainvuln.mri.preprocess import preprocess_session
    from brainvuln.mri.gradcam import (gradcam_3d, save_cam_nifti,
                                       save_planes_png)

    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    records["model"] = {"architecture": "ResNet18Binary",
                        "epoch": int(ckpt["epoch"]), "val_auc": float(ckpt["val_auc"]),
                        "target_layer": "layer4", "target_class": "AD (positive logit)"}
    print(f"loaded epoch {ckpt['epoch']} val_auc {ckpt['val_auc']}")

    subjects = {"OAS1_0013": "CN", "OAS1_0028": "AD"}
    camps: dict[str, np.ndarray] = {}
    vols: dict[str, np.ndarray] = {}

    for subj, cls in subjects.items():
        vol_path = find_volume(subj)
        rec: dict = {"class": cls, "input_path": str(vol_path),
                     "input_sha256": sha256_file(vol_path),
                     "checkpoint_sha256": ident["sha256"],
                     "preprocessing": ("preprocess_session: T88 reference, 2 mm isotropic "
                                       "resample, brain-bbox crop, centre-pad to 128^3, "
                                       "z-score within brain mask"),
                     "gradcam_target_layer": "layer4",
                     "gradcam_target_class": "AD positive logit"}
        print(f"\n=== {subj} ({cls}) {vol_path.name} ===")

        # ---------------- 1.3 + 1.4 baseline + run-twice consistency ----------------
        volume = preprocess_session(vol_path)                      # [1,128,128,128]
        # determinism anchor: must equal the training cache bit-for-bit
        import re as _re
        m = _re.search(r"OAS1_\d+_MR\d+", vol_path.name)
        cache = None
        if m:
            cache = PROJECT_ROOT / "data" / "derived" / "oasis1_128" / f"{m.group(0)}.npy"
        if cache is not None and cache.exists():
            c = np.load(cache)
            c = c[0] if c.ndim == 4 else c
            cache_diff = float(np.abs(c - volume[0]).max())
            rec["cache_max_abs_diff"] = cache_diff
            assert cache_diff == 0.0, f"preprocessing does not reproduce cache: {cache_diff}"
            print(f"cache check: bit-identical to {cache.name}")
        x = torch.from_numpy(volume[None].astype(np.float32))      # [1,1,128,128,128]
        with torch.no_grad():
            p1 = float(torch.sigmoid(model(x)).item())
        m1 = gradcam_3d(model, x)
        with torch.no_grad():
            p2 = float(torch.sigmoid(model(x)).item())
        m2 = gradcam_3d(model, x)
        assert np.isfinite(p1) and np.isfinite(m1).all(), "non-finite values"
        assert m1.max() > 0, "all-zero CAM"
        assert float(m1.std()) > 1e-6, "constant CAM"
        cam_maxdiff = float(np.abs(m1 - m2).max())
        prob_stable = abs(p1 - p2) <= 1e-6
        cam_stable = cam_maxdiff <= 1e-5
        print(f"P1={p1:.6f} P2={p2:.6f} stable={prob_stable} | "
              f"cam maxdiff={cam_maxdiff:.3e} stable={cam_stable} | "
              f"cam max={m1.max():.3f} nonzero={float((m1 > 0).mean()):.3f}")
        rec.update({"probability": p1, "probability_rerun": p2, "cam_max": float(m1.max()),
                    "cam_nonzero_frac": float((m1 > 0).mean()),
                    "rerun_prob_identical": bool(prob_stable),
                    "rerun_cam_max_abs_diff": cam_maxdiff,
                    "rerun_cam_identical": bool(cam_stable)})

        # NIfTI round-trip on the true grid
        nii = OUT / f"{subj}_gradcam.nii.gz"
        save_cam_nifti(m1, vol_path, nii)
        import nibabel as nib
        img = nib.load(str(nii))
        aff = img.affine
        affine_ok = bool(np.isfinite(aff).all() and abs(np.linalg.det(aff[:3, :3])) > 1e-6)
        data_back = np.asanyarray(img.dataobj, dtype=np.float32)
        print(f"nifti reopen: shape={img.shape} affine_det={np.linalg.det(aff[:3, :3]):.6f} "
              f"affine_ok={affine_ok}")
        rec.update({"nifti_shape": list(img.shape), "affine_det": float(np.linalg.det(aff[:3, :3])),
                    "affine_ok": affine_ok,
                    "nifti_nonzero_frac": float((data_back > 0).mean())})
        png = OUT / f"{subj}_gradcam_planes.png"
        save_planes_png(m1, volume, png)
        rec["outputs"] = {"nifti": str(nii), "png": str(png)}
        camps[subj], vols[subj] = m1, volume[0]

        # ---------------- 1.5 perturbation sanity (data-driven loci) ----------------
        brain = volume[0] > 0.1
        idx = np.unravel_index(np.argmax(m1), m1.shape)
        # farthest brain voxel from the CAM peak (control locus)
        coords = np.argwhere(brain)
        far = coords[int(np.argmax(((coords - np.array(idx)) ** 2).sum(1)))]
        print(f"perturbation loci: peak={idx} farthest={tuple(far)}")

        def perturbed(center, radius=9):
            v = volume.copy()
            g = np.ogrid[tuple(slice(0, s) for s in v.shape[1:])]
            d = sum((gr - c) ** 2 for gr, c in zip(g, center))
            v[0][d <= radius ** 2] = 0.0
            return v

        results = {}
        for name, center in (("peak", idx), ("far_control", tuple(far))):
            xp = torch.from_numpy(perturbed(center).astype(np.float32)[None])
            with torch.no_grad():
                p = float(torch.sigmoid(model(xp)).item())
            mp = gradcam_3d(model, xp)
            sim = cam_similarity(m1, mp, brain)
            results[name] = {"prob": p, "delta_prob": p - p1,
                             "cam_spearman_vs_baseline": sim["spearman"],
                             "cam_pearson_vs_baseline": sim["pearson"]}
            print(f"  perturb {name}: P={p:.6f} dP={p - p1:+.6f} "
                  f"cam_spearman={sim['spearman']:.4f}")
        rec["perturbation"] = results
        # production cross-reference: probability recorded by the audited run
        pred_csv = (PROJECT_ROOT / "results" / "ml" / "resnet_seed42" / "predictions"
                    / "test_predictions.csv")
        if pred_csv.exists():
            pdf = pd.read_csv(pred_csv)
            sid_col = next((c for c in pdf.columns
                            if subj in str(pdf[c].iloc[0]) or c == "subject_id"), None)
            row = None
            for c in pdf.columns:
                hit = pdf[pdf[c].astype(str).str.contains(subj, na=False)]
                if len(hit):
                    row = hit.iloc[0]
                    break
            if row is not None:
                pcol = next((c for c in pdf.columns if "prob" in c.lower()), None)
                if pcol:
                    rec["production_probability"] = float(row[pcol])
                    print(f"production probability ({pcol}): {row[pcol]:.6f}")
        records["subjects"][subj] = rec

    # ---------------- 1.6 randomization sanity ----------------
    print("\n=== 1.6 weight randomization sanity ===")
    subj = "OAS1_0013"
    x = torch.from_numpy(vols[subj][None].astype(np.float32)[None])
    base_cam = camps[subj]
    torch.manual_seed(20240924)
    rmodel = ResNet18Binary()
    for m in rmodel.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            try:
                m.reset_parameters()
            except Exception:  # noqa: BLE001 - some buffers legitimately lack it
                pass
        if hasattr(m, "reset_running_stats") and callable(m.reset_running_stats):
            m.reset_running_stats()
    rmodel.eval()
    with torch.no_grad():
        rp = float(torch.sigmoid(rmodel(x)).item())
    rcam = gradcam_3d(rmodel, x)
    brain = vols[subj] > 0.1
    sim = cam_similarity(base_cam, rcam, brain)
    print(f"random model: P={rp:.6f} cam_spearman_vs_trained={sim['spearman']:.4f} "
          f"pearson={sim['pearson']:.4f}")
    critical = bool(np.isfinite(sim["spearman"]) and sim["spearman"] > 0.9)
    print(f"CRITICAL_FLAG={critical} (trained vs destroyed-model CAM nearly identical)")
    records["checks"]["randomization"] = {**sim, "random_model_prob": rp,
                                          "critical_flag": critical}

    # ---------------- 1.7 regional mapping (complete atlas) ----------------
    print("\n=== 1.7 regional mapping (87-parcel atlas, no selection) ===")
    from brainvuln.disease_maps import parcellate_nifti
    info = pd.read_csv(ATLAS_INFO)
    rows = []
    for subj in subjects:
        series, qc, _notes = parcellate_nifti(
            OUT / f"{subj}_gradcam.nii.gz", ATLAS, atlas_info_path=ATLAS_INFO,
            mask_zero=False)
        vox = None
        for cand in ("n_voxels", "n_voxels_in_map", "voxels", "n"):
            if isinstance(qc, pd.DataFrame) and cand in qc.columns:
                vox = qc[cand]
                break
        for _, r in info.iterrows():
            lab = r["label"]
            rows.append({
                "subject": subj,
                "parcel_id": int(r["id"]),
                "region": lab,
                "structure": r["structure"],
                "mean_relevance": float(series.get(lab, np.nan)),
                "voxel_count": int(vox.get(lab, r["n_voxels"])) if vox is not None
                else int(r["n_voxels"]),
            })
    tsv = PROJECT_ROOT / "audit_artifacts" / "phase_10_region_relevance.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    print(f"wrote {tsv} ({len(rows)} rows; parcels={len(info)})")

    # ---------------- 1.8 cross-subject stability (27 production CSVs) ----------------
    print("\n=== 1.8 cross-subject stability (existing 27 per-subject regional CSVs) ===")
    per_subject = {}
    for f in sorted((PROJECT_ROOT / "results" / "ml" / "resnet_seed42" /
                     "regional_relevance").glob("OAS1_*.csv")):
        s = pd.read_csv(f, index_col=0).iloc[:, 0]
        per_subject[f.stem] = s
    mat = pd.DataFrame(per_subject)  # parcels x subjects
    stab = pd.DataFrame({
        "mean_relevance": mat.mean(axis=1),
        "median_relevance": mat.median(axis=1),
        "sd": mat.std(axis=1, ddof=1),
        "cv": mat.std(axis=1, ddof=1) / mat.mean(axis=1).abs().replace(0, np.nan),
        "n_subjects": mat.notna().sum(axis=1),
    })
    stab_path = PROJECT_ROOT / "audit_artifacts" / "phase_10_cross_subject_stability.tsv"
    stab.to_csv(stab_path, sep="\t")
    top = stab.dropna(subset=["cv"]).sort_values("mean_relevance", ascending=False).head(5)
    print("top-5 mean-relevance regions (mean / cv):")
    for lab, r in top.iterrows():
        print(f"  {lab}: {r.mean_relevance:.4f} / cv={r.cv:.3f}")
    records["checks"]["cross_subject"] = {
        "n_subjects": int(mat.shape[1]), "n_parcels": int(mat.shape[0]),
        "stability_tsv": str(stab_path),
        "top_mean_regions": {str(k): {"mean": float(v.mean_relevance), "cv": None
                                      if pd.isna(v.cv) else float(v.cv)}
                             for k, v in top.iterrows()}}

    # ---------------- 1.9 Grad-CAM vs occlusion (distributions) ----------------
    print("\n=== 1.9 Grad-CAM vs occlusion agreement (per-subject distributions) ===")
    gdir = PROJECT_ROOT / "results" / "ml" / "resnet_seed42" / "gradcam"
    odir = PROJECT_ROOT / "results" / "ml" / "resnet_seed42" / "occlusion"
    rows = []
    for g in sorted(gdir.glob("OAS1_*_gradcam.nii.gz")):
        subj = g.name.replace("_gradcam.nii.gz", "")
        o = odir / f"{subj}_occlusion.nii.gz"
        if not o.exists():
            continue
        cam_s, _, _ = parcellate_nifti(g, ATLAS, atlas_info_path=ATLAS_INFO, mask_zero=False)
        occ_s, _, _ = parcellate_nifti(o, ATLAS, atlas_info_path=ATLAS_INFO, mask_zero=False)
        j = pd.concat([cam_s.rename("cam"), occ_s.rename("occ")], axis=1).dropna()
        from scipy.stats import pearsonr, spearmanr
        rho_s = float(spearmanr(j["cam"], j["occ"]).statistic)
        rho_p = float(pearsonr(j["cam"], j["occ"]).statistic)
        top_cam = set(j["cam"].nlargest(10).index)
        top_occ = set(j["occ"].nlargest(10).index)
        rows.append({"subject": subj, "n_regions": len(j),
                     "spearman": rho_s, "pearson": rho_p,
                     "top10_overlap": len(top_cam & top_occ)})
    agree = pd.DataFrame(rows)
    agree_path = PROJECT_ROOT / "audit_artifacts" / "phase_10_method_agreement_per_subject.tsv"
    agree.to_csv(agree_path, sep="\t", index=False)
    print(agree.to_string(index=False))
    print("distribution: spearman min={:.3f} median={:.3f} max={:.3f} | "
          "top10_overlap min={} median={} max={}".format(
              agree.spearman.min(), agree.spearman.median(), agree.spearman.max(),
              agree.top10_overlap.min(), agree.top10_overlap.median(),
              agree.top10_overlap.max()))
    records["checks"]["method_agreement"] = {
        "n_common_subjects": int(len(agree)),
        "spearman_min": float(agree.spearman.min()),
        "spearman_median": float(agree.spearman.median()),
        "spearman_max": float(agree.spearman.max()),
        "top10_overlap_min": int(agree.top10_overlap.min()),
        "top10_overlap_median": float(agree.top10_overlap.median()),
        "top10_overlap_max": int(agree.top10_overlap.max()),
        "per_subject_tsv": str(agree_path)}

    with open(OUT / "consistency_and_sanity.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nwrote {OUT / 'consistency_and_sanity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
