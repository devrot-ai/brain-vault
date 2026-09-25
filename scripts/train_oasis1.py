#!/usr/bin/env python
"""Train and evaluate the OASIS-1 MRI classifier (Parts D–L).

Runs the full ML path for one model configuration:

    python scripts/train_oasis1.py --model resnet --seed 42 \
        [--extracted-root data/raw/oasis1/extracted] [--epochs 40] [--quick]

Stages:
  1. preprocess + cache all cohort scans (deterministic, resumable)
  2. train on the train partition (early stop on VALIDATION ROC-AUC only)
  3. freeze → evaluate on TEST with full metrics + subject-bootstrap CIs
  4. Grad-CAM + occlusion sensitivity for every test subject
  5. regional M_CNN map on the predefined atlas

Nothing in this script reads the test partition before stage 3, and nothing
reads the GWAS/AHBA layers at all — the bridge is a separate script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from brainvuln.config import ensure_output_tree
from brainvuln.mri.augment import training_transforms
from brainvuln.mri.dataset import OASISDataset, collate_items, load_cohort, load_split_tables
from brainvuln.mri.models import AgeSexBaseline, ResNet18Binary, SimpleCNN3D
from brainvuln.mri.train import TrainConfig, train_model
from brainvuln.mri.evaluate import (
    classification_metrics,
    predict_loader,
    save_predictions,
    select_threshold,
    subject_bootstrap_ci,
)


def make_loader(ds, batch_size, shuffle, seed, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_items,
                      generator=g if shuffle else None,
                      worker_init_fn=None if not shuffle else
                      lambda wid: np.random.seed(seed + wid))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet", choices=["resnet", "simplecnn", "agesex"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1,
                    help="LR warmup epochs before cosine decay")
    ap.add_argument("--cos-epochs", type=int, default=0,
                    help="cosine horizon in epochs; 0 = full --epochs budget "
                         "(legacy). Set to the realistic early-stopping window "
                         "so LR actually anneals in short runs.")
    ap.add_argument("--tag", default="",
                    help="suffix on the output dir (sweeps/ablations); "
                         "empty keeps the canonical <model>_seed<seed> dir")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--extracted-root", default="data/raw/oasis1/extracted")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--cache-dir", default="data/derived/oasis1_128")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-augment", action="store_true",
                    help="ablation: disable training augmentation")
    ap.add_argument("--aug-strength", default="full", choices=["full", "light"],
                    help="full: documented default (±7°, ±3 vox, ±5%% scale, "
                         "±10%% intensity, noise 0.02). light: regularization "
                         "kept but fitting drag reduced (±4°, ±2 vox, ±3%% "
                         "scale, ±5%% intensity, no noise) — use when the "
                         "model underfits from scratch.")
    ap.add_argument("--normalization", default="brain_z",
                    choices=["brain_z", "global_minmax"],
                    help="ablation axis: intensity normalization")
    ap.add_argument("--threshold-method", default="youden",
                    choices=["youden", "f1"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--occlusion-subjects", type=int, default=8,
                    help="occlusion sensitivity is O(10^2) forwards per "
                         "subject; it runs on the first N test subjects "
                         "(documented subset). Grad-CAM runs for all.")
    ap.add_argument("--quick", action="store_true",
                    help="smoke path: tiny epoch budget for wiring checks")
    ap.add_argument("--skip-train", action="store_true",
                    help="resume: load checkpoints/best.pt and run stages 3-5 "
                         "only (used after an interrupted run)")
    ap.add_argument("--resume-epochs", action="store_true",
                    help="resume an epoch-crashed training run: continue from "
                         "checkpoints/last_state.pt (per-epoch full optimizer "
                         "state) instead of restarting from scratch")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = ensure_output_tree()
    splits = load_split_tables(args.splits_dir)
    cohort = load_cohort(args.splits_dir)
    out_root = Path(args.out_dir) / "ml" / f"{args.model}_seed{args.seed}"
    if args.no_augment:
        out_root = out_root.with_name(out_root.name + "_noaug")
    if args.aug_strength != "full":
        out_root = out_root.with_name(out_root.name + "_" + args.aug_strength + "aug")
    if args.normalization != "brain_z":
        out_root = out_root.with_name(out_root.name + "_" + args.normalization)
    if args.tag:
        out_root = out_root.with_name(out_root.name + "_" + args.tag)
    out_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- 1. preprocessing (train partition logs progress) ----
    print(f"[1/5] preprocessing cache: {args.cache_dir}", flush=True)
    t0 = time.time()
    train_ds = OASISDataset(splits["train"], args.extracted_root, args.cache_dir,
                            cohort=cohort)
    val_ds = OASISDataset(splits["val"], args.extracted_root, args.cache_dir,
                          cohort=cohort)
    test_ds = OASISDataset(splits["test"], args.extracted_root, args.cache_dir,
                           cohort=cohort)
    n = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"  sessions: {n} (train {len(train_ds)} / val {len(val_ds)} / "
          f"test {len(test_ds)}) in {time.time() - t0:.0f}s", flush=True)

    if args.no_augment:
        augment_fn = None
    elif args.aug_strength == "light":
        augment_fn = training_transforms(
            seed=args.seed,
            rotate_range=4.0 * np.pi / 180.0,
            translate_range=2.0,
            scale_range=0.03,
            intensity_scale=0.05,
            intensity_shift=0.05,
            noise_sigma=0.0,
        )
    else:
        augment_fn = training_transforms(seed=args.seed)
    if augment_fn is not None:
        # MONAI Compose operates on dicts; the dataset hands it {'img': tensor}
        train_ds.augment = augment_fn  # contract: tensor -> tensor (dict-wrap inside)

    train_loader = make_loader(train_ds, args.batch_size, True, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.seed)
    test_loader = make_loader(test_ds, args.batch_size, False, args.seed)

    # ---------------- 2. train ------------------------------------------
    print(f"[2/5] training {args.model} (seed {args.seed}, device {device})",
          flush=True)
    if args.model == "resnet":
        model = ResNet18Binary()
    elif args.model == "simplecnn":
        model = SimpleCNN3D()
    else:
        print("agesex baseline: closed-form below", flush=True)
    ckpt_dir = out_root / "checkpoints"
    cfg = TrainConfig(
        lr=args.lr, batch_size=args.batch_size, max_epochs=3 if args.quick else args.epochs,
        patience=args.patience, warmup_epochs=args.warmup,
        cosine_epochs=args.cos_epochs,
        resume=args.resume_epochs,
        resume_path=str(ckpt_dir / "last_state.pt"),
        device=str(device), seed=args.seed)

    if args.model == "agesex":
        # closed-form logistic regression on (age, sex) from TRAIN subjects
        from sklearn.linear_model import LogisticRegression
        tr = cohort[cohort["split"] == "train"]
        va = cohort[cohort["split"] == "val"]
        te = cohort[cohort["split"] == "test"]
        X = lambda d: np.stack([d["Age"].to_numpy(float),
                                (d["Sex"] == "M").to_numpy(float)], axis=1)
        clf = LogisticRegression(max_iter=1000).fit(X(tr), tr["label"])
        rows = {}
        for name, part in (("val", va), ("test", te)):
            probs = clf.predict_proba(X(part))[:, 1]
            thr = select_threshold(part["label"].to_numpy(), probs,
                                   args.threshold_method)
            rows[name] = classification_metrics(part["label"].to_numpy(), probs, thr)
            if name == "val":
                val_thr = thr
        (out_root / "metrics").mkdir(parents=True, exist_ok=True)
        (out_root / "metrics" / "test_metrics.json").write_text(json.dumps(
            {"test": rows["test"], "val": rows["val"],
             "threshold": val_thr, "coef": clf.coef_.tolist()}, indent=2))
        print(f"  age+sex baseline: test ROC-AUC "
              f"{rows['test']['roc_auc']:.3f} (frozen threshold {val_thr:.2f})",
              flush=True)
        return 0

    ckpt_path = out_root / "checkpoints" / "best.pt"
    if args.skip_train and ckpt_path.exists():
        print("[2/5] resume: loading existing checkpoint "
              f"{ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        result = None
    else:
        result = train_model(model, train_loader, val_loader, cfg,
                             checkpoint_path=ckpt_path)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    if result is not None:
        (out_root / "metrics" / "train_history.json").write_text(json.dumps(
            {"best_val_auc": result.best_val_auc,
             "best_epoch": result.best_epoch,
             "epochs_run": result.epochs_run,
             "history": result.history}, indent=2))
        print(f"  best val_auc {result.best_val_auc:.4f} "
              f"(epoch {result.best_epoch})", flush=True)
    else:
        print(f"  resumed from checkpoint (best val_auc "
              f"{json.loads((out_root / 'metrics' / 'train_history.json').read_text())['best_val_auc']:.4f})",
              flush=True)

    # ---------------- 3. threshold (val) + frozen test -------------------
    print("[3/5] threshold on validation, then frozen test", flush=True)
    yv, pv, _, _ = predict_loader(model, val_loader, device)
    threshold = select_threshold(yv, pv, args.threshold_method)
    yt, pt, sids, subs = predict_loader(model, test_loader, device)
    test_metrics = classification_metrics(yt, pt, threshold)
    ci = subject_bootstrap_ci(yt, pt, subs, threshold, n_boot=args.n_boot,
                              seed=args.seed)
    test_metrics["bootstrap_ci"] = ci
    from brainvuln.config import checkpoint_identity
    (out_root / "metrics" / "test_metrics.json").write_text(json.dumps(
        {"test": test_metrics, "threshold": threshold,
         "threshold_method": args.threshold_method,
         "checkpoint_identity": checkpoint_identity(str(ckpt_path)),
         "val": classification_metrics(yv, pv, threshold)}, indent=2))
    (out_root / "predictions").mkdir(exist_ok=True)
    save_predictions(out_root / "predictions" / "test_predictions.csv",
                     sids, subs, yt, pt)
    print(f"  test ROC-AUC {test_metrics['roc_auc']:.3f} "
          f"[{ci['roc_auc']['lo']:.3f}, {ci['roc_auc']['hi']:.3f}] "
          f"PR-AUC {test_metrics['pr_auc']:.3f} "
          f"(threshold {threshold:.2f})", flush=True)

    # ---------------- 4. Grad-CAM + occlusion ----------------------------
    print("[4/5] Grad-CAM + occlusion for test subjects", flush=True)
    from brainvuln.mri.gradcam import (
        gradcam_3d, occlusion_sensitivity, save_cam_nifti, save_planes_png)
    cam_dir = out_root / "gradcam"
    occ_dir = out_root / "occlusion"
    cam_by_subj: dict[str, np.ndarray] = {}
    occ_by_subj: dict[str, np.ndarray] = {}
    seen_subjects: set[str] = set()
    ref_scan = {r.session_id: r.scan_path for r in test_ds.records}
    n_occ = 0
    for i, (volumes, labels, _a, _s, meta) in enumerate(test_loader):
        for b in range(volumes.shape[0]):
            vol = volumes[b:b + 1]
            sid, sub = meta["session_id"][b], meta["subject_id"][b]
            if sub in seen_subjects:
                continue  # one explanation per subject (first session)
            seen_subjects.add(sub)
            cam = gradcam_3d(model, vol)
            cam_by_subj[sub] = cam
            save_cam_nifti(cam, ref_scan[sid],
                           cam_dir / f"{sub}_gradcam.nii.gz")
            save_planes_png(cam, volumes.numpy()[b], cam_dir / f"{sub}_planes.png")
            if n_occ < args.occlusion_subjects:
                occ = occlusion_sensitivity(model, vol, patch=32, stride=32,
                                            device=str(device))
                occ_by_subj[sub] = occ
                save_cam_nifti(occ, ref_scan[sid],
                               occ_dir / f"{sub}_occlusion.nii.gz")
                n_occ += 1
            print(f"  explained {sub} ({sid})"
                  f"{' (+occlusion)' if sub in occ_by_subj else ''}",
                  flush=True)

    # ---------------- 5. regional maps -----------------------------------
    print("[5/5] regional M_CNN on the predefined atlas", flush=True)
    from brainvuln.mri.regional import (
        aggregate_regional, method_agreement, regionalize_cam)
    atlas_img = Path("data/derived/atlas_dk68_tianS1.nii.gz")
    atlas_info = Path("data/derived/atlas_dk68_tianS1_info.csv")
    if not atlas_img.exists():
        print("  atlas missing — run build_expression.py first; "
              "regional maps skipped", file=sys.stderr)
        return 0
    reg_dir = out_root / "regional_relevance"
    reg_dir.mkdir(parents=True, exist_ok=True)
    per_subject, per_subject_occ = {}, {}
    for sub in sorted(seen_subjects):
        cam_nii = cam_dir / f"{sub}_gradcam.nii.gz"
        per_subject[sub] = regionalize_cam(cam_nii, atlas_img, atlas_info)
        per_subject[sub].to_csv(reg_dir / f"{sub}.csv")
        occ_nii = occ_dir / f"{sub}_occlusion.nii.gz"
        if occ_nii.exists():
            per_subject_occ[sub] = regionalize_cam(occ_nii, atlas_img,
                                                   atlas_info)
    table, m_cnn = aggregate_regional(per_subject)
    table.to_csv(reg_dir / "M_CNN_regional.csv")
    occ_table, m_occ = aggregate_regional(per_subject_occ)
    occ_table.to_csv(reg_dir / "M_CNN_occlusion_regional.csv")
    agreement = method_agreement(m_cnn, m_occ)
    (reg_dir / "method_agreement.json").write_text(json.dumps(agreement, indent=2))
    print(f"  M_CNN: {int(m_cnn.notna().sum())} parcels; "
          f"Grad-CAM vs occlusion regional Spearman = "
          f"{agreement['spearman']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
