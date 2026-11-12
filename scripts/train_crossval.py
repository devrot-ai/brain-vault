#!/usr/bin/env python
"""K-fold cross-validation training for robust evaluation on small datasets.

With only ~44 subjects total, a single train/val/test split gives high-variance
performance estimates. Cross-validation trains K models, each on a different
split, and averages their predictions for more reliable metrics.

    python scripts/train_crossval.py --model resnet --k 5 --seed 42

This creates K different splits, trains K models, and evaluates each one.
The final ensemble prediction averages probabilities across all K models.

RESEARCH USE ONLY — this is not a clinical diagnostic system.
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
from brainvuln.mri.dataset import OASISDataset, collate_items, load_cohort
from brainvuln.mri.models import (
    AgeSexBaseline,
    ResNet18Binary,
    ResNet18CBAM,
    ResNet18WithDemographics,
    SimpleCNN3D,
    count_parameters,
)
from brainvuln.mri.train import TrainConfig, train_model
from brainvuln.mri.evaluate import (
    classification_metrics,
    predict_loader,
    save_predictions,
    select_threshold,
    subject_bootstrap_ci,
)


def make_k_folds(cohort_df, k: int, seed: int):
    """Create K stratified folds from the cohort DataFrame.

    Args:
        cohort_df: DataFrame with 'subject', 'label', and demographic columns.
        k: number of folds.
        seed: random seed for reproducibility.

    Returns:
        List of K tuples (train_indices, val_indices, test_indices).
    """
    np.random.seed(seed)

    # Group by subject for stratification
    subjects = cohort_df["subject"].unique()
    np.random.shuffle(subjects)

    # Create stratified folds based on label
    ad_subjects = cohort_df[cohort_df["label"] == 1]["subject"].unique()
    normal_subjects = cohort_df[cohort_df["label"] == 0]["subject"].unique()

    np.random.shuffle(ad_subjects)
    np.random.shuffle(normal_subjects)

    # Distribute subjects into K folds
    ad_folds = np.array_split(ad_subjects, k)
    normal_folds = np.array_split(normal_subjects, k)

    folds = []
    for i in range(k):
        test_subjects = list(ad_folds[i]) + list(normal_folds[i])
        val_subjects_pool = []
        train_subjects_pool = []

        for j in range(k):
            if j == i:
                continue
            elif j == (i + 1) % k:
                val_subjects_pool.extend(list(ad_folds[j][:1]))
                val_subjects_pool.extend(list(normal_folds[j][:1]))
            else:
                train_subjects_pool.extend(list(ad_folds[j]))
                train_subjects_pool.extend(list(normal_folds[j]))

        # Remaining from test fold goes to val or train
        remaining_ad = list(ad_folds[i][1:])
        remaining_normal = list(normal_folds[i][1:])
        val_subjects_pool.extend(remaining_ad[:1])
        val_subjects_pool.extend(remaining_normal[:1])
        train_subjects_pool.extend(remaining_ad[1:])
        train_subjects_pool.extend(remaining_normal[1:])

        folds.append((train_subjects_pool, val_subjects_pool, test_subjects_pool))

    return folds


def build_model(model_name: str, widen_factor: float, demo_fuse: bool):
    """Build a model from name string."""
    if model_name == "resnet":
        if demo_fuse:
            return ResNet18WithDemographics(widen_factor=widen_factor)
        return ResNet18Binary(widen_factor=widen_factor)
    elif model_name == "resnet-cbam":
        return ResNet18CBAM(widen_factor=widen_factor)
    elif model_name == "simplecnn":
        return SimpleCNN3D()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet",
                    choices=["resnet", "resnet-cbam", "simplecnn"])
    ap.add_argument("--k", type=int, default=5,
                    help="number of cross-validation folds (default: 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cos-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--extracted-root", default="data/raw/oasis1/extracted")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--cache-dir", default="data/derived/oasis1_128")
    ap.add_argument("--out-dir", default="results/crossval")
    ap.add_argument("--widen-factor", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--loss-type", default="bce", choices=["bce", "focal"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--demo-fuse", action="store_true")
    ap.add_argument("--aug-strength", default="full", choices=["full", "light"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny epoch budget")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit temperature scaling on validation data after "
                         "training to improve probability calibration")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load full cohort
    cohort = load_cohort(args.splits_dir)
    print(f"Cohort: {len(cohort)} subjects, "
          f"{(cohort['label']==1).sum()} AD / {(cohort['label']==0).sum()} normal",
          flush=True)

    # Create K folds
    folds = make_k_folds(cohort, args.k, args.seed)
    print(f"Created {args.k} folds:", flush=True)
    for i, (tr, va, te) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}, val={len(va)}, test={len(te)}", flush=True)

    # Setup augmentation
    if args.aug_strength == "light":
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

    # Track results across folds
    fold_results = []
    checkpoint_paths = []

    for fold_idx in range(args.k):
        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fold_idx}/{args.k - 1}", flush=True)
        print(f"{'='*60}", flush=True)

        train_subs, val_subs, test_subs = folds[fold_idx]

        # Create fold-specific cohort
        fold_cohort = cohort.copy()
        fold_cohort["split"] = "train"
        fold_cohort.loc[fold_cohort["subject"].isin(val_subs), "split"] = "val"
        fold_cohort.loc[fold_cohort["subject"].isin(test_subs), "split"] = "test"

        # Create datasets
        train_ds = OASISDataset(
            fold_cohort[fold_cohort["split"] == "train"],
            args.extracted_root, args.cache_dir, cohort=fold_cohort
        )
        val_ds = OASISDataset(
            fold_cohort[fold_cohort["split"] == "val"],
            args.extracted_root, args.cache_dir, cohort=fold_cohort
        )
        test_ds = OASISDataset(
            fold_cohort[fold_cohort["split"] == "test"],
            args.extracted_root, args.cache_dir, cohort=fold_cohort
        )

        if augment_fn is not None:
            train_ds.augment = augment_fn

        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_items, generator=g
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_items
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_items
        )

        # Build model
        model = build_model(args.model, args.widen_factor, args.demo_fuse)
        model = model.to(device)
        n_params = count_parameters(model)
        print(f"Model: {type(model).__name__} ({n_params:,} params)", flush=True)

        # Train
        ckpt_dir = out_root / f"fold_{fold_idx}" / "checkpoints"
        cfg = TrainConfig(
            lr=args.lr, batch_size=args.batch_size,
            max_epochs=3 if args.quick else args.epochs,
            patience=args.patience, warmup_epochs=args.warmup,
            cosine_epochs=args.cos_epochs,
            device=str(device), seed=args.seed,
            label_smoothing=args.label_smoothing,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
        )

        t0 = time.time()
        result = train_model(
            model, train_loader, val_loader, cfg,
            ckpt_dir=ckpt_dir,
        )
        print(f"Fold {fold_idx} training completed in {time.time() - t0:.0f}s", flush=True)

        # Temperature calibration (optional)
        fold_temperature = 1.0
        if args.calibrate:
            from brainvuln.mri.calibrate import (
                fit_temperature, save_calibration_fit)
            print(f"Fold {fold_idx}: temperature calibration on validation", flush=True)
            fold_temperature = fit_temperature(model, val_loader, device=str(device))
            calib_dir = out_root / f"fold_{fold_idx}" / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            save_calibration_fit(fold_temperature, calib_dir / "calibration_fit.json")

        # Evaluate on test
        ckpt_path = ckpt_dir / "best.pt"
        checkpoint_paths.append(str(ckpt_path))

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        model.eval()

        # Get test predictions
        test_probs, test_labels = predict_loader(model, test_loader, device=str(device))
        threshold = select_threshold(test_labels, test_probs, "youden")
        metrics = classification_metrics(test_labels, test_probs, threshold)

        fold_results.append({
            "fold": fold_idx,
            "metrics": metrics,
            "threshold": float(threshold),
            "temperature": fold_temperature,
            "checkpoint": str(ckpt_path),
        })

        print(f"Fold {fold_idx} test accuracy: {metrics['accuracy']:.3f}, "
              f"ROC-AUC: {metrics['roc_auc']:.3f}, "
              f"F1: {metrics['f1']:.3f}", flush=True)

    # Save checkpoint list for ensemble inference
    with open(out_root / "checkpoints.txt", "w") as f:
        for p in checkpoint_paths:
            f.write(p + "\n")

    # Aggregate results across folds
    all_accuracies = [r["metrics"]["accuracy"] for r in fold_results]
    all_aucs = [r["metrics"]["roc_auc"] for r in fold_results]
    all_f1s = [r["metrics"]["f1"] for r in fold_results]

    summary = {
        "k_folds": args.k,
        "model": args.model,
        "seed": args.seed,
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean(all_accuracies)),
        "std_accuracy": float(np.std(all_accuracies)),
        "mean_roc_auc": float(np.mean(all_aucs)),
        "std_roc_auc": float(np.std(all_aucs)),
        "mean_f1": float(np.mean(all_f1s)),
        "std_f1": float(np.std(all_f1s)),
        "checkpoints": checkpoint_paths,
    }

    with open(out_root / "crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"CROSS-VALIDATION SUMMARY ({args.k}-fold)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}", flush=True)
    print(f"ROC-AUC:  {np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f}", flush=True)
    print(f"F1:       {np.mean(all_f1s):.3f} ± {np.std(all_f1s):.3f}", flush=True)
    print(f"\nCheckpoints saved to: {out_root}/checkpoints.txt", flush=True)
    print(f"Use scripts/ensemble_predict.py --sweep-dir {out_root} for ensemble inference", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
