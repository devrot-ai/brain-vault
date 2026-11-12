#!/usr/bin/env python
"""End-to-end training pipeline combining all accuracy improvements.

This script automates the complete training recipe with all implemented
improvements: demographics fusion, wider model, label smoothing, focal loss,
temperature calibration, and optional ensemble training.

    # Single model with all improvements:
    python scripts/train_full_recipe.py

    # Multi-seed ensemble (5 seeds):
    python scripts/train_full_recipe.py --ensemble --n-seeds 5

    # Quick smoke test:
    python scripts/train_full_recipe.py --quick

Each trained model includes:
- Demographics fusion (age + sex)
- Wider backbone (2.0x channels)
- Label smoothing (0.1)
- Focal loss for hard examples
- Temperature calibration on validation
- Cosine LR annealing with warmup

RESEARCH USE ONLY — this is not a clinical diagnostic system.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def build_command(args, seed: int) -> list[str]:
    """Build the train_oasis1.py command line for one seed."""
    cmd = [
        sys.executable, "scripts/train_oasis1.py",
        "--model", args.model,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--warmup", str(args.warmup),
        "--cos-epochs", str(args.cos_epochs),
        "--patience", str(args.patience),
        "--widen-factor", str(args.widen_factor),
        "--label-smoothing", str(args.label_smoothing),
        "--loss-type", args.loss_type,
        "--focal-gamma", str(args.focal_gamma),
        "--focal-alpha", str(args.focal_alpha),
        "--demo-fuse",
        "--aug-strength", args.aug_strength,
        "--threshold-method", args.threshold_method,
        "--calibrate",
        "--extracted-root", args.extracted_root,
        "--splits-dir", args.splits_dir,
        "--cache-dir", args.cache_dir,
        "--out-dir", args.out_dir,
    ]

    if args.quick:
        cmd.append("--quick")

    return cmd


def run_training(cmd: list[str], seed: int) -> int:
    """Run a single training command and return exit code."""
    print(f"\n{'='*70}", flush=True)
    print(f"Training seed {seed}:", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    elapsed = time.time() - t0

    print(f"\nSeed {seed} finished in {elapsed:.0f}s (exit code {result.returncode})",
          flush=True)
    return result.returncode


def collect_results(out_dir: str, seeds: list[int]) -> dict:
    """Collect test metrics from all trained seeds."""
    results = {}
    out_root = Path(out_dir) / "ml"

    for seed in seeds:
        metrics_path = out_root / f"resnet_seed{seed}" / "metrics" / "test_metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
            results[seed] = {
                "accuracy": data.get("test", {}).get("accuracy"),
                "roc_auc": data.get("test", {}).get("roc_auc"),
                "f1": data.get("test", {}).get("f1"),
                "precision": data.get("test", {}).get("precision"),
                "recall": data.get("test", {}).get("recall"),
                "specificity": data.get("test", {}).get("specificity"),
                "threshold": data.get("threshold"),
                "temperature": data.get("temperature"),
            }
            print(f"  Seed {seed}: acc={results[seed]['accuracy']:.3f}, "
                  f"AUC={results[seed]['roc_auc']:.3f}, "
                  f"F1={results[seed]['f1']:.3f}", flush=True)
        else:
            print(f"  Seed {seed}: metrics not found at {metrics_path}", flush=True)

    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model selection
    ap.add_argument("--model", default="resnet",
                    choices=["resnet", "resnet-cbam"],
                    help="model architecture (default: resnet)")

    # Ensemble
    ap.add_argument("--ensemble", action="store_true",
                    help="train multiple seeds for ensemble inference")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds for ensemble (default: 5)")
    ap.add_argument("--start-seed", type=int, default=42,
                    help="starting seed number (default: 42)")

    # Training hyperparameters (optimized defaults)
    ap.add_argument("--epochs", type=int, default=60,
                    help="max training epochs (default: 60)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="batch size (default: 4)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="learning rate (default: 5e-5 for wider model)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup epochs (default: 3)")
    ap.add_argument("--cos-epochs", type=int, default=15,
                    help="cosine annealing horizon (default: 15)")
    ap.add_argument("--patience", type=int, default=10,
                    help="early stopping patience (default: 10)")

    # Architecture improvements
    ap.add_argument("--widen-factor", type=float, default=2.0,
                    help="backbone channel width (default: 2.0)")

    # Loss improvements
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="label smoothing epsilon (default: 0.1)")
    ap.add_argument("--loss-type", default="focal",
                    choices=["bce", "focal"],
                    help="loss function (default: focal)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal loss gamma (default: 2.0)")
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="focal loss alpha (default: 0.25)")

    # Other
    ap.add_argument("--aug-strength", default="full",
                    choices=["full", "light"],
                    help="augmentation strength (default: full)")
    ap.add_argument("--threshold-method", default="f1",
                    choices=["youden", "f1"],
                    help="threshold selection method (default: f1)")
    ap.add_argument("--extracted-root", default="data/raw/oasis1/extracted")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--cache-dir", default="data/derived/oasis1_128")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny epoch budget")

    args = ap.parse_args(argv)

    # Determine seeds
    if args.ensemble:
        seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    else:
        seeds = [args.start_seed]

    print("=" * 70, flush=True)
    print("FULL RECIPE TRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"Model:          {args.model}", flush=True)
    print(f"Seeds:          {seeds}", flush=True)
    print(f"Widen factor:   {args.widen_factor}", flush=True)
    print(f"Label smoothing:{args.label_smoothing}", flush=True)
    print(f"Loss type:      {args.loss_type} "
          f"(gamma={args.focal_gamma}, alpha={args.focal_alpha})", flush=True)
    print(f"Demographics:   fused", flush=True)
    print(f"Calibration:    temperature scaling", flush=True)
    print(f"Threshold:      {args.threshold_method}", flush=True)
    print(f"Augmentation:   {args.aug_strength}", flush=True)
    print(f"Epochs:         {args.epochs}", flush=True)
    print(f"LR:             {args.lr} (warmup={args.warmup}, "
          f"cos={args.cos_epochs})", flush=True)
    print(f"Patience:       {args.patience}", flush=True)
    if args.quick:
        print("QUICK MODE: tiny epoch budget for wiring checks", flush=True)
    print("=" * 70, flush=True)

    # Run training for each seed
    t0 = time.time()
    exit_codes = []
    for seed in seeds:
        cmd = build_command(args, seed)
        code = run_training(cmd, seed)
        exit_codes.append(code)

        # Stop on first failure (unless quick mode for debugging)
        if code != 0 and not args.quick:
            print(f"\nTraining failed for seed {seed}. Stopping.", flush=True)
            return code

    total_time = time.time() - t0
    print(f"\nAll training completed in {total_time:.0f}s", flush=True)

    # Collect and summarize results
    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    results = collect_results(args.out_dir, seeds)

    if results:
        accuracies = [r["accuracy"] for r in results.values() if r["accuracy"] is not None]
        aucs = [r["roc_auc"] for r in results.values() if r["roc_auc"] is not None]
        f1s = [r["f1"] for r in results.values() if r["f1"] is not None]

        if accuracies:
            import numpy as np
            print(f"\nAccuracy: {np.mean(accuracies):.3f} "
                  f"± {np.std(accuracies):.3f}", flush=True)
            print(f"ROC-AUC:  {np.mean(aucs):.3f} "
                  f"± {np.std(aucs):.3f}", flush=True)
            print(f"F1:       {np.mean(f1s):.3f} "
                  f"± {np.std(f1s):.3f}", flush=True)

    # Save ensemble checkpoint list if applicable
    if args.ensemble and results:
        out_root = Path(args.out_dir) / "ml"
        ensemble_dir = out_root / "ensemble_full_recipe"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        ckpt_paths = []
        for seed in seeds:
            ckpt = out_root / f"resnet_seed{seed}" / "checkpoints" / "best.pt"
            if ckpt.exists():
                ckpt_paths.append(str(ckpt))

        if ckpt_paths:
            (ensemble_dir / "checkpoints.txt").write_text("\n".join(ckpt_paths) + "\n")
            print(f"\nEnsemble checkpoints saved to: "
                  f"{ensemble_dir / 'checkpoints.txt'}", flush=True)
            print(f"Use: python scripts/ensemble_predict.py "
                  f"--sweep-dir {ensemble_dir}", flush=True)

    print("\nDone!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
