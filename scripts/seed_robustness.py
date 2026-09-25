#!/usr/bin/env python
"""Multi-seed training robustness (Part V) + explanation agreement (Part W).

Thin driver around the tested ``train_oasis1.py`` pipeline: it launches the
primary ResNet-18 configuration once per seed as a subprocess, then aggregates
from disk:

* per-seed test ROC-AUC / PR-AUC (selection never saw test — the child
  process is the tested frozen-test path);
* between-seed attribution agreement: Spearman correlation between each pair
  of population M_CNN regional maps;
* per-seed Grad-CAM vs occlusion agreement (Part W).

Seeds whose metrics already exist are skipped (resumable); ``--force`` re-runs.

Usage:
    python scripts/seed_robustness.py --seeds 0 1 2 3 4 --epochs 30
    python scripts/seed_robustness.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_oasis1.py"


def run_seed(seed: int, epochs: int, force: bool, extra: list[str]) -> Path:
    out = ROOT / "results" / "ml" / f"resnet_seed{seed}"
    done = out / "metrics" / "test_metrics.json"
    if done.exists() and not force:
        print(f"[seed {seed}] already complete -> {done}", flush=True)
        return out
    print(f"[seed {seed}] launching training ({epochs} epochs)", flush=True)
    cmd = [sys.executable, "-u", str(TRAIN), "--model", "resnet",
           "--seed", str(seed), "--epochs", str(epochs), *extra]
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not done.exists():
        raise RuntimeError(f"seed {seed} finished without {done}")
    return out


def aggregate(seeds: list[int], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, maps, agree_rows = [], {}, []
    for seed in seeds:
        mdir = ROOT / "results" / "ml" / f"resnet_seed{seed}"
        mfile = mdir / "metrics" / "test_metrics.json"
        if not mfile.exists():
            print(f"[seed {seed}] no metrics yet — skipped", file=sys.stderr)
            continue
        m = json.loads(mfile.read_text())
        t = m["test"]
        rows.append({"seed": seed,
                     "roc_auc": t["roc_auc"],
                     "pr_auc": t["pr_auc"],
                     "balanced_accuracy": t.get("balanced_accuracy"),
                     "threshold": m.get("threshold"),
                     "best_val_auc": m.get("val", {}).get("roc_auc")})
        rfile = mdir / "regional_relevance" / "M_CNN_regional.csv"
        if rfile.exists():
            maps[seed] = pd.read_csv(rfile, index_col=0)["M_CNN"]
        # Part W: Grad-CAM vs occlusion agreement for this seed
        agree_file = mdir / "regional_relevance" / "method_agreement.json"
        if agree_file.exists():
            a = json.loads(agree_file.read_text())
            agree_rows.append({"seed": seed, **a})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "seed_metrics.csv", index=False)
    if agree_rows:
        pd.DataFrame(agree_rows).to_csv(out_dir / "method_agreement.csv",
                                        index=False)

    # ---- between-seed attribution agreement ------------------------------
    pairs = []
    for a, b in combinations(sorted(maps), 2):
        j = pd.concat([maps[a].rename("a"), maps[b].rename("b")],
                      axis=1).dropna()
        if len(j) >= 5 and j["a"].std() > 0 and j["b"].std() > 0:
            rho = j["a"].corr(j["b"], method="spearman")
            pairs.append({"seed_a": a, "seed_b": b,
                          "spearman_rho": float(rho), "n_parcels": len(j)})
    pd.DataFrame(pairs).to_csv(out_dir / "attribution_agreement.csv",
                               index=False)

    summary = {"n_seeds": len(df), "seeds": df["seed"].tolist()}
    if len(df):
        summary.update({
            "roc_auc_mean": float(df["roc_auc"].mean()),
            "roc_auc_sd": float(df["roc_auc"].std(ddof=1)) if len(df) > 1 else None,
            "roc_auc_min": float(df["roc_auc"].min()),
            "roc_auc_max": float(df["roc_auc"].max()),
            "pr_auc_mean": float(df["pr_auc"].mean()),
        })
    if pairs:
        rhos = pd.DataFrame(pairs)["spearman_rho"]
        summary["attribution_agreement_mean_rho"] = float(rhos.mean())
        summary["attribution_agreement_min_rho"] = float(rhos.min())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; aggregate existing seed results")
    ap.add_argument("--out", default="results/multi_seed")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra args passed through to train_oasis1.py")
    args = ap.parse_args()

    if not args.aggregate_only:
        for seed in args.seeds:
            run_seed(seed, args.epochs, args.force, args.extra)
    aggregate(args.seeds, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
