#!/usr/bin/env python
"""Materialize the Part AD required-output tree from the implemented layout.

The pipeline writes per-model artifacts under ``results/ml/<model>_seed<seed>/``
(so multiple seeds/ablations coexist). Part AD specifies a flat tree
(``results/metrics``, ``results/gradcam``, ...). This collector copies the
primary model's artifacts into that layout without moving anything, materializes
derived artifacts that only exist inline (confusion matrix, calibration), and
records a manifest of every source -> destination pair so provenance stays
explicit. Idempotent: re-running overwrites the spec copies in place.

Usage:
    python scripts/collect_spec_tree.py [--model-dir results/ml/resnet_seed42]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFUSION_ANALYTICS = """
Confusion-matrix and calibration artifacts are derived from
predictions/test_predictions.csv of the primary model. Threshold comes from
metrics/test_metrics.json (selected on validation only).
"""


def collect(model_dir: Path) -> int:
    from brainvuln.config import ensure_output_tree

    dirs = ensure_output_tree(ROOT)
    spec = {
        "metrics": ROOT / "results" / "metrics",
        "predictions": ROOT / "results" / "predictions",
        "confusion_matrix": ROOT / "results" / "confusion_matrix",
        "calibration": ROOT / "results" / "calibration",
        "gradcam": ROOT / "results" / "gradcam",
        "regional_relevance": ROOT / "results" / "regional_relevance",
        "ahba": ROOT / "results" / "ahba",
        "gwas": ROOT / "results" / "gwas",
        "matched_null": ROOT / "results" / "matched_null",
        "spatial_null": ROOT / "results" / "spatial_null",
        "donor_robustness": ROOT / "results" / "donor_robustness",
        "external_validation": ROOT / "results" / "external_validation",
    }
    for d in spec.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    copied: list[tuple[Path, Path]] = []

    def cp(src: Path | None, dst_dir: Path, name: str | None = None) -> None:
        if src is None or not src.exists():
            return
        dst = dst_dir / (name or src.name)
        if dst.resolve() == src.resolve():
            manifest[str(dst.relative_to(ROOT))] = str(src.relative_to(ROOT))
            return  # already in the spec location
        shutil.copy2(src, dst)
        copied.append((src, dst))
        manifest[str(dst.relative_to(ROOT))] = str(src.relative_to(ROOT))

    # ---- per-model artifacts -------------------------------------------
    for sub, spec_name in (("metrics", "metrics"),
                           ("predictions", "predictions"),
                           ("gradcam", "gradcam"),
                           ("occlusion", "gradcam"),
                           ("regional_relevance", "regional_relevance")):
        src_dir = model_dir / sub
        if src_dir.is_dir():
            for f in sorted(src_dir.iterdir()):
                if f.is_file():
                    cp(f, spec[spec_name])

    # ---- confusion matrix + calibration (derived from predictions) ------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix

    preds_csv = model_dir / "predictions" / "test_predictions.csv"
    metrics_json = model_dir / "metrics" / "test_metrics.json"
    if preds_csv.exists() and metrics_json.exists():
        m = json.loads(metrics_json.read_text())
        thr = m["test"]["threshold"]
        p = pd.read_csv(preds_csv)
        ycol = next(c for c in p.columns if c in ("y_true", "label", "y"))
        pcol = next(c for c in p.columns if c in ("p", "prob", "p_ad",
                                                  "proba", "probability"))
        y = p[ycol].to_numpy(int)
        prob = p[pcol].to_numpy(float)
        cm = confusion_matrix(y, (prob >= thr).astype(int))
        fig, ax = plt.subplots(figsize=(4.4, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred CN", "pred AD"])
        ax.set_yticks([0, 1], ["true CN", "true AD"])
        ax.set_title(f"Confusion matrix (threshold {thr:.2f}, frozen)")
        fig.colorbar(im)
        fig.savefig(spec["confusion_matrix"] / "confusion_matrix.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)
        (spec["confusion_matrix"] / "confusion_matrix.json").write_text(
            json.dumps({"matrix": cm.tolist(), "threshold": float(thr),
                        "labels": ["CN", "AD"]}, indent=2))
        manifest["results/confusion_matrix/confusion_matrix.png"] = \
            "derived from " + str(preds_csv.relative_to(ROOT))
        manifest["results/confusion_matrix/confusion_matrix.json"] = \
            "derived from " + str(preds_csv.relative_to(ROOT))

        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=8,
                                                strategy="quantile")
        fig, ax = plt.subplots(figsize=(4.4, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(mean_pred, frac_pos, "o-", color="#2b5d8c",
                label=f"model (Brier {m['test']['brier']:.3f})")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed AD fraction")
        ax.set_title("Calibration (held-out test)")
        ax.legend(fontsize=8)
        fig.savefig(spec["calibration"] / "calibration.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)
        manifest["results/calibration/calibration.png"] = \
            "derived from " + str(preds_csv.relative_to(ROOT))

    # ---- transcriptomics / nulls (already produced by the legacy pipeline)
    derived = ROOT / "data" / "derived"
    for f in sorted(derived.glob("expression_dk68_tianS1*")):
        cp(f, spec["ahba"])
    cp(ROOT / "results" / "gene_sets" / "alzheimer_gwascat.json", spec["gwas"])
    cp(ROOT / "results" / "gene_sets" / "alzheimer_gwascat_metadata.json",
       spec["gwas"])
    bridge = ROOT / "results" / "bridge"
    if bridge.is_dir():
        for f in sorted(bridge.glob("*.json")):
            cp(f, spec["matched_null"] if "matched" in f.name or
               f.name == "bridge_results.json" else spec["spatial_null"])
        for f in sorted(bridge.glob("*.npy")):
            cp(f, spec["matched_null"])
    cp(ROOT / "results" / "donor_robustness" / "cnn_gene_loo.json",
       spec["donor_robustness"])
    for f in sorted((ROOT / "results" / "multi_seed").glob("summary*.json")):
        cp(f, spec["metrics"])

    (ROOT / "results" / "spec_tree_manifest.json").write_text(
        json.dumps({"model_dir": str(model_dir), "sources": manifest},
                   indent=2))
    print(f"collected {len(copied)} artifact groups into the Part AD tree")
    for dst, src in sorted(manifest.items()):
        print(f"  {dst}  <-  {src}")
    missing = [k for k in spec
               if not any(Path(p).parts[1] == k for p in manifest)]
    if missing:
        print("spec dirs without new artifacts: " + ", ".join(missing),
              file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="results/ml/resnet_seed42",
                    help="primary model run to mirror into the spec tree")
    args = ap.parse_args(argv)
    return collect(ROOT / args.model_dir)


if __name__ == "__main__":
    sys.exit(main())
