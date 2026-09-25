#!/usr/bin/env python
"""Subject-level evaluation of the FROZEN model (REAL MODEL EVALUATION).

    python scripts/eval_subject_level.py [--n-boot 2000] [--seed 42]

Runs, in order, entirely on the canonical checkpoint
(results/ml/resnet_seed42/checkpoints/best.pt, SHA-pinned):

  A. threshold re-derivation on VALIDATION subjects only (Youden's J) +
     validation-only subject-level bootstrap stability
     -> audit_artifacts/threshold_selection.{json,md}          (phases 1-2)
  B. frozen-threshold evaluation on the untouched TEST subjects
     -> results/ml/evaluation/                                 (phases 5-8)
  C. error analysis, age/sex confound checks, calibration      (phases 9-12)
  D. attribution table from the frozen run's Grad-CAM regional maps
     -> results/ml/evaluation/attribution_table.csv            (phases 13-14)

The model is loaded, never trained; the canonical checkpoint file is opened
read-only. The threshold step completes BEFORE any test loader is built, and
no test label or probability is consulted anywhere in phase A.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainvuln.config import (  # noqa: E402
    checkpoint_identity,
    resolve_checkpoint,
)
from brainvuln.mri.dataset import (  # noqa: E402
    OASISDataset,
    collate_items,
    load_cohort,
    load_split_tables,
)
from brainvuln.mri.evaluate import (  # noqa: E402
    classification_metrics,
    predict_loader,
    select_threshold,
    subject_bootstrap_ci,
)
from brainvuln.mri.models import ResNet18Binary  # noqa: E402
from brainvuln.mri.preprocess import PREPROCESS_VERSION  # noqa: E402

AUDIT_DIR = ROOT / "audit_artifacts"
EVAL_DIR = ROOT / "results" / "ml" / "evaluation"
CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
RUN_DIR = ROOT / "results" / "ml" / "resnet_seed42"


def make_loader(ds, batch_size, seed):
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_items, generator=g)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# phase A — threshold on validation only
# ---------------------------------------------------------------------------

def threshold_phase(model, device, args, identity) -> dict:
    thr_file = AUDIT_DIR / "threshold_selection.json"
    if thr_file.exists():
        rec = json.loads(thr_file.read_text(encoding="utf-8"))
        if (rec.get("checkpoint_sha256") == identity["sha256"]
                and rec.get("agreement_with_run_freeze", {}).get("match")):
            print("[A] resume: threshold_selection.json exists -> reusing "
                  "(no validation re-inference)", flush=True)
            return {"threshold": float(rec["selected_threshold"]),
                    "method": rec["selection"]["method"],
                    "val_metrics": rec["validation_metrics_at_threshold"],
                    "val_roc": rec["validation_roc_auc"],
                    "val_pr": rec["validation_pr_auc"]}
        print("[A] existing threshold_selection.json does not match this "
              "checkpoint -> recomputing", flush=True)
    print("[A] threshold on VALIDATION subjects only (no test contact)",
          flush=True)
    splits = load_split_tables(ROOT / "data" / "splits")
    cohort = load_cohort(ROOT / "data" / "splits")
    val_ds = OASISDataset(splits["val"], ROOT / "data" / "raw" / "oasis1" / "extracted",
                          ROOT / "data" / "derived" / "oasis1_128", cohort=cohort)
    val_loader = make_loader(val_ds, args.batch_size, args.seed)
    yv, pv, _, vsubs = predict_loader(model, val_loader, device)

    method = "youden"
    threshold = select_threshold(yv, pv, method)
    val_metrics = classification_metrics(yv, pv, threshold)
    from sklearn.metrics import roc_auc_score, average_precision_score
    val_roc = float(roc_auc_score(yv, pv))
    val_pr = float(average_precision_score(yv, pv))

    # validation-only subject bootstrap: stability of the selected threshold
    rng = np.random.default_rng(args.seed)
    vdf = pd.DataFrame({"y": yv, "p": pv, "s": list(vsubs)})
    subj_ids = vdf["s"].unique()
    by_subj = {key: g[["y", "p"]].to_numpy() for key, g in vdf.groupby("s")}
    t0 = time.time()
    boot = []
    for _ in range(args.n_boot):
        pick = rng.choice(subj_ids, size=len(subj_ids), replace=True)
        ys = np.concatenate([by_subj[s][:, 0] for s in pick])
        ps = np.concatenate([by_subj[s][:, 1] for s in pick])
        if len(set(ys.tolist())) < 2:
            continue
        boot.append(select_threshold(ys, ps, method))
    boot = np.asarray(boot, dtype=float)
    print(f"    bootstrap of the selection: {len(boot)} draws in "
          f"{time.time() - t0:.0f}s", flush=True)

    # agreement with the value frozen by the original run
    metrics_file = RUN_DIR / "metrics" / "test_metrics.json"
    run_frozen = float(json.loads(metrics_file.read_text(encoding="utf-8"))["threshold"])

    record = {
        "purpose": ("REAL MODEL EVALUATION phases 1-2: operating-point "
                    "re-derivation on validation subjects only"),
        "created_utc": now_utc(),
        "checkpoint_sha256": identity["sha256"],
        "checkpoint_identity": identity,
        "preprocess_version": PREPROCESS_VERSION,
        "selection": {
            "method": method,
            "definition": ("maximize sensitivity + specificity - 1 (Youden's J) "
                           "over a 197-point grid on [0.01, 0.99]"),
            "data": f"validation subjects only (n={len(yv)} sessions/subjects)",
            "rationale": (
                "Youden's J treats sensitivity and specificity symmetrically and "
                "is the method already fixed in configs/oasis1_resnet3d.yaml "
                "(threshold.method: youden) BEFORE this evaluation; re-using it "
                "keeps the operating point a documented, data-driven choice made "
                "without any test information."),
        },
        "selected_threshold": float(threshold),
        "validation_metrics_at_threshold": val_metrics,
        "validation_roc_auc": val_roc,
        "validation_pr_auc": val_pr,
        "bootstrap": {
            "unit": "subject",
            "n_boot_requested": args.n_boot,
            "n_boot_used": int(len(boot)),
            "seed": args.seed,
            "median": float(np.median(boot)),
            "ci95_lo": float(np.percentile(boot, 2.5)),
            "ci95_hi": float(np.percentile(boot, 97.5)),
            "note": ("percentile interval of thresholds re-selected on validation "
                     "subject resamples; quantifies operating-point stability"),
        },
        "agreement_with_run_freeze": {
            "run_frozen_threshold": run_frozen,
            "recomputed_threshold": float(threshold),
            "match": bool(abs(run_frozen - threshold) < 1e-9),
        },
        "test_contact": False,
    }
    AUDIT_DIR.mkdir(exist_ok=True)
    (AUDIT_DIR / "threshold_selection.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    write_threshold_md(record)
    print(f"    selected threshold {threshold:.2f} "
          f"(run freeze: {run_frozen:.2f}, match={record['agreement_with_run_freeze']['match']})",
          flush=True)
    return {"threshold": float(threshold), "method": method,
            "val_metrics": val_metrics, "val_roc": val_roc, "val_pr": val_pr}


def write_threshold_md(rec: dict) -> None:
    v = rec["validation_metrics_at_threshold"]
    b = rec["bootstrap"]
    lines = [
        "# Threshold selection (REAL MODEL EVALUATION, phases 1-2)",
        "",
        f"*Created {rec['created_utc']} — validation subjects only; no test contact "
        f"(`test_contact: false` in threshold_selection.json).*",
        "",
        "## Method",
        "",
        f"* **Method:** Youden's J — maximize sensitivity + specificity − 1 over a "
        f"197-point grid on [0.01, 0.99] (the tested `select_threshold(..., \"youden\")`).",
        "* **Why:** symmetric treatment of sensitivity/specificity; already fixed as "
        "`threshold.method: youden` in `configs/oasis1_resnet3d.yaml` before this phase, "
        "so the re-derivation changes no pre-registered choice.",
        f"* **Data:** validation subjects only (n = {v['n']}). The threshold was frozen "
        "before the test loader was ever constructed in this evaluation.",
        f"* **Checkpoint:** `{rec['checkpoint_identity']['filename']}` "
        f"(sha256 `{rec['checkpoint_sha256'][:16]}…`), "
        f"preprocessing `{rec['preprocess_version']}`.",
        "",
        "## Selected operating point",
        "",
        f"* **Threshold = {rec['selected_threshold']:.2f}**",
        f"* Validation ROC-AUC {rec['validation_roc_auc']:.3f} "
        f"(PR-AUC {rec['validation_pr_auc']:.3f}) — discrimination context only.",
        "",
        "| Validation metric @ threshold | Value |",
        "| --- | ---: |",
        f"| Sensitivity | {v['sensitivity']:.3f} |",
        f"| Specificity | {v['specificity']:.3f} |",
        f"| Balanced accuracy | {v['balanced_accuracy']:.3f} |",
        f"| Accuracy | {v['accuracy']:.3f} |",
        f"| F1 | {v['f1']:.3f} |",
        "",
        "## Bootstrap stability (validation subjects only)",
        "",
        f"* {b['n_boot_used']} resamples of the {v['n']} validation subjects "
        f"(unit = subject, seed {b['seed']}); Youden re-selected per resample.",
        f"* **Median** {b['median']:.3f} — **95% interval** "
        f"[{b['ci95_lo']:.3f}, {b['ci95_hi']:.3f}]",
        "* Interpretation: with n=27 validation subjects the operating point is only "
        "moderately stable; the interval quantifies that uncertainty and is NOT a "
        "recommendation to re-tune per dataset.",
        "",
        "## Freeze",
        "",
        f"* The value {rec['selected_threshold']:.2f} is frozen as "
        "`inference.threshold.value` in config/analysis.yaml; serving code resolves it "
        "through `brainvuln.config.resolve_threshold()` "
        "(tests/test_threshold_frozen.py pins the value).",
        f"* Agreement with the original run's frozen threshold: "
        f"{rec['agreement_with_run_freeze']['run_frozen_threshold']:.2f} "
        f"(match = {rec['agreement_with_run_freeze']['match']}).",
        "",
        "## Test-set integrity",
        "",
        "No test label, prediction, metric, ROC curve, or confusion matrix was used in "
        "any step above. Test evaluation happens only in phase B, after this file was "
        "written.",
    ]
    (AUDIT_DIR / "threshold_selection.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# phase B — frozen test evaluation
# ---------------------------------------------------------------------------

def test_phase(model, device, args, identity, threshold_info) -> dict:
    threshold = threshold_info["threshold"]
    print(f"[B] frozen-threshold evaluation on TEST subjects (threshold {threshold:.2f})",
          flush=True)
    splits = load_split_tables(ROOT / "data" / "splits")
    cohort = load_cohort(ROOT / "data" / "splits")
    # resume support: cache the (expensive, CPU) test predictions so a crash
    # in later stages never repeats inference. The cache is keyed to nothing
    # but this process run; the checkpoint SHA is asserted in main() before
    # any of this executes.
    cache = EVAL_DIR / "test_arrays_cache.npz"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        yt, pt = z["y"], z["p"]
        tsids = [str(x) for x in z["sids"]]
        tsubs = [str(x) for x in z["subs"]]
        print(f"[B] resume: reusing cached test predictions ({len(yt)} subjects)",
              flush=True)
    else:
        test_ds = OASISDataset(splits["test"],
                               ROOT / "data" / "raw" / "oasis1" / "extracted",
                               ROOT / "data" / "derived" / "oasis1_128", cohort=cohort)
        test_loader = make_loader(test_ds, args.batch_size, args.seed)
        yt, pt, tsids, tsubs = predict_loader(model, test_loader, device)
        np.savez(cache, y=yt, p=pt, sids=np.array(tsids), subs=np.array(tsubs))

    # subject-level unit check: one canonical scan per subject
    per_subject = pd.Series(tsubs).value_counts()
    multi = per_subject[per_subject > 1]
    assert len(multi) == 0, (
        f"test subjects with multiple sessions would need an aggregation rule; "
        f"found {multi.to_dict()}")
    n_ad = int((yt == 1).sum())
    n_cn = int((yt == 0).sum())

    test_metrics = classification_metrics(yt, pt, threshold)
    ci = subject_bootstrap_ci(yt, pt, tsubs, threshold, n_boot=args.n_boot,
                              seed=args.seed)
    test_metrics["bootstrap_ci"] = ci

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "subject_id": list(tsubs),
        "session_id": list(tsids),
        "diagnosis": np.where(yt == 1, "AD", "CN"),
        "probability": pt,
        "prediction": np.where(pt >= threshold, "AD", "CN"),
        "threshold": threshold,
        "checkpoint_sha256": identity["sha256"],
        "preprocess_version": PREPROCESS_VERSION,
    })
    pred_df.to_csv(EVAL_DIR / "test_predictions_subject_level.csv", index=False)

    metrics_record = {
        "purpose": "REAL MODEL EVALUATION phases 5-8 (subject-level test evaluation)",
        "created_utc": now_utc(),
        "checkpoint_identity": identity,
        "checkpoint_sha256": identity["sha256"],
        "preprocess_version": PREPROCESS_VERSION,
        "threshold": threshold,
        "threshold_method": threshold_info["method"],
        "threshold_selected_on": "validation (see audit_artifacts/threshold_selection.md)",
        "subject_unit": ("canonical scan per subject: the test partition contains "
                         "exactly one session per subject (verified by assertion)"),
        "n_subjects": int(len(yt)), "n_ad": n_ad, "n_cn": n_cn,
        "test_metrics": test_metrics,
    }
    (EVAL_DIR / "test_metrics_subject_level.json").write_text(
        json.dumps(metrics_record, indent=2), encoding="utf-8")

    curves(yt, pt, test_metrics, threshold)
    print(f"    test ROC-AUC {test_metrics['roc_auc']:.3f} "
          f"[{ci['roc_auc']['lo']:.3f}, {ci['roc_auc']['hi']:.3f}] "
          f"PR-AUC {test_metrics['pr_auc']:.3f}", flush=True)
    return {"pred_df": pred_df, "metrics": test_metrics, "ci": ci,
            "yt": yt, "pt": pt, "n_ad": n_ad, "n_cn": n_cn}


def curves(yt, pt, m, threshold) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                                 roc_curve)
    from sklearn.linear_model import LogisticRegression

    # ROC
    fpr, tpr, roc_thr = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC — frozen checkpoint (subject level)")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(EVAL_DIR / "roc_curve.png", dpi=150); plt.close(fig)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(
        EVAL_DIR / "roc_curve_data.csv", index=False)

    # PR
    prec, rec, pr_thr = precision_recall_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(rec, prec, lw=2, label=f"PR (AP = {m['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Test precision-recall — frozen checkpoint")
    ax.legend(); fig.tight_layout()
    fig.savefig(EVAL_DIR / "pr_curve.png", dpi=150); plt.close(fig)
    # prec/rec have one more point than pr_thr; pad thresholds with NaN for
    # the anchor point (precision=1, recall=0)
    pd.DataFrame({"precision": prec, "recall": rec,
                  "threshold": list(pr_thr) + [float("nan")]}).to_csv(
        EVAL_DIR / "pr_curve_data.csv", index=False)

    # Confusion matrix at the frozen threshold
    cm = confusion_matrix(yt, (pt >= threshold).astype(int), labels=[0, 1])
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["CN", "AD"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Test confusion (frozen threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)
    pd.DataFrame(cm, index=["true_CN", "true_AD"],
                 columns=["pred_CN", "pred_AD"]).to_csv(EVAL_DIR / "confusion_matrix.csv")

    # Calibration (reliability): quantile bins for n=27
    n_bins = 4
    frac_pos, mean_prob = calibration_curve(yt, pt, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfectly calibrated")
    ax.plot(mean_prob, frac_pos, "o-", label=f"model (quantile bins, {n_bins})")
    ax.set_xlabel("Mean predicted AD probability"); ax.set_ylabel("Observed AD fraction")
    ax.set_title("Reliability diagram — test subjects")
    ax.legend(); fig.tight_layout()
    fig.savefig(EVAL_DIR / "calibration.png", dpi=150); plt.close(fig)

    # slope/intercept: logistic recalibration on the logit of the prediction
    eps = 1e-6
    lg = np.log(np.clip(pt, eps, 1 - eps) / (1 - np.clip(pt, eps, 1 - eps)))
    cal = LogisticRegression().fit(lg.reshape(-1, 1), yt)
    binned = pd.DataFrame({"mean_predicted_prob": mean_prob, "observed_fraction": frac_pos,
                           "n_bins": n_bins, "strategy": "quantile"})
    binned.to_csv(EVAL_DIR / "calibration_curve.csv", index=False)
    (EVAL_DIR / "calibration.json").write_text(json.dumps({
        "created_utc": now_utc(),
        "brier": m["brier"],
        "calibration_slope": float(cal.coef_[0][0]),
        "calibration_intercept": float(cal.intercept_[0]),
        "slope_intercept_note": ("logistic fit of the label on logit(p): slope 1 / "
                                 "intercept 0 = perfectly calibrated; slope < 1 = "
                                 "overconfident, slope > 1 = underconfident"),
        "reliability_bins": binned.to_dict(orient="records"),
        "note": ("discrimination (ROC-AUC) and calibration are separate properties; "
                 "a high AUC does not imply well-calibrated probabilities"),
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# phase C — error analysis + confounds
# ---------------------------------------------------------------------------

def error_and_confounds(pred_df, m, threshold_info) -> None:
    print("[C] error analysis + age/sex confound checks", flush=True)
    cohort = pd.read_csv(ROOT / "data" / "splits" / "matched_cohort.csv")
    df = pred_df.merge(cohort[["subject", "Age", "Sex", "eTIV", "MMSE", "Educ"]],
                       left_on="subject_id", right_on="subject", how="left")
    y = (df["diagnosis"] == "AD").astype(int).to_numpy()
    pred = (df["probability"] >= df["threshold"]).astype(int).to_numpy()
    group = np.where((y == 1) & (pred == 1), "TP",
                     np.where((y == 0) & (pred == 0), "TN",
                              np.where((y == 0) & (pred == 1), "FP", "FN")))
    df["error_group"] = group
    df.drop(columns=["subject"]).to_csv(EVAL_DIR / "error_analysis.csv", index=False)

    # ---- error group summary ----
    lines = ["# Error analysis (frozen checkpoint, test subjects)", "",
             f"Threshold {df['threshold'].iloc[0]:.2f} "
             f"(frozen, validation-selected). Probabilities are mean predicted "
             f"AD-probabilities. Age/Sex from data/splits/matched_cohort.csv.", "",
             "| Group | n | mean prob | prob range | mean age | age range | F | M |",
             "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |"]
    summary = {}
    for g in ("TP", "TN", "FP", "FN"):
        sub = df[df["error_group"] == g]
        summary[g] = {
            "n": int(len(sub)),
            "probability": {"mean": float(sub["probability"].mean()),
                            "min": float(sub["probability"].min()),
                            "max": float(sub["probability"].max())},
            "age": {"mean": float(sub["Age"].mean()),
                    "sd": float(sub["Age"].std()),
                    "min": float(sub["Age"].min()), "max": float(sub["Age"].max())},
            "sex": {"F": int((sub["Sex"] == "F").sum()), "M": int((sub["Sex"] == "M").sum())},
            "subjects": sub["subject_id"].tolist(),
        }
        lines.append(
            f"| {g} | {len(sub)} | {sub['probability'].mean():.3f} | "
            f"[{sub['probability'].min():.3f}, {sub['probability'].max():.3f}] | "
            f"{sub['Age'].mean():.1f} | [{sub['Age'].min():.0f}, {sub['Age'].max():.0f}] | "
            f"{(sub['Sex'] == 'F').sum()} | {(sub['Sex'] == 'M').sum()} |")
    lines += ["", "## False positives (CN called AD by the model)", ""]
    for _, r in df[df["error_group"] == "FP"].iterrows():
        lines.append(f"* `{r['subject_id']}` — prob {r['probability']:.3f}, "
                     f"age {r['Age']:.0f}, sex {r['Sex']}, MMSE {r['MMSE']}")
    lines += ["", "## False negatives (AD called CN by the model)", ""]
    for _, r in df[df["error_group"] == "FN"].iterrows():
        lines.append(f"* `{r['subject_id']}` — prob {r['probability']:.3f}, "
                     f"age {r['Age']:.0f}, sex {r['Sex']}, MMSE {r['MMSE']}")
    lines += ["", "The test set exists to reveal these errors; no subject was "
                  "re-labeled, removed, or 'fixed'.", ""]
    (EVAL_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- age confounding ----
    from scipy.stats import pearsonr, spearmanr
    age = df["Age"].to_numpy(float)
    prob = df["probability"].to_numpy(float)
    sp = spearmanr(prob, age); pe = pearsonr(prob, age)
    bands = [("60-69", 60, 70), ("70-79", 70, 80), ("80+", 80, 200)]
    band_rows = []
    for name, lo, hi in bands:
        sel = (age >= lo) & (age < hi)
        band_rows.append({"band": name, "n": int(sel.sum()),
                          "mean_prob": float(prob[sel].mean()) if sel.any() else None,
                          "mean_prob_sd": float(prob[sel].std()) if sel.sum() > 1 else None})

    # age/sex comparator fitted on TRAIN subjects only
    from sklearn.linear_model import LogisticRegression
    tr = cohort[cohort["split"] == "train"]
    Xtr = np.stack([tr["Age"].to_numpy(float), (tr["Sex"] == "M").to_numpy(float)], axis=1)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
    Xte = np.stack([age, (df["Sex"] == "M").to_numpy(float)], axis=1)
    base_prob = clf.predict_proba(Xte)[:, 1]
    from sklearn.metrics import roc_auc_score
    base_auc = float(roc_auc_score(y, base_prob))
    va = cohort[cohort["split"] == "val"]
    Xva = np.stack([va["Age"].to_numpy(float), (va["Sex"] == "M").to_numpy(float)], axis=1)
    base_val_thr = select_threshold(va["label"].to_numpy(int),
                                    clf.predict_proba(Xva)[:, 1], "youden")
    base_metrics = classification_metrics(y, base_prob, base_val_thr)

    confounds = {
        "created_utc": now_utc(),
        "age_probability_association": {
            "spearman": {"rho": float(sp.statistic), "p_value": float(sp.pvalue)},
            "pearson": {"r": float(pe.statistic), "p_value": float(pe.pvalue)},
            "note": ("correlation of predicted AD-probability with age in the test "
                     "set; association, NOT causation. Purpose: detect age-detector "
                     "behaviour, not to interpret age as a mechanism."),
        },
        "age_bands": band_rows,
        "age_sex_baseline_train_fitted": {
            "model": "logistic regression on (Age, Sex M), TRAIN subjects only",
            "val_youden_threshold": float(base_val_thr),
            "test_roc_auc": base_auc,
            "test_sensitivity": base_metrics["sensitivity"],
            "test_specificity": base_metrics["specificity"],
            "test_balanced_accuracy": base_metrics["balanced_accuracy"],
            "comparison_note": ("if the CNN's test ROC-AUC is not clearly above this "
                                "baseline, much of its signal may be demographic"),
        },
        "cnn_test_roc_auc": m["roc_auc"],
        "sex_stratified": {},
        "small_n_caveat": ("test n=27; per-sex subgroups are smaller still. Any "
                           "sex-stratified number is descriptive only; no strong "
                           "claims are made from these tiny subgroups."),
    }
    for sex in ("F", "M"):
        sel = (df["Sex"] == sex).to_numpy()
        ys, ps = y[sel], prob[sel]
        entry = {"n": int(sel.sum()),
                 "n_ad": int(ys.sum()), "n_cn": int((ys == 0).sum())}
        if len(set(ys.tolist())) == 2:
            entry["roc_auc"] = float(roc_auc_score(ys, ps))
        else:
            entry["roc_auc"] = None
            entry["roc_auc_note"] = "single class present — AUC undefined"
        preds = (ps >= df["threshold"].iloc[0]).astype(int)
        tp = int(((ys == 1) & (preds == 1)).sum()); fn = int(((ys == 1) & (preds == 0)).sum())
        tn = int(((ys == 0) & (preds == 0)).sum()); fp = int(((ys == 0) & (preds == 1)).sum())
        entry["sensitivity"] = tp / (tp + fn) if (tp + fn) else None
        entry["specificity"] = tn / (tn + fp) if (tn + fp) else None
        confounds["sex_stratified"][sex] = entry
    (EVAL_DIR / "age_sex_confounds.json").write_text(json.dumps(confounds, indent=2),
                                                     encoding="utf-8")
    print(f"    age-prob Spearman rho {sp.statistic:+.3f} (p {sp.pvalue:.3f}); "
          f"age+sex baseline test AUC {base_auc:.3f}", flush=True)


# ---------------------------------------------------------------------------
# phase D — attribution table from the frozen run's Grad-CAM regional maps
# ---------------------------------------------------------------------------

def attribution_table(pred_df, identity) -> dict:
    print("[D] attribution table from frozen-run Grad-CAM regional maps", flush=True)
    reg_dir = RUN_DIR / "regional_relevance"
    cam_dir = RUN_DIR / "gradcam"
    atlas = pd.read_csv(ROOT / "data" / "derived" / "atlas_dk68_tianS1_info.csv")
    label_by_id = dict(zip(atlas["label"].astype(str), atlas["structure"].astype(str)))

    rows, missing_cam = [], []
    for _, r in pred_df.iterrows():
        sub = r["subject_id"]
        cam_nii = cam_dir / f"{sub}_gradcam.nii.gz"
        if not cam_nii.exists():
            missing_cam.append(sub)
        per_subj = reg_dir / f"{sub}.csv"
        if not per_subj.exists():
            rows.append({"subject_id": sub, "diagnosis": r["diagnosis"],
                         "prediction": r["prediction"], "probability": r["probability"],
                         "top_region_1": None, "top_region_2": None, "top_region_3": None,
                         "mean_relevance": None, "n_parcels_valid": 0,
                         "cam_file_exists": cam_nii.exists()})
            continue
        tab = pd.read_csv(per_subj, index_col=0)
        val_col = tab.columns[0]
        s = tab[val_col].dropna().sort_values(ascending=False)
        top = s.index[:3].tolist()
        rows.append({
            "subject_id": sub, "diagnosis": r["diagnosis"],
            "prediction": r["prediction"], "probability": r["probability"],
            "top_region_1": top[0] if len(top) > 0 else None,
            "top_region_2": top[1] if len(top) > 1 else None,
            "top_region_3": top[2] if len(top) > 2 else None,
            "mean_relevance": float(s.mean()),
            "n_parcels_valid": int(len(s)),
            "cam_file_exists": cam_nii.exists(),
        })
    attr = pd.DataFrame(rows)
    attr.to_csv(EVAL_DIR / "attribution_table.csv", index=False)

    m_cnn = pd.read_csv(reg_dir / "M_CNN_regional.csv", index_col=0)["M_CNN"].dropna()
    top_regions = m_cnn.sort_values(ascending=False).head(10)
    coverage = {
        "created_utc": now_utc(),
        "checkpoint_sha256": identity["sha256"],
        "n_subjects_expected": int(len(pred_df)),
        "n_subjects_with_regional_csv": int((attr["n_parcels_valid"] > 0).sum()),
        "n_subjects_with_cam_file": int(attr["cam_file_exists"].sum()),
        "missing_cam_subjects": missing_cam,
        "region_names_source": "data/derived/atlas_dk68_tianS1_info.csv (atlas-derived; not hard-coded)",
        "population_top_regions": [
            {"region": k, "M_CNN": float(v),
             "structure": label_by_id.get(str(k), "")}
            for k, v in top_regions.items()],
        "method_agreement_gradcam_vs_occlusion": json.loads(
            (reg_dir / "method_agreement.json").read_text(encoding="utf-8")),
    }
    (EVAL_DIR / "gradcam_coverage.json").write_text(json.dumps(coverage, indent=2),
                                                    encoding="utf-8")
    print(f"    CAM files present {coverage['n_subjects_with_cam_file']}/"
          f"{coverage['n_subjects_expected']}; population top region: "
          f"{top_regions.index[0]}", flush=True)
    return coverage


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # canonical checkpoint: explicit path, read-only, SHA-pinned
    ckpt_path = resolve_checkpoint()
    identity = checkpoint_identity(str(ckpt_path))
    if identity["sha256"] != CANONICAL_SHA:
        print(f"FATAL: canonical checkpoint SHA mismatch: {identity['sha256']} "
              f"!= {CANONICAL_SHA} — aborting before any evaluation", file=sys.stderr)
        return 3
    print(f"[0] canonical checkpoint OK: {ckpt_path.name} sha {identity['sha256'][:16]}… "
          f"({identity['size_bytes'] / 1e6:.1f} MB)", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    threshold_info = threshold_phase(model, device, args, identity)
    test_info = test_phase(model, device, args, identity, threshold_info)
    error_and_confounds(test_info["pred_df"], test_info["metrics"], threshold_info)
    coverage = attribution_table(test_info["pred_df"], identity)

    m = test_info["metrics"]
    print("\n=== REAL MODEL EVALUATION (phases 1-14) complete ===", flush=True)
    print(f"  threshold (validation, {threshold_info['method']}): "
          f"{threshold_info['threshold']:.2f}", flush=True)
    print(f"  test n={m['n']} (AD {test_info['n_ad']} / CN {test_info['n_cn']}) "
          f"ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}", flush=True)
    print(f"  sensitivity {m['sensitivity']:.3f}  specificity {m['specificity']:.3f}  "
          f"brier {m['brier']:.3f}", flush=True)
    print(f"  confusion {m['confusion']}", flush=True)
    print(f"  Grad-CAM coverage {coverage['n_subjects_with_cam_file']}/"
          f"{coverage['n_subjects_expected']} subjects", flush=True)
    print(f"  artifacts: {EVAL_DIR} + {AUDIT_DIR}/threshold_selection.*", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
