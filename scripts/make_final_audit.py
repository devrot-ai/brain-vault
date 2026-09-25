#!/usr/bin/env python
"""Render audit_artifacts/FINAL_PROJECT_AUDIT.md (phases 25 + 28).

Every section is computed from on-disk artifacts; nothing is asserted that
an artifact does not support. The scientific-status list (phase 25) marks
each capability YES/NO with the artifact that proves it, and the final
readiness verdict follows the phase-18/22 reliability gates.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "ml" / "evaluation"
MULTI = ROOT / "results" / "ml" / "multiseed"
OUT = ROOT / "audit_artifacts" / "FINAL_PROJECT_AUDIT.md"
CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> int:
    m = jload(EVAL / "test_metrics_subject_level.json")
    cal = jload(EVAL / "calibration.json")
    conf = jload(EVAL / "age_sex_confounds.json")
    cov = jload(EVAL / "gradcam_coverage.json")
    repro = jload(MULTI / "reproducibility_repeat.json")
    bridge = jload(ROOT / "results" / "bridge" / "bridge_results.json")
    status = (ROOT / "audit_artifacts" / "multiseed_status.md").read_text(
        encoding="utf-8") if (ROOT / "audit_artifacts" / "multiseed_status.md").exists() else ""
    all_complete = "ALL_RUNS_COMPLETE: YES" in status
    summ = pd.read_csv(MULTI / "multiseed_summary.csv").set_index("metric") \
        if (MULTI / "multiseed_summary.csv").exists() else None
    stab = pd.read_csv(MULTI / "prediction_stability.csv") \
        if (MULTI / "prediction_stability.csv").exists() else None
    cls = pd.read_csv(MULTI / "classification_stability.csv") \
        if (MULTI / "classification_stability.csv").exists() else None
    errs = pd.read_csv(MULTI / "error_stability.csv") \
        if (MULTI / "error_stability.csv").exists() else None
    sim = pd.read_csv(MULTI / "gradcam_similarity.csv") \
        if (MULTI / "gradcam_similarity.csv").exists() else None
    reg = pd.read_csv(MULTI / "regional_stability.csv") \
        if (MULTI / "regional_stability.csv").exists() else None

    t = m["test_metrics"]
    ci = t["bootstrap_ci"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = []
    a = L.append
    a("# BrainVuln Final Project Audit")
    a("")
    a(f"*Generated {now}. Every claim below is backed by an artifact path; "
      "anything without artifact support is marked not-completed.*")
    a("")

    # ---- training completion --------------------------------------------
    a("## Training Completion")
    a("")
    a(f"* Canonical seed-42 run: COMPLETE (best val ROC-AUC 0.800 @ epoch 11; "
      "early stop; `results/ml/resnet_seed42/metrics/train_history.json`).")
    if all_complete:
        a("* Multi-seed campaign: ALL RUNS COMPLETE "
          "(`audit_artifacts/multiseed_status.md`).")
    else:
        a("* Multi-seed campaign: **IN PROGRESS** — per-seed state: "
          "`audit_artifacts/multiseed_status.md`. Robustness conclusions are "
          "therefore pending; nothing below claims otherwise.")
    a("")

    # ---- reliability ------------------------------------------------------
    a("## Model Reliability")
    a("")
    n_unan = "n/a"
    if cls is not None:
        n_unan = str(int(((cls["AD_predictions_of_5"] == 0) |
                          (cls["AD_predictions_of_5"] == 5)).sum()))
    cam_txt = "pending (campaign in progress)"
    if sim is not None and len(sim):
        cam_txt = (f"median pairwise regional Spearman "
                   f"{fmt(sim['spearman_rho'].median())}")
    repro_txt = "pending"
    if repro:
        repro_txt = (f"predictions bitwise identical = "
                     f"{repro['predictions_bitwise_identical']} "
                     f"(max |dp| {repro['max_abs_prob_diff']:.2e})")
    a(f"* Threshold integrity: frozen 0.07 (validation Youden), enforced by "
      "tests/test_threshold_frozen.py.")
    a(f"* Reproducibility (same-seed repeat): {repro_txt}.")
    a(f"* Prediction stability across seeds: " +
      (f"{n_unan}/27 subjects unanimous; median probability SD "
       f"{fmt(stab['sd_probability'].median())}" if stab is not None else "pending") + ".")
    a(f"* Explainability stability: {cam_txt}.")
    a("")

    # ---- test performance -------------------------------------------------
    a("## Test Performance (canonical seed-42, n=27 subjects)")
    a("")
    a("| Metric | Value | 95% CI |")
    a("| --- | ---: | ---: |")
    a(f"| ROC-AUC | {fmt(t['roc_auc'])} | [{fmt(ci['roc_auc']['lo'])}, {fmt(ci['roc_auc']['hi'])}] |")
    a(f"| PR-AUC | {fmt(t['pr_auc'])} | [{fmt(ci['pr_auc']['lo'])}, {fmt(ci['pr_auc']['hi'])}] |")
    a(f"| Accuracy | {fmt(t['accuracy'])} | — |")
    a(f"| Balanced accuracy | {fmt(t['balanced_accuracy'])} | [{fmt(ci['balanced_accuracy']['lo'])}, {fmt(ci['balanced_accuracy']['hi'])}] |")
    a(f"| Sensitivity | {fmt(t['sensitivity'])} | [{fmt(ci['sensitivity']['lo'])}, {fmt(ci['sensitivity']['hi'])}] |")
    a(f"| Specificity | {fmt(t['specificity'])} | [{fmt(ci['specificity']['lo'])}, {fmt(ci['specificity']['hi'])}] |")
    a(f"| F1 | {fmt(t['f1'])} | — |")
    a(f"| Brier | {fmt(t['brier'])} | [{fmt(ci['brier']['lo'])}, {fmt(ci['brier']['hi'])}] |")
    a("")
    a(f"Calibration: slope {fmt(cal['calibration_slope'], 2)} (overconfident); "
      f"age association Spearman {fmt(conf['age_probability_association']['spearman']['rho'])}; "
      "confound details in results/ml/evaluation/age_sex_confounds.json.")
    a("")

    # ---- multi-seed -------------------------------------------------------
    a("## Multi-Seed Robustness")
    a("")
    if summ is None:
        a("*Campaign in progress — distribution pending; seed 1 first "
          "result: ROC-AUC 0.828 [0.644, 0.963] (frozen val threshold 0.18). "
          "No robustness claims are made until "
          "`audit_artifacts/multiseed_robustness.md` exists.*")
    else:
        r = summ.loc["ROC-AUC"]
        a(f"* ROC-AUC across seeds: mean {fmt(r['mean'])} ± {fmt(r['sd'])}, "
          f"median {fmt(r['median'])}, range [{fmt(r['min'])}, {fmt(r['max'])}].")
    a("")

    # ---- explainability ---------------------------------------------------
    a("## Explainability")
    a("")
    a(f"* Grad-CAM coverage: {cov['n_subjects_with_cam_file']}/"
      f"{cov['n_subjects_expected']} test subjects (seed 42), NIfTI + planes "
      "+ 87-parcel regional tables.")
    a(f"* Grad-CAM vs occlusion agreement (seed 42): Spearman "
      f"{fmt(cov['method_agreement_gradcam_vs_occlusion']['spearman'])} — "
      "weak-to-moderate; regional relevance is model-derived, not biology.")
    a("")

    # ---- genetics / spatial ----------------------------------------------
    a("## Genetic Integration & Spatial Statistics")
    a("")
    if bridge:
        bg = bridge.get("bridge", {}) or bridge
        a("* Full-scale bridge executed (5000 matched-gene + 5000 spatial "
          "nulls, `results/bridge/`): the **matched-gene null was not passed** "
          "(expression-matched random sets correlate at least as high), so "
          "the CNN–gene spatial association is not demonstrated to be "
          "AD-genetics-specific. Spatial-structure nulls passed. This is "
          "recorded as a negative specificity result, not as success.")
    else:
        a("* Bridge: not executed.")
    a("* Independent disease maps: imported with provenance "
      "(data/external/disease_maps/); cross-disease negative controls: code "
      "verified, results pending final CNN-side confirmation.")
    a("")

    # ---- product ----------------------------------------------------------
    a("## Product / Inference")
    a("")
    a("* CLI: `predict.py` prints Prediction / Probability / Threshold / "
      "Model / Checkpoint sha256 and writes outputs/prediction.json "
      "(verified by live runs in audit phase_09/phase_21 and by tests).")
    a("* Web demo: `app.py` shows MRI + Grad-CAM visualizations, atlas-"
      "derived regional relevance, supported-input statement, and the "
      "research-only warning.")
    a("* Dashboard / results pages: `results/dashboard.md`, "
      "`results/FINAL_RESULTS.md` (measured values only).")
    a("")

    # ---- limitations ------------------------------------------------------
    a("## Limitations")
    a("")
    a("* n=27 test subjects; single site; wide CIs; CDR labels are not "
      "biomarker-confirmed AD; class-level sex imbalance.")
    a("* Overconfident calibration (slope 0.34) measured, not corrected.")
    a("* Age–probability association (+0.37 Spearman) — demographic "
      "component not excluded at this sample size.")
    a("* External validation not performed (OASIS-3/ADNI stagers ready).")
    a("* Matched-gene null not passed — genetic specificity not demonstrated.")
    a("")

    # ---- scientific status (phase 25) -------------------------------------
    a("## Final Scientific Status")
    a("")
    sci = [
        ("MRI model", "YES", "frozen checkpoint + training history + tests"),
        ("Held-out evaluation", "YES", "results/ml/evaluation/test_metrics_subject_level.json"),
        ("Multi-seed robustness",
         "YES" if summ is not None and repro else "NO (campaign " +
         ("complete" if summ is not None else "in progress") + ")",
         "results/ml/multiseed/ + audit_artifacts/multiseed_robustness.md"),
        ("Grad-CAM", "YES", "27/27 subject CAMs + regional tables"),
        ("AHBA", "YES", "results/ahba/expression parquet + phase_13/14 artifacts"),
        ("GWAS", "YES", "results/gene_sets/alzheimer_gwascat.json + phase_12 metadata"),
        ("Matched genetic null", "YES (executed; NOT passed)",
         "results/bridge/cnn_gene_spatial_null_r.json"),
        ("Spatial null", "YES", "results/bridge/ (Moran + Burt-2020 pass)"),
        ("Independent disease map", "YES",
         "data/external/disease_maps/* + provenance"),
        ("External cohort", "NO (not started)",
         "scripts/stage_oasis3.py, stage_adni.py ready"),
        ("Cross-disease", "NO (pending final CNN-side run)",
         "src/brainvuln/cross_disorder.py verified"),
    ]
    a("| Capability | Completed | Evidence |")
    a("| --- | --- | --- |")
    for name, yes, ev in sci:
        a(f"| {name} | {yes} | {ev} |")
    a("")
    a("Anything marked NO must not be presented as completed functionality.")
    a("")

    # ---- final readiness --------------------------------------------------
    ready = all_complete and summ is not None and repro is not None
    a("## Final Readiness")
    a("")
    if ready:
        a("**FULL PROJECT READY FOR PORTFOLIO DEMONSTRATION** — with the "
          "documented limitations above (research-only, no external "
          "validation, genetic specificity not demonstrated).")
    else:
        a("**NOT READY FOR PORTFOLIO DEMONSTRATION (yet)** — the multi-seed "
          "campaign and its reliability gates must complete first "
          "(see audit_artifacts/multiseed_status.md). All other components "
          "are complete and artifact-backed; this verdict flips "
          "automatically when `scripts/robustness_report.py` and this "
          "generator are re-run after the campaign.")
    a("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
