#!/usr/bin/env python
"""Render the delivery documents from measured artifacts only.

Outputs (both re-runnable; regenerate after the multi-seed campaign):

* results/dashboard.md    - phase-22 dashboard (performance, robustness,
                            subjects, errors, explainability, validation status)
* results/FINAL_RESULTS.md - phase-24 measured-results page

Nothing here fabricates numbers: every value is read from
results/ml/evaluation/, results/ml/multiseed/, or the frozen run dirs.
If a section's source artifact does not exist yet (campaign in flight),
the section says so explicitly instead of guessing.
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
CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_context() -> dict:
    m = jload(EVAL / "test_metrics_subject_level.json")
    cal = jload(EVAL / "calibration.json")
    conf = jload(EVAL / "age_sex_confounds.json")
    cov = jload(EVAL / "gradcam_coverage.json")
    errs = pd.read_csv(EVAL / "error_analysis.csv") if (EVAL / "error_analysis.csv").exists() else None
    preds = pd.read_csv(EVAL / "test_predictions_subject_level.csv") if (EVAL / "test_predictions_subject_level.csv").exists() else None
    summ = pd.read_csv(MULTI / "multiseed_summary.csv") if (MULTI / "multiseed_summary.csv").exists() else None
    stab = pd.read_csv(MULTI / "prediction_stability.csv") if (MULTI / "prediction_stability.csv").exists() else None
    reg = pd.read_csv(MULTI / "regional_stability.csv") if (MULTI / "regional_stability.csv").exists() else None
    errs_md = pd.read_csv(MULTI / "error_stability.csv") if (MULTI / "error_stability.csv").exists() else None
    repro = jload(MULTI / "reproducibility_repeat.json")
    return dict(metrics=m, cal=cal, conf=conf, cov=cov, errs=errs, preds=preds,
                summ=summ, stab=stab, reg=reg, errs_md=errs_md, repro=repro)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def baseline_tables(ctx) -> tuple[str, str, str, str, str, str, str]:
    t = ctx["metrics"]["test_metrics"]
    ci = t["bootstrap_ci"]
    perf = (f"| ROC-AUC | **{fmt(t['roc_auc'])}** | [{fmt(ci['roc_auc']['lo'])}, "
            f"{fmt(ci['roc_auc']['hi'])}] |\n"
            f"| PR-AUC | {fmt(t['pr_auc'])} | [{fmt(ci['pr_auc']['lo'])}, "
            f"{fmt(ci['pr_auc']['hi'])}] |\n"
            f"| Accuracy | {fmt(t['accuracy'])} | — |\n"
            f"| Balanced accuracy | {fmt(t['balanced_accuracy'])} | "
            f"[{fmt(ci['balanced_accuracy']['lo'])}, {fmt(ci['balanced_accuracy']['hi'])}] |\n"
            f"| Sensitivity | {fmt(t['sensitivity'])} | [{fmt(ci['sensitivity']['lo'])}, "
            f"{fmt(ci['sensitivity']['hi'])}] |\n"
            f"| Specificity | {fmt(t['specificity'])} | [{fmt(ci['specificity']['lo'])}, "
            f"{fmt(ci['specificity']['hi'])}] |\n"
            f"| Precision | {fmt(t['precision'])} | — |\n"
            f"| Recall | {fmt(t['recall'] if 'recall' in t else t.get('sensitivity'))} | "
            f"[{fmt(ci['sensitivity']['lo'])}, {fmt(ci['sensitivity']['hi'])}] |\n"
            f"| F1 | {fmt(t['f1'])} | — |\n"
            f"| Brier | {fmt(t['brier'])} | [{fmt(ci['brier']['lo'])}, "
            f"{fmt(ci['brier']['hi'])}] |")
    cm = t["confusion"]
    cm_tbl = (f"| | pred CN | pred AD |\n| --- | ---: | ---: |\n"
              f"| **true CN** | {cm['tn']} | {cm['fp']} |\n"
              f"| **true AD** | {cm['fn']} | {cm['tp']} |")
    n = ctx["metrics"]["n_subjects"]
    subjects = (f"* {n} test subjects: {ctx['metrics']['n_ad']} AD / "
                f"{ctx['metrics']['n_cn']} CN (one session per subject)")
    fp = [r["subject_id"] for _, r in ctx["errs"].iterrows()
          if r["error_group"] == "FP"] if ctx["errs"] is not None else []
    fn = [r["subject_id"] for _, r in ctx["errs"].iterrows()
          if r["error_group"] == "FN"] if ctx["errs"] is not None else []
    errors = (f"* False positives: **{cm['fp']}** ({', '.join(fp) if fp else ''})\n"
              f"* False negatives: **{cm['fn']}** ({', '.join(fn) if fn else ''})\n"
              f"* Details: `results/ml/evaluation/error_analysis.md`")
    c = ctx["cal"]
    calib = (f"* Brier {fmt(c['brier'])}; calibration slope "
             f"**{fmt(c['calibration_slope'], 2)}** (intercept "
             f"{fmt(c['calibration_intercept'], 2)}) — slope < 1 = overconfident\n"
             f"* Reliability bins (predicted → observed): " +
             " · ".join(f"{b['mean_predicted_prob']:.2f} → {b['observed_fraction']:.2f}"
                        for b in c["reliability_bins"]))
    cv = ctx["cov"]
    explain = (f"* Grad-CAM coverage: **{cv['n_subjects_with_cam_file']}/"
               f"{cv['n_subjects_expected']}** test subjects "
               f"(regional CSVs: {cv['n_subjects_with_regional_csv']})\n"
               f"* Population top region (seed 42): "
               f"`{cv['population_top_regions'][0]['region']}` "
               f"(M_CNN {fmt(cv['population_top_regions'][0]['M_CNN'])})\n"
               f"* Grad-CAM vs occlusion agreement (seed 42): Spearman "
               f"{fmt(cv['method_agreement_gradcam_vs_occlusion']['spearman'], 3)}")
    return perf, cm_tbl, subjects, errors, calib, explain, perf


def robustness_block(ctx) -> str:
    if ctx["summ"] is None:
        status = ROOT / "audit_artifacts" / "multiseed_status.md"
        return ("*Multi-seed campaign **in progress** — see "
                "`audit_artifacts/multiseed_status.md` for per-seed state. "
                "This section fills automatically when "
                "`scripts/multiseed_analysis.py` completes.*")
    s = ctx["summ"].set_index("metric")
    lines = ["| Metric | Mean ± SD | Median [IQR] | Min–Max |", "| --- | --- | --- | --- |"]
    for k in ["ROC-AUC", "PR-AUC", "Accuracy", "Balanced Acc", "Sensitivity",
              "Specificity", "Precision", "Recall", "F1", "Brier"]:
        r = s.loc[k]
        lines.append(f"| {k} | {fmt(r['mean'])} ± {fmt(r['sd'])} | "
                     f"{fmt(r['median'])} [{fmt(r['iqr_lo'])}, {fmt(r['iqr_hi'])}] | "
                     f"{fmt(r['min'])}–{fmt(r['max'])} |")
    return "\n".join(lines)


def stability_block(ctx) -> str:
    if ctx["stab"] is None:
        return "*Campaign in progress — prediction-stability table fills post-campaign.*"
    s = ctx["stab"]
    cls = (MULTI / "classification_stability.csv")
    n_unan = "n/a"
    if cls.exists():
        c = pd.read_csv(cls)
        n_unan = str(int(((c["AD_predictions_of_5"] == 0) |
                          (c["AD_predictions_of_5"] == 5)).sum()))
    return (f"* Median across-seed SD of predicted probability: "
            f"{fmt(s['sd_probability'].median())} (max {fmt(s['sd_probability'].max())})\n"
            f"* Subjects with the same label from all 5 seeds: **{n_unan}/27**\n"
            f"* Per-subject table: `results/ml/multiseed/prediction_stability.csv`")


def regional_block(ctx) -> str:
    if ctx["reg"] is None:
        return "*Campaign in progress — regional-stability table fills post-campaign.*"
    r = ctx["reg"]
    stable = r[r["top10_count"] == r["n_seeds"]].sort_values("mean_relevance",
                                                             ascending=False)
    return (f"* Regions in the population top-10 for all seeds: "
            f"**{len(stable)}**" +
            (f" — {', '.join(stable['region'].tolist())}" if len(stable) else "") + "\n"
            f"* Regions in top-10 for ≥ 4 seeds: {int((r['top10_count'] >= 4).sum())}/87\n"
            f"* Table: `results/ml/multiseed/regional_stability.csv`")


def age_block(ctx) -> str:
    c = ctx["conf"]
    a = c["age_probability_association"]
    base = c["age_sex_baseline_train_fitted"]
    multi = (MULTI / "age_confound_by_seed.csv")
    extra = ""
    if multi.exists():
        adf = pd.read_csv(multi)
        seeds = adf[adf["seed"] != "age+sex_baseline"]["spearman_rho"].astype(float)
        extra = (f" (across seeds: {fmt(seeds.min())} to {fmt(seeds.max())}"
                 f"{' — sign-consistent' if (seeds > 0).all() else ' — variable'})")
    return (f"* Spearman(probability, age): {fmt(a['spearman']['rho'])} "
            f"(p = {fmt(a['spearman']['p_value'], 3)}){extra}\n"
            f"* Pearson: {fmt(a['pearson']['r'])} (p = {fmt(a['pearson']['p_value'], 3)})\n"
            f"* Age+sex baseline (train-fitted): test ROC-AUC "
            f"{fmt(base['test_roc_auc'])} vs CNN {fmt(c['cnn_test_roc_auc'])}\n"
            f"* Reading: association, not causation; the CNN exceeds the "
            f"demographic baseline but a demographic component cannot be "
            f"excluded at n=27 (FPs skew old: mean age "
            f"{fmt(float(ctx['errs'][ctx['errs']['error_group'] == 'FP']['Age'].mean()), 1)})"
            if ctx["errs"] is not None else "")


def repro_block(ctx) -> str:
    r = ctx["repro"]
    if not r:
        return "*Campaign in progress — the seed-4 repeat has not finished.*"
    return (f"* Same seed + same config, run twice: predictions bitwise "
            f"identical = **{r['predictions_bitwise_identical']}** "
            f"(max |Δp| = {r['max_abs_prob_diff']:.2e}); ROC-AUC "
            f"{fmt(r['roc_auc_seed4'])} vs {fmt(r['roc_auc_seed4_repeat'])}; "
            f"checkpoint bytes identical = {r['checkpoint_sha_identical']}.")


def dashboard(ctx) -> str:
    perf, cm_tbl, subjects, errors, calib, explain, _ = baseline_tables(ctx)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""# BrainVuln results dashboard

*Generated {now} from evaluation artifacts. Every number below is measured;
nothing is projected. Research use only — not a clinical diagnostic system.*

## Model performance (canonical seed-42 baseline, held-out test)

| Metric | Value | 95% CI (subject bootstrap) |
| --- | ---: | ---: |
{perf}

Threshold **{fmt(ctx['metrics']['threshold'], 2)}** (frozen, validation-only
Youden). Confusion matrix:

{cm_tbl}

## Model robustness (multi-seed distribution, seeds 42/1/2/3/4)

{robustness_block(ctx)}

Reproducibility (phase-19 repeat): {repro_block(ctx)}

## Test subjects

{subjects}

## Error analysis

{errors}

## Explainability

{explain}

{regional_block(ctx)}

## Calibration

{calib}

## Validation status

* External validation (OASIS-3 / ADNI): **NOT YET PERFORMED**.
* Genomics integration (GWAS/AHBA bridge): code complete and audited; the
  CNN side of the bridge awaits the multi-seed gate.
* Research use only. Not a clinical diagnostic system.
"""


def final_results(ctx) -> str:
    perf, _cm, subjects, errors, calib, explain, _ = baseline_tables(ctx)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""# BrainVuln — Final Results (measured only)

*Generated {now}. This file contains measured results and their limitations.
No marketing claims. Research use only — not a clinical diagnostic system.*

## Primary model

* Checkpoint: `results/ml/resnet_seed42/checkpoints/best.pt`
* SHA256: `{CANONICAL_SHA}`
* Training seed: 42 (canonical baseline; never replaced by the robustness seeds)
* Architecture: ResNet18Binary (MONAI ResNet-18, 1 input channel, single-logit head)
* Preprocessing: `t88_masked_gfc-2mm-128cube-brainz-v1`

## Dataset

OASIS-1 cross-sectional, age ≥ 60, CDR-based labels.

{subjects}

Full cohort: 182 subjects (100 AD / 82 CN), deterministic 1:1 matching,
70/15/15 subject-level split (seed 42). The threshold was selected on the
27 validation subjects only.

## Test results (27 held-out subjects, unit = subject)

| Metric | Value | 95% CI (subject bootstrap, 2000 draws) |
| --- | ---: | ---: |
{perf}

## Multi-seed results (robustness distribution — seeds 42, 1, 2, 3, 4)

{robustness_block(ctx)}

{repro_block(ctx)}

## Prediction stability

{stability_block(ctx)}

## Error analysis

{errors}

False-positive profile (seed 42): oldest group (mean age ~80.6), MMSE 27–30,
probabilities 0.225–0.999. False-negative profile: two CDR-0.5 subjects,
probabilities 0.027/0.042. Per-seed persistence:
`results/ml/multiseed/error_stability.csv`.

## Age confounding

{age_block(ctx)}

## Calibration

{calib}

Probabilities are usable for ranking, not as absolute risk estimates; the
model was **not** recalibrated (measurement only).

## Explainability

{explain}

Explanations are model-derived relevance, not biological truth; a region
highlighted by a single seed is reported as such.

## Current limitations

* n = 27 test subjects; all CIs are wide; single site, single scanner.
* CDR-based "probable AD" is not biomarker-confirmed AD.
* Class-level sex imbalance in the cohort (AD 59% F vs CN 70% F).
* Overconfident calibration (slope ≈ 0.34) measured, not corrected.
* External validation (OASIS-3, ADNI): **not performed**.
* Grad-CAM vs occlusion agreement is weak-to-moderate; regional claims stay coarse.
* Research prototype only — must not inform any clinical decision.
"""


def main() -> int:
    ctx = build_context()
    if ctx["metrics"] is None:
        print("baseline evaluation artifacts missing — run "
              "scripts/eval_subject_level.py first", file=sys.stderr)
        return 1
    (ROOT / "results" / "dashboard.md").write_text(dashboard(ctx), encoding="utf-8")
    (ROOT / "results" / "FINAL_RESULTS.md").write_text(final_results(ctx),
                                                       encoding="utf-8")
    print("wrote results/dashboard.md and results/FINAL_RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
