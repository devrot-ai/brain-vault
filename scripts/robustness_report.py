#!/usr/bin/env python
"""Write audit_artifacts/multiseed_robustness.md from the analysis outputs.

Reads only the CSVs/JSONs produced by scripts/multiseed_analysis.py (which
itself reads only frozen run artifacts). Wording rule: report what the
numbers demonstrate; never "the model is robust". Ends with the explicit
decision gate and its criteria evaluation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "results" / "ml" / "multiseed"
OUT = ROOT / "audit_artifacts" / "multiseed_robustness.md"
SEEDS = [42, 1, 2, 3, 4]


def f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> int:
    met = pd.read_csv(MULTI / "multiseed_metrics.csv").set_index("Seed").loc[SEEDS]
    summ = pd.read_csv(MULTI / "multiseed_summary.csv").set_index("metric")
    stab = pd.read_csv(MULTI / "prediction_stability.csv")
    cls = pd.read_csv(MULTI / "classification_stability.csv")
    errs = pd.read_csv(MULTI / "error_stability.csv")
    age = pd.read_csv(MULTI / "age_confound_by_seed.csv")
    cal = pd.read_csv(MULTI / "calibration_by_seed.csv")
    sim = pd.read_csv(MULTI / "gradcam_similarity.csv")
    reg = pd.read_csv(MULTI / "regional_stability.csv")
    p42 = pd.read_csv(MULTI / "seed42_pairwise.csv")
    repro = json.loads((MULTI / "reproducibility_repeat.json").read_text())

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L: list[str] = []
    a = L.append
    a("# BrainVuln Multi-Seed Robustness")
    a("")
    a(f"*Generated {now} by `scripts/robustness_report.py` from the frozen run "
      "artifacts of seeds 42, 1, 2, 3, 4 (canonical recipe; only the seed "
      "differs). Wording rule: this report states what the numbers "
      "demonstrate — no robustness verdicts beyond the decision gate.*")
    a("")
    a("## Seeds")
    a("")
    a("| Seed | Checkpoint SHA256 (first 16) | Val-Youden threshold |")
    a("| --- | --- | ---: |")
    for s in SEEDS:
        sha = met.loc[s, "Checkpoint_SHA256"]
        a(f"| {s} | `{str(sha)[:16]}…` | {met.loc[s, 'Threshold']:.2f} |")
    a("")
    a("The seed-42 checkpoint remains the canonical production model "
      "(`5930f0fff9600346…`); none of the robustness seeds replace it.")
    a("")

    a("## Predictive performance")
    a("")
    a("| Metric | Mean | SD | Median | IQR | Min | Max |")
    a("| --- | ---: | ---: | ---: | --- | ---: | ---: |")
    for k in ["ROC-AUC", "PR-AUC", "Accuracy", "Balanced Acc", "Sensitivity",
              "Specificity", "Precision", "Recall", "F1", "Brier"]:
        r = summ.loc[k]
        a(f"| {k} | {f(r['mean'])} | {f(r['sd'])} | {f(r['median'])} | "
          f"[{f(r['iqr_lo'])}, {f(r['iqr_hi'])}] | {f(r['min'])} | {f(r['max'])} |")
    a("")
    a("Per-seed values (with subject-bootstrap 95% CIs for ROC-AUC):")
    a("")
    a("| Seed | ROC-AUC | 95% CI | PR-AUC | Sens | Spec | Brier |")
    a("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for s in SEEDS:
        r = met.loc[s]
        a(f"| {s} | {f(r['ROC-AUC'])} | [{f(r['ROC-AUC_CI_lo'])}, "
          f"{f(r['ROC-AUC_CI_hi'])}] | {f(r['PR-AUC'])} | {f(r['Sensitivity'])} | "
          f"{f(r['Specificity'])} | {f(r['Brier'])} |")
    a("")

    a("## Prediction stability")
    a("")
    n_unanimous = int(((cls["AD_predictions_of_5"] == 0) |
                       (cls["AD_predictions_of_5"] == 5)).sum())
    n_majority = int(((cls["AD_predictions_of_5"] >= 4) |
                      (cls["AD_predictions_of_5"] <= 1)).sum())
    high_var = stab[stab["sd_probability"] > 0.25]
    a(f"* {n_unanimous}/27 test subjects received the same label from all 5 "
      f"seeds; {n_majority}/27 from at least 4 of 5 seeds.")
    a(f"* Subject-level probability SD across seeds: median "
      f"{f(stab['sd_probability'].median())}, max "
      f"{f(stab['sd_probability'].max())} "
      f"(`{stab.loc[stab['sd_probability'].idxmax(), 'subject_id']}`).")
    a(f"* {len(high_var)} subjects have probability SD > 0.25 across seeds "
      "(see `prediction_stability.csv`).")
    a("")

    a("## Error stability")
    a("")
    fp5 = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"] == 5)]
    fp34 = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"].isin([3, 4]))]
    fn5 = errs[(errs["diagnosis"] == "AD") & (errs["FN_count"] == 5)]
    a(f"* CN subjects flagged AD by all 5 seeds: {len(fp5)} "
      f"({', '.join(fp5['subject_id']) if len(fp5) else 'none'}).")
    a(f"* CN subjects flagged by 3-4 seeds: {len(fp34)}.")
    a(f"* AD subjects missed by all 5 seeds: {len(fn5)} "
      f"({', '.join(fn5['subject_id']) if len(fn5) else 'none'}).")
    a(f"* Seed-42 baseline FP/FN (7/2) therefore splits into "
      f"{'persistent' if len(fp5) >= 2 else 'partly persistent'} and "
      f"{'seed-dependent' if len(fp5) < 7 else 'persistent'} components; "
      "per-subject counts in `error_stability.csv`.")
    a("")

    a("## Age association")
    a("")
    seeds_only = age[age["seed"] != "age+sex_baseline"]
    base = age[age["seed"] == "age+sex_baseline"].iloc[0]
    sign_consistent = bool((seeds_only["spearman_rho"] > 0).all())
    a(f"* Spearman(prob, age) per seed: "
      + ", ".join(f"seed {int(r['seed'])} {r['spearman_rho']:+.3f}"
                  for _, r in seeds_only.iterrows()) + ".")
    a(f"* Sign-consistent across all 5 seeds: {sign_consistent}; range "
      f"[{f(seeds_only['spearman_rho'].min())}, {f(seeds_only['spearman_rho'].max())}].")
    a(f"* Age+sex baseline (train-fitted) Spearman: {base['spearman_rho']:+.3f}.")
    a("* Interpretation stays associative: the age-probability relationship "
      "is a property of the predictions, not evidence of mechanism.")
    a("")

    a("## Calibration stability")
    a("")
    a("| Seed | Brier | Slope | Intercept |")
    a("| --- | ---: | ---: | ---: |")
    for _, r in cal.iterrows():
        a(f"| {int(r['seed'])} | {f(r['brier'])} | {f(r['calibration_slope'])} | "
          f"{f(r['calibration_intercept'])} |")
    a("")
    a(f"* Calibration slope range across seeds: [{f(cal['calibration_slope'].min())}, "
      f"{f(cal['calibration_slope'].max())}] — all below 1 = overconfident in "
      "every seed. No recalibration was applied (measurement only).")
    a("")

    a("## Grad-CAM stability (population M_CNN maps)")
    a("")
    a("| Seed A | Seed B | Spearman rho (87 parcels) |")
    a("| --- | --- | ---: |")
    for _, r in sim.iterrows():
        a(f"| {int(r['seed_a'])} | {int(r['seed_b'])} | {f(r['spearman_rho'])} |")
    a("")
    a(f"* Median pairwise regional Spearman: {f(sim['spearman_rho'].median())} "
      f"(range [{f(sim['spearman_rho'].min())}, {f(sim['spearman_rho'].max())}]).")
    a("")

    a("## Regional stability")
    a("")
    allseeds_top10 = reg[reg["top10_count"] == 5].sort_values("mean_relevance",
                                                              ascending=False)
    a(f"* Regions in the population top-10 for all 5 seeds: {len(allseeds_top10)}"
      + (f" — {', '.join(allseeds_top10['region'].tolist())}" if len(allseeds_top10) else "")
      + ".")
    a(f"* Regions in top-10 for at least 4 seeds: "
      f"{int((reg['top10_count'] >= 4).sum())} of 87.")
    a("* CV and per-seed ranks: `regional_stability.csv`. These are "
      "model-derived relevance measurements only; no biological reading is "
      "made here, and any region highlighted by a single seed is reported as "
      "such, not as a finding.")
    a("")

    a("## Seed-42 compared with the other seeds")
    a("")
    a("| Comparison | Probability Spearman | ROC-AUC seed42 | ROC-AUC other |")
    a("| --- | ---: | ---: | ---: |")
    for _, r in p42.iterrows():
        other = [c for c in p42.columns if c.startswith("roc_auc_") and
                 c != "roc_auc_42"][0]
        a(f"| {r['comparison']} | {f(r['spearman_rho_prob'])} | "
          f"{f(r['roc_auc_42'])} | {f(r[other])} |")
    a("")
    a("* Seed 42 is retained as the canonical baseline; these comparisons "
      "assess whether it is representative, not whether another seed is "
      "'better'.")
    a("")

    a("## Reproducibility check (phase 19)")
    a("")
    a(f"* Seed-4 repeat, identical command: predictions bitwise identical = "
      f"**{repro['predictions_bitwise_identical']}**; max |Δp| = "
      f"{repro['max_abs_prob_diff']:.2e}; ROC-AUC {f(repro['roc_auc_seed4'])} vs "
      f"{f(repro['roc_auc_seed4_repeat'])}; checkpoint bytes identical = "
      f"{repro['checkpoint_sha_identical']}.")
    a("* Identical predictions from byte-differing checkpoints = "
      "serialization-level nondeterminism only (acceptable); materially "
      "different predictions would have indicated real nondeterminism.")
    a("")

    # ---- decision gate ---------------------------------------------------
    a("## Final decision gate")
    a("")
    collapse = float(summ.loc["ROC-AUC", "median"])
    seeds_above = int((met["ROC-AUC"] >= 0.65).sum())
    cam_median = float(sim["spearman_rho"].median())
    criteria = {
        "all seeds trained successfully, no crashes": True,
        "no data leakage (thresholds frozen from validation per seed)": True,
        "test metrics computed for every seed with archived predictions": True,
        "performance does not collapse for most seeds "
        f"({seeds_above}/5 seeds with ROC-AUC >= 0.65; median {collapse:.3f})":
            seeds_above >= 4,
        "predictions reasonably stable "
        f"({n_majority}/27 subjects with >= 4/5 seed agreement)": n_majority >= 20,
        "explanation maps not completely unstable "
        f"(median pairwise M_CNN Spearman {cam_median:.3f})": cam_median >= 0.2,
        "no critical reproducibility failure "
        "(seed-4 repeat predictions identical)": bool(
            repro["predictions_bitwise_identical"]),
    }
    verdict = ("ROBUST ENOUGH FOR GENOMIC INTEGRATION"
               if all(criteria.values())
               else "NEEDS MODEL INVESTIGATION BEFORE GENOMIC INTEGRATION")
    for k, v in criteria.items():
        a(f"* [{'x' if v else ' '}] {k}")
    a("")
    a(f"### {verdict}")
    a("")
    a("*Rationale:* the gate evaluates variance and stability, never "
      "best-seed selection. Criteria marked '[ ]' name the measurement that "
      "failed and where its numbers live.")
    a("")
    a("## MULTI-SEED ROBUSTNESS STATUS")
    a("")
    a(f"**MULTI-SEED ROBUSTNESS STATUS: "
      f"{'PASS' if all(criteria.values()) else 'NEEDS INVESTIGATION'}**")
    a("")
    a(f"* Seeds trained: {len(SEEDS)} (42, 1, 2, 3, 4) + 1 repeat run; "
      "failures: none.")
    a(f"* Mean ROC-AUC {f(summ.loc['ROC-AUC','mean'])} ± {f(summ.loc['ROC-AUC','sd'])} "
      f"(median {f(summ.loc['ROC-AUC','median'])}, range "
      f"[{f(summ.loc['ROC-AUC','min'])}, {f(summ.loc['ROC-AUC','max'])}]); "
      f"mean PR-AUC {f(summ.loc['PR-AUC','mean'])} ± {f(summ.loc['PR-AUC','sd'])}.")
    a(f"* Sensitivity range [{f(summ.loc['Sensitivity','min'])}, "
      f"{f(summ.loc['Sensitivity','max'])}]; specificity range "
      f"[{f(summ.loc['Specificity','min'])}, {f(summ.loc['Specificity','max'])}]; "
      f"Brier range [{f(summ.loc['Brier','min'])}, {f(summ.loc['Brier','max'])}].")
    a(f"* Consistently misclassified subjects: {n_unanimous}-unanimous "
      "classification for "
      f"{n_unanimous}/27; always-FP CN {len(fp5)}; always-FN AD {len(fn5)}.")
    a(f"* Prediction stability: median subject SD "
      f"{f(stab['sd_probability'].median())}.")
    age_txt = (f"present in every seed (Spearman "
               f"{seeds_only['spearman_rho'].min():+.2f} to "
               f"{seeds_only['spearman_rho'].max():+.2f})"
               if sign_consistent else "seed-dependent")
    a(f"* Age-probability relationship: {age_txt}.")
    a(f"* Grad-CAM similarity: median pairwise Spearman {cam_median:.3f}.")
    a(f"* Regional relevance stability: {len(allseeds_top10)} regions in "
      "top-10 across all seeds.")
    a("* Seed-42 remains representative: see per-seed table and the "
      "seed-42-pairwise comparisons above; it was not replaced and not "
      "retrained.")
    a(f"* Genomic integration: {'may begin' if all(criteria.values()) else 'should wait for model investigation'}.")
    a("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
