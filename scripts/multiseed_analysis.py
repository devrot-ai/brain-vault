#!/usr/bin/env python
"""Multi-seed robustness analysis (phases 7-19 of the robustness spec).

Aggregates the frozen seed-42 baseline plus seeds 1-4 (and the seed-4
repeat) from results/ml/resnet_seed*/ into:

    results/ml/multiseed/multiseed_metrics.csv        phase 7
    results/ml/multiseed/multiseed_summary.csv        phase 7 (mean/SD/median/IQR/min/max)
    results/ml/multiseed/prediction_stability.csv     phase 9
    results/ml/multiseed/classification_stability.csv phase 10
    results/ml/multiseed/error_stability.csv          phase 11
    results/ml/multiseed/age_confound_by_seed.csv     phase 12
    results/ml/multiseed/calibration_by_seed.csv      phase 13
    results/ml/multiseed/gradcam_similarity.csv       phase 14
    results/ml/multiseed/M_CNN_mean_by_seed.csv       phase 15
    results/ml/multiseed/regional_stability.csv       phase 16
    results/ml/multiseed/seed42_pairwise.csv          phase 18
    results/ml/multiseed/reproducibility_repeat.json  phase 19

and prints a summary for the scorecard (audit_artifacts/multiseed_robustness.md).
Every number is computed from the on-disk artifacts; nothing is re-trained,
and the seed-42 checkpoint is only ever opened read-only (SHA-verified).

Statistical unit: SUBJECT (each run dir holds exactly one session per
test subject; asserted here). No best seed is ever selected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brainvuln.mri.evaluate import classification_metrics, subject_bootstrap_ci  # noqa: E402

MULTI = ROOT / "results" / "ml" / "multiseed"
SEEDS = [42, 1, 2, 3, 4]
REPEAT_SEED = 4
REPEAT_DIR = ROOT / "results" / "ml" / "resnet_seed4_repeat"
CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
EVAL_DIR = ROOT / "results" / "ml" / "evaluation"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(seed: int, run_dir: Path | None = None) -> dict | None:
    """Load one seed's frozen outputs (predictions + metrics)."""
    run_dir = run_dir or (ROOT / "results" / "ml" / f"resnet_seed{seed}")
    pred_file = run_dir / "predictions" / "test_predictions.csv"
    metrics_file = run_dir / "metrics" / "test_metrics.json"
    if not (pred_file.exists() and metrics_file.exists()):
        return None
    m = json.loads(metrics_file.read_text(encoding="utf-8"))
    pred = pd.read_csv(pred_file)
    assert len(pred) == 27, f"{run_dir}: expected 27 subject rows, got {len(pred)}"
    assert pred["subject_id"].is_unique, f"{run_dir}: duplicate subjects in predictions"
    ident = m.get("checkpoint_identity", {})
    sha = str(ident.get("sha256", ""))
    if not sha:
        # older runs (seed-42 canonical) predate the embedded identity;
        # hash the checkpoint file directly (read-only)
        ck = run_dir / "checkpoints" / "best.pt"
        sha = sha256_file(ck) if ck.exists() else ""
    return {
        "seed": seed,
        "dir": run_dir,
        "pred": pred,
        "threshold": float(m["threshold"]),
        "metrics_file": m,
        "sha": sha,
    }


def main() -> int:
    MULTI.mkdir(parents=True, exist_ok=True)

    # ---- phase 1 verifications ------------------------------------------
    ckpt = ROOT / "results" / "ml" / "resnet_seed42" / "checkpoints" / "best.pt"
    sha_now = sha256_file(ckpt)
    assert sha_now == CANONICAL_SHA, (
        f"canonical seed-42 checkpoint changed: {sha_now}")
    baseline_archived = (EVAL_DIR / "test_metrics_subject_level.json").exists() and \
        (ROOT / "audit_artifacts" / "subject_level_evaluation.md").exists()
    assert baseline_archived, "baseline subject-level evaluation artifacts missing"
    print(f"[1] baseline OK: seed-42 sha {sha_now[:16]}…, baseline artifacts archived")

    runs = {}
    for seed in SEEDS:
        r = load_run(seed)
        if r is None:
            print(f"[!] seed {seed}: not complete yet (no predictions/metrics)")
        else:
            runs[seed] = r
            print(f"[load] seed {seed}: threshold {r['threshold']:.2f}, "
                  f"sha {r['sha'][:16]}…")
    repeat = load_run(REPEAT_SEED, REPEAT_DIR)
    if repeat is None:
        print("[!] seed-4 repeat: not complete yet")
    missing = [s for s in SEEDS if s not in runs]
    if missing:
        print(f"INCOMPLETE: seeds {missing} still pending — analysis aborted "
              f"(no partial scorecard is written)")
        return 1

    cohort = pd.read_csv(ROOT / "data" / "splits" / "matched_cohort.csv")
    demo = cohort.set_index("subject")[["Age", "Sex", "eTIV", "nWBV", "ASF"]]
    diag = {r.subject_id: r.diagnosis for r in runs[42]["pred"].itertuples()}

    # ---- phase 7: metric table ------------------------------------------
    rows = []
    for seed in SEEDS:
        r = runs[seed]
        pred = r["pred"]
        y = pred["label"].to_numpy(int)
        p = pred["probability"].to_numpy(float)
        m = classification_metrics(y, p, r["threshold"])
        # cross-check against the stored frozen metrics
        stored = r["metrics_file"]["test"]
        for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "brier"):
            assert abs(m[k] - stored[k]) < 1e-9, \
                f"seed {seed}: recomputed {k} {m[k]} != stored {stored[k]}"
        ci = subject_bootstrap_ci(y, p, pred["subject_id"].tolist(),
                                  r["threshold"], n_boot=2000, seed=42)
        rows.append({
            "Seed": seed,
            "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
            "Accuracy": m["accuracy"], "Balanced Acc": m["balanced_accuracy"],
            "Sensitivity": m["sensitivity"], "Specificity": m["specificity"],
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
            "Brier": m["brier"],
            "ROC-AUC_CI_lo": ci["roc_auc"]["lo"], "ROC-AUC_CI_hi": ci["roc_auc"]["hi"],
            "Threshold": r["threshold"],
            "Checkpoint_SHA256": r["sha"],
        })
    met = pd.DataFrame(rows).sort_values("Seed")
    met.to_csv(MULTI / "multiseed_metrics.csv", index=False)

    metric_cols = ["ROC-AUC", "PR-AUC", "Accuracy", "Balanced Acc",
                   "Sensitivity", "Specificity", "Precision", "Recall", "F1", "Brier"]
    summary_rows = []
    for k in metric_cols:
        v = met[k].to_numpy(float)
        summary_rows.append({
            "metric": k, "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)),
            "iqr_lo": float(np.percentile(v, 25)), "iqr_hi": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())})
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(MULTI / "multiseed_summary.csv", index=False)
    print("[7] metric table + summary written")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- phases 9-11: prediction / classification / error stability ------
    wide = {}
    for seed in SEEDS:
        s = runs[seed]["pred"].set_index("subject_id")["probability"]
        wide[f"prob_seed{seed}"] = s
    w = pd.DataFrame(wide).sort_index()
    assert len(w) == 27
    stability = pd.DataFrame({
        "subject_id": w.index,
        "diagnosis": [diag[s] for s in w.index],
        "mean_probability": w.mean(axis=1).to_numpy(),
        "sd_probability": w.std(axis=1, ddof=1).to_numpy(),
        "min_probability": w.min(axis=1).to_numpy(),
        "max_probability": w.max(axis=1).to_numpy(),
    })
    stability.to_csv(MULTI / "prediction_stability.csv", index=False)

    ad_votes = np.zeros(len(w), dtype=int)
    for seed in SEEDS:
        thr = runs[seed]["threshold"]
        ad_votes += (w[f"prob_seed{seed}"].to_numpy() >= thr).astype(int)
    cls = pd.DataFrame({
        "subject_id": w.index, "diagnosis": [diag[s] for s in w.index],
        "AD_predictions_of_5": ad_votes, "CN_predictions_of_5": 5 - ad_votes})
    cls.to_csv(MULTI / "classification_stability.csv", index=False)

    err_rows = []
    for i, subj in enumerate(w.index):
        y_i = 1 if diag[subj] == "AD" else 0
        fp = fn = 0
        for seed in SEEDS:
            thr = runs[seed]["threshold"]
            pred_i = int(w.iloc[i][f"prob_seed{seed}"] >= thr)
            fp += int(y_i == 0 and pred_i == 1)
            fn += int(y_i == 1 and pred_i == 0)
        err_rows.append({"subject_id": subj, "diagnosis": diag[subj],
                         "FP_count": fp, "FN_count": fn,
                         "mean_probability": float(w.iloc[i].mean()),
                         "probability_sd": float(w.iloc[i].std(ddof=1))})
    errs = pd.DataFrame(err_rows).sort_values(["FP_count", "FN_count"],
                                              ascending=False)
    errs.to_csv(MULTI / "error_stability.csv", index=False)
    n_unanimous = int((ad_votes == 0).sum() + (ad_votes == 5).sum())
    n_majority = int(((ad_votes >= 4) | (ad_votes <= 1)).sum())
    always_fp = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"] == 5)]
    never_fp = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"] == 0)]
    always_fn = errs[(errs["diagnosis"] == "AD") & (errs["FN_count"] == 5)]
    print(f"[9-11] unanimous 5/5 classification: {n_unanimous}/27; "
          f"4-5/5 majority: {n_majority}/27; always-FP CN: {len(always_fp)}; "
          f"never-FP CN: {len(never_fp)}; always-FN AD: {len(always_fn)}")

    # ---- phase 12: age confound per seed ---------------------------------
    from scipy.stats import pearsonr, spearmanr
    age = demo.loc[w.index, "Age"].to_numpy(float)
    age_rows = []
    for seed in SEEDS:
        p = w[f"prob_seed{seed}"].to_numpy()
        sp, pe = spearmanr(p, age), pearsonr(p, age)
        age_rows.append({"seed": seed, "spearman_rho": float(sp.statistic),
                         "spearman_p": float(sp.pvalue),
                         "pearson_r": float(pe.statistic),
                         "pearson_p": float(pe.pvalue)})
    # age+sex baseline (train-fitted logistic regression) for reference
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tr = cohort[cohort["split"] == "train"]
    Xtr = np.stack([tr["Age"].to_numpy(float), (tr["Sex"] == "M").to_numpy(float)], 1)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
    Xte = np.stack([age, (demo.loc[w.index, "Sex"] == "M").to_numpy(float)], 1)
    base_p = clf.predict_proba(Xte)[:, 1]
    sp, pe = spearmanr(base_p, age), pearsonr(base_p, age)
    age_rows.append({"seed": "age+sex_baseline", "spearman_rho": float(sp.statistic),
                     "spearman_p": float(sp.pvalue), "pearson_r": float(pe.statistic),
                     "pearson_p": float(pe.pvalue)})
    agedf = pd.DataFrame(age_rows)
    agedf.to_csv(MULTI / "age_confound_by_seed.csv", index=False)
    rho = agedf[agedf["seed"] != "age+sex_baseline"]["spearman_rho"].astype(float)
    sign_consistent = bool((rho > 0).all() or (rho < 0).all())
    print(f"[12] age-prob Spearman across seeds: {rho.min():+.3f} … {rho.max():+.3f} "
          f"(sign-consistent={sign_consistent}); baseline "
          f"{agedf.iloc[-1]['spearman_rho']:+.3f}")

    # ---- phase 13: calibration per seed ----------------------------------
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression as LR
    cal_rows, bins_all = [], {}
    for seed in SEEDS:
        y = runs[seed]["pred"]["label"].to_numpy(int)
        p = w[f"prob_seed{seed}"].to_numpy()
        eps = 1e-6
        lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        c = LR().fit(lg.reshape(-1, 1), y)
        frac, meanp = calibration_curve(y, p, n_bins=4, strategy="quantile")
        bins_all[seed] = {"mean_predicted": meanp, "observed": frac}
        cal_rows.append({"seed": seed,
                         "brier": float(brier(y, p)),
                         "calibration_slope": float(c.coef_[0][0]),
                         "calibration_intercept": float(c.intercept_[0])})
    caldf = pd.DataFrame(cal_rows)
    caldf.to_csv(MULTI / "calibration_by_seed.csv", index=False)
    slopes = caldf["calibration_slope"].to_numpy()
    print(f"[13] calibration slope across seeds: {slopes.min():.3f} … {slopes.max():.3f}; "
          f"Brier {caldf['brier'].min():.3f} … {caldf['brier'].max():.3f}")

    # ---- phases 14-16: Grad-CAM / regional stability ---------------------
    maps, have_maps = {}, []
    for seed in SEEDS:
        f = ROOT / "results" / "ml" / f"resnet_seed{seed}" / "regional_relevance" / "M_CNN_regional.csv"
        if f.exists():
            maps[seed] = pd.read_csv(f, index_col=0)["M_CNN"]
            have_maps.append(seed)
        else:
            print(f"[!] seed {seed}: M_CNN_regional.csv missing (CAM stage pending?)")
    pair_rows = []
    for a, b in combinations(sorted(maps), 2):
        j = pd.concat([maps[a].rename("a"), maps[b].rename("b")], axis=1).dropna()
        if len(j) >= 5 and j["a"].std() > 0 and j["b"].std() > 0:
            pair_rows.append({"seed_a": a, "seed_b": b,
                              "spearman_rho": float(j["a"].corr(j["b"], method="spearman")),
                              "n_regions": len(j)})
    pd.DataFrame(pair_rows).to_csv(MULTI / "gradcam_similarity.csv", index=False)
    if pair_rows:
        rhos = pd.DataFrame(pair_rows)["spearman_rho"]
        print(f"[14] pairwise M_CNN Spearman: median {rhos.median():.3f}, "
              f"range [{rhos.min():.3f}, {rhos.max():.3f}] over {len(pair_rows)} pairs")

    if maps:
        allmap = pd.DataFrame({f"seed{s}": maps[s] for s in sorted(maps)})
        allmap["mean"] = allmap.mean(axis=1)
        allmap["sd"] = allmap.std(axis=1, ddof=1)
        allmap.to_csv(MULTI / "M_CNN_mean_by_seed.csv")
        atlas = pd.read_csv(ROOT / "data" / "derived" / "atlas_dk68_tianS1_info.csv")
        label2struct = dict(zip(atlas["label"].astype(str), atlas["structure"].astype(str)))
        stab_rows = []
        for region in allmap.index:
            vals = allmap.loc[region, [c for c in allmap.columns if c.startswith("seed")]].astype(float)
            mean_v, sd_v = float(vals.mean()), float(vals.std(ddof=1))
            ranks = []
            top10 = top20 = 0
            for s in sorted(maps):
                ranked = maps[s].sort_values(ascending=False)
                r = list(ranked.index).index(region) + 1
                ranks.append(r)
                top10 += int(r <= 10)
                top20 += int(r <= 20)
            stab_rows.append({
                "region": region, "structure": label2struct.get(region, ""),
                "mean_relevance": mean_v, "sd": sd_v,
                "cv": (sd_v / mean_v) if abs(mean_v) > 1e-12 else float("nan"),
                "mean_rank": float(np.mean(ranks)), "best_rank": int(min(ranks)),
                "worst_rank": int(max(ranks)),
                "top10_count": top10, "top20_count": top20,
                "n_seeds": len(maps)})
        st = pd.DataFrame(stab_rows).sort_values("mean_relevance", ascending=False)
        st.to_csv(MULTI / "regional_stability.csv", index=False)
        stable10 = st[st["top10_count"] == len(maps)]
        print(f"[16] regions in top-10 for ALL seeds: {len(stable10)} "
              f"({', '.join(stable10['region'].head(5))})")

    # ---- phase 18: seed-42 vs others -------------------------------------
    cmp_rows = []
    for seed in [s for s in SEEDS if s != 42]:
        a = w["prob_seed42"].to_numpy()
        b = w[f"prob_seed{seed}"].to_numpy()
        cmp_rows.append({"comparison": f"seed42_vs_seed{seed}",
                         "pearson_r_prob": float(np.corrcoef(a, b)[0, 1]),
                         "spearman_rho_prob": float(spearmanr(a, b).statistic),
                         "roc_auc_42": float(met.loc[met['Seed'] == 42, 'ROC-AUC'].iloc[0]),
                         f"roc_auc_{seed}": float(met.loc[met['Seed'] == seed, 'ROC-AUC'].iloc[0])})
    pd.DataFrame(cmp_rows).to_csv(MULTI / "seed42_pairwise.csv", index=False)
    print("[18] seed-42 vs others probability Spearman: " +
          ", ".join(f"{r['spearman_rho_prob']:.3f}" for r in cmp_rows))

    # ---- phase 19: reproducibility (seed-4 repeat) -----------------------
    repro = {}
    if repeat is not None:
        sha4, shar = runs[4]["sha"], repeat["sha"]
        p4 = runs[4]["pred"].sort_values("subject_id")["probability"].to_numpy()
        pr = repeat["pred"].sort_values("subject_id")["probability"].to_numpy()
        repro = {
            "checkpoint_sha_seed4": sha4,
            "checkpoint_sha_seed4_repeat": shar,
            "checkpoint_sha_identical": sha4 == shar,
            "predictions_bitwise_identical": bool(np.array_equal(p4, pr)),
            "max_abs_prob_diff": float(np.max(np.abs(p4 - pr))),
            "roc_auc_seed4": float(met.loc[met['Seed'] == 4, 'ROC-AUC'].iloc[0]),
            "roc_auc_seed4_repeat": classification_metrics(
                repeat["pred"]["label"].to_numpy(int), pr,
                repeat["threshold"])["roc_auc"],
            "note": ("identical predictions with differing checkpoint bytes = "
                     "serialization-level nondeterminism only (acceptable per "
                     "the reproducibility rule); materially different "
                     "predictions would indicate real nondeterminism"),
        }
        (MULTI / "reproducibility_repeat.json").write_text(json.dumps(repro, indent=2))
        print(f"[19] repeat: predictions bitwise identical = "
              f"{repro['predictions_bitwise_identical']}, "
              f"max|dp| = {repro['max_abs_prob_diff']:.2e}")

    print("ANALYSIS_COMPLETE")
    return 0


def brier(y, p):
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y, p)


if __name__ == "__main__":
    sys.exit(main())
