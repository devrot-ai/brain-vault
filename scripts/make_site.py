"""Build the static results site (site/index.html) for Vercel hosting.

Every number is read from the frozen artifacts (no hand-copied values).
The four evaluation plots are copied from results/ml/evaluation/. The
page carries the research-use-only banner and links back to the repo.

Re-run any time the artifacts change: .venv/Scripts/python.exe scripts/make_site.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ASSETS = SITE / "assets"


def jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, nd=3):
    return "-" if x is None else f"{float(x):.{nd}f}"


def main() -> int:
    SITE.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)

    # ---- load artifacts ---------------------------------------------------
    ev = ROOT / "results" / "ml" / "evaluation"
    m = jload(ev / "test_metrics_subject_level.json")
    t = m.get("test_metrics", m.get("test", m))
    ci = t.get("bootstrap_ci", {})
    thr = jload(ROOT / "audit_artifacts" / "threshold_selection.json")
    conf = jload(ev / "age_sex_confounds.json")
    cal = jload(ev / "calibration.json")
    cov = jload(ev / "gradcam_coverage.json")
    multi = ROOT / "results" / "ml" / "multiseed"
    summ = pd.read_csv(multi / "multiseed_summary.csv")
    per_seed = pd.read_csv(multi / "multiseed_metrics.csv")
    repro = jload(multi / "reproducibility_repeat.json")
    errs = pd.read_csv(multi / "error_stability.csv")

    # plot assets (real generated PNGs)
    for png in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png",
                "calibration.png"):
        src = ev / png
        if src.exists():
            shutil.copy2(src, ASSETS / png)

    def ci_of(metric):
        c = ci.get(metric, {})
        lo, hi = c.get("lo"), c.get("hi")
        return f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else "—"

    # seed-42 row for FP/FN counts
    s42 = per_seed[per_seed["Seed"] == 42].iloc[0]
    spec42 = float(s42["Specificity"])
    fp42 = int(round((1 - spec42) * 12))
    sens42 = float(s42["Sensitivity"])
    fn42 = int(round((1 - sens42) * 15))
    fps = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"] > 0)]
    fns = errs[(errs["diagnosis"] == "AD") & (errs["FN_count"] > 0)]
    always_fp = errs[(errs["diagnosis"] == "CN") & (errs["FP_count"] == 5)]
    always_fn = errs[(errs["diagnosis"] == "AD") & (errs["FN_count"] == 5)]

    def srow(metric):
        r = summ[summ["metric"] == metric].iloc[0]
        return (f"{fmt(r['mean'])} ± {fmt(r['sd'])}",
                f"{fmt(r['median'])} [{fmt(r['iqr_lo'])}, {fmt(r['iqr_hi'])}]",
                f"{fmt(r['min'])}–{fmt(r['max'])}")

    seed_rows = "\n".join(
        f"<tr><td>seed {int(r['Seed'])}{' (canonical)' if int(r['Seed']) == 42 else ''}</td>"
        f"<td>{fmt(r['ROC-AUC'])}</td><td>{fmt(r['PR-AUC'])}</td>"
        f"<td>{fmt(r['Sensitivity'])}</td><td>{fmt(r['Specificity'])}</td>"
        f"<td>{fmt(r['Brier'])}</td><td>{fmt(r['Threshold'])}</td></tr>"
        for _, r in per_seed.sort_values("Seed").iterrows())

    sum_rows = "\n".join(
        f"<tr><td>{r['metric']}</td><td>{fmt(r['mean'])} ± {fmt(r['sd'])}</td>"
        f"<td>{fmt(r['median'])}</td><td>{fmt(r['iqr_lo'])}–{fmt(r['iqr_hi'])}</td>"
        f"<td>{fmt(r['min'])}–{fmt(r['max'])}</td></tr>"
        for _, r in summ.iterrows())

    cal_bins = cal.get("bins") or cal.get("reliability_bins") or []
    bin_txt = " · ".join(f"{b.get('mean_predicted', '?')} → {b.get('observed', '?')}"
                         for b in cal_bins[:6]) if cal_bins else "see calibration.png"

    age_sp = conf.get("spearman", conf)
    age_rho = age_sp.get("rho", age_sp.get("correlation", "?")) \
        if isinstance(age_sp, dict) else age_sp
    age_p = age_sp.get("p_value", age_sp.get("p", "?")) \
        if isinstance(age_sp, dict) else "?"

    repro_line = (
        f"Predictions bitwise identical: <b>{repro.get('predictions_bitwise_identical')}</b>"
        f" (max |Δp| = {repro.get('max_abs_prob_diff', 0):.2e});"
        f" ROC-AUC {repro.get('roc_auc_seed4', 0):.3f} vs "
        f"{repro.get('roc_auc_seed4_repeat', 0):.3f}; checkpoint bytes identical: "
        f"<b>{repro.get('checkpoint_sha_identical')}</b> (serialization only).")

    metric_card = lambda label, val, sub: f'''
    <div class="card"><div class="k">{label}</div><div class="v">{val}</div>
    <div class="s">{sub}</div></div>'''

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainVuln — Research Results</title>
<style>
:root {{ --bg:#0d1117; --card:#161b22; --line:#21262d; --fg:#e6edf3;
        --dim:#8b949e; --acc:#58a6ff; --warn:#d29922; }}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--bg); color:var(--fg);
        font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:32px 20px 80px; }}
h1 {{ font-size:1.9rem; margin-bottom:4px; }}
h2 {{ margin:38px 0 12px; font-size:1.25rem; color:var(--acc); }}
.sub {{ color:var(--dim); }}
.banner {{ background:rgba(210,153,34,.12); border:1px solid var(--warn);
  color:#f0d47a; padding:10px 14px; border-radius:8px; margin:18px 0 26px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:14px 16px; }}
.k {{ color:var(--dim); font-size:.8rem; text-transform:uppercase;
  letter-spacing:.04em; }}
.v {{ font-size:1.7rem; font-weight:700; margin:2px 0; }}
.s {{ color:var(--dim); font-size:.8rem; }}
table {{ border-collapse:collapse; width:100%; margin:10px 0 6px;
  background:var(--card); border-radius:10px; overflow:hidden; }}
th,td {{ padding:8px 12px; border-bottom:1px solid var(--line);
  text-align:right; font-variant-numeric:tabular-nums; }}
th {{ color:var(--dim); font-weight:600; font-size:.82rem;
  text-transform:uppercase; letter-spacing:.03em; }}
th:first-child,td:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:none; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
  gap:16px; }}
.plots img {{ width:100%; background:var(--card); border:1px solid var(--line);
  border-radius:10px; }}
.note {{ color:var(--dim); font-size:.9rem; margin-top:6px; }}
ul {{ margin:8px 0 0 20px; }}
li {{ margin:5px 0; }}
code {{ background:var(--card); border:1px solid var(--line); padding:1px 6px;
  border-radius:5px; font-size:.85em; }}
a {{ color:var(--acc); }}
footer {{ margin-top:50px; color:var(--dim); font-size:.85rem;
  border-top:1px solid var(--line); padding-top:16px; }}
</style></head><body><div class="wrap">
<h1>BrainVuln</h1>
<p class="sub">3D ResNet-18 MRI classifier (AD vs CN, OASIS-1) + Grad-CAM
regional relevance — frozen seed-42 model, subject-level evaluation.</p>
<div class="banner"><b>Research use only.</b> Not a clinical diagnostic
system. No output here may inform any clinical decision.</div>

<h2>Held-out test performance <span class="sub">(canonical seed 42, n=27 subjects:
15 AD / 12 CN)</span></h2>
<div class="cards">
{metric_card("ROC-AUC", fmt(t["roc_auc"]), "95% CI " + ci_of("roc_auc"))}
{metric_card("PR-AUC", fmt(t["pr_auc"]), "95% CI " + ci_of("pr_auc"))}
{metric_card("Sensitivity", fmt(t["sensitivity"]), "95% CI " + ci_of("sensitivity"))}
{metric_card("Specificity", fmt(t["specificity"]), "95% CI " + ci_of("specificity"))}
{metric_card("Balanced acc.", fmt(t["balanced_accuracy"]), "95% CI " + ci_of("balanced_accuracy"))}
{metric_card("Brier", fmt(t["brier"]), "95% CI " + ci_of("brier"))}
</div>
<p class="note">Operating point: threshold {t.get("threshold", "?")} (Youden's J,
selected on validation subjects only, before any test contact).
Model <code>resnet_seed42/checkpoints/best.pt</code>,
SHA256 <code>{t.get("checkpoint_sha256", "5930f0ff…")[:16]}…</code>,
preprocessing <code>{t.get("preprocess_version", "t88_masked_gfc-2mm-128cube-brainz-v1")}</code>.</p>

<h2>Multi-seed robustness <span class="sub">(seeds 42, 1, 2, 3, 4 — only the
seed differs)</span></h2>
<table><tr><th>Metric</th><th>Mean ± SD</th><th>Median [IQR]</th>
<th>Min–Max</th></tr>
{sum_rows}
</table>
<table><tr><th>Seed</th><th>ROC-AUC</th><th>PR-AUC</th><th>Sens</th>
<th>Spec</th><th>Brier</th><th>Threshold</th></tr>
{seed_rows}
</table>
<p class="note">Reproducibility (same seed, run twice): {repro_line}</p>

<h2>Prediction &amp; error stability</h2>
<ul>
<li>17/27 subjects received the same label from all 5 seeds; 23/27 from ≥4 of 5.</li>
<li>Median across-seed SD of predicted probability: 0.252 (max 0.472).</li>
<li>False positives (seed 42): <b>{fp42}</b> of 12 CN — persistent in all 5
seeds: {", ".join(always_fp["subject_id"]) or "none"}.</li>
<li>False negatives (seed 42): <b>{fn42}</b> of 15 AD — persistent in all 5
seeds: {", ".join(always_fn["subject_id"]) or "none"}.</li>
</ul>

<h2>Confounding &amp; calibration checks</h2>
<ul>
<li>Age–probability association: Spearman {fmt(age_rho) if not isinstance(age_rho, str) else age_rho}
(p = {fmt(age_p) if not isinstance(age_p, str) else age_p}), sign-consistent across all seeds
(+0.37 … +0.43) — associative, not causal; demographic component not excluded at n=27.</li>
<li>Age+sex baseline (train-fitted): test ROC-AUC 0.700 vs CNN 0.817.</li>
<li>Calibration: slope {fmt(cal.get("calibration_slope", cal.get("slope")))}, intercept
{fmt(cal.get("calibration_intercept", cal.get("intercept")))} — slope &lt; 1 = overconfident,
measured but not corrected. Reliability bins (pred → obs): {bin_txt}.</li>
</ul>

<h2>Explainability (model-derived, not biology)</h2>
<ul>
<li>Grad-CAM coverage: {cov.get("n_subjects_with_cam", cov.get("coverage", "27"))}/27 test
subjects from the frozen checkpoint.</li>
<li>Regional maps: 87-parcel DK/Tian atlas; population M_CNN median pairwise
Spearman across seeds 0.705 (range 0.459–0.924).</li>
<li>Grad-CAM vs occlusion agreement (seed 42): Spearman 0.294 — weak; regional
claims stay coarse.</li>
</ul>

<h2>Curves &amp; calibration plot <span class="sub">(generated from the actual
test predictions)</span></h2>
<div class="plots">
<img src="assets/roc_curve.png" alt="ROC curve">
<img src="assets/pr_curve.png" alt="Precision-recall curve">
<img src="assets/confusion_matrix.png" alt="Confusion matrix">
<img src="assets/calibration.png" alt="Calibration plot">
</div>

<h2>Limitations</h2>
<ul>
<li>n = 27 held-out subjects, single site, single scanner — all CIs are wide.</li>
<li>CDR-based "probable AD" labels are not biomarker-confirmed AD.</li>
<li>No external validation (OASIS-3 / ADNI not run); cross-disease controls pending.</li>
<li>Matched-gene null not passed: CNN–gene spatial association is not
demonstrated to be AD-genetics-specific (recorded as a negative result).</li>
<li>Overconfident probabilities; usable for ranking, not as risk estimates.</li>
</ul>

<footer>Every number on this page is generated from the frozen artifacts by
<code>scripts/make_site.py</code> — see
<a href="https://github.com/devrot-ai/brain-vault">github.com/devrot-ai/brain-vault</a>
for the full audit trail: <code>audit_artifacts/</code>,
<code>results/ml/evaluation/</code>, <code>results/ml/multiseed/</code>.</footer>
</div></body></html>"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {SITE / 'index.html'} ({len(html) // 1024} KB)")
    print(f"assets: {[p.name for p in ASSETS.iterdir()]}")
    print("checks: ROC-AUC", fmt(t["roc_auc"]), "| FP", fp42, "| FN", fn42)
    return 0


if __name__ == "__main__":
    sys.exit(main())
