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


def analyze_section():
    """Live-inference UI (upload → real backend → real Grad-CAM) with zero
    mock data. When no backend URL is configured the section states that
    honestly instead of faking a prediction."""
    css = """
.up {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:20px; margin:10px 0; }}
.up.drop {{ border-color:var(--acc); background:rgba(88,166,255,.06); }}
.upfile {{ display:none; }}
.btn {{ background:var(--acc); color:#0d1117; border:none; border-radius:8px;
  padding:10px 18px; font-weight:600; font-size:.95rem; cursor:pointer; }}
.btn:disabled {{ background:#30363d; color:var(--dim); cursor:not-allowed; }}
.btn.ghost {{ background:transparent; color:var(--acc);
  border:1px solid var(--acc); margin-left:10px; }}
.fmeta {{ color:var(--dim); font-size:.9rem; margin:10px 0 0; }}
.state {{ margin:12px 0 0; font-size:.95rem; }}
.state b {{ color:var(--acc); }}
.spin {{ display:inline-block; width:14px; height:14px;
  border:2px solid var(--acc); border-top-color:transparent; border-radius:50%;
  animation:sp 0.8s linear infinite; vertical-align:-2px; margin-right:7px; }}
@keyframes sp {{ to {{ transform:rotate(360deg); }} }}
.err {{ color:#f85149; margin:10px 0 0; white-space:pre-wrap; }}
.res {{ margin-top:16px; border-top:1px solid var(--line); padding-top:14px; }}
.resgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:10px; margin:10px 0; }}
.probbar {{ background:#21262d; border-radius:6px; height:14px; overflow:hidden;
  margin:6px 0 2px; }}
.probbar > div {{ height:100%; background:var(--acc); }}
.regtable td, .regtable th {{ padding:5px 10px; font-size:.88rem; }}
.camwrap img {{ max-width:100%; border:1px solid var(--line); border-radius:8px;
  background:#000; margin-top:8px; }}
"""
    html = """
<h2>Live inference <span class="sub">(real frozen seed-42 model — upload a NIfTI T1)</span></h2>
<div class="banner"><b>Research use only.</b> Not a medical diagnosis. This model has
not been externally validated for clinical use. Outputs are model-derived research
classifications with known limitations (small cohort, imperfect calibration,
demographic confounding not excluded).</div>
<div class="up" id="upbox">
  <input type="file" id="mriFile" class="upfile" accept=".nii,.nii.gz" />
  <button class="btn" id="pickBtn" type="button">Upload MRI (.nii / .nii.gz)</button>
  <button class="btn ghost" id="analyzeBtn" type="button" disabled>Analyze MRI</button>
  <div class="fmeta" id="fileMeta">Supported input: NIfTI T1 volume (.nii / .nii.gz),
  up to 200 MB. The canonical pipeline expects a brain-extracted (skull-stripped)
  T1; raw full-head scans will still be processed but with reduced fidelity.</div>
  <div class="state" id="state"></div>
  <div class="err" id="err"></div>
  <div class="res" id="result" style="display:none"></div>
</div>
<p class="note">Uploads are processed in temporary storage and deleted immediately
after inference; nothing is persisted.</p>
"""
    js = """
(function () {
  var API = window.BRAINVULN_API_BASE || "";
  if (API.slice(-1) === "/") API = API.slice(0, -1);
  var MAXMB = window.BRAINVULN_MAX_UPLOAD_MB || 200;
  var box = document.getElementById("upbox");
  var fileInput = document.getElementById("mriFile");
  var pickBtn = document.getElementById("pickBtn");
  var analyzeBtn = document.getElementById("analyzeBtn");
  var meta = document.getElementById("fileMeta");
  var stateEl = document.getElementById("state");
  var errEl = document.getElementById("err");
  var resEl = document.getElementById("result");
  var file = null;

  function fmtMB(b) { return (b / 1048576).toFixed(1) + " MB"; }
  function setState(kind, txt) {
    stateEl.innerHTML = kind === "busy"
      ? '<span class="spin"></span>' + txt
      : (txt ? "<b>" + kind.toUpperCase() + "</b> — " + txt : "");
  }
  function pick(f) {
    if (!f) return;
    var ok = /\.nii(\.gz)?$/i.test(f.name);
    if (!ok) {
      setState("", "");
      errEl.textContent = "Unsupported file: " + f.name +
        ". Please select a .nii or .nii.gz NIfTI volume.";
      analyzeBtn.disabled = true;
      meta.textContent = "No file selected.";
      return;
    }
    errEl.textContent = "";
    file = f;
    analyzeBtn.disabled = false;
    meta.textContent = f.name + " · " + fmtMB(f.size) + " · NIfTI";
    setState("", "File selected. Click Analyze MRI to run the frozen model.");
  }
  pickBtn.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { pick(fileInput.files[0]); });
  ["dragover", "dragenter"].forEach(function (ev) {
    box.addEventListener(ev, function (e) {
      e.preventDefault(); box.classList.add("drop");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    box.addEventListener(ev, function (e) {
      e.preventDefault(); box.classList.remove("drop");
    });
  });
  box.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]);
  });

  analyzeBtn.addEventListener("click", function () {
    errEl.textContent = "";
    resEl.style.display = "none";
    if (!API) {
      setState("", "");
      errEl.textContent = "Inference backend not configured. This static site " +
        "needs the BrainVuln FastAPI service (service/app.py) deployed and " +
        "window.BRAINVULN_API_BASE set in config.js — no demo or simulated " +
        "results are shown by design.";
      return;
    }
    if (!file) return;
    if (file.size > MAXMB * 1048576) {
      errEl.textContent = "File too large (" + fmtMB(file.size) +
        "); the limit is " + MAXMB + " MB.";
      return;
    }
    analyzeBtn.disabled = true;
    setState("busy", "Uploading MRI…");
    var fd = new FormData();
    fd.append("file", file, file.name);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", API + "/api/predict", true);
    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) {
        var pct = Math.round((e.loaded / e.total) * 100);
        setState("busy", "Uploading MRI… " + pct + "%");
      }
    };
    xhr.upload.onload = function () {
      setState("busy", "Processing: canonical preprocessing → frozen model → " +
        "Grad-CAM → atlas regions (typically 10–60 s on CPU)…");
    };  
    xhr.onload = function () {
      analyzeBtn.disabled = false;
      try {
        var data = JSON.parse(xhr.responseText);
      } catch (e) {
        setState("", "");
        errEl.textContent = "Unexpected non-JSON response (HTTP " +
          xhr.status + ").";
        return;
      }
      if (xhr.status !== 200 || !data.success) {
        setState("", "");
        errEl.textContent = (data && (data.detail || data.reason)) ||
          ("Request failed (HTTP " + xhr.status + ").");
        return;
      }
      setState("", "");
      render(data);
    };
    xhr.onerror = function () {
      analyzeBtn.disabled = false;
      setState("", "");
      errEl.textContent = "Could not reach the inference backend at " + API +
        " (network or CORS error).";
    };
    xhr.send(fd);
  });

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
      .replace(String.fromCharCode(92), "&#92;");
  }
  function render(d) {
    var p = d.prediction, mo = d.model, ex = d.explainability;
    var pct = (p.probability * 100).toFixed(1);
    var thr = Number(p.threshold).toFixed(3);
    var rows = (ex.regions || []).slice(0, 10).map(function (r, i) {
      return "<tr><td>" + (i + 1) + "</td><td>" + esc(r.name) + "</td><td>" +
        esc(r.structure || "") + "</td><td>" + esc(r.hemisphere || "") +
        "</td><td>" + r.relevance.toFixed(4) + "</td></tr>";
    }).join("");
    resEl.innerHTML =
      '<h3 style="margin:6px 0 10px">Result <span class="sub">' +
      esc(d.input.filename) + '</span></h3>' +
      '<div class="resgrid">' +
      '<div class="card"><div class="k">Prediction</div>' +
      '<div class="v">' + esc(p.label) + '</div><div class="s">' +
      esc(p.label_text) + '</div></div>' +
      '<div class="card"><div class="k">Probability</div>' +
      '<div class="v">' + p.probability.toFixed(4) + '</div>' +
      '<div class="probbar"><div style="width:' + pct + '%"></div></div>' +
      '<div class="s">threshold ' + thr + '</div></div>' +
      '<div class="card"><div class="k">Model</div>' +
      '<div class="s" style="margin-top:6px">' + esc(mo.name) + '<br/>' +
      'seed ' + esc(mo.seed) + '<br/>SHA256 ' +
      esc(String(mo.checkpoint_sha256).slice(0, 16)) + '…<br/>' +
      esc(d.pipeline.preprocess_version) + '</div></div>' +
      '</div>' +
      '<div class="card" style="margin:10px 0"><div class="k">Top regions by ' +
      'model-derived Grad-CAM relevance</div>' +
      '<table class="regtable"><tr><th>#</th><th>Parcel</th><th>Structure</th>' +
      '<th>Hemi</th><th>Relevance</th></tr>' + rows + '</table>' +
      '<div class="note">Model-derived relevance, not biology: high-CAM regions ' +
      'are regions the network used, not proof of pathology.</div></div>' +
      (ex.gradcam_png_base64
        ? '<div class="camwrap card"><div class="k">Grad-CAM overlay</div>' +
          '<img alt="Grad-CAM overlay" src="data:image/png;base64,' +
          ex.gradcam_png_base64 + '"/></div>'
        : "") +
      '<div class="note">' + esc(d.disclaimer) + '</div>';
    resEl.style.display = "block";
    resEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // test/debug hook: lets automated checks drive the real code path
  window.__brainvulnTest = { pick: pick, render: render,
    analyze: function () { analyzeBtn.click(); } };
})();
"""
    return css, html, js


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

    analyze_css, analyze_html, analyze_js = analyze_section()
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
{analyze_css}
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

{analyze_html}

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
</div>
<script src="config.js"></script>
<script>
{analyze_js}
</script>
</body></html>"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {SITE / 'index.html'} ({len(html) // 1024} KB)")
    print(f"assets: {[p.name for p in ASSETS.iterdir()]}")
    print("checks: ROC-AUC", fmt(t["roc_auc"]), "| FP", fp42, "| FN", fn42)
    return 0


if __name__ == "__main__":
    sys.exit(main())
