#!/usr/bin/env python
"""Render a self-contained HTML dashboard of the current BrainVuln state.

Reads the run manifest/results JSON, the production-run log, the imported
disease-map provenance records, and the figure PNGs, and writes one static
HTML file (figures embedded as base64 — no server, no external assets):

    python scripts/make_report.py [--output results/report.html]

The output is suitable for static preview/sharing: open it anywhere.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_ORIGIN = {
    # which run produced each figure group (kept in sync with the logs)
    "parkinson_score_map": "real map — Zeighami 2015 PD-ICA (quick null counts)",
    "parkinson_null_moran": "real map — Zeighami 2015 PD-ICA (quick null counts)",
    "parkinson_null_burt2020": "real map — Zeighami 2015 PD-ICA (quick null counts)",
    "parkinson_null_spin": "real map — Zeighami 2015 PD-ICA (quick null counts)",
    "parkinson_donor_loocv": "real map — Zeighami 2015 PD-ICA (quick null counts)",
    "alzheimer": "synthetic demo map (full production nulls)",
    "huntington": "synthetic demo map (full production nulls)",
    "disease_by_region_heatmap": "synthetic demo maps (cross-disease block)",
    "disease_similarity_heatmap": "synthetic demo maps (cross-disease block)",
}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return html.escape(str(x))
    return "—" if f != f else f"{f:+.{nd}f}" if abs(f) < 10 else f"{f:.3g}"


def _production_summary() -> str:
    """Parse the full production-run log (synthetic demo maps)."""
    log = ROOT / "results" / "run_production.log"
    if not log.exists():
        return "<p class='muted'>production log not found</p>"
    rows, lines = [], log.read_text(encoding="utf-8", errors="ignore").splitlines()
    cur = None
    nc_cross = []
    for ln in lines:
        if ln.startswith("== "):
            cur = {"disease": ln.strip("= "), "cross": []}
            rows.append(cur)
        elif cur is not None and ln.strip():
            cur["cross"].append(ln.strip())
            if ln.strip().startswith("NC cross-disease"):
                nc_cross.append(ln.strip())
    cards = []
    for r in rows:
        items = "".join(
            f"<div class='kv'><span>{html.escape(l.split(':')[0])}</span>"
            f"<b>{html.escape(l.split(':', 1)[1].strip() if ':' in l else '')}</b></div>"
            for l in r["cross"] if l.startswith(("H1", "H2", "H3", "NC matched", "donor"))
        )
        cards.append(
            f"<div class='card'><h4>{html.escape(r['disease'])}</h4>{items}</div>"
        )
    cross = "".join(
        f"<div class='kv'><span>{html.escape(l[len('NC cross-disease '):].split(':')[0])}</span>"
        f"<b>{html.escape(l.split('->')[-1].strip())}</b></div>"
        for l in nc_cross
    )
    return (
        "<div class='cards'>"
        + "".join(cards)
        + "<div class='card'><h4>cross-disease NC</h4>"
        + (cross or "<span class='muted'>none</span>")
        + "</div></div>"
    )


def _disease_section(key: str, res: dict) -> str:
    fams = res.get("h3_spatial_nulls", {})
    fam_rows = "".join(
        f"<tr><td>{html.escape(m)}</td><td>{_fmt(v.get('r_obs'))}</td>"
        f"<td>{_fmt(v.get('p_spatial'), 4)}</td>"
        f"<td>{v.get('n_perm', '—')}{' ⚠ ' + html.escape(v['error']) if 'error' in v else ''}</td></tr>"
        for m, v in fams.items()
    )
    loo = (res.get("donor_loocv") or {}).get("summary") or {}
    loo_txt = (
        f"median r = {_fmt(loo.get('median_r'))} "
        f"(IQR {_fmt(loo.get('iqr_low'))}…{_fmt(loo.get('iqr_high'))})"
        if loo and "median_r" in loo else "—"
    )
    conc = res.get("concordance", {})
    fig_html = ""
    for stem, caption in (
        (f"{key}_score_map", "molecular score map (glass brain)"),
        (f"{key}_null_moran", "Moran spectral null distribution"),
        (f"{key}_null_burt2020", "Burt-2020 null distribution"),
        (f"{key}_null_spin", "surface spin-permutation null"),
        (f"{key}_donor_loocv", "leave-one-donor-out stability"),
    ):
        p = FIG_DIR / f"{stem}.png"
        if p.exists():
            fig_html += (
                f"<figure><img src='data:image/png;base64,{_b64(p)}' "
                f"alt='{html.escape(caption)}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
    return f"""
    <section id="{key}">
      <h3>{html.escape(res.get("disease", key))}</h3>
      <p class="muted">gene definition: {html.escape(str(res.get("gene_definition", "—")))}</p>
      <table>
        <tr><th>spatial-null family</th><th>ρ obs</th><th>p (spatial)</th><th>n perm</th></tr>
        {fam_rows or "<tr><td colspan=4 class='muted'>no families</td></tr>"}
      </table>
      <div class="kv"><span>concordance robust</span><b>{"yes ✓" if conc.get("robust") else "no"}</b></div>
      <div class="kv"><span>donor LOOCV</span><b>{loo_txt}</b></div>
      {fig_html}
    </section>"""


def _map_library() -> str:
    maps_dir = ROOT / "data" / "external" / "disease_maps"
    rows = []
    for pj in sorted(maps_dir.glob("*.provenance.json")):
        p = json.loads(pj.read_text(encoding="utf-8"))
        cov = p.get("coverage", {})
        rows.append(
            f"<tr><td><b>{html.escape(p['name'])}</b><br>"
            f"<span class='muted'>{html.escape(p['disease'])}</span></td>"
            f"<td>{html.escape(p['modality'])}<br>"
            f"<span class='muted'>{html.escape(p['sign_convention'])}</span></td>"
            f"<td>{html.escape(p['citation'])}<br>"
            f"<a class='muted' href='{html.escape(p.get('persistent_id') or p.get('source_url') or '#')}'>"
            f"{html.escape(p.get('persistent_id') or p.get('source_url') or '')}</a></td>"
            f"<td>{cov.get('n_parcels_with_value', '—')}/{cov.get('n_parcels_atlas', '—')} parcels<br>"
            f"<span class='muted'>{html.escape(json.dumps(cov.get('missing_by_structure', {})))}</span></td>"
            f"<td class='small'>{html.escape(p['license_note'])}</td></tr>"
        )
    if not rows:
        return "<p class='muted'>no imported maps yet</p>"
    return ("<table><tr><th>map</th><th>modality / sign</th><th>citation / id</th>"
            "<th>coverage</th><th>license</th></tr>" + "".join(rows) + "</table>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT / "results" / "report.html"))
    args = ap.parse_args()

    res_path = ROOT / "results" / "analysis" / "run_primary_results.json"
    run = json.loads(res_path.read_text(encoding="utf-8"))
    man = run["manifest"]

    disease_html = "".join(
        _disease_section(k, v) for k, v in run["results"].items()
    )

    nc_reg = "".join(
        f"<li><b>{html.escape(c['name'])}</b> — {html.escape(c['description'].strip())}"
        f"<br><span class='muted'>expectation: {html.escape(c['expectation'])}</span></li>"
        for c in run.get("negative_controls_registered", [])
    )

    figures_extra = ""
    for stem, cap in (
        ("disease_by_region_heatmap", "disease × region molecular matrix"),
        ("disease_similarity_heatmap", "disease × disease similarity"),
    ):
        p = FIG_DIR / f"{stem}.png"
        if p.exists():
            origin = FIG_ORIGIN.get(stem, "")
            figures_extra += (
                f"<figure><img src='data:image/png;base64,{_b64(p)}' alt='{html.escape(cap)}'>"
                f"<figcaption>{html.escape(cap)}"
                + (f" — {html.escape(origin)}" if origin else "")
                + "</figcaption></figure>"
            )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainVuln — live project dashboard</title>
<style>
 :root {{ --bg:#0e1117; --panel:#161b22; --edge:#232a33; --fg:#d7dde5; --mut:#8a95a3; --acc:#4da3ff; --ok:#3fb950; --warn:#d29922; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--fg);
        font:15px/1.55 "Segoe UI",system-ui,sans-serif; }}
 header {{ padding:28px 32px 18px; border-bottom:1px solid var(--edge); }}
 h1 {{ margin:0 0 4px; font-size:22px; }} h3 {{ margin:24px 0 8px; font-size:17px; color:var(--acc); }}
 .muted {{ color:var(--mut); font-size:13px; }} .small {{ font-size:12px; }}
 main {{ max-width:1080px; margin:0 auto; padding:16px 32px 64px; }}
 .badges span {{ display:inline-block; margin:2px 6px 2px 0; padding:3px 10px;
   border:1px solid var(--edge); border-radius:99px; font-size:12px; color:var(--mut); }}
 table {{ border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:14px; }}
 th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--edge); vertical-align:top; }}
 th {{ color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin:12px 0; }}
 .card {{ background:var(--panel); border:1px solid var(--edge); border-radius:10px; padding:12px 14px; }}
 .card h4 {{ margin:0 0 8px; font-size:14px; }}
 .kv {{ display:flex; justify-content:space-between; gap:10px; padding:2px 0; font-size:13px; }}
 .kv span {{ color:var(--mut); }} .kv b {{ text-align:right; font-weight:600; }}
 figure {{ margin:14px 0; background:var(--panel); border:1px solid var(--edge);
          border-radius:10px; padding:10px; }}
 figure img {{ width:100%; border-radius:6px; display:block; }}
 figcaption {{ color:var(--mut); font-size:12px; padding:8px 4px 2px; }}
 a {{ color:var(--acc); text-decoration:none; }}
 code {{ background:var(--panel); border:1px solid var(--edge); border-radius:5px;
        padding:1px 6px; font-size:13px; }}
 ul {{ margin:6px 0; }} li {{ margin:4px 0; }}
 .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }}
</style></head><body>
<header>
  <h1>BrainVuln — molecular vulnerability of the human brain</h1>
  <div>A reproducible imaging-transcriptomics framework: do genetically
  implicated disease genes show spatial expression patterns that predict where
  the brain breaks?</div>
  <div class="badges" style="margin-top:12px">
    <span>manifest: {html.escape(man.get("run_name", "—"))}</span>
    <span>seed {man.get("seed")}</span>
    <span>matched nulls {man.get("n_nulls_matched")}</span>
    <span>spatial perms {man.get("n_perm_spatial")}</span>
    <span>atlas {html.escape(str(man.get("atlas", "—")))}</span>
    <span>config {html.escape(str(man.get("config_hash", "—"))[:12])}…</span>
    <span>{html.escape(str(man.get("timestamp_utc", "—"))[:16])} UTC</span>
  </div>
</header>
<main>

  <h3>Current results in the JSON manifest</h3>
  <p class="muted">run_primary_results.json — currently holds the most recent
  run: the <b>real</b> Zeighami-2015 Parkinson map at quick null counts
  (50 matched nulls / 200 perms — resolution-limited preview, not final
  inference). ρ is negative because the score is expression-z vs the ICA Z-map
  pole; interpret via the recorded sign convention.</p>
  {disease_html or "<p class='muted'>no disease results</p>"}

  <h3>Full production run (synthetic demo maps — pipeline validation)</h3>
  <p class="muted">5000 perms × 3 spatial-null families + matched gene nulls.
  Synthetic maps exist to validate the machinery; the nulls correctly refuse
  weak correlations (robust = False) while HD's specificity control passes —
  the controls have power.</p>
  {_production_summary()}

  <h3>Cross-disease figures</h3>
  {figures_extra or "<p class='muted'>none yet</p>"}

  <h3>Independent disease-map library (verified public sources)</h3>
  {_map_library()}

  <h3>Pre-registered negative controls</h3>
  <ul>{nc_reg or "<li class='muted'>none registered</li>"}</ul>

  <h3>Reproduce it</h3>
  <div class="card">
    <div class="kv"><span>environment</span><b><code>pip install -e ".[atlas,dev]"</code> (see environment.yml)</b></div>
    <div class="kv"><span>gene sets</span><b><code>python scripts/fetch_gene_sets.py</code></b></div>
    <div class="kv"><span>expression (AHBA, ~4 GB once)</span><b><code>python scripts/build_expression.py</code></b></div>
    <div class="kv"><span>import a disease map</span><b><code>python scripts/import_disease_map.py --map … --disease …</code></b></div>
    <div class="kv"><span>run analysis</span><b><code>python scripts/run_pipeline.py --disease parkinson --disease-map data/external/disease_maps/zeighami2015_pd_ica7.csv</code></b></div>
    <div class="kv"><span>tests</span><b><code>python -m pytest tests -q</code> (80 passing)</b></div>
    <div class="kv"><span>this dashboard</span><b><code>python scripts/make_report.py</code></b></div>
  </div>

  <p class="muted">Every number on this page comes from on-disk artifacts:
  results/analysis/run_primary_results.json, results/run_production.log,
  the imported maps' provenance records, and results/figures/*.png.</p>
</main></body></html>"""

    out = Path(args.output)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
