#!/usr/bin/env python
"""Run the BrainVuln primary analysis for one or more diseases.

Given fetched gene sets, a built expression matrix, and a disease map (real or
synthetic), computes: matched-gene nulls (H1), the molecular score map, spatial
correspondence with the independent disease map (H2), Moran + Burt spatial
nulls (H3), donor LOOCV, score-method sensitivity, and the pre-registered
negative controls. Everything is written to results/analysis/ with a full run
manifest.

Usage:
    python scripts/run_pipeline.py --disease alzheimer --use-synthetic-map --quick
    python scripts/run_pipeline.py --disease alzheimer \
        --disease-map data/external/ad_vulnerability.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln import ahba, gene_sets, plotting, spatial  # noqa: E402
from brainvuln.config import (  # noqa: E402
    PROJECT_ROOT,
    ensure_output_tree,
    load_analysis_config,
    load_diseases,
    stamp_manifest,
)
from brainvuln.scoring import compute_score, score_methods_matrix  # noqa: E402
from brainvuln.surfaces import load_fsaverage_spin_assets  # noqa: E402
from brainvuln.validation import (  # noqa: E402
    donor_loocv,
    generate_synthetic_map,
    load_disease_map,
    score_method_sensitivity,
)


def _load_parcel_table(dirs, atlas_name: str) -> pd.DataFrame:
    info_path = dirs["data_derived"] / f"atlas_{atlas_name}_info.csv"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Atlas info not found: {info_path}; run scripts/build_expression.py first"
        )
    return pd.read_csv(info_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", action="append", required=True)
    parser.add_argument("--disease-map", default=None,
                        help="path to the independent disease map (CSV recommended)")
    parser.add_argument("--use-synthetic-map", action="store_true",
                        help="smoke-test mode: use the labeled synthetic demo map")
    parser.add_argument("--atlas", default=None)
    parser.add_argument("--gene-definition", default=None,
                        help="override: gwascat | opentargets | union")
    parser.add_argument("--analysis-config", default=None)
    parser.add_argument("--diseases-config", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="reduced null counts for smoke testing")
    args = parser.parse_args()

    analysis = load_analysis_config(args.analysis_config or
                                    PROJECT_ROOT / "config" / "analysis.yaml")
    diseases = load_diseases(args.diseases_config or
                             PROJECT_ROOT / "config" / "diseases.yaml")
    dirs = ensure_output_tree()

    if args.use_synthetic_map and args.disease_map:
        parser.error("--use-synthetic-map and --disease-map are mutually exclusive")
    if not args.use_synthetic_map and not args.disease_map:
        parser.error("provide --disease-map (real map) or --use-synthetic-map (smoke test)")

    if args.quick:
        analysis.gene_sets.n_nulls_matched = min(50, analysis.gene_sets.n_nulls_matched)
        analysis.spatial_nulls.n_perm = min(200, analysis.spatial_nulls.n_perm)

    atlas_name = args.atlas or analysis.primary_atlas
    parcel_table = _load_parcel_table(dirs, atlas_name)
    expr, donor_long, coverage = ahba.load_expression(dirs["data_derived"], atlas_name)
    expr, dropped_parcels = ahba.drop_unsampled_parcels(expr, coverage)
    parcel_table = parcel_table[parcel_table["label"].isin(expr.index)].copy()
    coords = (parcel_table.set_index("label")
              .loc[expr.index, ["centroid_x", "centroid_y", "centroid_z"]])

    # cortical surface assets for the spin-permutation sensitivity test
    # (cortical parcels only; fails soft so volumetric nulls always run)
    spin_assets: dict | None = None
    if analysis.spatial_nulls.surface_method:
        try:
            spin_assets = load_fsaverage_spin_assets(parcel_table)
            print("Surface spin assets loaded (fsaverage5 spheres + DK labels)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [spins] unavailable: {exc}", file=sys.stderr)
    gz_dim = analysis.scoring.gene_z_dimension
    null_method = analysis.scoring.primary_method
    if null_method == "p_weighted_mean_z":
        # matched null genes have no association p-values; fall back with a note
        null_method = "mean_z"

    print(f"Expression: {expr.shape[0]} parcels x {expr.shape[1]} genes "
          f"(dropped unsampled parcels: {len(dropped_parcels)})")

    results: dict[str, dict] = {}
    score_maps: dict[str, pd.Series] = {}
    disease_maps: dict[str, pd.Series] = {}
    fig_dir = dirs["results_figures"]
    run_seed = analysis.run.seed

    for disease_key in args.disease:
        if disease_key not in diseases:
            print(f"Unknown disease {disease_key!r}; known: {list(diseases)}",
                  file=sys.stderr)
            return 2
        spec = diseases[disease_key]
        definition = args.gene_definition or analysis.gene_sets.gene_definition
        gs = gene_sets.assemble_gene_set(disease_key, dirs["results_gene_sets"], definition)
        expr.columns = expr.columns.astype(str)
        genes_all = [str(g) for g in gs["genes"]]
        # Restrict to genes actually measured in the AHBA matrix (standard
        # imaging-transcriptomics step; the drop is recorded, never silent).
        genes = [g for g in genes_all if g in set(expr.columns)]
        n_unmeasured = len(genes_all) - len(genes)
        if not genes:
            print(f"No disease genes measured in the AHBA for {disease_key}; skipping",
                  file=sys.stderr)
            continue
        if n_unmeasured:
            print(f"  {disease_key}: {len(genes)}/{len(genes_all)} gene-set genes "
                  f"measured in AHBA ({n_unmeasured} unmeasured genes excluded)")

        gene_pvalues: dict[str, float] = {}
        try:
            raw_gs = gene_sets.load_gene_set_json(
                dirs["results_gene_sets"] / f"{disease_key}_gwascat.json"
            )
            for a in raw_gs.get("associations", []):
                p = a.get("p_value", a.get("pvalue"))
                if a.get("genes") and p not in (None, "NR"):
                    for g in a["genes"]:
                        gene_pvalues.setdefault(g, float(p))
        except Exception:  # noqa: BLE001 - p-weighting is optional
            gene_pvalues = {}

        # ---------------- disease map (independent of the gene set) ----------------
        if args.use_synthetic_map:
            disease_map, map_prov = generate_synthetic_map(
                parcel_table.set_index("label").reindex(expr.index).reset_index(),
                profile=spec.synthetic_map_profile,
                seed=run_seed,
            )
        else:
            disease_map, map_prov = load_disease_map(args.disease_map, list(expr.index))

        # ---------------- H1: matched-gene nulls ----------------
        background, bg_prov = gene_sets.get_background_genes(
            expr, analysis.gene_sets.background, dirs["data_raw"]
        )
        covariates = gene_sets.compute_gene_covariates(expr,
                                                       analysis.gene_sets.match_covariates)
        rng = np.random.default_rng(run_seed + abs(hash(disease_key)) % 100000)
        matched = gene_sets.match_genes_1to1(
            genes, background, covariates, analysis.gene_sets.n_nulls_matched, rng
        )
        null_scores = gene_sets.null_score_distribution(
            expr, matched["matched_sets"],
            lambda sub: compute_score(sub, list(sub.columns), null_method, gz_dim, None),
        )
        obs_score = compute_score(expr, genes, analysis.scoring.primary_method,
                                  gz_dim, gene_pvalues)
        # null_scores is (n_nulls x n_parcels): reduce over PARCELS (axis=1) to
        # get each null draw's peak/range, then compare the observed peak/range
        # against those 1000 per-draw values
        emp_p_peak = gene_sets.empirical_p_greater(
            float(obs_score.max()), null_scores.max(axis=1)  # per-draw max over parcels
        )
        emp_p_range = gene_sets.empirical_p_greater(
            float(np.nanmax(obs_score.to_numpy()) - np.nanmin(obs_score.to_numpy())),
            np.nanmax(null_scores, axis=1) - np.nanmin(null_scores, axis=1),
        )
        h1 = {
            "n_genes": len(genes),
            "n_genes_unmeasured_excluded": n_unmeasured,
            "n_background": len(background),
            "n_nulls": matched["n_nulls"],
            "covariates": matched["covariates"],
            "diagnostics": matched["diagnostics"],
            "background_provenance": bg_prov,
            "peak_score_statistic": {
                "observed": float(obs_score.max()),
                "p_empirical": float(emp_p_peak),
            },
            "range_score_statistic": {
                "observed": float(np.nanmax(obs_score.to_numpy())
                                  - np.nanmin(obs_score.to_numpy())),
                "p_empirical": float(emp_p_range),
            },
            "matched_sets_first10": matched["matched_sets"][:10],
        }

        # ---------------- H2: correspondence with the independent map ----------------
        r_obs = spatial.compare_maps(obs_score, disease_map)
        # Pre-registered negative control 1: matched-null SPATIAL SPECIFICITY.
        # Each matched null gene set's score map is correlated with the disease
        # map; the true rho must lie in the far right tail. This guards H2
        # against the shared-expression-structure confound (null sets match
        # mean/variance, so any signal beyond their tail is gene-set specific).
        null_map_rhos = np.array([
            spatial.compare_maps(
                pd.Series(null_scores[i], index=obs_score.index), disease_map)
            for i in range(null_scores.shape[0])
        ])
        nc_specificity_p = float((1.0 + (null_map_rhos >= r_obs).sum())
                                 / (1.0 + null_map_rhos.size))
        nc_specificity = {
            "control": "matched_gene_sets_spatial_specificity",
            "expectation": "observed_rho_gt_95pct_of_null_rhos",
            "observed_rho": float(r_obs),
            "null_rho_p95": float(np.percentile(null_map_rhos, 95)),
            "null_rho_max": float(null_map_rhos.max()),
            "p_empirical": nc_specificity_p,
            "passed": bool(r_obs > np.percentile(null_map_rhos, 95)),
        }
        h2 = {"r_obs_spearman": r_obs, "map_provenance": map_prov,
              "negative_control_matched_nulls": nc_specificity}

        # ---------------- H3: spatially constrained nulls ----------------
        spatial_results: dict[str, dict] = {}
        for method in analysis.spatial_nulls.volumetric_methods:
            try:
                spatial_results[method] = spatial.run_spatial_null(
                    obs_score, disease_map, coords,
                    method=method,
                    n_perm=analysis.spatial_nulls.n_perm,
                    seed=analysis.spatial_nulls.seed,
                )
            except Exception as exc:  # noqa: BLE001 - record method failures
                spatial_results[method] = {
                    "method": method, "error": f"{type(exc).__name__}: {exc}"
                }
        conc = spatial.concordance(spatial_results,
                                   required=analysis.spatial_nulls.concordance_required)

        if spin_assets is not None:
            try:
                spatial_results[analysis.spatial_nulls.surface_method] = \
                    spatial.surface_spin_pvalues(
                        obs_score, disease_map,
                        spin_assets["verts_lh"], spin_assets["verts_rh"],
                        spin_assets["labels_lh"], spin_assets["labels_rh"],
                        n_perm=analysis.spatial_nulls.n_perm,
                        seed=analysis.spatial_nulls.seed,
                    )
                conc = spatial.concordance(
                    spatial_results,
                    required=analysis.spatial_nulls.concordance_required)
            except Exception as exc:  # noqa: BLE001 - record method failures
                spatial_results[analysis.spatial_nulls.surface_method] = {
                    "method": analysis.spatial_nulls.surface_method,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        # ---------------- sensitivity axes ----------------
        sens = score_method_sensitivity(
            expr, genes,
            [analysis.scoring.primary_method] + analysis.scoring.sensitivity_methods,
            disease_map, gz_dim, gene_pvalues,
        )
        loocv_table, loocv_summary = (None, {})
        if donor_long is not None:
            try:
                loocv_table, loocv_summary = donor_loocv(
                    donor_long, genes, disease_map,
                    lambda df: compute_score(
                        df, [g for g in genes if g in df.columns], null_method,
                        gz_dim, None),
                )
            except Exception as exc:  # noqa: BLE001
                loocv_summary = {"error": f"{type(exc).__name__}: {exc}"}

        results[disease_key] = {
            "disease": spec.display_name,
            "gene_definition": definition,
            "h1_matched_nulls": h1,
            "h2_correspondence": h2,
            "h3_spatial_nulls": spatial_results,
            "concordance": conc,
            "sensitivity_score_methods": sens.to_dict(orient="records"),
            "donor_loocv": {
                "table": (loocv_table.to_dict(orient="records")
                          if loocv_table is not None else None),
                "summary": loocv_summary,
            },
        }
        score_maps[disease_key] = obs_score
        disease_maps[disease_key] = disease_map

        # ---------------- figures ----------------
        try:
            plotting.plot_score_map_glass_brain(
                obs_score,
                parcel_table.set_index("label").reindex(obs_score.index).reset_index(),
                fig_dir / f"{disease_key}_score_map.png",
                title=(f"{spec.display_name} molecular score "
                       f"[SYNTHETIC demo map]" if args.use_synthetic_map
                       else f"{spec.display_name} molecular score"),
            )
            for method, res in spatial_results.items():
                if "null_r" in res and res["null_r"] is not None:
                    plotting.plot_null_distribution(
                        res["r_obs"], np.asarray(res["null_r"]),
                        fig_dir / f"{disease_key}_null_{method}.png",
                        title=f"{spec.display_name}: {method} null (p={res['p_spatial']:.4f})",
                    )
            if loocv_table is not None:
                plotting.plot_loocv_bar(
                    loocv_table, fig_dir / f"{disease_key}_donor_loocv.png",
                    title=f"{spec.display_name}: leave-one-donor-out stability",
                )
        except Exception as exc:  # noqa: BLE001 - figures must not kill an analysis
            print(f"  [figures] warning: {exc}", file=sys.stderr)

        # save the score map itself
        obs_score.rename("score").to_csv(
            dirs["results_analysis"] / f"{disease_key}_score_map.csv"
        )

    # ---------------- cross-disease ----------------
    cross: dict = {}
    if len(score_maps) >= 2:
        matrix = pd.DataFrame(score_maps)
        matrix.to_csv(dirs["results_analysis"] / "disease_by_region_matrix.csv")
        try:
            from brainvuln.cross_disorder import (pairwise_similarity,
                                                  shared_and_specific_parcels)
            sim = pairwise_similarity(matrix, coords,
                                      n_perm=analysis.spatial_nulls.n_perm,
                                      seed=analysis.spatial_nulls.seed)
            sim.to_csv(dirs["results_analysis"] / "disease_similarity.csv", index=False)
            shared = shared_and_specific_parcels(matrix)
            pd.Series({k: str(v) for k, v in shared.items()}).to_csv(
                dirs["results_analysis"] / "shared_specific_parcels.csv")
            cross = {"similarity": sim.to_dict(orient="records"),
                     "shared_specific": shared}
            try:
                sim_mat = matrix.T.corr(method="spearman")
                plotting.plot_similarity_heatmap(
                    sim_mat, fig_dir / "disease_similarity_heatmap.png")
                plotting.plot_disease_region_heatmap(
                    matrix, fig_dir / "disease_by_region_heatmap.png")
            except Exception as exc:  # noqa: BLE001
                print(f"  [figures] cross-disease warning: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            cross = {"error": str(exc)}

    # ---------------- pre-registered control: cross-disease gene sets ----------------
    cross_ctl: dict = {}
    if len(score_maps) >= 2:
        for dk_a, res in results.items():
            try:
                rho_own = res["h2_correspondence"]["r_obs_spearman"]
                others = {dk_b: float(spatial.compare_maps(sm_b,
                                                           disease_maps[dk_a]))
                          for dk_b, sm_b in score_maps.items() if dk_b != dk_a}
                cross_ctl[dk_a] = {
                    "control": "cross_disease_gene_sets",
                    "expectation": "disease_set_rho_gt_unrelated_sets",
                    "own_rho": float(rho_own),
                    "unrelated_rhos": others,
                    "passed": bool(all(r < rho_own for r in others.values())),
                }
            except Exception as exc:  # noqa: BLE001
                cross_ctl[dk_a] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---------------- manifest + results ----------------
    manifest = stamp_manifest(analysis, extra={
        "step": "run_pipeline",
        "diseases": list(results),
        "use_synthetic_map": args.use_synthetic_map,
        "disease_map_path": args.disease_map,
        "atlas": atlas_name,
        "gene_definition": args.gene_definition or analysis.gene_sets.gene_definition,
        "n_nulls_matched": analysis.gene_sets.n_nulls_matched,
        "n_perm_spatial": analysis.spatial_nulls.n_perm,
    })
    out_path = dirs["results_analysis"] / f"run_{analysis.run.name}_results.json"
    payload = {
        "manifest": manifest,
        "negative_controls_registered": analysis.negative_controls,
        "negative_controls_computed": {
            "per_disease": {dk: res["h2_correspondence"].get(
                "negative_control_matched_nulls") for dk, res in results.items()},
            "cross_disease": cross_ctl,
        },
        "results": results,
        "cross_disease": cross,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # ---------------- console summary ----------------
    for key, res in results.items():
        h2 = res["h2_correspondence"]["r_obs_spearman"]
        h3 = {m: r.get("p_spatial") for m, r in res["h3_spatial_nulls"].items()}
        print(f"\n== {res['disease']} ==")
        print(f"  genes: {res['h1_matched_nulls']['n_genes']} ({res['gene_definition']})")
        print(f"  H1 matched-null p (peak / range): "
              f"{res['h1_matched_nulls']['peak_score_statistic']['p_empirical']:.4f} / "
              f"{res['h1_matched_nulls']['range_score_statistic']['p_empirical']:.4f}")
        print(f"  H2 rho vs disease map: {h2:.3f}")
        print(f"  H3 spatial p: {h3} -> robust={res['concordance']['robust']}")
        nc = res["h2_correspondence"].get("negative_control_matched_nulls")
        if nc:
            print(f"  NC matched-null specificity: rho {nc['observed_rho']:.3f} vs "
                  f"null p95 {nc['null_rho_p95']:.3f} "
                  f"-> {'PASS' if nc['passed'] else 'FAIL'}")
        s = res["donor_loocv"]["summary"]
        if s and "median_r" in s:
            print(f"  donor LOOCV: median r = {s['median_r']:.3f} "
                  f"(IQR {s['iqr_low']:.3f}-{s['iqr_high']:.3f})")
    for dk, ctl in cross_ctl.items():
        if "passed" in ctl:
            others = ", ".join(f"{k} {v:.3f}" for k, v in ctl["unrelated_rhos"].items())
            print(f"  NC cross-disease [{dk}]: own {ctl['own_rho']:.3f} vs {others} "
                  f"-> {'PASS' if ctl['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
