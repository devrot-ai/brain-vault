#!/usr/bin/env python
"""Compute and render the molecular brain map for one disease.

Standalone entry point (also packaged as ``brainvuln-make-score-map``):
recomputes the primary molecular score S_r for one disease gene set and writes
the parcellated CSV plus a glass-brain figure.

Usage:
    python scripts/make_score_map.py --disease alzheimer \
        [--method mean_z] [--gene-definition gwascat]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln import ahba, gene_sets, plotting  # noqa: E402
from brainvuln.config import (  # noqa: E402
    PROJECT_ROOT,
    ensure_output_tree,
    load_analysis_config,
    load_diseases,
    stamp_manifest,
)
from brainvuln.scoring import compute_score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--gene-definition", default=None)
    parser.add_argument("--atlas", default=None)
    parser.add_argument("--analysis-config", default=None)
    parser.add_argument("--diseases-config", default=None)
    args = parser.parse_args()

    analysis = load_analysis_config(args.analysis_config or
                                    PROJECT_ROOT / "config" / "analysis.yaml")
    diseases = load_diseases(args.diseases_config or
                             PROJECT_ROOT / "config" / "diseases.yaml")
    dirs = ensure_output_tree()

    if args.disease not in diseases:
        print(f"Unknown disease {args.disease!r}; known: {list(diseases)}", file=sys.stderr)
        return 2
    spec = diseases[args.disease]
    atlas_name = args.atlas or analysis.primary_atlas
    method = args.method or analysis.scoring.primary_method
    definition = args.gene_definition or analysis.gene_sets.gene_definition

    parcel_table = pd.read_csv(dirs["data_derived"] / f"atlas_{atlas_name}_info.csv")
    expr, _, coverage = ahba.load_expression(dirs["data_derived"], atlas_name)
    expr, dropped = ahba.drop_unsampled_parcels(expr, coverage)

    gs = gene_sets.assemble_gene_set(args.disease, dirs["results_gene_sets"], definition)
    gene_pvalues: dict[str, float] = {}
    try:
        raw_gs = gene_sets.load_gene_set_json(
            dirs["results_gene_sets"] / f"{args.disease}_gwascat.json"
        )
        for a in raw_gs.get("associations", []):
            p = a.get("p_value", a.get("pvalue"))
            if a.get("genes") and p not in (None, "NR"):
                for g in a["genes"]:
                    gene_pvalues.setdefault(g, float(p))
    except Exception:  # noqa: BLE001
        gene_pvalues = {}

    score = compute_score(expr, gs["genes"], method,
                          analysis.scoring.gene_z_dimension, gene_pvalues)

    out_csv = dirs["results_analysis"] / f"{args.disease}_score_map.csv"
    score.rename("score").to_csv(out_csv)
    parcel_indexed = parcel_table.set_index("label").reindex(score.index).reset_index()
    try:
        fig = plotting.plot_score_map_glass_brain(
            score, parcel_indexed,
            dirs["results_figures"] / f"{args.disease}_score_map.png",
            title=f"{spec.display_name} molecular score ({method})",
        )
        print(f"Figure: {fig}")
    except Exception as exc:  # noqa: BLE001
        print(f"Figure skipped: {exc}", file=sys.stderr)

    manifest = stamp_manifest(analysis, extra={
        "step": "make_score_map",
        "disease": args.disease,
        "method": method,
        "gene_definition": definition,
        "n_genes": len(gs["genes"]),
        "n_parcels": int(score.notna().sum()),
        "dropped_unsampled_parcels": dropped,
    })
    (dirs["results_analysis"] / f"manifest_score_map_{args.disease}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Score map: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
