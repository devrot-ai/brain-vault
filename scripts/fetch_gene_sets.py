#!/usr/bin/env python
"""Fetch disease gene sets from GWAS Catalog (Tier A) and Open Targets (Tier B).

For each disease in config/diseases.yaml (or --disease), retrieves associations
by explicit EFO id with explicit child-trait and gene-mapping policies, and
writes results/gene_sets/{disease}_gwascat.json and
results/gene_sets/{disease}_opentargets.json with full retrieval metadata.

Usage:
    python scripts/fetch_gene_sets.py [--disease alzheimer] [--tiers gwascat opentargets]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln.config import (  # noqa: E402
    DEFAULT_DISEASES_CONFIG,
    PROJECT_ROOT,
    ensure_output_tree,
    load_analysis_config,
    load_diseases,
)
from brainvuln.gwas import GWASCatalogError, fetch_disease_genes, write_gene_set  # noqa: E402
from brainvuln.opentargets import (  # noqa: E402
    OpenTargetsError,
    fetch_disease_genetic_genes,
    write_gene_set as write_ot_gene_set,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", action="append", default=[],
                        help="disease key(s) from config/diseases.yaml; default: all")
    parser.add_argument("--tiers", nargs="+", default=["gwascat", "opentargets"],
                        choices=["gwascat", "opentargets"],
                        help="which genetic tiers to fetch")
    parser.add_argument("--analysis-config", default=None)
    parser.add_argument("--diseases-config", default=str(DEFAULT_DISEASES_CONFIG))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    analysis = load_analysis_config(args.analysis_config or
                                    PROJECT_ROOT / "config" / "analysis.yaml")
    diseases = load_diseases(args.diseases_config)
    dirs = ensure_output_tree()

    wanted = args.disease or list(diseases)
    unknown = [d for d in wanted if d not in diseases]
    if unknown:
        print(f"Unknown disease key(s): {unknown}; known: {list(diseases)}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str, str]] = []
    for key in wanted:
        spec = diseases[key]
        print(f"== {spec.display_name} ({key}) ==")
        if "gwascat" in args.tiers and spec.gwas_catalog:
            try:
                result = fetch_disease_genes(
                    spec, analysis.gwas_catalog, verbose=args.verbose,
                    page_cache_dir=dirs["results_gene_sets"] / "page_cache",
                )
                out = dirs["results_gene_sets"] / f"{key}_gwascat.json"
                write_gene_set(result, out)
                md = result["metadata"]
                print(f"  [gwascat] {md['number_associations']} associations -> "
                      f"{md['number_unique_genes']} unique genes "
                      f"(child_traits={md['show_child_traits']}) -> {out.name}")
            except GWASCatalogError as exc:
                failures.append((key, "gwascat", str(exc)))
                print(f"  [gwascat] FAILED: {exc}", file=sys.stderr)
        if "opentargets" in args.tiers and spec.open_targets:
            try:
                result = fetch_disease_genetic_genes(spec, analysis.opentargets,
                                                     verbose=args.verbose)
                for threshold, out_name in (
                    (analysis.opentargets.primary_threshold, f"{key}_opentargets.json"),
                    *[(t, f"{key}_opentargets_{t:.2f}.json")
                      for t in analysis.opentargets.genetic_score_thresholds
                      if abs(t - analysis.opentargets.primary_threshold) > 1e-9],
                ):
                    per_thr = dict(result)
                    per_thr["genes"] = result["genes_by_threshold"][f"{threshold:.2f}"]
                    write_ot_gene_set(per_thr,
                                      dirs["results_gene_sets"] / out_name)
                n = len(result["genes"])
                print(f"  [opentargets] {n} genes at score >= "
                      f"{analysis.opentargets.primary_threshold:.2f} "
                      f"(thresholds also saved: "
                      f"{analysis.opentargets.genetic_score_thresholds})")
            except OpenTargetsError as exc:
                failures.append((key, "opentargets", str(exc)))
                print(f"  [opentargets] FAILED: {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures (recorded above); re-run to retry.", file=sys.stderr)
        return 1
    print("\nAll gene sets fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
