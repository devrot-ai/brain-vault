#!/usr/bin/env python
"""Verify the EFO / MONDO ids configured for each disease before any retrieval.

Queries the GWAS Catalog REST v2 for each configured efo_id and the Open
Targets GraphQL API for each mondo_id, and prints the resolved trait labels so
the ids in config/diseases.yaml can be checked against intent. This tool never
edits the config; it only reports.

Usage:
    python scripts/verify_diseases.py [--disease alzheimer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln.config import (  # noqa: E402
    DEFAULT_DISEASES_CONFIG,
    PROJECT_ROOT,
    load_analysis_config,
    load_diseases,
)
from brainvuln.gwas import (  # noqa: E402
    GWASCatalogError,
    fetch_ols_label,
    verify_catalog_trait,
)
from brainvuln.opentargets import OpenTargetsError, graphql_request  # noqa: E402

_DISEASE_QUERY = """
query($efoId: String!) {
  disease(efoId: $efoId) { id name }
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", action="append", default=[])
    parser.add_argument("--diseases-config", default=str(DEFAULT_DISEASES_CONFIG))
    args = parser.parse_args()

    analysis = load_analysis_config(PROJECT_ROOT / "config" / "analysis.yaml")
    diseases = load_diseases(args.diseases_config)
    wanted = args.disease or list(diseases)

    failures = 0
    for key in wanted:
        spec = diseases[key]
        print(f"== {key}: {spec.display_name} ==")
        try:
            ols = fetch_ols_label(spec.mondo_id, analysis.gwas_catalog)
            if ols:
                print(f"  OLS           {spec.mondo_id} -> {ols['label']!r} ({ols.get('iri')})")
            else:
                print(f"  OLS           {spec.mondo_id} -> could not resolve (check manually)")
        except Exception as exc:  # noqa: BLE001
            print(f"  OLS           {spec.mondo_id} -> FAILED: {exc}", file=sys.stderr)
        try:
            cat = verify_catalog_trait(spec, analysis.gwas_catalog)
            print(f"  GWAS Catalog  {cat['trait_id']} -> {cat['n_associations']} associations"
                  + (f" (trait label: {cat['catalog_trait_label']!r})"
                     if cat.get("catalog_trait_label") else ""))
            if cat["n_associations"] == 0:
                failures += 1
                print("                ^ no associations: the Catalog trait id looks wrong",
                      file=sys.stderr)
        except GWASCatalogError as exc:
            failures += 1
            print(f"  GWAS Catalog  -> FAILED: {exc}", file=sys.stderr)
        try:
            payload = graphql_request(_DISEASE_QUERY, {"efoId": spec.mondo_id or spec.efo_id},
                                      timeout=analysis.gwas_catalog.timeout_seconds,
                                      retries=2)
            d = payload["data"]["disease"]
            if d is None:
                print(f"  Open Targets  {spec.mondo_id} -> NOT FOUND")
                failures += 1
            else:
                print(f"  Open Targets  {d['id']} -> {d['name']!r}")
        except OpenTargetsError as exc:
            failures += 1
            print(f"  Open Targets  {spec.mondo_id} -> FAILED: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{failures} id(s) failed verification; fix config/diseases.yaml "
              "before fetching gene sets.", file=sys.stderr)
        return 1
    print("\nAll configured ids resolved. If any label looks wrong, fix "
          "config/diseases.yaml before running the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
