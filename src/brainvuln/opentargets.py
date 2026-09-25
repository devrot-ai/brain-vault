"""Open Targets Platform GraphQL client (Tier B gene definitions).

Tier B is a robustness layer: genes whose association with the disease is
supported by the Open Targets *genetic* evidence class above an explicit
overall-score threshold. Fine-mapping / locus-to-gene resources from Open
Targets Genetics are exposed in the platform data model; the API does not
serve L2G scores directly, so those require the offline OTG downloads and are
left as a documented extension (see README and paper/analysis_plan.md).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from .config import DiseaseSpec, OpenTargetsConfig

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_ASSOCIATIONS_QUERY = """
query diseaseAssociations($efoId: String!, $page: Pagination!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: $page, enableIndirect: false) {
      count
      rows {
        target { id approvedSymbol }
        score
        datasourceScores { id score }
        datatypeScores { id score }
      }
    }
  }
}
"""


class OpenTargetsError(RuntimeError):
    """Raised when the Open Targets GraphQL API cannot be queried."""


def graphql_request(query: str, variables: dict[str, Any], timeout: int = 30,
                    retries: int = 5) -> dict[str, Any]:
    """POST a GraphQL query with retry/backoff and error propagation."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                API_URL,
                json={"query": query, "variables": variables},
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("errors"):
                    raise OpenTargetsError(f"GraphQL errors: {payload['errors']}")
                return payload
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = OpenTargetsError(f"HTTP {resp.status_code} (attempt {attempt + 1})")
            else:
                raise OpenTargetsError(f"Open Targets API returned HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(min(60.0, 2.0 ** attempt))
    raise OpenTargetsError(f"Open Targets query failed after {retries + 1} attempts: {last_exc}")


def fetch_disease_genetic_genes(
    disease: DiseaseSpec,
    cfg: OpenTargetsConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fetch genetically-supported genes for a disease at all configured thresholds.

    Pages through ``disease.associations`` (100 rows/page), keeps only rows
    whose datasource is genetic (or whose datatype is genetic association),
    and emits a gene set per configured overall-score threshold.
    """
    efo_id = disease.mondo_id or disease.efo_id
    genetic_rows: dict[str, dict[str, Any]] = {}
    page_index = 0
    page_size = 100
    total: Optional[int] = None
    disease_name: Optional[str] = None

    while True:
        variables = {"efoId": efo_id, "page": {"size": page_size, "index": page_index}}
        payload = graphql_request(_ASSOCIATIONS_QUERY, variables)
        data = payload["data"]["disease"]
        if data is None:
            raise OpenTargetsError(
                f"Open Targets has no disease record for id {efo_id!r}; verify the "
                "mondo_id/efo_id in config/diseases.yaml"
            )
        disease_name = data.get("name")
        assoc = data["associatedTargets"]
        total = assoc["count"]
        for row in assoc["rows"]:
            symbol = (row.get("target") or {}).get("approvedSymbol")
            if not symbol:
                continue
            ds_scores = {d["id"]: float(d["score"])
                         for d in row.get("datasourceScores", []) if d.get("id")}
            dt_scores = {d["id"]: float(d["score"])
                         for d in row.get("datatypeScores", []) if d.get("id")}
            genetic = {**dt_scores, **ds_scores}  # datasource ids take precedence
            is_genetic = any(
                k.startswith("genetic_") or k in ("eva", "gwas_credible_sets",
                                                  "unpublished_gwas", "gwas_catalog_gene", "gwas_catalog")
                for k in genetic
            )
            if not is_genetic:
                continue
            prev = genetic_rows.get(symbol)
            if prev is None or float(row["score"]) > float(prev["score"]):
                genetic_rows[symbol] = {
                    "symbol": symbol,
                    "score": float(row["score"]),
                    "datasource_ids": sorted(ds_scores),
                    "datatype_ids": sorted(dt_scores),
                }
        page_index += 1
        if verbose:
            print(f"  [opentargets] {disease.key}: page {page_index}, "
                  f"{len(genetic_rows)} genetic rows so far")
        if page_index * page_size >= total or not assoc["rows"]:
            break

    gene_sets_by_threshold: dict[str, list[str]] = {}
    for threshold in cfg.genetic_score_thresholds:
        thr_str = f"{threshold:.2f}"
        gene_sets_by_threshold[thr_str] = sorted(
            (r["symbol"] for r in genetic_rows.values() if r["score"] >= threshold),
        )

    metadata = disease.gene_set_metadata(
        number_associations=total or 0,
        number_unique_genes=len(genetic_rows),
    )
    metadata.update(
        {
            "api_url": API_URL,
            "enable_indirect": False,
            "ot_score_thresholds": cfg.genetic_score_thresholds,
            "primary_threshold": cfg.primary_threshold,
            "trait_label": disease_name,
            "genetic_rows": sorted(genetic_rows.values(), key=lambda r: -r["score"]),
        }
    )
    return {
        "tier": "opentargets",
        "genes_by_threshold": gene_sets_by_threshold,
        "genes": gene_sets_by_threshold[f"{cfg.primary_threshold:.2f}"],
        "metadata": metadata,
    }


def write_gene_set(result: dict[str, Any], out_path) -> None:
    """Persist a Tier B gene-set result as JSON."""
    import json
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
