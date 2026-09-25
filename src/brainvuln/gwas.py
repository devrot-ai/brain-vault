"""GWAS Catalog REST API v2 client (Tier A gene definitions).

Queries associations by explicit EFO id with explicit child-trait and
gene-mapping policies, follows server pagination via ``next`` links, respects
the documented rate limit, and records full retrieval metadata for every
disease gene set.

API notes (verified LIVE against the GWAS Catalog v2 service, Sep 2026):
- Base path: ``/gwas/rest/api/v2``; filters are snake_case query params.
- The ``efo_id`` filter on ``/associations`` and ``/studies`` matches the
  ontology ids exposed in each association's ``efo_traits`` block — in
  practice MONDO ids (e.g. ``MONDO_0004975`` -> 6716 AD associations).
  Free text is not allowed anywhere.
- Trait labels in the Catalog are normalized WITHOUT apostrophes
  ('Alzheimer disease', 'Parkinson disease'); the ``efo_trait`` label filter
  exists but label matching is fragile, so the id filter is primary.
- ``show_child_traits`` toggles descendant-concept expansion (recorded, not
  assumed); ``extended_geneset=false`` maps variants to mapped + nearest
  upstream/downstream genes (the current web interface).
- Response rows carry flat ``mapped_genes`` symbol lists, ``p_value``,
  ``association_id`` and ``accession_id``; pagination is via the ``page``
  block (``number``/``totalPages``).
- There is no v2 trait-record endpoint; labels/IRIs for verification are
  resolved through the EBI OLS API instead.
- Documented limit: 15 queries per second.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import requests

from .config import DiseaseSpec, GWASClientConfig

OLS_BASE = "https://www.ebi.ac.uk/ols/api"
_OLS_PREFIX = {"MONDO": "mondo", "EFO": "efo", "HP": "hp", "NCIT": "ncit",
               "ORPHANET": "ordo"}


class GWASCatalogError(RuntimeError):
    """Raised when the GWAS Catalog API cannot be queried successfully."""


def _rate_limiter(max_rps: float):
    interval = 1.0 / max(0.1, max_rps)
    state = {"last": 0.0}

    def wait() -> None:
        now = time.monotonic()
        delta = now - state["last"]
        if delta < interval:
            time.sleep(interval - delta)
        state["last"] = time.monotonic()

    return wait


def _request_with_retries(
    url: str,
    params: Optional[dict[str, Any]],
    timeout: int,
    retries: int,
    wait_fn,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        wait_fn()
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"Accept": "application/json"})
        except requests.RequestException as exc:  # network-level failure
            last_exc = exc
            resp = None  # type: ignore[assignment]
        if resp is not None:
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = GWASCatalogError(
                    f"HTTP {resp.status_code} from GWAS Catalog (attempt {attempt + 1}/{retries + 1})"
                )
            else:
                raise GWASCatalogError(
                    f"GWAS Catalog returned HTTP {resp.status_code} for {resp.url}"
                )
        time.sleep(min(60.0, 2.0 ** attempt))
    raise GWASCatalogError(f"GWAS Catalog query failed after {retries + 1} attempts: {last_exc}")


def _iter_pages(
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    cfg: GWASClientConfig,
):
    """Yield JSON payloads for every page, following server-provided next links."""
    wait = _rate_limiter(cfg.max_requests_per_second)
    url: Optional[str] = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    first_params = dict(params)
    page_idx = 0
    while url:
        params_for_call = first_params if page_idx == 0 else None
        resp = _request_with_retries(url, params_for_call, cfg.timeout_seconds, cfg.retries, wait)
        payload = resp.json()
        yield payload
        page_idx += 1
        nxt = None
        links = payload.get("_links", {}) if isinstance(payload, dict) else {}
        if isinstance(links, dict):
            next_link = links.get("next")
            if isinstance(next_link, dict) and next_link.get("href"):
                nxt = next_link["href"]
        if nxt is None and isinstance(payload, dict):
            nxt = payload.get("next") or payload.get("_next")
        url = nxt

_EFO_ID_RE = re.compile(r"^(EFO|MONDO|Orphanet|HP|NCIT)_[A-Za-z0-9]+$")


def validate_efo_id(efo_id: str) -> str:
    """Accept explicit ontology ids like ``EFO_0000249``; reject free text."""
    efo_id = efo_id.strip()
    if not _EFO_ID_RE.match(efo_id):
        raise GWASCatalogError(
            f"efo_id {efo_id!r} does not look like an ontology id "
            "(expected e.g. 'EFO_0000249'). Text matching is not allowed: resolve the "
            "trait at https://www.ebi.ac.uk/gwas/rest/api/v2/efotraits first."
        )
    return efo_id


def fetch_ols_label(ontology_id: str, cfg: Optional[GWASClientConfig] = None) -> dict[str, Any]:
    """Resolve an ontology id (e.g. MONDO_0004975) to label/IRI via EBI OLS."""
    ontology_id = validate_efo_id(ontology_id)
    prefix, _, suffix = ontology_id.partition("_")
    ontology = _OLS_PREFIX.get(prefix.upper())
    if ontology is None:
        return {}
    wait = _rate_limiter(cfg.max_requests_per_second if cfg else 5.0)
    wait()
    url = f"{OLS_BASE}/ontologies/{ontology}/terms"
    try:
        resp = requests.get(url, params={"obo_id": f"{prefix}:{suffix}"},
                            timeout=30, headers={"Accept": "application/json"})
    except requests.RequestException:
        return {}
    if resp.status_code != 200:
        return {}
    terms = resp.json().get("_embedded", {}).get("terms", [])
    if not terms:
        return {}
    return {"label": terms[0].get("label"), "iri": terms[0].get("iri"),
            "source": "OLS"}


def fetch_efo_trait(ontology_id: str, cfg: GWASClientConfig) -> dict[str, Any]:
    """Trait-record lookup for verification (OLS-backed; the Catalog v2 API has
    no trait-record endpoint). Returns ``label``/``traitUri`` keys."""
    info = fetch_ols_label(ontology_id, cfg)
    if not info:
        raise GWASCatalogError(
            f"Could not resolve {ontology_id!r} to a trait record via OLS; "
            "verify the id manually before use"
        )
    return {"traitLabel": info["label"], "traitUri": info.get("iri"),
            "source": "OLS"}


def verify_catalog_trait(disease: DiseaseSpec,
                         cfg: GWASClientConfig) -> dict[str, Any]:
    """Confirm the Catalog trait id has associations before a full retrieval."""
    trait_id = disease.mondo_id or disease.efo_id
    validate_efo_id(trait_id)
    wait = _rate_limiter(cfg.max_requests_per_second)
    params = {"efo_id": trait_id, "size": 1,
              "show_child_traits": str(disease.show_child_traits).lower()}
    resp = _request_with_retries(f"{cfg.api_base.rstrip('/')}/associations",
                                 params, cfg.timeout_seconds, cfg.retries, wait)
    payload = resp.json()
    total = int(payload.get("page", {}).get("totalElements", 0))
    label = None
    rows = payload.get("_embedded", {}).get("associations", [])
    if rows:
        for t in rows[0].get("efo_traits", []):
            if str(t.get("efo_id", "")).upper() == trait_id.upper():
                label = t.get("efo_trait")
                break
    return {"trait_id": trait_id, "n_associations": total,
            "catalog_trait_label": label}


def parse_association_genes(association: dict[str, Any]) -> list[str]:
    """Extract unique mapped gene symbols from one association record.

    v2 rows carry a flat ``mapped_genes`` symbol list; older/other shapes
    (``loci.authorReportedGenes``, ``mappedGenes``) are kept as fallbacks.
    """
    genes: list[str] = []
    for g in association.get("mapped_genes") or []:
        if isinstance(g, str) and g:
            genes.append(g)
        elif isinstance(g, dict):
            symbol = g.get("geneSymbol") or g.get("geneName") or g.get("symbol")
            if symbol:
                genes.append(str(symbol))
    if not genes:
        loci = association.get("loci") or []
        if isinstance(loci, list):
            for locus in loci:
                author_genes = locus.get("authorReportedGenes") or []
                if isinstance(author_genes, list):
                    for g in author_genes:
                        symbol = g.get("geneSymbol") if isinstance(g, dict) else None
                        if symbol:
                            genes.append(str(symbol))
    if not genes:
        for g in association.get("mappedGenes") or []:
            if isinstance(g, dict):
                symbol = g.get("geneSymbol") or g.get("geneName")
                if symbol:
                    genes.append(str(symbol))
            elif isinstance(g, str):
                genes.append(g)
    seen: set[str] = set()
    ordered: list[str] = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered


def fetch_disease_genes(
    disease: DiseaseSpec,
    cfg: GWASClientConfig,
    verbose: bool = False,
    page_cache_dir=None,
) -> dict[str, Any]:
    """Retrieve all associations for the disease trait id and extract gene sets.

    The Catalog v2 ``efo_id`` filter matches the ontology ids in the
    association ``efo_traits`` block — in practice MONDO ids, so the
    ``mondo_id`` is used when available (with ``efo_id`` as fallback).
    Returns the gene list, per-association records, and full retrieval
    metadata required for reproducibility.
    """
    catalog_trait_id = (disease.mondo_id or disease.efo_id).strip()
    validate_efo_id(catalog_trait_id)
    params: dict[str, Any] = {
        "efo_id": catalog_trait_id,
        "show_child_traits": str(disease.show_child_traits).lower(),
        "extended_geneset": str(disease.extended_geneset).lower(),
        "size": cfg.page_size,
    }
    wait = _rate_limiter(cfg.max_requests_per_second)
    url = f"{cfg.api_base.rstrip('/')}/associations"
    genes: set[str] = set()
    associations: list[dict[str, Any]] = []
    n_pages = 0
    page_idx = 0
    total_pages: Optional[int] = None

    # Resumable page cache: each page's parsed rows are stored as JSON so an
    # interrupted retrieval (rate limits, long downloads, process restarts)
    # continues where it stopped instead of refetching everything.
    cache_dir = None
    if page_cache_dir is not None:
        from pathlib import Path as _Path

        cache_dir = _Path(page_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached_rows(idx: int):
        if cache_dir is None:
            return None
        f = cache_dir / f"{disease.key}_gwascat_p{idx}.json"
        if f.exists():
            import json as _json

            return _json.loads(f.read_text(encoding="utf-8"))
        return None

    def _store_rows(idx: int, rows: list[dict[str, Any]]) -> None:
        if cache_dir is None:
            return
        f = cache_dir / f"{disease.key}_gwascat_p{idx}.json"
        f.write_text(json.dumps(rows), encoding="utf-8")

    while True:
        cached = _cached_rows(page_idx)
        if cached is not None:
            rows = cached
            n_pages += 1
        else:
            page_params = dict(params)
            page_params["page"] = page_idx
            resp = _request_with_retries(url, page_params, cfg.timeout_seconds,
                                         cfg.retries, wait)
            payload = resp.json()
            n_pages += 1
            page_meta = payload.get("page", {})
            total_pages = page_meta.get("totalPages", total_pages)
            rows = payload.get("_embedded", {}).get("associations", [])
            if not rows and isinstance(payload.get("associations"), list):
                rows = payload["associations"]
            parsed = [
                {
                    "association_id": a.get("association_id") or a.get("accessionId"),
                    "accession_id": a.get("accession_id") or a.get("accessionId"),
                    "p_value": a.get("p_value") if a.get("p_value") is not None
                    else a.get("pvalue"),
                    "genes": parse_association_genes(a),
                }
                for a in rows
            ]
            _store_rows(page_idx, parsed)
            rows = parsed
        for assoc in rows:
            assoc_genes = assoc.get("genes") or parse_association_genes(assoc)
            associations.append(
                {
                    "association_id": assoc.get("association_id"),
                    "accession_id": assoc.get("accession_id"),
                    "p_value": assoc.get("p_value"),
                    "genes": assoc_genes,
                }
            )
            genes.update(assoc_genes)
        if verbose:
            print(f"  [gwascat] {disease.key}: page {page_idx + 1}/{total_pages}, "
                  f"running unique genes={len(genes)}")
        page_idx += 1
        if total_pages is not None and page_idx >= total_pages:
            break
        if not rows and total_pages is None:
            break

    unique_genes = sorted(genes)
    metadata = disease.gene_set_metadata(
        number_associations=len(associations),
        number_unique_genes=len(unique_genes),
    )
    metadata.update(
        {
            "api_base": cfg.api_base,
            "catalog_trait_id": catalog_trait_id,
            "pages_fetched": n_pages,
            "trait_label": None,
            "trait_uri": None,
        }
    )
    try:
        trait = fetch_ols_label(catalog_trait_id, cfg)
        metadata["trait_label"] = trait.get("label")
        metadata["trait_uri"] = trait.get("iri")
        metadata["label_source"] = "EBI OLS"
    except Exception:  # noqa: BLE001 - verification is best-effort, never invented
        pass

    return {
        "tier": "gwascat",
        "genes": unique_genes,
        "associations": associations,
        "metadata": metadata,
    }


def write_gene_set(result: dict[str, Any], out_path) -> None:
    """Persist a gene-set result (genes + associations + metadata) as JSON."""
    import json
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
