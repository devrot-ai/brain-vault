"""Tests for GWAS Catalog response parsing (offline, fixture-based)."""

from __future__ import annotations

import pandas as pd
import pytest

from brainvuln.gwas import (
    GWASCatalogError,
    parse_association_genes,
    validate_efo_id,
)

ASSOC_V2 = {
    "associationId": "abc123",
    "pvalue": 1.2e-9,
    "loci": [
        {
            "authorReportedGenes": [
                {"geneSymbol": "APOE"},
                {"geneSymbol": "TOMM40"},
                {"geneSymbol": "APOE"},  # duplicate within association
            ]
        }
    ],
}

ASSOC_V1_STYLE = {
    "associationId": "old1",
    "pvalue": None,
    "mappedGenes": ["SORL1", {"geneSymbol": "BIN1"}],
}

ASSOC_EMPTY = {"associationId": "x", "loci": [], "mappedGenes": []}


def test_parse_v2_loci_genes():
    genes = parse_association_genes(ASSOC_V2)
    assert genes == ["APOE", "TOMM40"]  # unique, order-preserving


def test_parse_v1_style_fallback():
    genes = parse_association_genes(ASSOC_V1_STYLE)
    assert "SORL1" in genes and "BIN1" in genes


def test_parse_empty_is_empty_list():
    assert parse_association_genes(ASSOC_EMPTY) == []


def test_validate_efo_id_accepts_ontology_ids():
    assert validate_efo_id("EFO_0000249") == "EFO_0000249"
    assert validate_efo_id("MONDO_0004975") == "MONDO_0004975"


@pytest.mark.parametrize("bad", ["alzheimer", "Alzheimer disease", "EFO-0000249", "", "efo:0000249"])
def test_validate_efo_id_rejects_free_text(bad):
    with pytest.raises(GWASCatalogError):
        validate_efo_id(bad)


def test_disease_metadata_schema(gene_set_json):
    """The retrieval-metadata schema required for reproducibility is present."""
    md = gene_set_json["result"]["metadata"]
    for key in ("disease", "efo_id", "show_child_traits", "extended_geneset",
                "retrieval_date", "number_associations", "number_unique_genes"):
        assert key in md, f"missing metadata key: {key}"
    assert md["number_unique_genes"] == len(gene_set_json["result"]["genes"])
