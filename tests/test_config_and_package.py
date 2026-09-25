"""Tests for config loading, validation errors, and package integrity."""

from __future__ import annotations

import pytest

from brainvuln import __version__, set_global_seed
from brainvuln.config import (
    ConfigError,
    config_hash,
    load_analysis_config,
    load_diseases,
    stamp_manifest,
)
from brainvuln.gene_sets import assemble_gene_set


def test_package_version():
    assert __version__ == "0.1.0"


def test_set_global_seed_runs():
    set_global_seed(42)


def test_load_default_analysis_config():
    cfg = load_analysis_config()
    assert cfg.primary_atlas == "dk68_tianS1"
    assert cfg.ahba.probe_selection == "diff_stability"
    assert cfg.ahba.missing is None
    assert "moran" in cfg.spatial_nulls.volumetric_methods
    assert "burt2020" in cfg.spatial_nulls.volumetric_methods
    assert cfg.spatial_nulls.n_perm == 5000
    assert len(cfg.negative_controls) >= 2
    assert cfg.gwas_catalog.max_requests_per_second <= 15


def test_load_default_diseases():
    diseases = load_diseases()
    assert set(diseases) >= {"alzheimer", "parkinson", "huntington"}
    for d in diseases.values():
        assert d.efo_id.startswith(("EFO_", "MONDO_"))
        assert d.mondo_id.startswith("MONDO_")


def test_missing_efo_id_raises(tmp_path):
    bad = tmp_path / "diseases.yaml"
    bad.write_text(
        "diseases:\n  alzheimer:\n    display_name: Alzheimer's disease\n"
    )
    with pytest.raises(ConfigError, match="efo_id"):
        load_diseases(bad)


def test_missing_analysis_section_raises(tmp_path):
    p = tmp_path / "analysis.yaml"
    p.write_text("run:\n  name: x\n")
    with pytest.raises(ConfigError, match="ahba"):
        load_analysis_config(p)


def test_filled_missing_parcels_rejected_in_primary(tmp_path):
    p = tmp_path / "analysis.yaml"
    p.write_text("ahba:\n  missing: interpolate\n")
    with pytest.raises(ConfigError, match="missing"):
        load_analysis_config(p)


def test_config_hash_stable():
    a = {"x": [1, 2, 3], "y": "s"}
    assert config_hash(a) == config_hash({"y": "s", "x": [1, 2, 3]})
    assert config_hash(a) != config_hash({"x": [1, 2]})


def test_stamp_manifest_fields():
    cfg = load_analysis_config()
    m = stamp_manifest(cfg, extra={"step": "test"})
    for key in ("timestamp_utc", "config_hash", "seed", "package_versions", "step"):
        assert key in m


def test_assemble_gene_set_gwascat(gene_set_json):
    gs = assemble_gene_set("alzheimer", gene_set_json["dir"], "gwascat")
    assert len(gs["genes"]) == 20
    assert gs["definition"] == "gwascat"


def test_assemble_gene_set_unknown_definition(gene_set_json):
    with pytest.raises(Exception, match="Unknown gene_definition"):
        assemble_gene_set("alzheimer", gene_set_json["dir"], "bogus")


def test_assemble_gene_set_missing_file(tmp_path):
    with pytest.raises(Exception, match="fetch_gene_sets"):
        assemble_gene_set("parkinson", tmp_path, "gwascat")


def test_union_atlas_info_schema():
    """The union atlas info carries the columns abagen and plotting rely on."""
    from conftest import build_parcel_table

    pt = build_parcel_table()
    for col in ("id", "label", "hemisphere", "structure",
                "centroid_x", "centroid_y", "centroid_z"):
        assert col in pt.columns
    assert set(pt["hemisphere"]).issubset({"L", "R", "B"})
    assert set(pt["structure"]).issubset({"cortex", "subcortex/brainstem", "cerebellum"})
