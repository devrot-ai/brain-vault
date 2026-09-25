"""Configuration loading, validation, and run-manifest stamping.

All inferential choices live in ``config/analysis.yaml`` and
``config/diseases.yaml``; this module turns them into typed objects, fails
loudly on invalid settings, and stamps every run with full provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "config" / "analysis.yaml"
DEFAULT_DISEASES_CONFIG = PROJECT_ROOT / "config" / "diseases.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# dataclass configs
# ---------------------------------------------------------------------------


@dataclass
class AhbaConfig:
    probe_selection: str = "diff_stability"
    donor_probes: str = "aggregate"
    region_agg: str = "donors"
    agg_metric: str = "mean"
    corrected_mni: bool = True
    reannotated: bool = True
    sample_norm: str = "srs"
    gene_norm: str = "srs"
    missing: Optional[str] = None
    ibf_threshold: float = 0.5
    lr_mirror: Optional[str] = None
    donors: Any = "all"
    data_dir: str = "data/raw/ahba"

    def to_abagen_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``abagen.get_expression_data`` (minus atlas)."""
        return dict(
            probe_selection=self.probe_selection,
            donor_probes=self.donor_probes,
            region_agg=self.region_agg,
            agg_metric=self.agg_metric,
            corrected_mni=self.corrected_mni,
            reannotated=self.reannotated,
            sample_norm=self.sample_norm,
            gene_norm=self.gene_norm,
            missing=self.missing,
            ibf_threshold=self.ibf_threshold,
            lr_mirror=self.lr_mirror,
            donors=self.donors,
            return_counts=True,
            return_donors=True,
            return_report=True,
        )


@dataclass
class AtlasConfig:
    name: str
    kind: str
    description: str = ""
    cortical: Optional[str] = None
    subcortical: Optional[str] = None


@dataclass
class GeneSetsConfig:
    gene_definition: str = "gwascat"
    background: str = "bing_scale"
    match_covariates: list[str] = field(
        default_factory=lambda: ["mean_expression", "expression_variance"]
    )
    n_nulls_matched: int = 1000
    matching: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringConfig:
    primary_method: str = "mean_z"
    gene_z_dimension: str = "genes"
    sensitivity_methods: list[str] = field(
        default_factory=lambda: ["mean_rank", "first_pc", "p_weighted_mean_z"]
    )


@dataclass
class SpatialNullsConfig:
    n_perm: int = 5000
    seed: int = 42
    volumetric_methods: list[str] = field(default_factory=lambda: ["moran", "burt2020"])
    surface_method: str = "spin"
    two_sided: bool = True
    fdr_method: str = "bh"
    fdr_alpha: float = 0.05
    concordance_required: int = 2


@dataclass
class OpenTargetsConfig:
    genetic_score_thresholds: list[float] = field(default_factory=lambda: [0.10, 0.30])
    primary_threshold: float = 0.10
    enable_indirect: bool = False
    datasource_filter: str = "genetic"


@dataclass
class GWASClientConfig:
    api_base: str = "https://www.ebi.ac.uk/gwas/rest/api/v2"
    page_size: int = 200
    max_requests_per_second: float = 8.0
    timeout_seconds: int = 30
    retries: int = 5
    extended_geneset: bool = False


@dataclass
class RunConfig:
    name: str = "primary"
    seed: int = 42
    output_dir: str = "results"


DEFAULT_CHECKPOINT_RELPATH = 'results/ml/resnet_seed42/checkpoints/best.pt'


@dataclass
class InferenceConfig:
    '''Canonical inference checkpoint - the single serving checkpoint source.

    Resolution never searches the filesystem and never uses mtime; see
    resolve_checkpoint().
    '''

    checkpoint_path: str = DEFAULT_CHECKPOINT_RELPATH


@dataclass
class AnalysisConfig:
    run: RunConfig = field(default_factory=RunConfig)
    ahba: AhbaConfig = field(default_factory=AhbaConfig)
    atlases: dict[str, AtlasConfig] = field(default_factory=dict)
    primary_atlas: str = "dk68_tianS1"
    gene_sets: GeneSetsConfig = field(default_factory=GeneSetsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    spatial_nulls: SpatialNullsConfig = field(default_factory=SpatialNullsConfig)
    negative_controls: list[dict[str, str]] = field(default_factory=list)
    opentargets: OpenTargetsConfig = field(default_factory=OpenTargetsConfig)
    gwas_catalog: GWASClientConfig = field(default_factory=GWASClientConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    source_path: Optional[Path] = None

    def to_manifest_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d


@dataclass
class DiseaseSpec:
    key: str
    display_name: str
    efo_id: str
    mondo_id: str
    show_child_traits: bool = False
    extended_geneset: bool = False
    gwas_catalog: bool = True
    open_targets: bool = True
    synthetic_map_profile: str = "anterior_medial_temporal"
    notes: str = ""

    def gene_set_metadata(self, number_associations: int, number_unique_genes: int) -> dict[str, Any]:
        """Retrieval-metadata schema shared by both genetic tiers."""
        return {
            "disease": self.display_name,
            "disease_key": self.key,
            "efo_id": self.efo_id,
            "mondo_id": self.mondo_id,
            "show_child_traits": self.show_child_traits,
            "extended_geneset": self.extended_geneset,
            "retrieval_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "number_associations": number_associations,
            "number_unique_genes": number_unique_genes,
        }


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return mapping[key]


def load_analysis_config(path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> AnalysisConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Analysis config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - yaml detail varies
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Analysis config {path} must be a YAML mapping")

    cfg = AnalysisConfig()

    run = raw.get("run", {})
    cfg.run = RunConfig(
        name=run.get("name", "primary"),
        seed=int(run.get("seed", 42)),
        output_dir=run.get("output_dir", "results"),
    )

    ahba_raw = raw.get("ahba")
    if ahba_raw is None:
        raise ConfigError(f"Missing required section 'ahba' in {path}")
    _missing = _require(ahba_raw, "missing", f"'ahba' section of {path}")
    if _missing not in (None, "centroids", "interpolate"):
        raise ConfigError(
            f"ahba.missing must be null, 'centroids', or 'interpolate' (got {_missing!r}); "
            "the primary analysis must not silently fill parcels"
        )
    if _missing is not None:
        raise ConfigError(
            "ahba.missing is set to "
            f"{_missing!r}; the primary analysis requires null (missing parcels stay missing). "
            "Amend the config explicitly if you intend a filled-imputation sensitivity run."
        )
    cfg.ahba = AhbaConfig(
        probe_selection=ahba_raw.get("probe_selection", "diff_stability"),
        donor_probes=ahba_raw.get("donor_probes", "aggregate"),
        region_agg=ahba_raw.get("region_agg", "donors"),
        agg_metric=ahba_raw.get("agg_metric", "mean"),
        corrected_mni=bool(ahba_raw.get("corrected_mni", True)),
        reannotated=bool(ahba_raw.get("reannotated", True)),
        sample_norm=ahba_raw.get("sample_norm", "srs"),
        gene_norm=ahba_raw.get("gene_norm", "srs"),
        missing=_missing,
        ibf_threshold=float(ahba_raw.get("ibf_threshold", 0.5)),
        lr_mirror=ahba_raw.get("lr_mirror"),
        donors=ahba_raw.get("donors", "all"),
        data_dir=ahba_raw.get("data_dir", "data/raw/ahba"),
    )

    atlases_raw = raw.get("atlases", {})
    cfg.primary_atlas = atlases_raw.get("primary", "dk68_tianS1")
    for name, spec in atlases_raw.items():
        if name == "primary" or not isinstance(spec, dict):
            continue
        cfg.atlases[name] = AtlasConfig(
            name=name,
            kind=spec.get("kind", "union"),
            description=str(spec.get("description", "")).strip(),
            cortical=spec.get("cortical"),
            subcortical=spec.get("subcortical"),
        )
    if cfg.primary_atlas not in cfg.atlases:
        raise ConfigError(
            f"atlases.primary '{cfg.primary_atlas}' is not defined in the atlases section"
        )

    gs_raw = raw.get("gene_sets", {})
    cfg.gene_sets = GeneSetsConfig(
        gene_definition=gs_raw.get("gene_definition", "gwascat"),
        background=gs_raw.get("background", "bing_scale"),
        match_covariates=list(gs_raw.get("match_covariates",
                                         ["mean_expression", "expression_variance"])),
        n_nulls_matched=int(gs_raw.get("n_nulls_matched", 1000)),
        matching=dict(gs_raw.get("matching", {})),
    )

    sc_raw = raw.get("scoring", {})
    cfg.scoring = ScoringConfig(
        primary_method=sc_raw.get("primary_method", "mean_z"),
        gene_z_dimension=sc_raw.get("gene_z_dimension", "genes"),
        sensitivity_methods=list(sc_raw.get("sensitivity_methods",
                                            ["mean_rank", "first_pc", "p_weighted_mean_z"])),
    )

    sn_raw = raw.get("spatial_nulls", {})
    cfg.spatial_nulls = SpatialNullsConfig(
        n_perm=int(sn_raw.get("n_perm", 5000)),
        seed=int(sn_raw.get("seed", 42)),
        volumetric_methods=list(sn_raw.get("volumetric_methods", ["moran", "burt2020"])),
        surface_method=sn_raw.get("surface_method", "spin"),
        two_sided=bool(sn_raw.get("two_sided", True)),
        fdr_method=sn_raw.get("fdr_method", "bh"),
        fdr_alpha=float(sn_raw.get("fdr_alpha", 0.05)),
        concordance_required=int(sn_raw.get("concordance_required", 2)),
    )

    cfg.negative_controls = list(raw.get("negative_controls", []))

    ot_raw = raw.get("opentargets", {})
    cfg.opentargets = OpenTargetsConfig(
        genetic_score_thresholds=[float(t) for t in ot_raw.get("genetic_score_thresholds",
                                                               [0.10, 0.30])],
        primary_threshold=float(ot_raw.get("primary_threshold", 0.10)),
        enable_indirect=bool(ot_raw.get("enable_indirect", False)),
        datasource_filter=ot_raw.get("datasource_filter", "genetic"),
    )

    gc_raw = raw.get("gwas_catalog", {})
    cfg.gwas_catalog = GWASClientConfig(
        api_base=gc_raw.get("api_base", "https://www.ebi.ac.uk/gwas/rest/api/v2"),
        page_size=int(gc_raw.get("page_size", 200)),
        max_requests_per_second=float(gc_raw.get("max_requests_per_second", 8.0)),
        timeout_seconds=int(gc_raw.get("timeout_seconds", 30)),
        retries=int(gc_raw.get("retries", 5)),
        extended_geneset=bool(gc_raw.get("extended_geneset", False)),
    )

    inf_raw = raw.get('inference', {}) or {}
    chk_raw = inf_raw.get('checkpoint', {}) or {}
    cfg.inference = InferenceConfig(
        checkpoint_path=str(chk_raw.get('path', DEFAULT_CHECKPOINT_RELPATH)),
    )

    cfg.source_path = path.resolve()
    return cfg


def load_diseases(path: str | Path = DEFAULT_DISEASES_CONFIG) -> dict[str, DiseaseSpec]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Diseases config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    diseases_raw = raw.get("diseases")
    if not isinstance(diseases_raw, dict) or not diseases_raw:
        raise ConfigError(f"'diseases' section missing or empty in {path}")

    diseases: dict[str, DiseaseSpec] = {}
    for key, spec in diseases_raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"Disease '{key}' in {path} must be a mapping")
        efo_id = spec.get("efo_id")
        if not efo_id:
            raise ConfigError(
                f"Disease '{key}' has no efo_id; the GWAS Catalog query requires an explicit "
                "ontology id (never a text match). Verify the id at "
                "https://www.ebi.ac.uk/gwas/rest/api/v2/efotraits and record it here."
            )
        diseases[str(key)] = DiseaseSpec(
            key=str(key),
            display_name=str(spec.get("display_name", key)),
            efo_id=str(efo_id),
            mondo_id=str(spec.get("mondo_id", "")),
            show_child_traits=bool(spec.get("show_child_traits", False)),
            extended_geneset=bool(spec.get("extended_geneset", False)),
            gwas_catalog=bool(spec.get("gwas_catalog", True)),
            open_targets=bool(spec.get("open_targets", True)),
            synthetic_map_profile=str(spec.get("synthetic_map_profile",
                                               "anterior_medial_temporal")),
            notes=str(spec.get("notes", "")).strip(),
        )
    return diseases


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# canonical checkpoint resolution (single source of truth for serving)
# ---------------------------------------------------------------------------

def resolve_checkpoint(path: str | Path | None = None,
                       config_path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> Path:
    '''Resolve THE configured inference checkpoint - never search, never mtime.

    With path given, resolves that exact file (CLI override). Otherwise
    reads inference.checkpoint.path from the analysis config. Relative
    paths are interpreted against the project root. Raises FileNotFoundError
    loudly when the resolved file does not exist; no fallback discovery of any
    kind is performed.
    '''
    if path is not None:
        configured = Path(path)
        origin = 'CLI --checkpoint argument'
    else:
        cfg = load_analysis_config(config_path)
        configured = Path(cfg.inference.checkpoint_path)
        origin = 'inference.checkpoint.path in ' + str(config_path)
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    configured = Path(os.path.normpath(str(configured)))
    if not configured.is_file():
        raise FileNotFoundError(
            'Configured checkpoint does not exist: ' + str(configured)
            + ' (source: ' + origin + '). No newest-file fallback is performed; '
            'fix inference.checkpoint.path in the analysis config or pass an '
            'explicit checkpoint path.')
    return configured.resolve()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    '''SHA-256 over the exact file bytes (streamed; deterministic).'''
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path | None = None,
                        config_path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> dict[str, Any]:
    '''Structured identity of the canonical checkpoint for run manifests.'''
    resolved = resolve_checkpoint(path, config_path)
    return {
        'path': str(resolved),
        'filename': resolved.name,
        'sha256': sha256_file(resolved),
        'size_bytes': int(resolved.stat().st_size),
    }


# paths and provenance
# ---------------------------------------------------------------------------


def ensure_output_tree(root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Create the repository output tree; returns the directory map."""
    dirs = {
        "root": root,
        "data_raw": root / "data" / "raw",
        "data_external": root / "data" / "external",
        "data_negative_controls": root / "data" / "external" / "negative_controls",
        "data_derived": root / "data" / "derived",
        "results_gene_sets": root / "results" / "gene_sets",
        "results_analysis": root / "results" / "analysis",
        "results_figures": root / "results" / "figures",
        "paper": root / "paper",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def config_hash(config_dict: dict[str, Any]) -> str:
    """Stable SHA-256 of a config dict (sorted keys, canonical JSON)."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(root: Path = PROJECT_ROOT) -> Optional[str]:
    """Best-effort current git commit hash; None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def package_versions() -> dict[str, str]:
    """Versions of the packages that materially affect results (best effort)."""
    result: dict[str, str] = {}
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "abagen", "neuromaps",
                "nibabel", "nilearn", "brainspace"):
        try:
            module = __import__(mod)
            result[mod] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - provenance must never crash a run
            continue
    return result


def stamp_manifest(analysis: AnalysisConfig, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Full provenance stamp for a run: commit, config hash, versions, seeds."""
    manifest = {
        "run_name": analysis.run.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python_version": sys.version.split()[0],
        "package_versions": package_versions(),
        "config_hash": config_hash(analysis.to_manifest_dict()),
        "seed": analysis.run.seed,
        "spatial_null_seed": analysis.spatial_nulls.seed,
        "analysis_config_source": str(analysis.source_path) if analysis.source_path else None,
    }
    if extra:
        manifest.update(extra)
    return manifest
