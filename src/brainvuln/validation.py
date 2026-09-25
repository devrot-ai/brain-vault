"""Independent disease-vulnerability maps, synthetic demo maps, and robustness.

The disease map is INDEPENDENT of the gene set by design: it is supplied by
the user (parcellated CSV / NIfTI / GIFTI) and is never used to define genes.
A clearly labeled synthetic generator exists only to smoke-test the pipeline
and is marked ``synthetic=true`` in every output it touches.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DiseaseMapError(RuntimeError):
    """Raised when a disease-vulnerability map cannot be loaded or matched."""


# ---------------------------------------------------------------------------
# loading real maps
# ---------------------------------------------------------------------------

_LABEL_COL_CANDIDATES = ("label", "region", "parcel", "roi", "name", "id", "acronym")
_VALUE_COL_CANDIDATES = ("value", "score", "vulnerability", "rho", "map", "stat", "z")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_disease_map_csv(
    path: str | Path,
    atlas_parcels: list[str],
) -> tuple[pd.Series, dict]:
    """Load a parcellated map from CSV and align it to the atlas parcel order."""
    path = Path(path)
    if not path.exists():
        raise DiseaseMapError(f"Disease map not found: {path}")
    df = pd.read_csv(path)
    label_col = _pick_column(list(df.columns), _LABEL_COL_CANDIDATES)
    value_col = _pick_column(list(df.columns), _VALUE_COL_CANDIDATES)
    if label_col is None:
        raise DiseaseMapError(
            f"No label column found in {path}; expected one of "
            f"{_LABEL_COL_CANDIDATES} (columns present: {list(df.columns)})"
        )
    if value_col is None:
        numeric = [c for c in df.columns if c != label_col
                   and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise DiseaseMapError(f"No numeric value column found in {path}")
        value_col = numeric[0]

    labels = df[label_col].astype(str).str.strip()
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=labels, name=Path(path).stem)

    lookup = {lbl.lower(): lbl for lbl in atlas_parcels}
    aligned = {}
    missing: list[str] = []
    for lbl, val in series.items():
        target = lookup.get(str(lbl).lower())
        if target is None:
            missing.append(str(lbl))
        else:
            aligned[target] = float(val)
    if not aligned:
        raise DiseaseMapError(
            f"None of the {len(series)} map labels matched the atlas parcels "
            f"(e.g. {list(series.index[:5])}); provide a mapping or rename labels"
        )
    out = pd.Series(aligned, name=series.name).reindex(atlas_parcels)
    provenance = {
        "source_path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "value_column": value_col,
        "label_column": label_col,
        "n_input_rows": int(len(series)),
        "n_matched_parcels": int(out.notna().sum()),
        "n_atlas_parcels": len(atlas_parcels),
        "unmatched_map_labels": missing[:25],
        "synthetic": False,
    }
    return out, provenance


def load_disease_map_nifti(
    map_path: str | Path,
    atlas_img_path: str | Path,
    atlas_info: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """Extract per-parcel means from a stat map on the atlas grid (NIfTI)."""
    import nibabel as nib

    map_path, atlas_img_path = Path(map_path), Path(atlas_img_path)
    if not map_path.exists():
        raise DiseaseMapError(f"Disease map not found: {map_path}")
    map_img = nib.load(str(map_path))
    atlas_img = nib.load(str(atlas_img_path))
    if map_img.shape != atlas_img.shape or not np.allclose(
        map_img.affine, atlas_img.affine, atol=1e-3
    ):
        raise DiseaseMapError(
            f"Map grid {map_img.shape} {map_img.affine.diagonal()[:3]} does not match "
            f"atlas grid {atlas_img.shape}; resample the map to the atlas grid first"
        )
    mdata = np.asanyarray(map_img.dataobj, dtype=float)
    adata = np.asanyarray(atlas_img.dataobj).astype(int)
    ids = atlas_info["id"].to_numpy()
    values = np.full(len(ids), np.nan)
    for i, pid in enumerate(ids):
        mask = adata == pid
        if mask.any():
            values[i] = float(np.nanmean(mdata[mask]))
    out = pd.Series(values, index=atlas_info["label"].astype(str), name=map_path.stem)
    provenance = {
        "source_path": str(map_path.resolve()),
        "sha256": _file_sha256(map_path),
        "transformation": "parcel_mean_on_atlas_grid",
        "n_matched_parcels": int(np.isfinite(values).sum()),
        "n_atlas_parcels": len(ids),
        "synthetic": False,
    }
    return out, provenance


def load_disease_map(
    path: str | Path,
    atlas_parcels: list[str],
    atlas_img_path: Optional[str | Path] = None,
    atlas_info: Optional[pd.DataFrame] = None,
) -> tuple[pd.Series, dict]:
    """Dispatch on file extension; CSV is the primary documented format."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_disease_map_csv(path, atlas_parcels)
    if suffix in (".nii", ".nii.gz"):
        if atlas_img_path is None or atlas_info is None:
            raise DiseaseMapError("NIfTI maps require atlas_img_path and atlas_info")
        return load_disease_map_nifti(path, atlas_img_path, atlas_info)
    raise DiseaseMapError(
        f"Unsupported disease-map format {suffix!r}; supported: .csv, .nii, .nii.gz "
        "(GIFTI support is a documented extension)"
    )


# ---------------------------------------------------------------------------
# synthetic demo maps (pipeline smoke tests only — never for inference)
# ---------------------------------------------------------------------------

_PROFILE_TARGETS = {
    # Gaussian-bump target points in MNI space: ((x, y, z), amplitude)
    "anterior_medial_temporal": {
        "targets": [((-24.0, -12.0, -28.0), 1.0), ((24.0, -12.0, -28.0), 1.0)],
        "structure_boost": {"cortex": 0.5},
        "name_keywords": ["entorhinal", "hippocampus", "parahippocampal", "temporal"],
    },
    "subcortical_nigral": {
        "targets": [((0.0, -20.0, -12.0), 1.2), ((-26.0, -2.0, 6.0), 0.8),
                    ((26.0, -2.0, 6.0), 0.8)],
        "structure_boost": {"subcortex/brainstem": 0.6},
        "name_keywords": ["nigra", "pallidum", "putamen", "brainstem"],
    },
    "striatal": {
        "targets": [((-12.0, 10.0, 4.0), 1.2), ((12.0, 10.0, 4.0), 1.2)],
        "structure_boost": {"subcortex/brainstem": 0.6},
        "name_keywords": ["caudate", "putamen", "accumbens", "pallidum"],
    },
}


def generate_synthetic_map(
    parcel_table: pd.DataFrame,
    profile: str = "anterior_medial_temporal",
    seed: int = 42,
) -> tuple[pd.Series, dict]:
    """Deterministic demo vulnerability map from parcel names/coords/structure.

    Weight = max over Gaussian bumps at profile target points, boosted for
    matching structures, + small seeded noise. Results from this map are
    smoke tests, NOT scientific findings; every output records
    ``synthetic: true`` and the profile name.
    """
    if profile not in _PROFILE_TARGETS:
        raise DiseaseMapError(
            f"Unknown synthetic profile {profile!r}; available: {sorted(_PROFILE_TARGETS)}"
        )
    spec = _PROFILE_TARGETS[profile]
    rng = np.random.default_rng(seed)
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    structures = parcel_table["structure"].astype(str).str.lower().to_numpy()
    labels = parcel_table["label"].astype(str).to_numpy()

    sigma2 = 2.0 * 18.0**2  # 18 mm spatial bump
    weights = np.zeros(len(labels))
    for (tx, ty, tz), amp in spec["targets"]:
        d2 = ((coords - np.array([tx, ty, tz])) ** 2).sum(axis=1)
        weights = np.maximum(weights, amp * np.exp(-d2 / sigma2))
    for structure, boost in spec["structure_boost"].items():
        weights[structures == structure.lower()] += boost
    lowered = [str(lbl).lower() for lbl in labels]
    for kw in spec["name_keywords"]:
        weights[[kw in lbl for lbl in lowered]] += 0.4
    weights += rng.normal(0, 0.03, size=len(labels))
    weights = np.clip(weights, 0, None)
    if weights.max() > 0:
        weights = weights / weights.max()

    out = pd.Series(weights, index=labels, name=f"synthetic_{profile}")
    provenance = {
        "profile": profile,
        "synthetic": True,
        "warning": "SYNTHETIC demo map — pipeline smoke test only, not a disease map",
        "seed": seed,
        "n_parcels": int(len(labels)),
    }
    return out, provenance


# ---------------------------------------------------------------------------
# donor leave-one-out and sensitivity orchestration
# ---------------------------------------------------------------------------


def donor_loocv(
    donor_expr: pd.DataFrame,
    genes: list[str],
    disease_map: pd.Series,
    score_fn,
) -> pd.DataFrame:
    """Leave-one-donor-out stability of the score/disease-map correlation.

    ``donor_expr`` is a long-format table with columns
    ``donor, label, gene, value`` (parcels x genes per donor). For each donor,
    the score is rebuilt from the mean across the remaining five donors,
    re-z-scored within that subset, and correlated (Spearman) with the map.
    """
    required = {"donor", "label", "gene", "value"}
    if not required.issubset(donor_expr.columns):
        raise DiseaseMapError(
            f"donor-level expression must have columns {sorted(required)}; "
            f"got {list(donor_expr.columns)}"
        )
    donors = sorted(donor_expr["donor"].unique())
    if len(donors) < 3:
        raise DiseaseMapError(f"Need >=3 donors for LOOCV; found {len(donors)}")
    rows = []
    for held_out in donors:
        sub = donor_expr[donor_expr["donor"] != held_out]
        wide = sub.pivot_table(index="label", columns="gene", values="value",
                               aggfunc="mean")
        wide = wide.reindex(columns=[g for g in genes if g in wide.columns])
        score = score_fn(wide)
        r = _spearman_aligned(score, disease_map)
        rows.append({"held_out_donor": str(held_out), "n_parcels_scored": int(score.notna().sum()),
                     "r": r})
    table = pd.DataFrame(rows)
    finite = table["r"].dropna()
    summary = {
        "median_r": float(finite.median()) if len(finite) else np.nan,
        "iqr_low": float(finite.quantile(0.25)) if len(finite) else np.nan,
        "iqr_high": float(finite.quantile(0.75)) if len(finite) else np.nan,
        "n_donors": len(donors),
    }
    return table, summary


def _spearman_aligned(score: pd.Series, disease_map: pd.Series) -> float:
    from scipy import stats as _stats

    joined = pd.concat([score.rename("s"), disease_map.rename("d")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    return float(_stats.spearmanr(joined["s"], joined["d"]).statistic)


def score_method_sensitivity(
    expr: pd.DataFrame,
    genes: list[str],
    methods: list[str],
    disease_map: pd.Series,
    gene_z_dimension: str = "genes",
    gene_pvalues: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Spatial correlation of S_r with the disease map across score methods."""
    from .scoring import compute_score

    rows = []
    for m in methods:
        try:
            score = compute_score(expr, genes, m, gene_z_dimension, gene_pvalues)
            r = _spearman_aligned(score, disease_map)
        except Exception as exc:  # noqa: BLE001 - sensitivity axes must not crash the run
            r, err = float("nan"), f"{type(exc).__name__}: {exc}"
        else:
            err = ""
        rows.append({"method": m, "r": r, "error": err})
    return pd.DataFrame(rows)
