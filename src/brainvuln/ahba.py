"""Allen Human Brain Atlas processing and the union cortical+subcortical atlas.

abagen is treated as a first-class part of the experiment: differential-
stability probe selection, donor-aware aggregation, corrected MNI coordinates,
reannotated labels, and — critically — ``missing=None`` so that parcels without
adequate AHBA sampling stay missing instead of being silently filled.

The primary anatomical space is the volumetric union of the Desikan-Killiany
68 cortical atlas (abagen-shipped) and the Tian 2020 Subcortex S1 atlas
(MNI152), constructed non-overlapping: subcortical Tian parcels take precedence
voxel-wise, and cortical parcels that are more than half subsumed by Tian
parcels are dropped entirely rather than kept as cortical shells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

TIAN_S1_URL = ("https://github.com/yetianmed/subcortex/raw/master/"
               "Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz")

DK_TIAN_OVERLAP_DROP_THRESHOLD = 0.5


class AtlasError(RuntimeError):
    """Raised when atlases cannot be fetched, built, or matched."""


# ---------------------------------------------------------------------------
# source atlases
# ---------------------------------------------------------------------------


def fetch_dk68(data_dir: str | Path) -> dict[str, Any]:
    """Fetch the abagen-shipped Desikan-Killiany atlas (volume + any surface)."""
    import abagen

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    try:
        fetched = abagen.fetch_desikan_killiany(data_dir=str(data_dir))
    except TypeError:  # older abagen without data_dir kwarg
        fetched = abagen.fetch_desikan_killiany()
    if isinstance(fetched, dict):
        out["image"] = fetched.get("image")
        out["info"] = fetched.get("info")
        for key in ("left", "right", "lh", "rh", "left_surface", "right_surface",
                    "surface_left", "surface_right", "pial_left", "pial_right"):
            if key in fetched:
                out[key] = fetched[key]
    else:  # Bunch-like object
        out["image"] = getattr(fetched, "image", None)
        out["info"] = getattr(fetched, "info", None)
        for key in ("left", "right", "lh", "rh"):
            out[key] = getattr(fetched, key, None)
    if out.get("image") is None or out.get("info") is None:
        raise AtlasError(
            "abagen.fetch_desikan_killiany did not return the expected "
            "'image'/'info' entries; check your abagen version"
        )
    # optional surface variant, if the installed abagen ships one
    if "left" not in out:
        try:
            surf = abagen.fetch_desikan_killiany(data_dir=str(data_dir), surface=True)
            if isinstance(surf, dict):
                for key in ("left", "right", "lh", "rh"):
                    if key in surf:
                        out[key] = surf[key]
        except (TypeError, AttributeError):
            pass
    return out


def fetch_tian_s1(data_dir: str | Path) -> Path:
    """Download the Tian 2020 Subcortex S1 volumetric atlas (cached)."""
    import requests

    data_dir = Path(data_dir) / "tian"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
    if target.exists() and target.stat().st_size > 10_000:
        return target
    resp = requests.get(TIAN_S1_URL, timeout=120)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


# ---------------------------------------------------------------------------
# union atlas construction
# ---------------------------------------------------------------------------


def _parcel_centroids_mni(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(data)
    ids = ids[ids > 0]
    centroids = np.zeros((len(ids), 3))
    counts = np.zeros(len(ids), dtype=int)
    for i, pid in enumerate(ids):
        vox = np.argwhere(data == pid)
        counts[i] = len(vox)
        centroids[i] = (affine[:3, :3] @ vox.mean(axis=0) + affine[:3, 3])
    return ids, centroids


def build_union_atlas(
    dk: dict[str, Any],
    tian_path: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build the DK68 + Tian S1 non-overlapping volumetric union.

    Returns (union_nifti_path, atlas_info_csv_path, parcel_table).
    """
    import nibabel as nib
    from nilearn.image import resample_img

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dk_img = nib.load(str(dk["image"]))
    tian_img = nib.load(str(tian_path))

    # Resample DK to the Tian grid (nearest-neighbour: label image)
    if dk_img.shape != tian_img.shape or not np.allclose(dk_img.affine, tian_img.affine, atol=1e-3):
        dk_img = resample_img(dk_img, target_affine=tian_img.affine,
                              target_shape=tian_img.shape, interpolation="nearest",
                              copy_header=True, force_resample=True)
    dk_data = np.rint(np.asanyarray(dk_img.dataobj)).astype(int)
    tian_data = np.rint(np.asanyarray(tian_img.dataobj)).astype(int)

    # Read DK info
    dk_info = pd.read_csv(dk["info"])
    id_col = "id" if "id" in dk_info.columns else dk_info.columns[0]
    dk_info[id_col] = dk_info[id_col].astype(int)
    if "label" not in dk_info.columns:
        dk_info["label"] = [f"DK{int(i)}" for i in dk_info[id_col]]

    tian_ids, tian_centroids = _parcel_centroids_mni(tian_data, tian_img.affine)

    # Decide which DK parcels survive (drop parcels mostly subsumed by Tian)
    dk_ids = np.unique(dk_data)
    dk_ids = dk_ids[dk_ids > 0]
    kept_dk: list[int] = []
    for pid in dk_ids:
        vox = dk_data == pid
        overlap = float((tian_data[vox] > 0).mean())
        if overlap <= DK_TIAN_OVERLAP_DROP_THRESHOLD:
            kept_dk.append(int(pid))

    # Assemble union volume: Tian first, then DK in remaining voxels
    union = np.zeros_like(tian_data)
    parcel_rows: list[dict[str, Any]] = []

    next_id = 1
    tian_id_map: dict[int, int] = {}
    for i, pid in enumerate(tian_ids):
        union[tian_data == pid] = next_id
        tian_id_map[int(pid)] = next_id
        cx, cy, cz = tian_centroids[i]
        parcel_rows.append({
            "id": next_id,
            "label": f"TianS1_{int(pid):02d}",
            "hemisphere": "L" if cx < 0 else "R",
            "structure": "subcortex/brainstem",
            "source": "tian_s1",
            "source_id": int(pid),
            "surface_dk_id": -1,
            "n_voxels": int(counts_from(union, next_id)),
            "centroid_x": float(cx), "centroid_y": float(cy), "centroid_z": float(cz),
        })
        next_id += 1

    dk_info_by_id = dk_info.set_index(id_col)
    dk_id_map: dict[int, int] = {}
    for pid in sorted(kept_dk):
        vox = (dk_data == pid) & (union == 0)
        if not vox.any():
            continue
        union[vox] = next_id
        dk_id_map[int(pid)] = next_id
        row = dk_info_by_id.loc[pid]
        label = str(row["label"]).strip() or f"DK{pid}"
        hemi = str(row.get("hemisphere", "?")).strip().upper()[:1]
        hemi = hemi if hemi in ("L", "R", "B") else "?"
        structure = str(row.get("structure", "cortex")).strip().lower()
        vox_mni = np.argwhere(vox)
        centroid = dk_img.affine[:3, :3] @ vox_mni.mean(axis=0) + dk_img.affine[:3, 3]
        parcel_rows.append({
            "id": next_id,
            "label": label,
            "hemisphere": hemi,
            "structure": structure,
            "source": "dk68",
            "source_id": int(pid),
            "surface_dk_id": int(pid),  # original DK id for surface spin tests
            "n_voxels": int(vox.sum()),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
        })
        next_id += 1

    parcel_table = pd.DataFrame(parcel_rows)

    # The abagen DK info reuses the same label for left/right hemisphere rows
    # (e.g. 'bankssts' twice); parcel labels must be unique for all downstream
    # joins, so disambiguate any duplicates with the hemisphere suffix.
    dup = parcel_table["label"].duplicated(keep=False)
    if dup.any():
        def _unique_label(row):
            if row["hemisphere"] in ("L", "R"):
                return f"{row['label']}_{row['hemisphere']}"
            return f"{row['label']}_{row['id']}"

        parcel_table.loc[dup, "label"] = parcel_table[dup].apply(_unique_label, axis=1)
    assert parcel_table["label"].is_unique, "union atlas labels must be unique"

    # Persist
    union_img = nib.Nifti1Image(union.astype(np.int16), tian_img.affine)
    union_path = out_dir / "atlas_union_dk68_tianS1.nii.gz"
    nib.save(union_img, str(union_path))
    info_path = out_dir / "atlas_union_dk68_tianS1_info.csv"
    parcel_table.to_csv(info_path, index=False)

    provenance = {
        "union_atlas": "dk68_tianS1",
        "dk_image": str(dk["image"]),
        "dk_info": str(dk["info"]),
        "tian_image": str(tian_path),
        "tian_url": TIAN_S1_URL,
        "n_tian_parcels": len(tian_id_map),
        "n_dk_parcels_kept": len(dk_id_map),
        "n_dk_parcels_dropped": int(len(dk_ids) - len(kept_dk)),
        "overlap_drop_threshold": DK_TIAN_OVERLAP_DROP_THRESHOLD,
        "n_union_parcels": len(parcel_table),
        "grid_shape": list(tian_img.shape),
    }
    prov_path = out_dir / "atlas_union_provenance.json"
    import json
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return union_path, info_path, parcel_table


def counts_from(data: np.ndarray, pid: int) -> int:
    return int((data == pid).sum())


# ---------------------------------------------------------------------------
# abagen expression
# ---------------------------------------------------------------------------


def get_regional_expression(
    atlas_path: str | Path,
    atlas_info_path: str | Path,
    ahba_cfg,
    data_dir: str | Path = "data/raw/ahba",
    verbose: int = 0,
) -> dict[str, Any]:
    """Run abagen.get_expression_data with the configured primary settings.

    Returns a dict with the group expression matrix (parcels x genes),
    per-donor expression, tissue-sample counts, the abagen report, and the
    parcel coverage table (which parcels stayed missing and why).
    """
    import abagen

    atlas_info = pd.read_csv(atlas_info_path)
    # 'label' must be included: abagen indexes the returned expression matrix
    # by atlas_info['label'] when present (otherwise by numeric ids).
    info_for_abagen = atlas_info[["id", "hemisphere", "structure", "label"]].copy()
    kwargs = ahba_cfg.to_abagen_kwargs()
    result = abagen.get_expression_data(
        atlas=str(atlas_path),
        atlas_info=info_for_abagen,
        data_dir=str(data_dir),
        verbose=verbose,
        **kwargs,
    )
    # abagen 0.1.3 packs: (microarray, [counts], [report]); with
    # return_donors=True the first element is a dict of per-donor
    # parcel x gene DataFrames, and the group matrix is the NaN-aware
    # mean across donors (matching agg_metric='mean').
    if len(result) == 3:
        microarray, counts, report = result
    elif len(result) == 2:
        microarray, counts = result
        report = {}
    else:  # pragma: no cover - abagen changed its return contract
        raise AtlasError(f"Unexpected abagen return arity: {len(result)}")

    if isinstance(microarray, dict):
        donors = microarray
        expr = pd.concat(donors.values()).groupby(level=0, sort=False).mean()
    else:  # return_donors=False path (not used by default)
        donors = {}
        expr = microarray
    if not isinstance(report, dict):
        report = {} if report is None else {"report": str(report)}

    # abagen keys rows by numeric atlas id; rename to parcel labels so all
    # downstream tables (disease maps, coords, plots) join on labels.
    id_to_label = atlas_info.set_index(atlas_info["id"].astype(int))["label"].astype(str)

    def _relabel(df: pd.DataFrame) -> pd.DataFrame:
        renamed = df.copy()
        renamed.index = pd.Index([str(id_to_label.get(int(i), str(i))) for i in df.index],
                                 name="label")
        return renamed

    expr = _relabel(expr)
    donors = {d: _relabel(df) for d, df in donors.items()}

    # Coverage: donors with non-NaN expression per parcel
    n_donors_sampled = {}
    for label in expr.index:
        count = 0
        for d in donors.values():
            rows = d.loc[[label]] if label in d.index else None
            if rows is not None and bool(rows.notna().any(axis=None)):
                count += 1
        n_donors_sampled[label] = count
    coverage = atlas_info[["id", "label", "structure"]].copy()
    coverage["n_donors_sampled"] = coverage["label"].map(n_donors_sampled)
    all_missing = coverage.loc[coverage["n_donors_sampled"] == 0, "label"].tolist()
    low_coverage = coverage.loc[
        (coverage["n_donors_sampled"] > 0) & (coverage["n_donors_sampled"] < 2),
        "label",
    ].tolist()

    return {
        "expr": expr,
        "donors": donors,
        "counts": counts,
        "report": report,
        "coverage": coverage,
        "all_missing_parcels": all_missing,
        "low_coverage_parcels": low_coverage,
        "atlas_info": atlas_info,
    }


def donor_long_format(donors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert per-donor parcel x gene matrices to long format for LOOCV."""
    frames = []
    for donor, df in donors.items():
        long = df.reset_index().melt(
            id_vars=df.index.name or "index", var_name="gene", value_name="value"
        )
        long = long.rename(columns={df.index.name or "index": "label"})
        long = long.dropna(subset=["value"])
        long.insert(0, "donor", str(donor))
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def save_expression_outputs(
    expression_result: dict[str, Any],
    derived_dir: str | Path,
    atlas_name: str,
) -> dict[str, Path]:
    """Persist expression matrices, donor-level data, and the coverage report."""
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    expr_path = derived_dir / f"expression_{atlas_name}.parquet"
    expression_result["expr"].to_parquet(expr_path)
    paths["expression"] = expr_path

    donor_long = donor_long_format(expression_result["donors"])
    donor_path = derived_dir / f"expression_{atlas_name}_donors_long.parquet"
    donor_long.to_parquet(donor_path)
    paths["donors_long"] = donor_path

    coverage_path = derived_dir / f"coverage_{atlas_name}.csv"
    expression_result["coverage"].to_csv(coverage_path, index=False)
    paths["coverage"] = coverage_path

    if isinstance(expression_result.get("counts"), pd.DataFrame):
        counts_path = derived_dir / f"sample_counts_{atlas_name}.csv"
        expression_result["counts"].to_csv(counts_path)
        paths["sample_counts"] = counts_path
    report = expression_result.get("report") or {}
    if report:
        import json

        report_path = derived_dir / f"abagen_report_{atlas_name}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str),
                               encoding="utf-8")
        paths["abagen_report"] = report_path
    return paths


def load_expression(
    derived_dir: str | Path,
    atlas_name: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load cached expression matrices (group + donor-level long)."""
    derived_dir = Path(derived_dir)
    expr_path = derived_dir / f"expression_{atlas_name}.parquet"
    if not expr_path.exists():
        raise AtlasError(
            f"Expression matrix not found: {expr_path}. "
            "Run scripts/build_expression.py first (first run downloads ~2-4 GB of "
            "AHBA microarray data and caches it under data/raw/ahba)."
        )
    expr = pd.read_parquet(expr_path)
    donor_path = derived_dir / f"expression_{atlas_name}_donors_long.parquet"
    donor_long = pd.read_parquet(donor_path) if donor_path.exists() else None
    coverage_path = derived_dir / f"coverage_{atlas_name}.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else None
    return expr, donor_long, coverage


def drop_unsampled_parcels(
    expr: pd.DataFrame, coverage: Optional[pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Drop parcels with no AHBA sampling (all-NaN rows); never impute."""
    usable = expr.dropna(axis=0, how="all")
    dropped = [str(i) for i in expr.index if str(i) not in set(map(str, usable.index))]
    if coverage is not None:
        cov_labels = set(coverage.loc[coverage["n_donors_sampled"] == 0, "label"].astype(str))
        dropped = sorted(set(dropped) | {str(i) for i in expr.index if str(i) in cov_labels})
    return usable, sorted(set(dropped))
