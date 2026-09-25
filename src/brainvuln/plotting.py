"""Publication-oriented figures: brain maps, null distributions, heatmaps.

Rendering uses nilearn for glass-brain/volume display and (when fsaverage
surfaces are available) surface projection; matplotlib/seaborn for
statistical graphics. Figures are saved under ``results/figures/``; every
function is defensive about the optional plotting dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class PlottingError(RuntimeError):
    """Raised when a figure cannot be produced."""


def _ensure_dirs(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_score_map_glass_brain(
    score_map: pd.Series,
    parcel_table: pd.DataFrame,
    out_path: str | Path,
    title: str = "Molecular score map",
) -> Path:
    """MNI glass-brain scatter/bubble view of the parcellated score map."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from nilearn import plotting as nplot
    except ImportError as exc:  # pragma: no cover
        raise PlottingError(f"nilearn/matplotlib required for glass-brain figures: {exc}") from exc

    joined = parcel_table.set_index("label").reindex(score_map.index)
    ok = score_map.notna()
    coords = joined.loc[ok, ["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    values = score_map[ok].to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 5))
    display = nplot.plot_glass_brain(
        None, display_mode="lyrz", figure=fig, alpha=0.6,
    )
    cmap = matplotlib.colormaps["RdYlBu_r"]
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    normed = (values - vmin) / (vmax - vmin)
    # One batched call with explicit arrays: passing a single RGBA tuple per
    # marker trips nilearn's hemisphere filtering (a length-4 color is
    # misread as 4 markers), and per-marker calls are slow anyway.
    display.add_markers(
        coords,
        marker_color=cmap(normed),
        marker_size=28 * (0.4 + 0.6 * normed),
    )
    plt.title(title)
    out_path = _ensure_dirs(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_null_distribution(
    r_obs: float,
    null_r: np.ndarray,
    out_path: str | Path,
    title: str = "Spatial-null distribution of Spearman rho",
) -> Path:
    """Histogram of null correlations with the observed value marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise PlottingError(f"matplotlib required for null-distribution figures: {exc}") from exc

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(null_r, bins=40, color="#9ecae1", edgecolor="#3182bd", alpha=0.9)
    ax.axvline(r_obs, color="#de2d26", lw=2,
               label=f"observed r = {r_obs:.3f}")
    ax.axvspan(np.percentile(null_r, 2.5), np.percentile(null_r, 97.5),
               color="grey", alpha=0.2, label="null 95% interval")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("null maps")
    ax.set_title(title)
    ax.legend(frameon=False)
    out_path = _ensure_dirs(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loocv_bar(
    loocv_table: pd.DataFrame,
    out_path: str | Path,
    title: str = "Leave-one-donor-out stability",
) -> Path:
    """Bar chart of per-donor-held-out correlations with median and IQR band."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise PlottingError(f"matplotlib required for LOOCV figures: {exc}") from exc

    r = loocv_table["r"].to_numpy(dtype=float)
    labels = loocv_table["held_out_donor"].astype(str).to_numpy()
    finite = r[np.isfinite(r)]
    median = float(np.median(finite)) if finite.size else np.nan

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#3182bd" if np.isfinite(v) else "#bdbdbd" for v in r]
    ax.bar(labels, r, color=colors, edgecolor="black", lw=0.5)
    if finite.size:
        ax.axhline(median, color="#de2d26", lw=1.5, ls="--",
                   label=f"median r = {median:.3f}")
        iqr_lo, iqr_hi = np.percentile(finite, [25, 75])
        ax.axhspan(iqr_lo, iqr_hi, color="#de2d26", alpha=0.08, label="IQR")
    ax.set_xlabel("held-out donor")
    ax.set_ylabel("Spearman rho vs disease map")
    ax.set_title(title)
    ax.legend(frameon=False)
    out_path = _ensure_dirs(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    out_path: str | Path,
    title: str = "Disease x disease molecular similarity (Spearman rho)",
) -> Path:
    """Symmetric heatmap with hierarchical leaf ordering if already clustered."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover
        raise PlottingError(f"seaborn required for similarity heatmaps: {exc}") from exc

    fig, ax = plt.subplots(figsize=(1.1 * len(similarity_matrix) + 2,
                                    0.9 * len(similarity_matrix) + 2))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    out_path = _ensure_dirs(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_disease_region_heatmap(
    matrix: pd.DataFrame,
    out_path: str | Path,
    top_parcels: int = 30,
    title: str = "D x R molecular vulnerability matrix",
) -> Path:
    """Top-variable parcels x diseases heatmap of the molecular matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover
        raise PlottingError(f"seaborn required for D x R heatmaps: {exc}") from exc

    var = matrix.var(axis=1, skipna=True)
    parcels = var.nlargest(min(top_parcels, len(var))).index
    fig, ax = plt.subplots(figsize=(0.9 * len(matrix.columns) + 3,
                                    0.24 * len(parcels) + 2))
    sns.heatmap(matrix.loc[parcels], cmap="RdYlBu_r", center=0, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    ax.set_ylabel("")
    out_path = _ensure_dirs(Path(out_path).parent) / Path(out_path).name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
