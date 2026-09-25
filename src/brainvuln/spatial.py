"""Spatially constrained null models for brain-map comparison (H3).

Parcellated brain maps are spatially autocorrelated: neighbouring parcels are
not independent, so naive Spearman p-values are anti-conservative. This module
wraps the two reference implementations of the parameterized data-null
families (Markello et al. 2021, "Comparing spatial null models for brain
maps"):

- ``moran``    : Moran spectral randomization (BrainSpace
                 ``MoranRandomization``, as used by neuromaps' Moran nulls) —
                 surrogates preserving the spatial autocorrelation structure
                 of the observed map via the eigendecomposition of an
                 inverse-distance spatial weights matrix;
- ``burt2020`` : variogram-matching surrogate maps (BrainSpace
                 ``SurrogateMaps``; Burt et al. 2020) — surrogates preserving
                 the observed spatial covariance function.

and a surface ``spin`` permutation wrapper (Alexander-Bloch et al. 2018) for
the cortical Desikan-Killiany representation. Empirical two-sided p-values use
the standard (1 + #|r_null| >= |r_obs|) / (1 + N) correction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


class SpatialNullError(RuntimeError):
    """Raised when a spatial null cannot be constructed."""


VOLUMETRIC_METHODS = ("moran", "burt2020")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def spatial_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with pairwise NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    r = stats.spearmanr(x[ok], y[ok]).statistic
    return float(r) if np.isfinite(r) else np.nan


def _prepare(
    x: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise SpatialNullError("coords must be (n_parcels, 3) MNI centroid coordinates")
    if len(x) != len(coords):
        raise SpatialNullError(
            f"map length {len(x)} != coords length {len(coords)}; parcel order must match"
        )
    ok = np.isfinite(x)
    if ok.sum() < 10:
        raise SpatialNullError(
            f"only {ok.sum()} valid parcels; spatial nulls need a usable map"
        )
    return x[ok], coords[ok], np.flatnonzero(ok)


def _row_standardized_weights(dist: np.ndarray) -> np.ndarray:
    """Row-standardized inverse-distance spatial weights (zero diagonal).

    Same distance decay as the BrainSpace Moran null (inverse distance),
    with self-weights removed so the statistic is a clean descriptive
    measure of spatial autocorrelation.
    """
    w = 1.0 / np.maximum(dist, 1e-9)
    np.fill_diagonal(w, 0.0)
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return w / row_sum


def morans_i(x: np.ndarray, coords: np.ndarray) -> float:
    """Observed Moran's I with row-standardized inverse-distance weights.

    With row-standardized W, S0 = n, so I = (z' W z) / (z' z).
    """
    xv, cv, _ = _prepare(x, coords)
    w = _row_standardized_weights(cdist(cv, cv))
    z = xv - xv.mean()
    denom = float(z @ z)
    if denom == 0:
        return np.nan
    return float((z @ w @ z) / denom)


def _moran_surrogates(xv: np.ndarray, coords_v: np.ndarray, n_perm: int,
                      seed: int) -> np.ndarray:
    """Moran spectral randomization surrogates (BrainSpace, per neuromaps)."""
    try:
        from brainspace.null_models.moran import MoranRandomization
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SpatialNullError(
            "brainspace is required for Moran surrogates "
            "(pip install brainvuln[atlas])"
        ) from exc
    dist = cdist(coords_v, coords_v).astype("float64")
    np.fill_diagonal(dist, 1)
    dist **= -1  # inverse-distance weights, as in neuromaps.nulls.moran
    mrs = MoranRandomization(procedure="singleton", joint=True, tol=1e-6,
                             n_rep=n_perm, random_state=seed)
    mrs.fit(dist)
    surrogates = np.asarray(mrs.randomize(xv), dtype=float)
    if surrogates.shape != (n_perm, len(xv)):  # pragma: no cover
        surrogates = surrogates.T
    return surrogates


def _burt2020_surrogates(xv: np.ndarray, coords_v: np.ndarray, n_perm: int,
                         seed: int) -> np.ndarray:
    """Variogram-matching surrogates (BrainSpace SurrogateMaps; Burt 2020)."""
    try:
        from brainspace.null_models.variogram import SurrogateMaps
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SpatialNullError(
            "brainspace is required for Burt-2020 surrogates "
            "(pip install brainvuln[atlas])"
        ) from exc
    dist = cdist(coords_v, coords_v)
    sm = SurrogateMaps(kernel="exp", pv=50, nh=25, random_state=seed)
    sm.fit(dist)
    surrogates = np.asarray(sm.randomize(xv, n_rep=n_perm), dtype=float)
    if surrogates.shape != (n_perm, len(xv)):  # pragma: no cover
        surrogates = surrogates.T
    return surrogates


def _rank_columns(m: np.ndarray) -> np.ndarray:
    """Rank each row (each surrogate map) across parcels."""
    return pd.DataFrame(m).rank(axis=1).to_numpy()


def _null_correlations(y: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    """Spearman correlation of the observed map against every null row.

    Vectorized: Pearson on ranks equals Spearman. ``nulls`` is
    (n_perm, n_parcels); returns (n_perm,).
    """
    y_ok = np.isfinite(y)
    x_ranks = stats.rankdata(y[y_ok])
    n_ranks = _rank_columns(nulls)
    xc = x_ranks - x_ranks.mean()
    nc = n_ranks - n_ranks.mean(axis=1, keepdims=True)
    num = nc @ xc  # (n_perm,)
    den = np.sqrt((xc**2).sum() * (nc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.nan_to_num(r, nan=0.0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def compare_maps(
    score_map: pd.Series,
    disease_map: pd.Series,
    method: str = "spearman",
) -> float:
    """Observed spatial correlation between two parcellated maps."""
    del method  # Spearman is the primary statistic throughout
    return spatial_corr(score_map.to_numpy(), disease_map.to_numpy())


def run_spatial_null(
    score_map: pd.Series,
    disease_map: pd.Series,
    coords: pd.DataFrame,
    method: str = "moran",
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """Test the score/disease-map correlation against one spatial null family.

    Surrogates are generated from the *disease* map (the independent
    comparison target); by symmetry this yields the same two-sided p as
    surrogating the score map, and it leaves the genetic side untouched.
    """
    if method not in VOLUMETRIC_METHODS:
        raise SpatialNullError(
            f"Unknown volumetric method {method!r}; expected one of {VOLUMETRIC_METHODS}"
        )
    y = disease_map.to_numpy(dtype=float)
    x = score_map.to_numpy(dtype=float)
    coords_arr = coords.to_numpy(dtype=float)
    xv, cv, valid_idx = _prepare(x, coords_arr)
    yv = y[valid_idx]

    if method == "moran":
        nulls = _moran_surrogates(yv, cv, n_perm, seed)
    else:
        nulls = _burt2020_surrogates(yv, cv, n_perm, seed)

    r_obs = spatial_corr(xv, yv)
    null_r = _null_correlations(yv, nulls)
    p = spatial_null_pvalue(r_obs, null_r)
    return {
        "method": method,
        "r_obs": r_obs,
        "p_spatial": p,
        "n_perm": n_perm,
        "n_valid_parcels": int(valid_idx.size),
        "null_r": null_r.tolist(),
        "null_r_mean": float(np.mean(null_r)),
        "null_r_p2.5": float(np.percentile(null_r, 2.5)),
        "null_r_p97.5": float(np.percentile(null_r, 97.5)),
        "moran_i_observed": morans_i(yv, cv),
    }


def spatial_null_pvalue(r_obs: float, null_r: np.ndarray, two_sided: bool = True) -> float:
    """Empirical p = (1 + #{|r_null| >= |r_obs|}) / (1 + N) (two-sided by default)."""
    null_r = np.asarray(null_r, dtype=float)
    if not np.isfinite(r_obs):
        return float("nan")
    if two_sided:
        exceed = np.abs(null_r) >= abs(r_obs)
    else:
        exceed = null_r >= r_obs
    return float((1 + exceed.sum()) / (1 + len(null_r)))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Vectorized BH-FDR adjusted p-values (stable ordering, monotone)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def concordance(results: dict[str, dict], required: int = 2) -> dict:
    """Do multiple spatial-null families agree on the conclusion?

    A family 'supports' the signal when its spatial p is below 0.05. The
    overall robust flag requires at least ``required`` agreeing families.
    """
    per_method = {
        m: {"p_spatial": r.get("p_spatial"), "r_obs": r.get("r_obs")}
        for m, r in results.items()
    }
    supporting = [m for m, r in per_method.items()
                  if r["p_spatial"] is not None and np.isfinite(r["p_spatial"])
                  and r["p_spatial"] < 0.05]
    return {
        "per_method": per_method,
        "methods_supporting": supporting,
        "n_methods_supporting": len(supporting),
        "robust": len(supporting) >= required,
        "concordance_required": required,
    }


def surface_spin_pvalues(
    score_map: pd.Series,
    disease_map: pd.Series,
    vertices_lh: np.ndarray,
    vertices_rh: np.ndarray,
    labels_lh: np.ndarray,
    labels_rh: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """Cortical spin-permutation test (Alexander-Bloch 2018) via BrainSpace.

    ``vertices_lh/rh`` are (m, 3) surface vertex coordinates (e.g. fsaverage5
    middle-cranial surfaces) and ``labels_lh/rh`` the per-vertex parcel labels,
    aligned with the parcel index of the maps for cortical parcels. Subcortical
    parcels (absent from the surface) are excluded from this sensitivity test.
    """
    try:
        from brainspace.null_models.spin import SpinPermutations
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SpatialNullError(
            "brainspace is required for spin permutations "
            "(pip install brainvuln[atlas])"
        ) from exc
    if vertices_lh is None or vertices_rh is None or labels_lh is None or labels_rh is None:
        raise SpatialNullError(
            "surface spin test requires vertex coordinates and per-vertex parcel labels"
        )

    x = score_map.to_numpy(dtype=float).ravel()
    y = disease_map.to_numpy(dtype=float).ravel()
    parcel_index = {str(lbl): i for i, lbl in enumerate(score_map.index)}
    labels_lh = np.asarray(labels_lh).astype(str)
    labels_rh = np.asarray(labels_rh).astype(str)
    n_lh = labels_lh.size
    vlabels = np.concatenate([labels_lh, labels_rh])
    in_map = np.array([lbl in parcel_index for lbl in vlabels])
    if in_map.sum() < 10:
        raise SpatialNullError("fewer than 10 surface vertices map to score-map parcels")
    # filter vertices/labels to mapped parcels BEFORE fitting, so spin indices
    # index into the same filtered vertex space as ``x_vert``
    in_lh, in_rh = in_map[:n_lh], in_map[n_lh:]
    labels_lh, labels_rh = labels_lh[in_lh], labels_rh[in_rh]
    verts_lh = np.asarray(vertices_lh, dtype=float)[in_lh][:, :3]
    verts_rh = np.asarray(vertices_rh, dtype=float)[in_rh][:, :3]
    vlabels = np.concatenate([labels_lh, labels_rh])
    parcel_names = sorted(set(vlabels))
    x_vert = np.array([x[parcel_index[lbl]] for lbl in vlabels])
    y_parc = np.array([y[parcel_index[lbl]] for lbl in parcel_names])
    ok = np.isfinite(y_parc)
    if ok.sum() < 10:
        raise SpatialNullError("fewer than 10 cortical parcels with finite values")

    sp = SpinPermutations(n_rep=n_perm, random_state=seed)
    if verts_rh.shape[0] >= 10 and verts_lh.shape[0] >= 10:
        # spin each hemisphere independently (Alexander-Bloch 2018 protocol)
        sp.fit(verts_lh, points_rh=verts_rh)
        spin_samples = np.hstack([np.asarray(sp.spin_lh_),
                                  np.asarray(sp.spin_rh_) + verts_lh.shape[0]])
    elif verts_lh.shape[0] >= 10:
        sp.fit(verts_lh)
        spin_samples = np.asarray(sp.spin_lh_)
    else:
        raise SpatialNullError("too few mapped vertices to fit surface spins")
    if spin_samples.ndim == 1:  # pragma: no cover
        spin_samples = spin_samples[None, :]

    # parcellate each spun vertex labeling: mean of source values per target parcel
    name_to_col = {lbl: k for k, lbl in enumerate(parcel_names)}
    vcols = np.array([name_to_col[lbl] for lbl in vlabels])
    counts = np.bincount(vcols, minlength=len(parcel_names)).astype(float)
    counts[counts == 0] = 1.0
    null_r = np.empty(spin_samples.shape[0])
    for p in range(spin_samples.shape[0]):
        vals = x_vert[spin_samples[p]]
        sums = np.bincount(vcols, weights=vals, minlength=len(parcel_names))
        surr = sums / counts
        null_r[p] = spatial_corr(surr[ok], y_parc[ok])

    r_obs = spatial_corr(
        np.array([x[parcel_index[lbl]] for lbl in parcel_names])[ok], y_parc[ok]
    )
    return {
        "method": "spin",
        "r_obs": r_obs,
        "p_spatial": spatial_null_pvalue(r_obs, null_r),
        "n_perm": n_perm,
        "n_valid_parcels": int(ok.sum()),
        "null_r": null_r.tolist(),
        "null_r_mean": float(np.mean(null_r)),
        "note": "cortical parcels only; subcortical values excluded",
    }
