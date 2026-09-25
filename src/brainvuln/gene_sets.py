"""Disease gene-set assembly and covariate-matched random gene nulls.

Null #1 (the *genetic* null): instead of comparing a disease gene set to
completely random genes, random sets are matched 1:1, without replacement,
for brain-expression covariates (mean expression, expression variance; the
design is extensible to gene length / GC content). Balance diagnostics are
computed for every draw so the null is auditable, never assumed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

BING_SCALE_URL = "https://zip.humanbrainproject.org/bing_scale.tsv.zip"
KNOWN_COVARIATES = ("mean_expression", "expression_variance")


class GeneSetError(RuntimeError):
    """Raised when gene sets or matched nulls cannot be constructed."""


# ---------------------------------------------------------------------------
# gene-set assembly
# ---------------------------------------------------------------------------


def load_gene_set_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise GeneSetError(
            f"Gene-set file not found: {path}. Run scripts/fetch_gene_sets.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_gene_set(
    disease_key: str,
    gene_sets_dir: str | Path,
    gene_definition: str = "gwascat",
) -> dict:
    """Load the gene list for one disease under the configured definition.

    gene_definition: ``gwascat`` (Tier A), ``opentargets`` (Tier B, primary
    threshold), or ``union`` (union of Tier A and Tier B).
    """
    gsets_dir = Path(gene_sets_dir)
    if gene_definition == "gwascat":
        result = load_gene_set_json(gsets_dir / f"{disease_key}_gwascat.json")
        genes = list(result["genes"])
        provenance = {"tier": "gwascat", "files": [str(result.get("metadata", {}).get("efo_id"))]}
    elif gene_definition == "opentargets":
        result = load_gene_set_json(gsets_dir / f"{disease_key}_opentargets.json")
        genes = list(result["genes"])
        provenance = {"tier": "opentargets"}
    elif gene_definition == "union":
        a = load_gene_set_json(gsets_dir / f"{disease_key}_gwascat.json")
        b = load_gene_set_json(gsets_dir / f"{disease_key}_opentargets.json")
        genes = sorted(set(a["genes"]) | set(b["genes"]))
        provenance = {"tier": "union"}
    else:
        raise GeneSetError(
            f"Unknown gene_definition {gene_definition!r}; "
            "expected 'gwascat', 'opentargets', or 'union'"
        )
    if not genes:
        raise GeneSetError(
            f"Gene set for '{disease_key}' ({gene_definition}) is empty; check the "
            "retrieval metadata in results/gene_sets/"
        )
    return {"genes": genes, "definition": gene_definition, "provenance": provenance}


# ---------------------------------------------------------------------------
# background gene universe
# ---------------------------------------------------------------------------


def background_from_expression(expr: pd.DataFrame) -> list[str]:
    """Background universe fallback: all genes measured in the expression matrix."""
    return [str(g) for g in expr.columns]


def download_bing_scale(data_dir: str | Path, timeout: int = 60) -> pd.DataFrame:
    """Download the Bing-scale whole-microarray gene list (with fallbacks).

    The primary source is the Human Brain Project ziplink of the Bing scale
    TSV (all probes/genes on the AHBA microarray platform, not only those in
    donor-consensus parcels). If the download fails, callers should fall back
    to the expressed-gene universe and record that in the manifest.
    """
    import requests

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "bing_scale.tsv"
    if cache.exists() and cache.stat().st_size > 1_000_000:
        return pd.read_csv(cache, sep="\t")
    resp = requests.get(BING_SCALE_URL, timeout=timeout)
    resp.raise_for_status()
    tmp = data_dir / "bing_scale.tsv.zip"
    tmp.write_bytes(resp.content)
    with zipfile.ZipFile(tmp) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith((".tsv", ".csv", ".txt")))
        with zf.open(member) as fh:
            df = pd.read_csv(fh, sep=None, engine="python")
    df.to_csv(cache, sep="\t", index=False)
    tmp.unlink(missing_ok=True)
    return df


def get_background_genes(
    expr: pd.DataFrame,
    background: str = "bing_scale",
    data_dir: str | Path = "data/raw",
) -> tuple[list[str], dict]:
    """Return (background gene list, provenance note). Falls back gracefully."""
    if background == "bing_scale":
        try:
            df = download_bing_scale(data_dir)
            gene_col = next(
                (c for c in df.columns if c.lower() in ("gene", "gene_symbol", "symbol", "genes")),
                df.columns[0],
            )
            genes = sorted(set(str(g) for g in df[gene_col].dropna()))
            if len(genes) < 10000:
                raise GeneSetError(f"Bing-scale download suspiciously small ({len(genes)} genes)")
            return genes, {"background": "bing_scale_download", "n": len(genes)}
        except Exception as exc:  # noqa: BLE001 - fallback must be recorded, not silent
            genes = background_from_expression(expr)
            return genes, {
                "background": "ahba_expressed_fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "n": len(genes),
            }
    genes = background_from_expression(expr)
    return genes, {"background": background, "n": len(genes)}


# ---------------------------------------------------------------------------
# covariates and matching
# ---------------------------------------------------------------------------


def compute_gene_covariates(expr: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Per-gene covariate table from a parcels x genes expression matrix."""
    unknown = [c for c in covariates if c not in KNOWN_COVARIATES]
    if unknown:
        raise GeneSetError(
            f"Unknown match covariates {unknown}; implemented: {list(KNOWN_COVARIATES)}. "
            "Gene length / SNP-density matching requires extra annotation data and is a "
            "documented extension, not a silent no-op."
        )
    cov = pd.DataFrame(index=[str(c) for c in expr.columns])
    if "mean_expression" in covariates:
        cov["mean_expression"] = expr.mean(axis=0).to_numpy()
    if "expression_variance" in covariates:
        cov["expression_variance"] = expr.var(axis=0, ddof=1).to_numpy()
    return cov


def match_genes_1to1(
    gene_set: list[str],
    background_genes: list[str],
    covariates: pd.DataFrame,
    n_nulls: int,
    rng: np.random.Generator,
    show_progress: bool = False,
    progress_every: int = 250,
) -> dict:
    """Greedy nearest-neighbour 1:1 matching without replacement.

    For each null draw, background genes (excluding the true set) are matched
    to disease genes in random order by standardized covariate distance.
    Distances come from a pairwise matrix computed once, so each greedy step
    is an O(pool) masked argmin instead of a full recomputation — identical
    results to the naive loop, ~50x faster for large background pools.
    Returns the matched sets plus balance diagnostics.
    """
    missing = [g for g in gene_set if g not in covariates.index]
    if missing:
        raise GeneSetError(
            f"{len(missing)} disease genes lack covariate values "
            f"(e.g. {missing[:5]}); they are absent from the expression matrix."
        )
    pool = [g for g in background_genes if g not in set(gene_set)]
    pool = [g for g in pool if g in covariates.index]
    if len(pool) < len(gene_set):
        raise GeneSetError(
            f"Background pool ({len(pool)}) smaller than gene set ({len(gene_set)})"
        )

    cov = covariates.loc[sorted(set(gene_set) | set(pool))]
    arr = cov.to_numpy(dtype=float)
    mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    z = (arr - mu) / sd
    idx = {g: i for i, g in enumerate(cov.index)}
    set_idx = np.array([idx[g] for g in gene_set])
    pool_idx = np.array([idx[g] for g in pool])
    z_set, z_pool = z[set_idx], z[pool_idx]

    # Pairwise covariate distances, computed once (n_set x n_pool).
    D = np.sqrt(((z_set[:, None, :] - z_pool[None, :, :]) ** 2).sum(axis=2))

    matched_sets: list[list[str]] = []
    all_dists: list[float] = []
    pool_labels = np.array(pool)
    for i_null in range(n_nulls):
        order = rng.permutation(len(gene_set))
        remaining = np.ones(len(pool), dtype=bool)
        matched: list[str] = [""] * len(gene_set)
        dists: list[float] = []
        for pos in order:
            masked = np.where(remaining, D[pos], np.inf)
            j = int(np.argmin(masked))          # full-pool index, tie -> lowest
            dists.append(float(D[pos, j]))
            matched[pos] = pool_labels[j]
            remaining[j] = False
        matched_sets.append(matched)
        all_dists.extend(dists)
        if show_progress and (i_null + 1) % progress_every == 0:
            print(f"  [matched-null] {i_null + 1}/{n_nulls} gene sets drawn",
                  flush=True)

    # Balance diagnostics: two-sample KS between matched-null and true covariates.
    diagnostics: dict[str, float] = {}
    for i, cov_name in enumerate(cov.columns):
        true_vals = z_set[:, i]
        # pooled matched values across draws (z space)
        matched_vals = np.concatenate([
            z[np.array([idx[g] for g in ms]), i] for ms in matched_sets
        ])
        ks = stats.ks_2samp(true_vals, matched_vals)
        diagnostics[f"ks_{cov_name}"] = float(ks.statistic)
        diagnostics[f"ks_p_{cov_name}"] = float(ks.pvalue)
    diagnostics["match_distance_mean"] = float(np.mean(all_dists))
    diagnostics["match_distance_p95"] = float(np.percentile(all_dists, 95))

    return {
        "matched_sets": matched_sets,
        "n_nulls": n_nulls,
        "covariates": list(cov.columns),
        "diagnostics": diagnostics,
    }


def null_score_distribution(
    expr: pd.DataFrame,
    matched_sets: list[list[str]],
    score_fn: Callable[[pd.DataFrame], pd.Series],
    show_progress: bool = False,
) -> np.ndarray:
    """Score every matched null gene set; returns (n_nulls x n_parcels) matrix."""
    rows = []
    for i, genes in enumerate(matched_sets):
        rows.append(score_fn(expr.loc[:, genes]).to_numpy())
        if show_progress and (i + 1) % 100 == 0:
            print(f"  [nulls] scored {i + 1}/{len(matched_sets)} matched sets")
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# empirical p-values
# ---------------------------------------------------------------------------


def empirical_p_greater(observed: float | np.ndarray, null_values: np.ndarray) -> float | np.ndarray:
    """One-sided empirical p: P(null >= observed), with +1 correction."""
    null_values = np.asarray(null_values, dtype=float)
    obs = np.asarray(observed, dtype=float)
    p = (1.0 + (null_values >= obs).sum(axis=0)) / (1.0 + null_values.shape[0])
    return p if np.ndim(observed) else float(p)
