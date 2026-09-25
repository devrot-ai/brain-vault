"""Shared fixtures: a small synthetic but spatially structured brain.

The fixture atlas is a 5x4x3 parcel grid in MNI-like space; expression is a
mixture of Gaussian spatial bumps per gene so that covariates (mean/variance)
vary meaningfully and spatial nulls have real autocorrelation to preserve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def build_parcel_table() -> pd.DataFrame:
    xs = np.arange(-40, 41, 20.0)
    ys = np.arange(-30, 31, 20.0)
    zs = np.arange(-20, 21, 20.0)
    rows = []
    pid = 1
    for z in zs:
        for y in ys:
            for x in xs:
                rows.append({
                    "id": pid,
                    "label": f"TestParcel_{pid:03d}",
                    "hemisphere": "L" if x < 0 else ("R" if x > 0 else "B"),
                    "structure": "cortex" if z >= 0 else "subcortex/brainstem",
                    "source": "test",
                    "source_id": pid,
                    "surface_dk_id": -1,
                    "n_voxels": 100,
                    "centroid_x": float(x), "centroid_y": float(y),
                    "centroid_z": float(z),
                })
                pid += 1
    return pd.DataFrame(rows)


def build_expression(parcel_table: pd.DataFrame, n_genes: int = 60,
                     seed: int = 7) -> pd.DataFrame:
    """Spatially autocorrelated synthetic expression: parcels x genes."""
    rng = np.random.default_rng(seed)
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    n_parcels = len(parcel_table)
    data = np.zeros((n_parcels, n_genes))
    for g in range(n_genes):
        center = rng.uniform(-40, 40, size=3)
        sigma2 = 2.0 * rng.uniform(15, 40) ** 2
        amp = rng.uniform(0.5, 2.0)
        bump = amp * np.exp(-((coords - center) ** 2).sum(axis=1) / sigma2)
        data[:, g] = bump + rng.normal(0, 0.25, size=n_parcels)
    return pd.DataFrame(data, index=parcel_table["label"].astype(str).to_list(),
                        columns=[f"GENE{g:03d}" for g in range(n_genes)])


@pytest.fixture
def parcel_table() -> pd.DataFrame:
    return build_parcel_table()


@pytest.fixture
def expr(parcel_table) -> pd.DataFrame:
    return build_expression(parcel_table)


@pytest.fixture
def coords(parcel_table) -> pd.DataFrame:
    return parcel_table.set_index("label").loc[
        build_parcel_table()["label"].tolist(),
        ["centroid_x", "centroid_y", "centroid_z"],
    ]


@pytest.fixture
def synthetic_disease_map(parcel_table) -> pd.Series:
    """A smooth gradient map along +y with a subcortical boost."""
    coords = parcel_table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    w = (coords[:, 1] + 30.0) / 60.0
    w[coords[:, 2] < 0] += 0.3
    return pd.Series(w, index=parcel_table["label"].astype(str).to_list(),
                     name="synthetic_gradient")


@pytest.fixture
def gene_set_json(tmp_path, expr) -> "dict":
    """A fake fetched gene set for the 'alzheimer' disease key."""
    import json

    result = {
        "tier": "gwascat",
        "genes": sorted(expr.columns[:20]),
        "associations": [
            {"association_id": f"abc{i}", "pvalue": 1e-8 + i * 1e-9,
             "genes": [f"GENE{i:03d}"]}
            for i in range(20)
        ],
        "metadata": {
            "disease": "Alzheimer's disease",
            "disease_key": "alzheimer",
            "efo_id": "EFO_0000249",
            "show_child_traits": False,
            "extended_geneset": False,
            "retrieval_date": "2026-09-19T00:00:00+00:00",
            "number_associations": 20,
            "number_unique_genes": 20,
        },
    }
    out_dir = tmp_path / "gene_sets"
    out_dir.mkdir()
    (out_dir / "alzheimer_gwascat.json").write_text(json.dumps(result))
    return {"dir": out_dir, "result": result}
