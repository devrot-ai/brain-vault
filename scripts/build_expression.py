#!/usr/bin/env python
"""Build the union atlas and run abagen to produce regional expression matrices.

Steps:
1. Fetch the abagen-shipped Desikan-Killiany atlas (volume + surface) and the
   Tian 2020 Subcortex S1 atlas; construct the non-overlapping union.
2. Run abagen.get_expression_data with the configured primary settings
   (diff_stability probes, corrected MNI, reannotation, missing=None).
3. Save the group expression matrix, donor-level expression, sample counts,
   the abagen report, and the parcel coverage table.

The first run downloads ~2-4 GB of AHBA microarray data into
data/raw/ahba and is fully cached afterwards.

Usage:
    python scripts/build_expression.py [--atlas dk68_tianS1] [--verbose 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln import ahba  # noqa: E402
from brainvuln.config import (  # noqa: E402
    PROJECT_ROOT,
    ensure_output_tree,
    load_analysis_config,
    stamp_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", default=None,
                        help="atlas key from config/analysis.yaml (default: atlases.primary)")
    parser.add_argument("--analysis-config", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    analysis = load_analysis_config(args.analysis_config or
                                    PROJECT_ROOT / "config" / "analysis.yaml")
    dirs = ensure_output_tree()
    atlas_name = args.atlas or analysis.primary_atlas
    atlas_cfg = analysis.atlases.get(atlas_name)
    if atlas_cfg is None:
        print(f"Unknown atlas {atlas_name!r}; defined: {list(analysis.atlases)}",
              file=sys.stderr)
        return 2

    derived = dirs["data_derived"]
    union_path = derived / f"atlas_{atlas_name}.nii.gz"
    info_path = derived / f"atlas_{atlas_name}_info.csv"

    if atlas_cfg.kind == "union" and not (union_path.exists() and info_path.exists()):
        print("Fetching source atlases (cached)...")
        dk = ahba.fetch_dk68(dirs["data_raw"])
        tian_path = ahba.fetch_tian_s1(dirs["data_raw"])
        print("Building union atlas (DK68 + Tian S1, non-overlapping)...")
        union_path, info_path, _ = ahba.build_union_atlas(dk, tian_path, derived)
        # persist under the atlas-specific name as well
        if union_path.name != f"atlas_{atlas_name}.nii.gz":
            new_union = derived / f"atlas_{atlas_name}.nii.gz"
            new_info = derived / f"atlas_{atlas_name}_info.csv"
            new_union.write_bytes(union_path.read_bytes())
            new_info.write_text(info_path.read_text(encoding="utf-8"), encoding="utf-8")
            union_path, info_path = new_union, new_info
        prov = json.loads((derived / "atlas_union_provenance.json").read_text(encoding="utf-8"))
        print(f"  union parcels: {prov['n_union_parcels']} "
              f"(tian {prov['n_tian_parcels']}, dk kept {prov['n_dk_parcels_kept']}, "
              f"dk dropped {prov['n_dk_parcels_dropped']})")
    elif atlas_cfg.kind != "union":
        print(f"Atlas kind {atlas_cfg.kind!r} (cortical_only) requires the abagen DK atlas "
              "directly; union build skipped.")
        dk = ahba.fetch_dk68(dirs["data_raw"])
        union_path, info_path = Path(dk["image"]), Path(dk["info"])
        union_path = Path(union_path)
        info_path = Path(info_path)

    print(f"Running abagen on {union_path.name} "
          f"(probe_selection={analysis.ahba.probe_selection}, missing={analysis.ahba.missing})...")
    result = ahba.get_regional_expression(
        union_path, info_path, analysis.ahba,
        data_dir=dirs["root"] / analysis.ahba.data_dir,
        verbose=args.verbose,
    )
    paths = ahba.save_expression_outputs(result, derived, atlas_name)

    expr = result["expr"]
    print(f"  expression matrix: {expr.shape[0]} parcels x {expr.shape[1]} genes")
    print(f"  parcels with NO sampling (kept missing): "
          f"{len(result['all_missing_parcels'])}")
    if result["all_missing_parcels"]:
        print(f"    -> {result['all_missing_parcels'][:8]}"
              f"{' ...' if len(result['all_missing_parcels']) > 8 else ''}")

    manifest = stamp_manifest(analysis, extra={
        "step": "build_expression",
        "atlas": atlas_name,
        "outputs": {k: str(v) for k, v in paths.items()},
        "n_parcels": int(expr.shape[0]),
        "n_genes": int(expr.shape[1]),
        "all_missing_parcels": result["all_missing_parcels"],
    })
    manifest_path = dirs["results_analysis"] / f"manifest_build_expression_{atlas_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Saved outputs; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
