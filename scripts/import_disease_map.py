#!/usr/bin/env python
"""Import an external disease-vulnerability map into BrainVuln format.

Converts a public MNI-space NIfTI group map (PET, atrophy, ...) or an
already-parcellated CSV into:

    <out-dir>/<name>.csv                  parcellated values (label,value)
    <out-dir>/<name>.provenance.json      full provenance + coverage report

The parcellated CSV is accepted directly by
``python scripts/run_pipeline.py --disease-map <csv>``.

Examples
--------
Import the verified La Joie 2020 Alzheimer atrophy map::

    python scripts/import_disease_map.py \
        --map data/external/disease_maps/raw/lajoie2020_ad_atrophy_mni152.nii.gz \
        --disease alzheimer --name lajoie2020_ad_atrophy \
        --modality "Structural MRI (longitudinal jacobian)" \
        --sign-convention "higher = faster gray-matter atrophy" \
        --citation "La Joie et al. 2020, Sci Transl Med 12:eaba5732" \
        --source-url "https://neurovault.org/images/64175/" \
        --persistent-id "https://identifiers.org/neurovault.image:64175" \
        --n-subjects 32

Import a literature-derived region table (CSV)::

    python scripts/import_disease_map.py \
        --map my_hd_regions.csv --format csv --disease huntington \
        --name hd_literature_regions --value-col "percent_volume_loss" \
        --modality "curated region table" \
        --sign-convention "higher = more volume loss" \
        --citation "Tabrizi et al. 2012, Lancet Neurol 11:42-53" \
        --source-url "<where the table was transcribed from>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brainvuln.disease_maps import (
    DiseaseMapImportError,
    import_disease_map,
    make_provenance,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--map", required=True, help="NIfTI (.nii[.gz]) or CSV map")
    p.add_argument("--format", default=None, choices=["nifti", "csv"],
                   help="default: inferred from extension")
    p.add_argument("--disease", required=True,
                   help="disease key, e.g. alzheimer / parkinson / huntington")
    p.add_argument("--name", required=True,
                   help="short unique name for the imported map")
    p.add_argument("--modality", required=True,
                   help="e.g. 'tau PET (flortaucipir SUVR)', 'Structural MRI'")
    p.add_argument("--sign-convention", required=True,
                   help="e.g. 'higher = more atrophy' — record, never assume")
    p.add_argument("--citation", required=True,
                   help="full citation of the paper the map comes from")
    p.add_argument("--source-url", default="",
                   help="URL the file was downloaded from")
    p.add_argument("--persistent-id", default="",
                   help="e.g. https://identifiers.org/neurovault.image:64175")
    p.add_argument("--license-note", default=None,
                   help="override the default no-explicit-license note")
    p.add_argument("--n-subjects", type=int, default=None)
    p.add_argument("--description", default="",
                   help="free-text description of the map")
    p.add_argument("--space", default="MNI152")
    p.add_argument("--atlas", default=None,
                   help="parcellation NIfTI (default: data/derived "
                        "dk68_tianS1 atlas built by build_expression.py)")
    p.add_argument("--atlas-info", default=None,
                   help="parcellation info CSV (default: matching _info.csv)")
    p.add_argument("--out-dir", default="data/external/disease_maps")
    p.add_argument("--no-mask-zero", action="store_true",
                   help="treat exact-0 voxels as data (default: mask them)")
    p.add_argument("--value-col", default=None,
                   help="CSV only: which column holds the values")
    p.add_argument("--label-col", default=None, help="CSV only: label column")
    args = p.parse_args(argv)

    from brainvuln.config import ensure_output_tree

    dirs = ensure_output_tree()
    atlas_name = "dk68_tianS1"
    atlas_img = Path(args.atlas) if args.atlas else (
        dirs["data_derived"] / f"atlas_{atlas_name}.nii.gz"
    )
    atlas_info = (
        Path(args.atlas_info) if args.atlas_info
        else dirs["data_derived"] / f"atlas_{atlas_name}_info.csv"
    )
    if not atlas_img.exists() or not atlas_info.exists():
        print("Atlas not found; run scripts/build_expression.py first.",
             file=sys.stderr)
        return 2
    fmt = args.format or (
        "csv" if Path(args.map).suffix.lower() == ".csv" else "nifti"
    )

    provenance = make_provenance(
        name=args.name,
        disease=args.disease,
        modality=args.modality,
        space=args.space,
        sign_convention=args.sign_convention,
        citation=args.citation,
        source_url=args.source_url,
        license_note=args.license_note or (
            "No explicit license set by the source repository; use for "
            "research with full citation and contact the authors before "
            "redistribution."
        ),
        raw_file=str(Path(args.map).resolve()),
        n_subjects=args.n_subjects,
        description=args.description,
        persistent_id=args.persistent_id,
    )

    try:
        series, prov, coverage = import_disease_map(
            args.map,
            provenance=provenance,
            atlas_img_path=atlas_img,
            atlas_info_path=atlas_info,
            out_dir=args.out_dir,
            source_format=fmt,
            mask_zero=not args.no_mask_zero,
            value_col=args.value_col,
            label_col=args.label_col,
        )
    except DiseaseMapImportError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1

    print(f"imported: {args.name} ({series.notna().sum()}/"
          f"{len(series)} parcels with values)")
    print(f"  coverage: {coverage['coverage_fraction']:.2f} "
          f"(missing: {coverage['missing_by_structure'] or 'none'})")
    print(f"  subcortical parcels with values: "
          f"{coverage['n_subcortical_with_value']}")
    if fmt == "nifti":
        notes = prov["import"]
        print(f"  resampling: {notes['resampling']}; "
              f"mask_zero={notes['mask_zero']} "
              f"({notes['n_voxels_masked_zero']} voxels masked)")
    out_csv = Path(args.out_dir) / f"{args.name}.csv"
    print(f"\nwrote {out_csv} and matching .provenance.json")
    print("run the pipeline on it with:")
    print(f"  python scripts/run_pipeline.py --disease {args.disease} "
          f"--disease-map {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
