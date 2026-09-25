#!/usr/bin/env python
"""Stage OASIS-3 for frozen external validation (Part X).

Access: request OASIS-3 at https://www.oasis-brains.org/ (data use
agreement). Nothing is downloaded here.

Staging contract — produce from the OASIS-3 download:

    <out>/imaging/<session_dir>/PROCESSED/...  T1 volumes (any layout the
                                               Part D finder can locate)
    <out>/labels.csv                           session_id, subject_id, label

Labeling rule (documented; OASIS-3 longitudinal "normalized" DxD/CDR):

* label 1: sessions from subjects whose normalized diagnosis is Alzheimer's
  disease (``DxDenc`` in {3, 4}: AD dementia / possible AD) OR baseline
  CDR >= 0.5.
* label 0: subjects with normalized diagnosis of cognitively normal
  (``DxDenc`` == 1) AND all observed CDR == 0.
* sessions of subjects with conflicting diagnoses are excluded (recorded).

The stager writes but NEVER tunes: the OASIS-1 model is loaded frozen by
scripts/validate_external.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

AD_DXD = {3, 4}          # dementia probs AD; poss. AD
NORMAL_DXD = 1


def stage(raw: Path, out: Path) -> int:
    subj_file = raw / "subj_tsvs"
    if not subj_file.is_dir():
        print(f"{raw} does not look like an extracted OASIS-3 download "
              "(expected subj_tsvs/); see --help", file=sys.stderr)
        return 2
    rows = []
    conflicts, included = [], 0
    for tsv in sorted(subj_file.glob("*.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        if "DxDenc" not in df.columns or "MR ID" not in df.columns:
            continue
        dx = df["DxDenc"].dropna()
        if dx.empty:
            continue
        subject = tsv.stem.replace("OAS3", "OAS3")
        dx_set = set(dx.astype(int))
        if dx_set <= {NORMAL_DXD}:
            label = 0
        elif dx_set & AD_DXD or (df.get("CDR", pd.Series(dtype=float)).dropna() >= 0.5).any():
            label = 1
        else:
            continue  # MCI/other: outside the AD-vs-CN binary task
        if {NORMAL_DXD} & dx_set and label == 1 and len(dx_set) > 1:
            conflicts.append(subject)
            continue
        included += 1
        for mr_id in df["MR ID"].dropna().astype(str):
            rows.append({"session_id": mr_id, "subject_id": subject,
                         "label": label})

    labels = pd.DataFrame(rows).drop_duplicates()
    (out / "labels.csv").parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / "labels.csv", index=False)
    print(f"staged labels for {labels['subject_id'].nunique()} subjects / "
          f"{len(labels)} sessions (excluded conflicting: {len(conflicts)})")
    print("NOW: link/copy the OASIS-3 T1 NIfTI sessions into "
          f"{out}/imaging/<MR ID>/ so scripts/validate_external.py can "
          "preprocess them; keep the frozen model frozen.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="extracted OASIS-3 download")
    ap.add_argument("--out", default="data/raw/oasis3")
    args = ap.parse_args(argv)
    return stage(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
