#!/usr/bin/env python
"""Stage ADNI for frozen external validation (Part X).

Access: request ADNI at https://adni.loni.usc.edu/ (DUA + approval).
Nothing is downloaded here.

Staging contract (from ADNIMERGE.csv + downloaded T1 NIfTI collection):

    <out>/imaging/<PTID>_<VISCODE>/...   T1 NIfTI volumes
    <out>/labels.csv                     session_id, subject_id, label

Labeling rule (documented; ADNI DXbl at the session's visit):

* label 1: ``DXbl`` in {AD (dementia)} — probable AD dementia
* label 0: ``DXbl`` == CN — cognitively normal
* MCI / SMC / EMCI / LMCI are excluded (outside the AD-vs-CN binary task;
  extending to 3-way classification is a documented future direction).

Optional covariates (Part Y): AGE, PTGENDER, ICV from ADNIMERGE are carried
into labels.csv when present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def stage(raw: Path, out: Path) -> int:
    merge = raw / "ADNIMERGE.csv"
    if not merge.exists():
        for cand in raw.rglob("ADNIMERGE.csv"):
            merge = cand
            break
    if not merge.exists():
        print(f"no ADNIMERGE.csv under {raw}", file=sys.stderr)
        return 2
    df = pd.read_csv(merge, low_memory=False)
    keep = df[df["DXbl"].isin(["AD", "CN"])].copy()
    rows = []
    for r in keep.itertuples():
        if pd.isna(getattr(r, "ORIGPROT", "")) or getattr(r, "ORIGPROT", "") != "ADNI1":
            # restrict to ADNI1 1.5T T1 protocol for preprocessing comparability
            continue
        rows.append({
            "session_id": f"{r.PTID}_{r.VISCODE}".replace("/", "_").replace(" ", ""),
            "subject_id": str(r.PTID),
            "label": 1 if r.DXbl == "AD" else 0,
            "age": getattr(r, "AGE", None),
            "sex": getattr(r, "PTGENDER", None),
            "icv": getattr(r, "ICV", None),
        })
    labels = pd.DataFrame(rows)
    (out / "labels.csv").parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / "labels.csv", index=False)
    print(f"staged {len(labels)} ADNI1 sessions "
          f"({labels['subject_id'].nunique()} subjects; "
          f"AD {int((labels.label == 1).sum())} / CN {int((labels.label == 0).sum())})")
    print("NOW: link the downloaded T1 NIfTI files into "
          f"{out}/imaging/<session_id>/ (name session dirs exactly as "
          "labels.csv session_id); then run scripts/validate_external.py")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="directory with ADNIMERGE.csv")
    ap.add_argument("--out", default="data/raw/adni")
    args = ap.parse_args(argv)
    return stage(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
