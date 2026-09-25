#!/usr/bin/env python
"""External validation CLI (Part X) — OASIS-3 / ADNI.

Data access (registration-gated; nothing is downloaded automatically):

* **OASIS-3** — request access at https://www.oasis-brains.org/ (DUA).
  After download, stage the cohort with::

      python scripts/stage_oasis3.py --raw <download dir> \
          --out data/raw/oasis3

* **ADNI** — request access at https://adni.loni.usc.edu/ (DUA + approval).
  Stage ADNIMERGE + T1 NIfTI collections, then::

      python scripts/stage_adni.py --raw <download dir> \
          --out data/raw/adni

Both stagers produce ``imaging/`` session dirs + ``labels.csv``; the frozen
OASIS-1 model is then evaluated without any tuning::

    python scripts/validate_external.py --cohort oasis3 \
        --staged-root data/raw/oasis3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COHORTS = {
    "oasis3": dict(name="oasis3", label_column="label",
                   note="labels from OASIS-3 normalized DxD/CDR; "
                        "rule documented by scripts/stage_oasis3.py"),
    "adni": dict(name="adni", label_column="label",
                 note="labels from ADNI DXbl; rule documented by "
                      "scripts/stage_adni.py"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    ap.add_argument("--staged-root", required=True,
                    help="staging output of scripts/stage_<cohort>.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default="results/external_validation")
    args = ap.parse_args(argv)

    from brainvuln.mri.external import ExternalCohort, evaluate_external

    root = Path(args.staged_root)
    if not (root / "imaging").is_dir() or not (root / "labels.csv").exists():
        print(f"{root} is not a staged cohort (need imaging/ + labels.csv); "
              "see scripts/validate_external.py --help for staging commands",
              file=sys.stderr)
        return 2
    from brainvuln.config import checkpoint_identity, resolve_checkpoint
    ckpt = resolve_checkpoint(args.checkpoint)
    identity = checkpoint_identity(str(ckpt))
    cache = args.cache_dir or f"data/derived/{args.cohort}_128"
    cohort = ExternalCohort(name=args.cohort, staged_root=root,
                            label_column=COHORTS[args.cohort]["label_column"])
    metrics = evaluate_external(ckpt, cohort, cache, args.out_dir)
    print(f"{args.cohort} (frozen model {ckpt}):")
    print("  checkpoint sha256 " + identity["sha256"])
    print(f"  ROC-AUC {metrics['roc_auc']:.3f} | PR-AUC "
          f"{metrics['pr_auc']:.3f} | balanced acc "
          f"{metrics['balanced_accuracy']:.3f} | n = {metrics['n']}")
    print(f"  threshold: frozen OASIS-1 validation value "
          f"({metrics['threshold']:.2f}) — never re-tuned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
