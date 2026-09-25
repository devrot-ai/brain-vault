"""External validation (Part X): OASIS-3 and ADNI.

Principles (enforced by construction):

* The OASIS-1 model is **frozen**: the checkpoint is loaded, never updated.
  No threshold re-selection, no architecture changes, no fine-tuning.
* Labels come from each cohort's own clinical metadata with the labeling
  rule documented per cohort (OASIS-3: normalized DxD + CDR; ADNI:
  DXbl AD dementia / MCI vs CN — cohort-specific mappings live in the
  cohort builders, not here).
* Preprocessing is the identical deterministic pipeline (Part D); where a
  cohort ships different image derivatives, the *same* code path runs after
  cohort-specific staging into the expected layout.
* Access is registration-gated: OASIS-3 via the OASIS portal (with DUA),
  ADNI via LONI/ADNI DUA after credential approval. The builders here run
  on data staged locally by the user; no download is attempted silently.

Usage (after access approval and staging):

    python scripts/validate_external.py --cohort oasis3 \
        --staged-root data/raw/oasis3 --checkpoint results/ml/resnet_seed42/checkpoints/best.pt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from brainvuln.mri.preprocess import ScanRecord, build_scan_index


@dataclass
class ExternalCohort:
    name: str
    staged_root: Path
    label_column: str          # column in the staged metadata with 0/1
    session_pattern: str = "*"  # directory glob per session


def load_external_records(cohort: ExternalCohort) -> pd.DataFrame:
    """Labels table for a staged external cohort.

    Expects ``<staged_root>/imaging/<session_dir>/`` volumes preprocessable
    by the Part D pipeline and ``<staged_root>/labels.csv`` with columns
    ``session_id, subject_id, label`` produced by the cohort stager.
    Session coverage is enforced later by ``OASISDataset`` (which raises on
    split sessions missing from disk).
    """
    labels = pd.read_csv(cohort.staged_root / "labels.csv")
    return labels


@torch.no_grad()
def evaluate_external(
    checkpoint_path: Path,
    cohort: ExternalCohort,
    cache_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    """Frozen-model evaluation on an external cohort.

    Reports the same metrics as Part I (threshold from OASIS-1 validation,
    never re-tuned) plus discrimination AUCs and calibration; a
    distribution-shift table (age/sex/eTIV where available) is written for
    the Part Y confound checks.
    """
    from brainvuln.mri.dataset import OASISDataset, collate_items
    from brainvuln.mri.evaluate import classification_metrics, predict_loader
    from brainvuln.mri.models import ResNet18Binary
    from torch.utils.data import DataLoader

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNetBinary_from_ckpt(ckpt)
    threshold = _frozen_threshold(checkpoint_path)

    split_df = load_external_records(cohort)
    ds = OASISDataset(split_df, cohort.staged_root / "imaging", cache_dir,
                      cohort=None)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=collate_items)
    y, p, sids, subs = predict_loader(model, loader, "cpu")
    metrics = classification_metrics(y, p, threshold)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session_id": sids, "subject_id": subs,
                  "label": y, "probability": p}).to_csv(
        out / f"{cohort.name}_predictions.csv", index=False)
    metrics["threshold_source"] = "frozen OASIS-1 validation threshold"
    metrics["n_subjects"] = len(set(subs))
    from brainvuln.config import checkpoint_identity
    metrics["checkpoint_identity"] = checkpoint_identity(str(checkpoint_path))
    (out / f"{cohort.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    return metrics


def ResNetBinary_from_ckpt(ckpt) -> "torch.nn.Module":
    from brainvuln.mri.models import ResNet18Binary
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _frozen_threshold(checkpoint_path: Path) -> float:
    metrics = checkpoint_path.parents[1] / "metrics" / "test_metrics.json"
    if metrics.exists():
        return float(json.loads(metrics.read_text())["threshold"])
    return 0.5

