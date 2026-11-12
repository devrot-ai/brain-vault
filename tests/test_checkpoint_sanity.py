"""Checkpoint sanity guards: an untrained or mismatched artifact must fail.

Why this file exists
--------------------
The canonical checkpoint was once overwritten by a freshly initialised
ResNet-18 that kept the old ``val_auc`` metadata. That file hashed cleanly,
loaded cleanly, and passed the SHA-vs-manifest test (the manifest had been
edited to bless it) while predicting ~0.53 for every subject. Byte-level
integrity is therefore not enough; these tests check the *contents*:

* the state dict loads strictly into the audited architecture and the
  parameter count matches the audited model;
* BatchNorm counters/running statistics prove the weights saw training data;
* the manifest pins the AUDITED canonical SHA256, rather than only agreeing
  with whatever hash it currently records, and its own fields agree;
* the serving constant accepts the same audited identity;
* on the cached validation split the probabilities are non-degenerate, the
  ROC-AUC clears a chance floor, and the checkpoint's recorded ``val_auc``
  is reproduced — skipped when the cache is not on disk.

A failure here means the configured artifact is not the audited model. The
fix is to restore the audited checkpoint or retrain and deliberately
re-pin ``AUDITED_CANONICAL_SHA256`` (and the manifest / serving constant)
as a recorded promotion — never to loosen these assertions.

Run:  pytest -q tests/test_checkpoint_sanity.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "audit_artifacts" / "canonical_checkpoint.json"
VAL_SPLIT = ROOT / "data" / "splits" / "val_subjects.csv"
CACHE = ROOT / "data" / "derived" / "oasis1_128"

# AUDIT-REVIEW-002 canonical artifact (see audit_artifacts/frozen_state.json
# and AUDIT_REVIEW_002.md). Pinning the audited value here — instead of
# trusting the mutable manifest — is the point of this file. Promoting a new
# model requires updating this constant, the manifest and the serving
# constant in the same deliberate change.
AUDITED_CANONICAL_SHA256 = (
    "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
)
AUDITED_PARAM_COUNT = 33_366_609  # ResNet18Binary, widen_factor=1.0


def _manifest() -> dict:
    assert MANIFEST.is_file(), f"canonical manifest missing: {MANIFEST}"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checkpoint() -> dict:
    torch = pytest.importorskip("torch")
    from brainvuln.config import resolve_checkpoint

    path = resolve_checkpoint()
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# contents: architecture agreement and evidence of training
# ---------------------------------------------------------------------------


def test_state_dict_matches_audited_architecture(checkpoint) -> None:
    pytest.importorskip("torch")
    from brainvuln.mri.models import ResNet18Binary, count_parameters

    model = ResNet18Binary()  # the audited architecture: widen_factor=1.0
    # strict=True raises on any missing key, unexpected key or shape mismatch
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert count_parameters(model) == AUDITED_PARAM_COUNT, (
        "configured checkpoint does not match the audited ResNet18Binary "
        f"parameter count ({count_parameters(model):,} != "
        f"{AUDITED_PARAM_COUNT:,}); a widened or otherwise different model "
        "must not occupy the canonical path"
    )
    assert model.fc.in_features == 512


def test_batch_norm_proves_the_weights_were_trained(checkpoint) -> None:
    torch = pytest.importorskip("torch")
    state = checkpoint["state_dict"]

    tracked = {k: int(v) for k, v in state.items()
               if k.endswith("num_batches_tracked")}
    assert tracked, "checkpoint has no BatchNorm buffers — wrong architecture?"

    never_updated = [k for k, v in tracked.items() if v == 0]
    assert not never_updated, (
        f"{len(never_updated)}/{len(tracked)} BatchNorm layers have "
        "num_batches_tracked == 0 "
        f"(e.g. {sorted(never_updated)[:3]}): this checkpoint is a freshly "
        "initialised network, not the audited trained model. Restore the "
        "audited artifact or retrain before promoting it."
    )

    running_vars = [state[k] for k in state if k.endswith("running_var")]
    assert running_vars, "checkpoint has no BatchNorm running statistics"
    at_init = sum(
        bool(torch.allclose(v, torch.ones_like(v))) for v in running_vars
    )
    assert at_init < len(running_vars), (
        "every BatchNorm running_var is exactly 1.0 (its initialisation "
        "value); the weights never saw training data"
    )


# ---------------------------------------------------------------------------
# identity: the manifest and the serving guard must agree with the audit
# ---------------------------------------------------------------------------


def test_manifest_pins_the_audited_checkpoint() -> None:
    from brainvuln.config import resolve_checkpoint, sha256_file

    manifest = _manifest()
    assert manifest["sha256"] == AUDITED_CANONICAL_SHA256, (
        "canonical_checkpoint.json no longer records the AUDIT-REVIEW-002 "
        f"checkpoint ({manifest['sha256'][:12]}… != "
        f"{AUDITED_CANONICAL_SHA256[:12]}…). Editing the manifest to accept "
        "a replacement is not a restore: restore the audited bytes or record "
        "a deliberate re-freeze that updates this test's constant too."
    )

    actual = sha256_file(resolve_checkpoint())
    assert actual == AUDITED_CANONICAL_SHA256, (
        "the configured checkpoint is NOT the audited canonical artifact "
        f"(got {actual[:12]}…, expected {AUDITED_CANONICAL_SHA256[:12]}…)"
    )


def test_manifest_is_internally_consistent() -> None:
    manifest = _manifest()
    assert manifest["sha256"] == manifest["canonical_sha_expected"], (
        "canonical_checkpoint.json contradicts itself: sha256 "
        f"{manifest['sha256'][:12]}… vs canonical_sha_expected "
        f"{manifest['canonical_sha_expected'][:12]}…"
    )
    assert manifest["sha_match"] is True, (
        "canonical_checkpoint.json records sha_match=false; the freeze is "
        "not valid"
    )


def test_serving_constant_matches_audited_checkpoint() -> None:
    service = _load_module("brainvault_service_predict_volume",
                           "service/predict_volume.py")
    assert service.CANONICAL_SHA256 == AUDITED_CANONICAL_SHA256, (
        "the serving guard would accept a checkpoint that is not the audited "
        f"artifact (CANONICAL_SHA256 {service.CANONICAL_SHA256[:12]}… != "
        f"{AUDITED_CANONICAL_SHA256[:12]}…)"
    )


# ---------------------------------------------------------------------------
# behaviour: the frozen model must still separate the cached validation split
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validation_predictions(checkpoint):
    """Frozen-checkpoint probabilities on the cached validation subjects."""
    torch = pytest.importorskip("torch")
    pandas = pytest.importorskip("pandas")

    if not VAL_SPLIT.is_file() or not CACHE.is_dir():
        pytest.skip("validation split or preprocessed cache not on disk")
    split = pandas.read_csv(VAL_SPLIT)
    missing = [str(s) for s in split["session_id"]
               if not (CACHE / f"{s}.npy").is_file()]
    if missing:
        pytest.skip(f"{len(missing)} cached validation volumes missing")

    from brainvuln.mri.models import ResNet18Binary

    model = ResNet18Binary()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    import numpy as np

    probs = []
    with torch.no_grad():
        for session_id in split["session_id"].astype(str):
            volume = np.load(CACHE / f"{session_id}.npy")
            x = torch.from_numpy(volume[None].astype(np.float32))
            probs.append(float(torch.sigmoid(model(x)).item()))
    return split, np.asarray(probs, dtype=float)


def test_cached_validation_predictions_are_not_degenerate(
    validation_predictions, checkpoint
) -> None:
    split, probs = validation_predictions
    from sklearn.metrics import roc_auc_score

    assert probs.std() > 0.02, (
        "the model emits a near-constant probability for every validation "
        f"subject (sd={probs.std():.4f}, range "
        f"{probs.min():.3f}–{probs.max():.3f}); the checkpoint carries no "
        "learned signal"
    )

    auc = float(roc_auc_score(split["label"].astype(int), probs))
    assert auc >= 0.60, (
        f"validation ROC-AUC is {auc:.3f}, at or below the chance floor; the "
        "configured checkpoint is not a trained classifier"
    )

    recorded = float(checkpoint["val_auc"])
    assert abs(auc - recorded) <= 0.05, (
        f"checkpoint metadata claims val_auc={recorded:.3f} but the cached "
        f"validation split reproduces {auc:.3f}; the recorded results do not "
        "belong to this artifact (or the preprocessing cache changed)"
    )
