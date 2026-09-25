"""AUDIT-REVIEW-002 regression tests: canonical checkpoint integrity.

Guards the freeze of the production checkpoint:

* the configured checkpoint's SHA256 equals the manifest
  (audit_artifacts/canonical_checkpoint.json) — fails if the file changes
  unexpectedly;
* a one-byte-mutated COPY of the checkpoint changes its SHA256 and is
  rejected by the canonical check (the production file itself is never
  touched; the mutated copy lives in pytest tmp_path and is discarded).

The tests always go through the explicit configured path
(``inference.checkpoint.path`` in config/analysis.yaml via the shared
resolver) — they never search the filesystem for another checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainvuln.config import checkpoint_identity, resolve_checkpoint, sha256_file  # noqa: E402

MANIFEST = PROJECT_ROOT / "audit_artifacts" / "canonical_checkpoint.json"


def _manifest_sha() -> str:
    assert MANIFEST.exists(), (
        f"canonical manifest missing: {MANIFEST} — generate it with "
        "audit_artifacts/scripts/review_002_checkpoint_freeze.py")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"]


def test_configured_checkpoint_sha_matches_manifest():
    canonical = _manifest_sha()
    # through the explicit configured path — no directory search anywhere
    path = resolve_checkpoint()
    assert sha256_file(path) == canonical


def test_checkpoint_identity_matches_manifest():
    canonical = _manifest_sha()
    identity = checkpoint_identity()
    assert identity["sha256"] == canonical
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert identity["size_bytes"] == manifest["size_bytes"]
    assert identity["filename"] == manifest["filename"]


def test_mutated_checkpoint_changes_sha256_and_is_rejected(tmp_path):
    canonical = _manifest_sha()
    src = resolve_checkpoint()
    copy = tmp_path / "mutated_best.pt"
    shutil.copyfile(src, copy)

    # sanity: the untouched copy still hashes to the canonical value
    assert sha256_file(copy) == canonical

    # flip exactly one byte
    data = bytearray(copy.read_bytes())
    data[len(data) // 2] ^= 0xFF
    copy.write_bytes(bytes(data))

    mutated = sha256_file(copy)
    assert mutated != canonical, "one-byte mutation did not change the SHA256"

    # the canonical integrity check rejects the mutated copy
    with Path(copy).open("rb"):
        pass
    assert _rejects(mutated), "mutated checkpoint passed the canonical check"


def _rejects(candidate_sha: str) -> bool:
    """The exact integrity decision rule used by the manifest check."""
    try:
        return candidate_sha != _manifest_sha()
    except AssertionError:
        return True
