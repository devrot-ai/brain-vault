"""Optional checkpoint bootstrap for deployed environments.

The canonical checkpoint is deliberately NOT committed to git (128 MB,
GitHub hard limit is 100 MB) and must never be exposed publicly. On hosts
that build from the repository (Render/Railway/Cloud Run), the service can
fetch the frozen artifact at startup when configured:

    BRAINVULN_CHECKPOINT_URL    direct URL to best.pt (private, pre-signed,
                                or token-authenticated)
    BRAINVULN_CHECKPOINT_TOKEN  optional Authorization: Bearer <token>

The file is downloaded to the canonical configured path, then verified
against the frozen SHA256 before the service will serve anything. A wrong
hash aborts startup loudly — the API never serves an unverified checkpoint.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "service") not in sys.path:
    sys.path.insert(0, str(ROOT / "service"))

from predict_volume import CANONICAL_CHECKPOINT_RELPATH, CANONICAL_SHA256  # noqa: E402


def target_path() -> Path:
    return ROOT / CANONICAL_CHECKPOINT_RELPATH


def download_checkpoint_if_needed() -> Path | None:
    """Fetch the frozen checkpoint when absent and BRAINVULN_CHECKPOINT_URL
    is set. Returns the verified path, or None when nothing was needed."""
    url = os.environ.get("BRAINVULN_CHECKPOINT_URL", "").strip()
    if not url:
        return None
    dest = target_path()
    if dest.is_file() and _sha(dest) == CANONICAL_SHA256:
        return dest  # already present and verified
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests

    headers = {"Accept": "application/octet-stream"}  # GitHub asset API
    token = os.environ.get("BRAINVULN_CHECKPOINT_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"[bootstrap] downloading frozen checkpoint from configured URL ...",
          flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=120,
                      allow_redirects=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=str(dest.parent),
                                         delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
    got = _sha(tmp_path)
    if got != CANONICAL_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded checkpoint has SHA256 {got}, expected "
            f"{CANONICAL_SHA256} — refusing to serve. Check "
            "BRAINVULN_CHECKPOINT_URL points at the frozen seed-42 best.pt.")
    tmp_path.replace(dest)
    print(f"[bootstrap] checkpoint verified ({CANONICAL_SHA256[:12]}…)", flush=True)
    return dest


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
