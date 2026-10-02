"""Publish the BrainVuln inference service to a free Hugging Face Space.

Publishes a WHITELIST of exactly the files the Docker Space build needs:
the canonical service, the science code, config, and atlas. Nothing else
is sent (no MRI data, no checkpoints, no local artifacts, no secrets).

The Space runs the EXISTING service/Dockerfile unmodified except for one
build mechanical: HF builds from the Space repo root, so
`service/Dockerfile` is uploaded to the Space root as `Dockerfile` and
`hf_space/README.md` is uploaded as the Space `README.md`. All COPY
paths inside the Dockerfile stay valid because the layout under /app is
identical to the repo layout. HF routes public traffic to the container
via the README `app_port` metadata (the image listens on PORT, which HF
sets to app_port; the Dockerfile already honors $PORT).

Auth: the HF token is taken from the local environment (HF_TOKEN or
HUGGINGFACE_TOKEN, or --token to type it interactively). It is used only
to push via huggingface_hub on this machine. It is never written to any
file in the repo, never echoed, and never committed. Space runtime
secrets (checkpoint URL/token, CORS origin, upload cap) are entered by
the user in the Space settings web UI, never by this script.

Usage (from the repo root):
  python scripts/sync_hf_space.py --space <user>/<space>            # publish
  python scripts/sync_hf_space.py --space <user>/<space> --dry-run  # preview
  python scripts/sync_hf_space.py --space <user>/<space> --token    # prompt

Exit codes: 0 ok, 1 missing files/input, 2 usage/credentials error.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Whitelist: exactly what a Docker Space build needs, nothing more.
# Mirrors .dockerignore's allow-list plus package metadata (pyproject.toml,
# README.md) and the Space's own metadata README. Atlas files are tracked in
# git already; the CAM reference + frozen metrics are audited artifacts the
# canonical pipeline requires inside the container.
# ---------------------------------------------------------------------------
# Each entry is (source path in this repo, destination path in the Space).
WHITELIST = [
    ("service/app.py", "service/app.py"),
    ("service/predict_volume.py", "service/predict_volume.py"),
    ("service/checkpoint_bootstrap.py", "service/checkpoint_bootstrap.py"),
    ("service/requirements.txt", "service/requirements.txt"),
    ("service/Dockerfile", "Dockerfile"),
    ("hf_space/README.md", "README.md"),
    ("data/derived/atlas_dk68_tianS1.nii.gz", "data/derived/atlas_dk68_tianS1.nii.gz"),
    ("data/derived/atlas_dk68_tianS1_info.csv", "data/derived/atlas_dk68_tianS1_info.csv"),
    ("results/gradcam/OAS1_0013_gradcam.nii.gz", "results/gradcam/OAS1_0013_gradcam.nii.gz"),
    ("results/ml/resnet_seed42/metrics/test_metrics.json",
     "results/ml/resnet_seed42/metrics/test_metrics.json"),
    ("pyproject.toml", "pyproject.toml"),
    ("README.md", "PROJECT_README.md"),
]

# Whole directories published as-is (destination mirrors the source layout).
WHITELIST_DIRS = [
    ("src", "src"),      # canonical science package, unchanged
    ("config", "config"),  # analysis.yaml: checkpoint + threshold resolution
]

# Hard guard against whitelist mistakes (checkpoints, data, frontend, VCS,
# caches). Path components in DENY_SEGMENTS must match exactly (so a file
# like config/.gitkeep is fine), while DENY_SUBSTRINGS match anywhere in the
# POSIX path (so any *best.pt* or *.env* file is refused).
DENY_SEGMENTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".vercel",
    "audit_artifacts", "site", "hf_space",
}
# "data/raw" stays a substring deny so raw OASIS-1 files are refused
# anywhere; the only data/ destinations allowed are the two whitelisted
# atlas files that service/Dockerfile COPYs (tracked in git already).
DENY_SUBSTRINGS = ["best.pt", ".env", "data/raw"]

README_SPACE = (
    "---\n"
    "title: BrainVault API\n"
    "emoji: \U0001F9E0\n"
    "colorFrom: blue\n"
    "colorTo: purple\n"
    "sdk: docker\n"
    "app_port: 7860\n"
    "---\n"
    "\n"
    "# BrainVault API — Hugging Face Space\n"
    "\n"
    "Docker Space serving the canonical, frozen BrainVuln inference API\n"
    "(seed-42 best.pt, SHA256-verified at startup; frozen threshold 0.07).\n"
    "Research use only — not a clinical diagnostic system.\n"
    "\n"
    "Published by `scripts/sync_hf_space.py` from the development repo.\n"
    "The scientific pipeline (service/predict_volume.py ->\n"
    "brainvuln.mri.preprocess / models / gradcam / regional) is copied\n"
    "unchanged; nothing is re-implemented here.\n"
    "\n"
    "## Endpoints\n"
    "\n"
    "| Method | Path              | Purpose                                       |\n"
    "|--------|-------------------|-----------------------------------------------|\n"
    "| GET    | `/api/health`     | liveness + checkpoint identity (no weights)   |\n"
    "| GET    | `/api/model-info` | frozen checkpoint / threshold / atlas details |\n"
    "| POST   | `/api/predict`    | multipart `file` = `.nii` / `.nii.gz`         |\n"
    "\n"
    "`/api/predict` runs the unchanged canonical chain:\n"
    "preprocess_session -> ResNet18Binary(best.pt) -> gradcam_3d ->\n"
    "regionalize_cam, and returns probability, frozen threshold, label,\n"
    "top atlas regions, and a Grad-CAM overlay PNG (base64). Uploaded\n"
    "volumes exist only in a temp directory deleted after every request.\n"
    "\n"
    "Free CPU Basic hardware sleeps when idle and auto-wakes on request.\n"
    "The checkpoint is fetched at startup from the private release asset\n"
    "and SHA256-verified against\n"
    "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9;\n"
    "a mismatch aborts startup — the API never serves an unverified model.\n"
    "\n"
    "Research use only — not a clinical diagnostic system.\n"
)


def _check_deny(path: str) -> None:
    posix = path.replace("\\", "/")
    segments = posix.lower().split("/")
    for seg in segments:
        if seg in DENY_SEGMENTS:
            raise SystemExit(
                f"refusing to publish: {path!r} matches deny segment {seg!r}")
    lowered = posix.lower()
    for needle in DENY_SUBSTRINGS:
        if needle in lowered:
            raise SystemExit(
                f"refusing to publish: {path!r} matches deny rule {needle!r}")


def _existing_files() -> list[tuple[Path, str]]:
    """Resolve the whitelist against disk; raise on anything missing.

    Destinations are always deny-checked. Explicit whitelist sources are
    developer-reviewed constants and skip the source-path check (e.g.
    `hf_space/README.md` is deliberately mapped to the Space's README.md);
    directory-walked files are checked on both sides.
    """
    out: list[tuple[Path, str]] = []
    for src, dest in WHITELIST:
        p = ROOT / src
        if not p.is_file():
            raise SystemExit(f"missing required file: {src}")
        _check_deny(dest)
        out.append((p, dest))
    for src, dest in WHITELIST_DIRS:
        d = ROOT / src
        if not d.is_dir():
            raise SystemExit(f"missing required directory: {src}")
        _check_deny(src + "/")
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(d).as_posix()
            parts = rel.split("/")
            if any(p.endswith(".egg-info") for p in parts):
                continue  # local build metadata, not source
            if rel.endswith((".pyc", ".pyo")) or rel == ".gitkeep":
                continue
            dest_rel = f"{dest}/{rel}"
            _check_deny(dest_rel)
            out.append((f, dest_rel))
    return out


def _token_from_env_or_prompt(explicit: bool) -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if tok.strip():
        return tok.strip()
    if explicit:
        print("Enter your Hugging Face WRITE token (input hidden, not stored):")
        return getpass.getpass("HF token: ").strip()
    print("No HF token found. Set HF_TOKEN in your shell, or re-run with --token "
          "to enter it interactively.")
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish the BrainVuln inference service to a Hugging Face Space.")
    ap.add_argument("--space", required=True, metavar="USER/SPACE",
                    help="Space repo id, e.g. devrot-ai/brainvault-api")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact upload plan without uploading")
    ap.add_argument("--token", action="store_true",
                    help="prompt for the HF token interactively (input hidden)")
    args = ap.parse_args()

    if "/" not in args.space or args.space.count("/") != 1:
        print("--space must look like USER/SPACE")
        return 2

    pairs = _existing_files()
    print(f"upload plan: {len(pairs)} files -> {args.space}")
    for p, dest in pairs:
        size_kb = p.stat().st_size / 1024
        print(f"  {dest:<55} {size_kb:9.1f} KB   (from {p.relative_to(ROOT)})")

    if args.dry_run:
        print("dry run: nothing uploaded.")
        return 0

    token = _token_from_env_or_prompt(args.token)
    if not token:
        print("empty token; aborting.")
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed in this venv.")
        print("install it with:  .venv/Scripts/python.exe -m pip install -r "
              "scripts/requirements-hf.txt")
        return 2

    api = HfApi(token=token)
    user = api.whoami()["name"]
    print(f"authenticated as {user}")

    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="docker",
                    private=False, exist_ok=True)
    print(f"space {args.space} ready (sdk=docker)")

    with tempfile.TemporaryDirectory(prefix="hf_space_upload_") as td:
        stage = Path(td)
        for p, dest in pairs:
            target = stage / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        (stage / "README.md").write_text(README_SPACE, encoding="utf-8")
        api.upload_folder(repo_id=args.space, repo_type="space",
                          folder_path=str(stage), commit_message=(
                              "Publish canonical inference service (science unchanged)"))
    print(f"done. build progress: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
