#!/usr/bin/env python
"""AUDIT-REVIEW-002 — PART 2: canonical checkpoint manifest + serving-path proof.

Creates audit_artifacts/canonical_checkpoint.json (2.1) and verifies every
production entry point resolves the same canonical checkpoint (2.2).
Read-only: never touches the checkpoint itself.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"


def main() -> int:
    from brainvuln.config import checkpoint_identity, resolve_checkpoint

    # ---------------- 2.1 manifest ----------------
    ident = checkpoint_identity()
    resolved = resolve_checkpoint()
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("config", {})
    manifest = {
        "purpose": "production baseline",
        "recorded_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resolved_path": str(resolved),
        "filename": ident["filename"],
        "sha256": ident["sha256"],
        "size_bytes": ident["size_bytes"],
        "model_architecture": "ResNet18Binary (MONAI ResNet backbone, 1 input channel, binary head)",
        "seed": int(train_cfg.get("seed", 42)),
        "training_config": {
            "source": "checkpoint-embedded snapshot (ckpt['config'])",
            "lr": train_cfg.get("lr"),
            "weight_decay": train_cfg.get("weight_decay"),
            "batch_size": train_cfg.get("batch_size"),
            "warmup_epochs": train_cfg.get("warmup_epochs"),
            "cosine_epochs": train_cfg.get("cosine_epochs"),
            "device": train_cfg.get("device"),
            "epoch_of_checkpoint": int(ckpt.get("epoch", -1)),
            "val_auc_of_checkpoint": float(ckpt.get("val_auc", float("nan"))),
        },
        "training_config_path": "config/analysis.yaml",
        "preprocessing_config_path": "src/brainvuln/mri/preprocess.py "
                                     "(input: OASIS-1 *_t88_masked_gfc; 2 mm, crop/pad 128^3, brain z-score)",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "monai_version": _pkg_version("monai"),
        "numpy_version": _pkg_version("numpy"),
        "platform": platform.platform(),
        "canonical_sha_expected": CANONICAL_SHA,
        "sha_match": ident["sha256"] == CANONICAL_SHA,
    }
    out = PROJECT_ROOT / "audit_artifacts" / "canonical_checkpoint.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(manifest, indent=2)[:400])
    if not manifest["sha_match"]:
        print("FATAL: canonical sha mismatch")
        return 2

    # ---------------- 2.2 serving-path proof ----------------
    print("\n=== 2.2 serving-path proof ===")
    results = {}

    # (a) app.py serving path
    import app
    model, app_path, threshold, app_ident = app._get_model()
    results["app.py _get_model"] = {
        "path": str(app_path), "sha256": app_ident["sha256"],
        "matches": app_ident["sha256"] == CANONICAL_SHA}
    del model

    # (b) predict.py CLI (full real run; detailed comparison happens in PART 5)
    vol = sorted(PROJECT_ROOT.glob(
        "data/raw/oasis1/extracted/disc*/disc*/OAS1_0013_MR1/"
        "PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img"))[0]
    r = subprocess.run(
        [sys.executable, "predict.py", "--image", str(vol)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    sha_line = next((l.strip() for l in r.stdout.splitlines()
                     if l.strip().startswith("5930f0ff")), "")
    results["predict.py CLI"] = {
        "returncode": r.returncode, "sha_line": sha_line,
        "matches": r.returncode == 0 and CANONICAL_SHA in sha_line}

    # (c) scripts/validate_external.py — uses the SAME shared resolver
    # (verified by import inspection below; a functional run needs a staged
    # external cohort, which does not exist yet — honest limitation)
    src = (PROJECT_ROOT / "scripts" / "validate_external.py").read_text(encoding="utf-8")
    results["scripts/validate_external.py"] = {
        "uses_shared_resolver": "resolve_checkpoint" in src and "checkpoint_identity" in src,
        "no_mtime_glob": "st_mtime" not in src and ".glob(" not in src}

    # (d) any other production torch.load entry points
    for rel in ("scripts/train_oasis1.py", "src/brainvuln/mri/external.py"):
        t = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        results[rel] = {
            "loads_checkpoints": "torch.load" in t,
            "no_mtime_glob_discovery": "st_mtime" not in t and "*.pt" not in t,
            "note": "train_oasis1 writes its own out_dir/checkpoints/best.pt; "
                    "external.py loads the path handed to it by validate_external.py "
                    "(canonical resolver)"}

    ok = True
    for k, v in results.items():
        flat = json.dumps(v)
        good = "false" not in flat.lower()
        ok &= good
        print(f"{k}: {json.dumps(v)}")
    print(f"\nALL SERVING PATHS CANONICAL: {ok}")
    return 0 if ok else 3


def _pkg_version(name: str) -> str:
    try:
        import importlib
        m = importlib.import_module(name)
        return str(getattr(m, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
