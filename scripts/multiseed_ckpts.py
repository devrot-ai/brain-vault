"""Checkpoint manifest for all multi-seed runs (final-project Phase 2).

Records path, size, SHA256, architecture shape, embedded seed and
embedded validation AUC for every best.pt. Read-only.

Writes audit_artifacts/multiseed_checkpoints.md.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "42 (canonical)": ROOT / "results" / "ml" / "resnet_seed42",
    "1": ROOT / "results" / "ml" / "resnet_seed1",
    "2": ROOT / "results" / "ml" / "resnet_seed2",
    "3": ROOT / "results" / "ml" / "resnet_seed3",
    "4": ROOT / "results" / "ml" / "resnet_seed4",
    "4 repeat": ROOT / "results" / "ml" / "resnet_seed4_repeat",
}
CANONICAL_SHA = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines = [
        "# BrainVuln Multi-Seed Checkpoint Manifest",
        "",
        "| Run | Checkpoint | Size (MB) | SHA256 | Embedded seed | "
        "Embedded val AUC | Loads | First-layer shape |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for name, d in RUNS.items():
        ck = d / "checkpoints" / "best.pt"
        if not ck.exists():
            lines.append(f"| {name} | - | - | - | - | - | MISSING | - |")
            continue
        sha = sha256_file(ck)
        size_mb = ck.stat().st_size / (1 << 20)
        note = ""
        if name.startswith("42"):
            note = " **(canonical — must equal frozen SHA)**"
        loads, seed_v, val_v, shape = "?", "?", "?", "-"
        try:
            p = torch.load(ck, map_location="cpu", weights_only=False)
            sd = p.get("model", p.get("state_dict", p))
            k0 = next(iter(sd))
            shape = str(tuple(sd[k0].shape))
            seed_v = (p.get("config") or {}).get("seed", "?")
            val_v = p.get("val_auc", "?")
            loads = "yes"
        except Exception as exc:  # noqa: BLE001
            loads = f"FAIL ({type(exc).__name__})"
        short = sha if name.startswith("42") else sha[:16] + "…"
        lines.append(
            f"| {name} | {ck.relative_to(ROOT)} | {size_mb:.1f} | `{short}`{note} "
            f"| {seed_v} | {val_v} | {loads} | {shape} |")
    lines += ["",
              f"Canonical seed-42 SHA256: `{CANONICAL_SHA}`"]
    out = ROOT / "audit_artifacts" / "multiseed_checkpoints.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    for name, d in RUNS.items():
        ck = d / "checkpoints" / "best.pt"
        if ck.exists():
            print(f"  {name}: {sha256_file(ck)[:16]}… ({ck.stat().st_size >> 20} MB)")
        else:
            print(f"  {name}: no checkpoint yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
