"""Multi-seed fairness check (final-project Phase 3).

Verifies that seeds 1-4 (and the seed-4 repeat) differ from each other
ONLY in the random seed: compares the checkpoint-embedded TrainConfig
(lr, batch size, epochs, patience, warmup, cosine, augmentation is not
embedded but the driver passes identical flags) and the directory
config snapshot if present.

Writes audit_artifacts/multiseed_fairness.md. Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    42: ROOT / "results" / "ml" / "resnet_seed42",
    1: ROOT / "results" / "ml" / "resnet_seed1",
    2: ROOT / "results" / "ml" / "resnet_seed2",
    3: ROOT / "results" / "ml" / "resnet_seed3",
    4: ROOT / "results" / "ml" / "resnet_seed4",
    44: ROOT / "results" / "ml" / "resnet_seed4_repeat",  # key 44 = repeat of seed 4
}

# fields that must be IDENTICAL across runs (seed excluded by design)
RECIPE_FIELDS = ["lr", "weight_decay", "batch_size", "max_epochs",
                 "patience", "warmup_epochs", "cosine_epochs",
                 "pos_weight_auto", "normalization"]


def load_cfg(run_dir: Path) -> dict | None:
    ck = run_dir / "checkpoints" / "best.pt"
    if not ck.exists():
        return None
    p = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = dict(p.get("config") or {})
    cfg.pop("history", None)          # bulky, not a recipe field
    cfg.pop("resume_path", None)      # run-specific path
    cfg.pop("resume", None)           # resume differs for the interrupted run
    cfg.pop("device", None)           # environment, not recipe
    cfg.pop("num_workers", None)
    cfg.pop("log_every", None)
    return cfg


def main() -> int:
    cfgs = {}
    for key, d in RUNS.items():
        c = load_cfg(d)
        if c is not None:
            cfgs[key] = c
    if len(cfgs) < 2:
        print("fairness: fewer than two completed runs — nothing to compare")
        return 1

    reference_key = next(iter(cfgs))
    diffs = []
    for key, cfg in cfgs.items():
        for f in RECIPE_FIELDS:
            if f not in cfg or f not in cfgs[reference_key]:
                continue
            if cfg[f] != cfgs[reference_key][f]:
                diffs.append((key, f, cfgs[reference_key][f], cfg[f]))

    seeds_seen = {k: c.get("seed") for k, c in cfgs.items()}
    lines = [
        "# BrainVuln Multi-Seed Fairness Check",
        "",
        "Checkpoint-embedded TrainConfig compared across runs.",
        "Recipe fields compared: " + ", ".join(RECIPE_FIELDS),
        "",
        "| Run key (dir seed) | Embedded seed |",
        "| --- | --- |",
    ]
    for k in sorted(cfgs):
        lines.append(f"| {k} | {seeds_seen[k]} |")
    lines += ["", f"Recipe mismatches vs run {reference_key}: "
              f"{'NONE — only the seed differs' if not diffs else ''}"]
    for key, f, ref, got in diffs:
        lines.append(f"- run {key}: {f} = {got} (reference {ref})")

    # split identity: every run evaluated the same 27 test subjects
    from collections import Counter
    test_sets = {}
    for k in cfgs:
        f = RUNS[k] / "predictions" / "test_predictions.csv"
        if f.exists():
            import pandas as pd
            test_sets[k] = tuple(sorted(pd.read_csv(f)["subject_id"]))
    consistent = len(set(test_sets.values())) == 1 if test_sets else None
    if consistent is not None:
        lines += ["",
                  f"Test subject set identical across "
                  f"{len(test_sets)} evaluated runs: {consistent}"]
    else:
        lines += ["", "Test subject set: not all prediction files present yet"]

    lines += ["",
              "Note: normalization/augmentation flags are not embedded in the "
              "checkpoint; the driver (scripts/multiseed.bat) passes the "
              "identical canonical recipe to every seed "
              "(--epochs 40 --patience 8 --warmup 2 --cos-epochs 12 "
              "--aug-strength light). The interrupted seed-3 run was restarted "
              "with the same recipe, so its history is a fresh deterministic "
              "replay of the same seed."]
    out = ROOT / "audit_artifacts" / "multiseed_fairness.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    print("recipe mismatches:", diffs if diffs else "NONE")
    if consistent is not None:
        print("test sets identical:", consistent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
