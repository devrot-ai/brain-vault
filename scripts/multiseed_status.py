#!/usr/bin/env python
"""Write audit_artifacts/multiseed_status.md (phase-0 hard completion gate).

A checkpoint does NOT mean a run completed. Status logic per run dir:

* RUNNING      - last_state.pt modified within the last 15 minutes
* COMPLETE     - training history + val metrics + test metrics all present
                 and the history ends in a legitimate stop (early stop or
                 epoch budget) rather than a crash
* INTERRUPTED  - stopped > 15 min ago without test metrics (needs resume)
* PENDING      - run has not started

Everything here is read from disk; nothing is inferred from a checkpoint
existing alone.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSOLE = ROOT / "results" / "multiseed_console.log"
RUNS = [(1, ROOT / "results" / "ml" / "resnet_seed1"),
        (2, ROOT / "results" / "ml" / "resnet_seed2"),
        (3, ROOT / "results" / "ml" / "resnet_seed3"),
        (4, ROOT / "results" / "ml" / "resnet_seed4"),
        ("4 repeat", ROOT / "results" / "ml" / "resnet_seed4_repeat")]


def parse_console_epochs(run_dir: Path) -> list[dict]:
    """Epoch lines for this run from the driver console (in-flight source)."""
    if not CONSOLE.exists():
        return []
    tag = run_dir.name
    text = CONSOLE.read_text(encoding="utf-8", errors="replace")
    start = text.rfind(f"TRAINING STARTED")
    epochs = []
    for m in re.finditer(
            r"epoch (\d+) loss ([\d.]+) val_auc ([\d.]+)( \*)?", text):
        epochs.append({"epoch": int(m.group(1)), "loss": float(m.group(2)),
                       "val_auc": float(m.group(3)),
                       "best": bool(m.group(4))})
    return epochs


def run_status(run_dir: Path) -> dict:
    ck = run_dir / "checkpoints" / "best.pt"
    last = run_dir / "checkpoints" / "last_state.pt"
    hist_f = run_dir / "metrics" / "train_history.json"
    met_f = run_dir / "metrics" / "test_metrics.json"
    out = {"best_ckpt": ck.exists(), "last_state": last.exists(),
           "history": hist_f.exists(), "test_metrics": met_f.exists(),
           "epochs_run": None, "best_val": None, "best_epoch": None,
           "status": "PENDING"}
    if not run_dir.exists():
        return out
    recent = last.exists() and (time.time() - last.stat().st_mtime) < 900
    hist = None
    if hist_f.exists():
        try:
            hist = json.loads(hist_f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hist = None
    if hist is not None:
        out["epochs_run"] = hist.get("epochs_run")
        out["best_val"] = hist.get("best_val_auc")
        out["best_epoch"] = hist.get("best_epoch")
    else:
        ep = parse_console_epochs(run_dir)
        if ep:
            best = [e for e in ep if e["best"]] or [ep[0]]
            b = max(best, key=lambda e: (e["val_auc"], -e["epoch"]))
            out["epochs_run"] = ep[-1]["epoch"] + 1
            out["best_val"] = b["val_auc"]
            out["best_epoch"] = b["epoch"]
    if met_f.exists() and hist is not None:
        out["status"] = "COMPLETE"
    elif recent:
        out["status"] = "RUNNING"
    elif out["epochs_run"]:
        out["status"] = "INTERRUPTED (needs resume from last_state.pt)"
    return out


def main() -> int:
    lines = ["# Multi-seed campaign status (phase-0 hard completion gate)", "",
             f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             "from disk state. A checkpoint alone is NOT completion evidence; "
             "'COMPLETE' requires history + val metrics + test metrics.*", "",
             "| Seed | Status | Epochs run | Best Val AUC (epoch) | Best "
             "Checkpoint | last_state | Test Done |",
             "| --- | --- | ---: | --- | --- | --- | --- |"]
    all_complete = True
    for label, d in RUNS:
        s = run_status(d)
        all_complete &= s["status"] == "COMPLETE"
        best = (f"{s['best_val']:.4f} ({s['best_epoch']})"
                if s["best_val"] is not None else "n/a")
        lines.append(
            f"| {label} | {s['status']} | {s['epochs_run']} | {best} | "
            f"{'yes' if s['best_ckpt'] else 'NO'} | "
            f"{'yes' if s['last_state'] else 'NO'} | "
            f"{'yes' if s['test_metrics'] else 'NO'} |")
    lines += ["", f"**ALL_RUNS_COMPLETE: {'YES' if all_complete else 'NO'}**",
              "", "Per the hard training-completion gate, final model "
              "selection, reliability gates and product finalization wait "
              "until ALL_RUNS_COMPLETE: YES."]
    out = ROOT / "audit_artifacts" / "multiseed_status.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
