"""Multi-seed training health report (final-project Phase 1).

For every seed run (42 canonical, 1-4, and the seed-4 repeat) inspect
train_history.json and the checkpoints:
  - NaN / non-finite losses
  - exploding losses (loss > 5x first-epoch loss)
  - validation collapse (val AUC < 0.5 after epoch 0)
  - best epoch (validation-only selection) and early-stop evidence
  - checkpoint loads + architecture/seed embedded identity

Writes audit_artifacts/multiseed_training_health.md. Read-only: never
touches checkpoints. Exits 0 even if some runs are incomplete (status is
reported per run); the HARD gate lives in the caller (finalize_all).
"""

from __future__ import annotations

import json
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
    "4 repeat (reproducibility)": ROOT / "results" / "ml" / "resnet_seed4_repeat",
}


def check_run(name: str, run_dir: Path) -> dict:
    out: dict = {"run": name, "dir": str(run_dir.relative_to(ROOT))}
    hist_file = run_dir / "metrics" / "train_history.json"
    if not hist_file.exists():
        out["status"] = "INCOMPLETE (no train_history.json)"
        return out
    hist = json.loads(hist_file.read_text(encoding="utf-8"))
    if isinstance(hist, dict):
        hist = hist.get("epochs", hist.get("history", []))
    n = len(hist)
    out["epochs_run"] = n
    losses = [e.get("loss", e.get("train_loss")) for e in hist]
    vaucs = [e.get("val_auc", e.get("val_roc_auc")) for e in hist]
    out["nan_loss"] = any(l is None or l != l for l in losses)
    out["nonfinite"] = any(
        l is None or abs(l) == float("inf") for l in losses)
    if losses and losses[0]:
        out["exploding"] = any(l > 5 * losses[0] for l in losses if l is not None)
    else:
        out["exploding"] = None
    late_auc = [a for i, a in enumerate(vaucs) if a is not None and i > 0]
    out["val_collapse"] = bool(late_auc and min(late_auc) < 0.5)
    best = max((a, i) for i, a in enumerate(vaucs) if a is not None)
    out["best_val_auc"] = round(best[0], 4)
    out["best_epoch"] = best[1]
    stopped = n < 40  # recipe allows up to 40 epochs; <40 means early stop
    out["early_stop"] = stopped
    # last epochs flat? (early-stopping corroboration)
    tail = vaucs[best[1] + 1:]
    out["epochs_after_best"] = len(tail)
    ck = run_dir / "checkpoints" / "best.pt"
    if ck.exists():
        try:
            payload = torch.load(ck, map_location="cpu", weights_only=False)
            sd = payload.get("model", payload.get("state_dict", payload))
            out["ckpt_loads"] = True
            out["n_tensors"] = len(sd)
            emb = payload.get("config", {}) if isinstance(payload, dict) else {}
            out["ckpt_seed"] = emb.get("seed", payload.get("seed", "?"))
            out["ckpt_arch"] = emb.get("model", "?")
            out["ckpt_val_auc"] = payload.get("val_auc", payload.get("best_auc", "?"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the report
            out["ckpt_loads"] = False
            out["ckpt_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["ckpt_loads"] = False
        out["ckpt_error"] = "best.pt missing"
    out["status"] = "OK" if out["ckpt_loads"] and not out["nan_loss"] \
        and not out["nonfinite"] and not out["val_collapse"] else "CHECK"
    return out


def main() -> int:
    results = [check_run(name, d) for name, d in RUNS.items()]
    lines = [
        "# BrainVuln Multi-Seed Training Health",
        "",
        "Generated from train_history.json + checkpoint files (read-only).",
        "",
        "| Run | Epochs | Best Val AUC (epoch) | Early stop | NaN | Explode | "
        "Val collapse | Checkpoint |",
        "| --- | -----: | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if "epochs_run" not in r:
            lines.append(f"| {r['run']} | - | - | - | - | - | - | "
                         f"{r['status']} |")
            continue
        if r["ckpt_loads"]:
            ck_txt = (f"loads ({r.get('n_tensors', '?')} tensors, "
                      f"seed {r.get('ckpt_seed', '?')})")
        else:
            ck_txt = f"FAIL: {r.get('ckpt_error')}"
        lines.append(
            f"| {r['run']} | {r['epochs_run']} | {r['best_val_auc']} "
            f"(epoch {r['best_epoch']}) | {'yes' if r['early_stop'] else 'no'} "
            f"({r['epochs_after_best']} after best) | "
            f"{'NO' if not r['nan_loss'] else 'YES'} | "
            f"{'no' if not r['exploding'] else 'YES'} | "
            f"{'no' if not r['val_collapse'] else 'YES'} | {ck_txt} |")
    lines += ["", "## Per-run detail", ""]
    for r in results:
        lines.append(f"- **{r['run']}** — {json.dumps(r, default=str)}")
    bad = [r for r in results if r.get("status") not in ("OK",)]
    lines += ["", f"Overall: {len(results) - len(bad)}/{len(results)} runs healthy."
              + ("" if not bad else
                 f" Flags: {', '.join(r['run'] for r in bad)}.")]
    out = ROOT / "audit_artifacts" / "multiseed_training_health.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    for r in results:
        print(f"  {r['run']}: {r.get('status')} "
              f"(epochs {r.get('epochs_run', '-')}), "
              f"best val {r.get('best_val_auc', '-')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
