#!/usr/bin/env python
"""Phase 23: performance & resource audit (CPU-only environment).

Measures, with real timings on this machine:
  1. preprocessing latency  (preprocess_session on a raw OASIS-1 T88 volume,
     median of 3 runs after 1 warmup; cache-load path timed separately)
  2. inference latency      (ResNet18Binary forward, batch 1, warm + 10 timed)
  3. full predict path      (torch.load checkpoint + one volume, end to end)
  4. parameter count        (raw total, dead backbone.fc excluded -> effective)
  5. checkpoint size        (results/ml/resnet_seed42/checkpoints/best.pt)
  6. memory                 (working set + peak working set via Psapi; no psutil)

Output: audit_artifacts/phase_23_performance.txt
"""
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from brainvuln.mri.models import ResNet18Binary
from brainvuln.mri.preprocess import build_scan_index, preprocess_session

OUT = Path("audit_artifacts/phase_23_performance.txt")
LINES: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(str(s))


def mem_kb() -> tuple[int, int]:
    """(current working set, peak working set) in KB via Psapi on Windows."""

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi = ctypes.WinDLL("Psapi.dll")
    k32 = ctypes.WinDLL("kernel32.dll")
    k32.GetCurrentProcess.restype = ctypes.c_void_p  # pseudo-handle, keep 64-bit
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                      ctypes.byref(pmc), pmc.cb):
        return 0, 0
    return pmc.WorkingSetSize, pmc.PeakWorkingSetSize


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


log("PHASE 23 — PERFORMANCE & RESOURCES (CPU-only)")
log("=" * 62)
log(f"torch {torch.__version__} | device cpu | threads {torch.get_num_threads()} "
    f"| pid {os.getpid()}")
ws0, peak0 = mem_kb()
log(f"memory at script start: working set {fmt_bytes(ws0)}, "
    f"peak {fmt_bytes(peak0)}")
log("")

# ---------------------------------------------------------------- checkpoint
CKPT = Path("results/ml/resnet_seed42/checkpoints/best.pt")
ckpt_bytes = CKPT.stat().st_size
log(f"[checkpoint] {CKPT}")
log(f"  size on disk: {ckpt_bytes:,} bytes ({fmt_bytes(ckpt_bytes)})")
t0 = time.perf_counter()
state = torch.load(CKPT, map_location="cpu", weights_only=False)
t_load = time.perf_counter() - t0
log(f"  torch.load time: {t_load * 1e3:.0f} ms (map_location=cpu)")
log("  keys: " + ", ".join(sorted(state.keys())))
log("")

# ------------------------------------------------------------------- model
model = ResNet18Binary()
model.load_state_dict(state["state_dict"])
model.eval()
named = [(n, p.numel()) for n, p in model.named_parameters()]
total = sum(n for _, n in named)
dead_fc = sum(n for name, n in named if ".fc." in name)
effective = total - dead_fc
log("[parameters]")
log(f"  raw total parameters:        {total:,}")
log(f"  dead backbone.fc parameters: {dead_fc:,}  (no gradient in train; "
    "excluded per phase 7)")
log(f"  effective (trainable) params: {effective:,}")
log("")

# ------------------------------------------------------- preprocessing latency
recs = list(build_scan_index(Path("data/raw/oasis1/extracted")))
rec = recs[0]
cache = Path("data/derived/oasis1_128") / f"{rec.session_id}.npy"
log("[preprocessing latency]")
log(f"  input: {rec.scan_path.name}  (session {rec.session_id})")
preprocess_session(rec.scan_path)  # warmup (import nibabel, JIT resampler)
times = []
for _ in range(3):
    t0 = time.perf_counter()
    vol = preprocess_session(rec.scan_path)
    times.append(time.perf_counter() - t0)
log(f"  preprocess_session: warmup + 3 runs -> median "
    f"{statistics.median(times):.2f} s (runs: "
    + ", ".join(f"{t:.2f}" for t in times) + ")")
log(f"  output shape {tuple(vol.shape)} dtype {vol.dtype} "
    f"({vol.nbytes / 1e6:.1f} MB per volume)")
if cache.exists():
    t0 = time.perf_counter()
    volc = np.load(cache)
    t_np = time.perf_counter() - t0
    log(f"  cache hit path (np.load of {cache.name}): {t_np * 1e3:.1f} ms, "
        f"shape {tuple(volc.shape)}")
else:
    log(f"  cache miss for {cache.name} (expected one of 182 cached volumes)")
log("")

# --------------------------------------------------------- inference latency
x = torch.from_numpy(np.load(cache)).float().unsqueeze(0)  # [1,1,128,128,128]
with torch.no_grad():
    for _ in range(3):
        model(x)  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        logits = model(x)
        times.append(time.perf_counter() - t0)
prob = torch.sigmoid(logits).item()
ws1, peak1 = mem_kb()
log("[inference latency (ResNet18Binary, batch 1, CPU)]")
log(f"  forward, 10 timed runs: median {statistics.median(times) * 1e3:.0f} ms "
    f"(min {min(times) * 1e3:.0f}, max {max(times) * 1e3:.0f})")
log(f"  example output: logit {logits.item():.4f} -> p(AD) {prob:.4f} "
    f"(threshold 0.07 -> {'AD' if prob >= 0.07 else 'CN'})")
log(f"  memory after inference: working set {fmt_bytes(ws1)} "
    f"(peak {fmt_bytes(peak1)})")
log("")

# --------------------------------------------------------- full predict path
t0 = time.perf_counter()
state2 = torch.load(CKPT, map_location="cpu", weights_only=False)
m2 = ResNet18Binary()
m2.load_state_dict(state2["state_dict"])
m2.eval()
with torch.no_grad():
    out = m2(torch.from_numpy(np.load(cache)).float().unsqueeze(0))
t_total = time.perf_counter() - t0
log("[full predict path (cold checkpoint load -> probability)]")
log(f"  load + build model + forward: {t_total * 1e3:.0f} ms total "
    f"(of which torch.load {t_load * 1e3:.0f} ms)")
log("")

log("[notes]")
log("  - all numbers measured on this machine, CPU-only (torch 2.4.1+cpu);")
log("    GPU numbers are NOT verifiable in this environment (see phase 22 R11).")
log("  - preprocessing is cached to data/derived/oasis1_128/*.npy; a cache hit")
log("    is bit-identical to fresh compute (phase 5), so steady-state serving")
log("    cost is cache load + forward.")
log(f"  - phase-22 risk R1 applies: predict latency is irrelevant if the app")
log(f"    picks the wrong checkpoint by mtime; R1 fix recommended post-audit.")

ws_final, peak_final = mem_kb()
log("")
log(f"memory at end: working set {fmt_bytes(ws_final)}, peak "
    f"{fmt_bytes(peak_final)}")
log("PHASE23_VERDICT: MEASURED (descriptive artifact, no pass/fail)")

OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}", flush=True)
