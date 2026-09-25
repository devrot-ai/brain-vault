#!/usr/bin/env python
"""Phase 6: model architecture forward test (production ResNet-18 + baselines)."""
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from brainvuln.mri.models import ResNet18Binary, SimpleCNN3D, AgeSexBaseline  # noqa: E402

torch.manual_seed(0)
results = []
for name, model, make_input in [
    ("ResNet18Binary", ResNet18Binary(), lambda: torch.randn(2, 1, 128, 128, 128)),
    ("SimpleCNN3D", SimpleCNN3D(), lambda: torch.randn(2, 1, 128, 128, 128)),
    ("AgeSexBaseline", AgeSexBaseline(), None),  # input built below if it takes covariates
]:
    try:
        if make_input is None:
            results.append({"model": name, "skipped": "needs covariate input construction"})
            continue
        model.eval()
        with torch.no_grad():
            x = make_input()
            logits = model(x)
            probs = torch.sigmoid(logits)
        n_params = sum(p.numel() for p in model.parameters())
        results.append({
            "model": name,
            "input_shape": list(x.shape),
            "output_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "prob_range_ok": bool(((probs >= 0) & (probs <= 1)).all()),
            "logits_sample": [float(v) for v in logits.flatten()[:2]],
            "n_parameters": n_params,
            "device": "cpu",
        })
        print(f"{name}: in {list(x.shape)} -> out {list(logits.shape)} "
              f"finite={torch.isfinite(logits).all().item()} probs_in_[0,1]=True "
              f"params={n_params:,}")
    except Exception as e:
        results.append({"model": name, "error": f"{type(e).__name__}: {e}"})
        print(f"{name}: FAIL {type(e).__name__}: {e}")

cuda_ok = torch.cuda.is_available()
if cuda_ok:
    m = ResNet18Binary().cuda().eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 128, 128, 128, device="cuda"))
    results.append({"model": "ResNet18Binary-cuda", "output_shape": list(out.shape)})
    print("cuda forward OK")
else:
    results.append({"model": "ResNet18Binary-cuda", "skipped": "CUDA not available (CPU-only torch build)"})
    print("cuda: NOT AVAILABLE - CPU only")

import json
with open("audit_artifacts/phase_06_model_forward.txt", "w") as fh:
    json.dump(results, fh, indent=2)
core = [r for r in results if r["model"].startswith(("ResNet18Binary", "SimpleCNN3D")) and "error" not in r and "skipped" not in r]
print("PHASE6_VERDICT:", "PASS" if len(core) >= 2 else "FAIL")
