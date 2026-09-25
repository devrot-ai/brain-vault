#!/usr/bin/env python
"""Phase 7: backward pass / trainability test (one real optimizer step)."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
from brainvuln.mri.models import ResNet18Binary  # noqa: E402

torch.manual_seed(42)
model = ResNet18Binary()
x = torch.randn(2, 1, 128, 128, 128)
y = torch.tensor([1.0, 0.0])

before = {n: p.detach().clone() for n, p in model.named_parameters()}
logits = model(x)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, y)
loss.backward()

grad_stats = {}
nan_grads = []
DEAD_LAYERS = {"backbone.fc.weight", "backbone.fc.bias"}  # MONAI ResNet's own
# unused head; our forward replaces it with self.fc. Dead (never-used)
# parameters legitimately receive no gradient and never update.
for n, p in model.named_parameters():
    if p.grad is None:
        tag = n + ":NO_GRAD"
        if n not in DEAD_LAYERS:
            nan_grads.append(tag)
        continue
    if not torch.isfinite(p.grad).all():
        nan_grads.append(n + ":NONFINITE")
    grad_stats[n] = (float(p.grad.abs().mean()), float(p.grad.abs().max()))

gvals = list(grad_stats.values())
gmean = sum(g[0] for g in gvals) / len(gvals)
gmax = max(g[1] for g in gvals)

opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
opt.step()

changed, unchanged = 0, []
for n, p in model.named_parameters():
    if not torch.equal(before[n], p):
        changed += 1
    else:
        unchanged.append(n)

report = {
    "loss": float(loss),
    "loss_finite": bool(torch.isfinite(loss)),
    "n_params_with_grad": len(grad_stats),
    "grad_abs_mean": gmean,
    "grad_abs_max": gmax,
    "grad_issues": nan_grads,
    "params_changed_after_step": changed,
    "params_unchanged": unchanged[:10],
    "grad_exploding": bool(gmax > 1e3),
}
print(f"loss={report['loss']:.4f} finite={report['loss_finite']}")
print(f"grads: {len(grad_stats)} tensors, abs_mean={gmean:.3e}, abs_max={gmax:.3e}")
print(f"grad issues: {nan_grads or 'none'}")
print(f"params changed after SGD step: {changed}")
print(f"unchanged: {unchanged or 'none'}")

import json
with open("audit_artifacts/phase_07_backprop.txt", "w") as fh:
    json.dump(report, fh, indent=2)

ok = (report["loss_finite"] and not nan_grads and changed > 0
      and not report["grad_exploding"])
print("PHASE7_VERDICT:", "PASS" if ok else "FAIL")
