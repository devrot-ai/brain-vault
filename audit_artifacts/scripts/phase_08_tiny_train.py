#!/usr/bin/env python
"""Phase 8: tiny end-to-end training on 8 real OASIS-1 subjects, 2 epochs.

Exercises dataset -> loader -> augment -> model -> loss -> backward ->
optimizer -> checkpoint -> validation -> metrics with REAL preprocessed MRI.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from brainvuln.mri.dataset import OASISDataset, collate_items  # noqa: E402
from brainvuln.mri.models import ResNet18Binary  # noqa: E402
from brainvuln.mri.evaluate import classification_metrics  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

cohort = pd.read_csv("data/splits/matched_cohort.csv")
# 4 AD + 4 CN from the TRAIN partition only (no test contact)
tr = cohort[(cohort.split == "train")].groupby("label", group_keys=False).apply(
    lambda g: g.head(4))
# need session ids: OASIS sessions for each subject (first session on disk)
import glob, os
sess = {}
for v in sorted(glob.glob("data/derived/oasis1_128/*.npy")):
    stem = os.path.basename(v)[:-4]
    sess.setdefault(stem.split("_MR")[0], []).append(stem)
tr = tr.copy()
tr["session_id"] = tr.subject.map(lambda s: sorted(sess[s])[0])

cache = Path("data/derived/oasis1_128")
ds = OASISDataset(tr, "data/raw/oasis1/extracted", cache)
print(f"dataset ok: {len(ds)} scans")
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True,
                                     collate_fn=collate_items)

model = ResNet18Binary()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([82.0 / 100.0]))

model.train()
losses = []
for epoch in range(2):
        ep_loss = []
        for volumes, labels, _ages, _sexes, _meta in loader:
            opt.zero_grad()
            logits = model(volumes)
            loss = loss_fn(logits, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(float(loss))
        losses.append(float(np.mean(ep_loss)))
        print(f"epoch {epoch}: train loss {losses[-1]:.4f}")

# validation-style evaluation on the same tiny set (pipeline check only)
model.eval()
probs, ys = [], []
with torch.no_grad():
    for volumes, labels, _ages, _sexes, _meta in loader:
        probs.extend(torch.sigmoid(model(volumes)).tolist())
        ys.extend(labels.tolist())
metrics = classification_metrics(np.array(ys), np.array(probs), threshold=0.5)
print("metrics:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

ckpt = {"model_state": model.state_dict(), "epoch": 1, "tiny": True,
        "data": "real OASIS-1 (8 subjects, train partition only)"}
ckpt_path = Path("audit_artifacts/phase_08_smoke_checkpoint.pt")
torch.save(ckpt, ckpt_path)
print("checkpoint saved:", ckpt_path, ckpt_path.stat().st_size, "bytes")

log = {
    "losses_per_epoch": losses,
    "loss_finite_all": all(np.isfinite(l) for l in losses),
    "loss_changed": losses[1] != losses[0],
    "metrics": metrics,
    "checkpoint_bytes": ckpt_path.stat().st_size,
    "n_subjects": 8,
    "data": "real OASIS-1 preprocessed volumes",
}
with open("audit_artifacts/phase_08_training_log.txt", "w") as fh:
    json.dump(log, fh, indent=2)
ok = log["loss_finite_all"] and log["loss_changed"] and ckpt_path.exists()
print("PHASE8_VERDICT:", "PASS" if ok else "FAIL")
