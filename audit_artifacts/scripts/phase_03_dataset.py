#!/usr/bin/env python
"""Phase 3 dataset audit: cohort vs volumes, duplicates, demographics."""
import glob
import hashlib
import os
import sys

import pandas as pd

cohort = pd.read_csv("data/splits/matched_cohort.csv")
vols = sorted(glob.glob("data/derived/oasis1_128/*.npy"))
vol_names = {os.path.basename(v)[:-4] for v in vols}

# ---- per-subject volume lookup (subject prefix OAS1_XXXX matches OAS1_XXXX_MRn) ----
rows = []
for _, r in cohort.iterrows():
    matches = sorted(v for v in vol_names if v.startswith(r["subject"] + "_MR"))
    rows.append({
        "subject": r["subject"], "label": r["label"], "cohort_class": r["cohort_class"],
        "Age": r["Age"], "Sex": r["Sex"], "CDR": r["CDR"], "split": r["split"],
        "volume": ";".join(matches),
    })
out = pd.DataFrame(rows)
out.to_csv("audit_artifacts/phase_03_dataset_audit.csv", index=False)
missing = out[~out["volume"].str.len().gt(0)]
print("subjects without a preprocessed volume:", len(missing))

# ---- duplicate volume content (md5) ----
hashes, dups = {}, []
for v in vols:
    h = hashlib.md5(open(v, "rb").read()).hexdigest()
    if h in hashes:
        dups.append((os.path.basename(v), os.path.basename(hashes[h])))
    hashes[h] = v
print("content-duplicate volumes:", len(dups))

# ---- demographics ----
for lab, name in ((1, "AD"), (0, "CN")):
    s = out[out["label"] == lab]
    print(f"{name}: n={len(s)} age mean={s['Age'].mean():.1f} sd={s['Age'].std():.1f} "
          f"min={s['Age'].min()} sex={dict(s['Sex'].value_counts())}")
age_smd = (out[out.label == 1]["Age"].mean() - out[out.label == 0]["Age"].mean()) / \
    (((out[out.label == 1]["Age"].var(ddof=1) + out[out.label == 0]["Age"].var(ddof=1)) / 2) ** 0.5)
print(f"age SMD (AD vs CN): {age_smd:+.2f}")

# ---- volume sanity: shapes/dtypes ----
import numpy as np
shapes = set()
for v in vols[:20]:
    a = np.load(v, mmap_mode="r")
    shapes.add((a.shape, str(a.dtype)))
print("first-20 volume (shape, dtype) set:", shapes)
