#!/usr/bin/env python
"""Phase 4: subject-level split and leakage validation."""
import glob
import os

import pandas as pd

cohort = pd.read_csv("data/splits/matched_cohort.csv")
tr = pd.read_csv("data/splits/train_subjects.csv")
va = pd.read_csv("data/splits/val_subjects.csv")
te = pd.read_csv("data/splits/test_subjects.csv")

def ids(df):
    c = [c for c in df.columns if "subject" in c.lower()][0]
    return set(df[c])

S_tr, S_va, S_te = ids(tr), ids(va), ids(te)
print(f"train/val/test subjects: {len(S_tr)}/{len(S_va)}/{len(S_te)}")
print("train INTERSECT val:", len(S_tr & S_va))
print("train INTERSECT test:", len(S_tr & S_te))
print("val INTERSECT test:", len(S_va & S_te))
print("union == cohort:", (S_tr | S_va | S_te) == set(cohort["subject"]))

# cohort table split column must agree with the per-split tables
mismatch = cohort[(cohort.split == "train") & ~cohort.subject.isin(S_tr)]
print("cohort-table split disagreement:", len(mismatch))

# every subject must have exactly one split label
counts = cohort.groupby("subject")["split"].nunique()
print("subjects with >1 split label:", int((counts > 1).sum()))

# session-level: any subject with multiple preprocessed volumes? all must be same split
vols = sorted(glob.glob("data/derived/oasis1_128/*.npy"))
by_subj = {}
for v in vols:
    stem = os.path.basename(v)[:-4]
    subj = stem.split("_MR")[0]
    by_subj.setdefault(subj, []).append(stem)
multi = {k: v for k, v in by_subj.items() if len(v) > 1}
print("subjects with multiple preprocessed volumes:", len(multi), list(multi)[:5])

# label balance per split
for name, S in (("train", S_tr), ("val", S_va), ("test", S_te)):
    s = cohort[cohort.subject.isin(S)]
    n1, n0 = int((s.label == 1).sum()), int((s.label == 0).sum())
    print(f"{name}: AD={n1} CN={n0} ({n1 / (n1 + n0):.0%} positive)")

# age/sex balance across splits (demographic leakage check)
for name, S in (("train", S_tr), ("val", S_va), ("test", S_te)):
    s = cohort[cohort.subject.isin(S)]
    print(f"{name}: age {s.Age.mean():.1f}±{s.Age.std():.1f}, sex {dict(s.Sex.value_counts())}")
