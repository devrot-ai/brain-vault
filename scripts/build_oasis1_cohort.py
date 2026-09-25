#!/usr/bin/env python
"""Build the OASIS-1 analysis cohort and subject-level splits.

Labeling rule (documented, from the official 2024 metadata revision):

* **Subject unit**: the numeric prefix of the session ID
  (``OAS1_0001_MR1`` and ``OAS1_0001_MR2`` belong to subject ``OAS1_0001``).
* **Class 1 (AD)**: sessions with ``CDR`` in {0.5, 1, 2} ("very mild",
  "moderate") — i.e. clinically diagnosed probable Alzheimer's disease —
  and ``Age >= 60``.
* **Class 0 (CN)**: sessions with ``CDR == 0`` and ``Age >= 60``.
* Sessions without a CDR (young healthy cohorts) are excluded.
* A subject with conflicting CDR labels across sessions is excluded
  (none exist in the official data; the check is a guard).

Age confound control (Part B of the design): controls are matched to AD
subjects 1:1 on age and sex by deterministic greedy nearest-age matching
(fallback ladder +-2, +-3, +-5 years, documented in the output manifest).

Splits (Part C): 70/15/15 at the SUBJECT level, stratified by class,
seeded RNG (``--seed``). All sessions of a subject stay in one partition.

Outputs:
    data/splits/matched_cohort.csv
    data/splits/train_subjects.csv, val_subjects.csv, test_subjects.csv
    data/splits/cohort_manifest.json   (balancing report + SMDs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AGE_TOLERANCE_LADDER = (2, 3, 5)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def load_metadata(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    df = df.rename(columns={"M/F": "Sex"})
    df["subject"] = df["ID"].astype(str).str.split("_MR").str[0]
    # subject-level demographics are constant across sessions (verified);
    # take the first non-null value per subject
    agg = (df.sort_values("ID")
             .groupby("subject", as_index=False)
             .first())
    # conflicting-CDR guard
    conflicts = df.groupby("subject")["CDR"].agg(
        lambda s: s.dropna().nunique() > 1)
    n_conf = int(conflicts.sum())
    if n_conf:
        print(f"excluding {n_conf} subjects with conflicting CDR labels",
              file=sys.stderr)
        agg = agg[~agg["subject"].isin(conflicts[conflicts].index)]
    return agg


def label_subjects(agg: pd.DataFrame, min_age: int) -> pd.DataFrame:
    old = agg[agg["Age"] >= min_age].copy()
    old["label"] = np.where(old["CDR"] > 0, 1, 0)
    ad = old[old["CDR"].notna() & (old["CDR"] > 0)]
    cn = old[old["CDR"] == 0]
    # keep only subjects that carry an explicit CDR label in the metadata
    return pd.concat([ad, cn]).sort_values("subject").reset_index(drop=True)


def match_controls(cohort: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministic greedy 1:1 age/sex matching of CN to AD subjects."""
    rng = np.random.default_rng(seed)
    ad = cohort[cohort["label"] == 1].sort_values("subject").reset_index(drop=True)
    cn_pool = cohort[cohort["label"] == 0].sort_values("subject").reset_index(drop=True)
    taken: set[int] = set()
    matched_rows = []
    fallback_used = {}
    for _, row in ad.iterrows():
        for tol in AGE_TOLERANCE_LADDER:
            pool = cn_pool.drop(index=taken, errors="ignore")
            pool = pool[(pool["Sex"] == row["Sex"])
                        & ((pool["Age"] - row["Age"]).abs() <= tol)]
            if len(pool):
                # nearest age, tie-break deterministically by subject ID
                pool = pool.assign(
                    d=(pool["Age"] - row["Age"]).abs(),
                    sid=pool["subject"])
                best = pool.sort_values(["d", "sid"]).iloc[0]
                taken.add(best.name)
                matched_rows.append(best)
                if tol != AGE_TOLERANCE_LADDER[0]:
                    fallback_used[row["subject"]] = tol
                break
        else:
            print(f"  [warn] no control within "
                  f"{AGE_TOLERANCE_LADDER[-1]} y for {row['subject']}",
                  file=sys.stderr)
    matched_cn = pd.DataFrame(matched_rows)
    out = pd.concat([ad.assign(cohort_class="AD"),
                     matched_cn.assign(cohort_class="CN")])
    return out.sort_values("subject").reset_index(drop=True), fallback_used


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference (absolute value, pooled SD)."""
    a, b = a.astype(float), b.astype(float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled)


def stratified_subject_split(
        cohort: pd.DataFrame, seed: int
) -> tuple[dict[str, list[str]], dict]:
    rng = np.random.default_rng(seed)
    test_frac = 1 - TRAIN_FRAC - VAL_FRAC
    split = {"train": [], "val": [], "test": []}
    balance = {}
    for label, grp in cohort.groupby("label"):
        subj = grp.sort_values("subject")["subject"].tolist()
        subj = [s for s in subj if s not in balance]  # defensive
        rng.shuffle(subj)
        n = len(subj)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * VAL_FRAC))
        # guarantee at least one of each class in val/test when possible
        n_val = max(n_val, 1) if n >= 2 else n_val
        n_test = max(n_test, 1) if n >= 2 else n_test
        split["test"] += subj[:n_test]
        split["val"] += subj[n_test:n_test + n_val]
        split["train"] += subj[n_test + n_val:]
        balance[f"label_{label}"] = {
            "n": n, "train": n - n_test - n_val, "val": n_val, "test": n_test}
    return split, balance


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="data/raw/oasis1/oasis_cross-sectional.xlsx")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_metadata(Path(args.metadata))
    labeled = label_subjects(agg, args.min_age)
    print(f"age >= {args.min_age} with explicit CDR: "
          f"{(labeled['label'] == 1).sum()} AD, {(labeled['label'] == 0).sum()} CN")

    cohort, fallback = match_controls(labeled, args.seed)
    split, balance = stratified_subject_split(cohort, args.seed)

    # session inventory per subject (all sessions of a subject share its split)
    sessions = pd.read_excel(Path(args.metadata))
    sessions["subject"] = sessions["ID"].astype(str).str.split("_MR").str[0]

    cohort_out = cohort[["subject", "label", "cohort_class", "Age", "Sex",
                         "CDR", "MMSE", "eTIV", "nWBV", "ASF", "Educ"]].copy()
    cohort_out["split"] = cohort_out["subject"].map(
        {s: k for k, v in split.items() for s in v})
    cohort_out["n_sessions"] = cohort_out["subject"].map(
        sessions.groupby("subject").size()).fillna(1).astype(int)
    cohort_out.to_csv(out_dir / "matched_cohort.csv", index=False)

    for name, subjects in split.items():
        rows = []
        for s in sorted(subjects):
            ids = sessions[sessions["subject"] == s]["ID"].tolist()
            for i in ids:
                rows.append({"subject": s, "session_id": i,
                             "label": int(cohort_out.loc[
                                 cohort_out["subject"] == s, "label"].iloc[0])})
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_subjects.csv", index=False)

    # balancing report
    cn = cohort_out[cohort_out["cohort_class"] == "CN"]
    ad = cohort_out[cohort_out["cohort_class"] == "AD"]
    manifest = {
        "labeling_rule": {
            "subject_unit": "numeric OAS1_XXXX prefix (session = _MRn suffix)",
            "class_1_AD": "CDR in {0.5,1,2} and Age >= 60 (probable AD)",
            "class_0_CN": "CDR == 0 and Age >= 60",
            "excluded": "sessions without CDR; subjects with conflicting CDR",
        },
        "matching": {
            "method": "deterministic greedy 1:1 nearest-age, same sex",
            "age_tolerance_ladder_years": list(AGE_TOLERANCE_LADDER),
            "fallback_cases": fallback,
            "tie_break": "subject ID ascending",
        },
        "split": {
            "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                          "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
            "unit": "subject", "seed": args.seed, **balance,
        },
        "balance_report": {
            "AD": {"n": len(ad), "mean_age": float(ad["Age"].mean()),
                   "sd_age": float(ad["Age"].std(ddof=1)),
                   "pct_female": float((ad["Sex"] == "F").mean() * 100)},
            "CN": {"n": len(cn), "mean_age": float(cn["Age"].mean()),
                   "sd_age": float(cn["Age"].std(ddof=1)),
                   "pct_female": float((cn["Sex"] == "F").mean() * 100)},
            "SMD": {
                "age": smd(ad["Age"], cn["Age"]),
                "sex_female_pct": abs((ad["Sex"] == "F").mean()
                                      - (cn["Sex"] == "F").mean()) * 100,
                "eTIV": smd(ad["eTIV"], cn["eTIV"]),
                "education": smd(ad["Educ"], cn["Educ"]),
            },
        },
    }
    (out_dir / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["balance_report"], indent=2))
    print(f"\nwrote {out_dir}/matched_cohort.csv, "
          f"train/val/test_subjects.csv, cohort_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
