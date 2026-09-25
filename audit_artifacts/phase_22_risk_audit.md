# Phase 22 — Risk Audit (code hygiene + scientific risks)

**Date:** 2026-09-24
**Scope:** `src/`, `scripts/`, `app.py`, `predict.py`, `config/` — greps for leftover scaffolding; then an honest register of every risk we know about.
**Verdict: PASS (hygiene) — with 12 recorded risks (3 HIGH, 4 MEDIUM, 5 LOW), none silent, all documented here and in AUDIT_REPORT.md.**

---

## 1. Code hygiene greps

| Pattern | Hits in src/ scripts/ app.py predict.py config/ |
|---|---|
| `TODO` / `FIXME` / `XXX` / `HACK` | **0** |
| `mock` / `placeholder` / `dummy` / `stub` (case-insens.) | **0** |
| `hardcod*` | **0** |

No dead scaffolding, no debug leftovers, no commented-out branches. The only
quarantined/dead artifacts live outside the code tree (`audit_artifacts/quarantined_*`,
`results/ml/resnet_seed42_misalignedCAM_*`, `results/bridge_M_CNN_misaligned`) and are
clearly named as such.

## 2. Risk register

### HIGH

**R1. Checkpoint auto-selection by mtime in three entry points — a stray directory hijacks the model.**
`app.py:26`, `predict.py:31`, `scripts/validate_external.py:60` all do
`sorted(Path("results/ml").glob("*/checkpoints/best.pt"), key=mtime, reverse=True)[0]`.
Any probe/sweep/partial run landing in `results/ml/` becomes "the model" for the app,
the CLI, and external validation. This *actually happened* during phase 21: an
interrupted `_lightaug` run (epoch 0, val AUC 0.60) hijacked `_get_model()`; the app
fell back to threshold 0.5 (its `metrics/test_metrics.json` was absent/foreign) and
produced an all-zero CAM until the run was quarantined.
Aggravating race: `train_oasis1.py` creates `checkpoints/best.pt` at run start
(checkpointing code writes the first best immediately), so during a normal training run
the app would serve an untrained model. Mitigation in place: naming convention
(`{model}_seed{seed}[_lightaug|_noaug]`) plus manual quarantine; partial runs now live
in `audit_artifacts/quarantined_interrupted_lightaug_run/`.
Fix (small, recommended post-audit): pin a canonical checkpoint path in config, or
require `metrics/test_metrics.json` to exist and pass a sanity floor (val/test AUC ≥ 0.5)
before a run is selectable.

**STATUS (checkpoint-freeze follow-up): FIXED.** All three entry points now resolve the
single configured checkpoint `inference.checkpoint.path` in `config/analysis.yaml` via
the one canonical `brainvuln.config.resolve_checkpoint()` - no globbing, no mtime, no
newest-file fallback; a missing configured checkpoint raises `FileNotFoundError`
loudly. Checkpoint identity (path, sha256, size) is stamped into
`outputs/prediction.json`, external-validation metrics, and training
`test_metrics.json`. Regression tests in `tests/test_checkpoint_resolution.py`
(newer-stray-file ignored, loud-missing, deterministic sha256, app==CLI identity,
static no-mtime check) all pass; full suite 103 passed. Canonical:
`results/ml/resnet_seed42/checkpoints/best.pt`, sha256
`5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` (133,540,267 bytes).


**R2. Genomics bridge headline is a NULL result — the pipeline must not be quoted as "validated".**
Corrected-M_CNN bridge (full 5000/5000 nulls, `results/bridge/bridge_results.json`):
ρ_CNN–GENE = −0.137, matched-gene null p = 0.303, Moran p = 0.459, Burt p = 0.323,
spin p = 0.862 → no CNN–transcriptomic association; concordance robust = False;
all four disease-map correlations (CNN↔La Joie atrophy −0.330, CNN↔tau −0.451,
GENE↔atrophy −0.323, GENE↔tau −0.110) are ns. This is the *honest* outcome after the
affine fix (the earlier ρ = +0.383 "spatially significant" result was a pure artifact of
the CAM affine bug, which the matched-gene null had already refused). Risk: any summary
that drops the null caveat misrepresents the science. Status: accepted and documented;
no re-run can manufacture significance.

**R3. Grad-CAM↔occlusion agreement fell 0.965 → 0.294 after the affine fix — attribution method choice materially changes M_CNN.**
The old 0.965 was inflated by shared misalignment (both maps squeezed into the same 14
right-hemisphere parcels). At 0.294, Grad-CAM and occlusion identify substantially
different regions; the production M_CNN uses Grad-CAM alone. Any scientific claim built
on M_CNN inherits this method sensitivity and should be re-checked against the occlusion
map (`results/ml/resnet_seed42/occlusion/`) before publication.

### MEDIUM

**R4. Small test set (n = 27) and correspondingly wide CIs.** Test ROC-AUC 0.817
[0.618, 0.972] (subject-level bootstrap, 27 subjects, 15 AD / 12 CN). The point
estimate is credible; precise claims (e.g. "AUC > 0.8") are not supported. Specificity
0.417 at the frozen threshold 0.07 reflects the val-set threshold choice on a small val
set (n = 27) — the model is tuned as a high-sensitivity screen (sens 0.867), not a
balanced classifier.

**R5. Cohort confound: sex imbalance within class.** Matched cohort (age ≥ 62, SMD
+0.04 on age) is 100 AD / 82 CN but sex is imbalanced within class (AD ~41% M vs CN
~30% M). The age+sex baseline exists precisely to quantify this; test AUC should be
interpreted relative to that baseline. No sex matching was attempted (would shrink the
cohort further).

**R6. GWAS background pool fallback.** The matched-gene null's preferred background
(bing gene-set coverage) could not be downloaded during the audit (host unreachable, n
_days unspecified); the code correctly fell back to all 15,633 AHBA-expressed genes and
*recorded the fallback in provenance* rather than failing silently. Consequence: the
matched-gene null controls for expression profile, not for gene-set external coverage;
p-values may differ slightly under the Bing pool. Everything else about the null
(precomputed z-matrix, precomputed pairwise distances, 5000 draws) is full-strength.

**R7. GWAS Catalog trait identity is MONDO, not EFO.** `EFO_0000249` returns 0 studies
on REST v2 (verified live); the client uses `MONDO_0004975` (6716 associations, label
"Alzheimer disease"). Live refetch reproduced the stored 3560-gene set exactly
(3560/3560 overlap). Residual risk: show_child_traits=false means newer AD sub-trait
associations may be excluded; documented in `config/diseases.yaml`.

**R8. Attribute-map overlap with CNN map.** Phase 11 verified correct voxel→parcel
mapping (87/87 parcels: 35/35 L, 35/35 R, 16/16 Tian on the true preprocessed grid
affine). Residual risk: `preprocessed_grid_affine()` replicates the preprocess geometry
by construction; a future change to `preprocess_session` (crop/pad logic, voxel size)
would silently desynchronize it. It mirrors rather than imports, so the invariant is
enforced only by discipline — noted as a coupling risk in the code.

### LOW

**R9. Corrupted NIfTI input surfaces a raw traceback.** Phase 9: three invalid inputs
fail gracefully; a truncated/corrupt `.nii.gz` raises nibabel's raw exception to the
console (CLI) — ugly but honest; the Gradio app catches all exceptions and shows the
message. Cosmetic robustness gap only.

**R10. One unsampled parcel (1/87) stays NaN in M_GENE.** AHBA coverage: 86/87 parcels
sampled in ≥1 donor (6 donors); the gap is carried honestly (NaN, not 0) and excluded
from correlations. Donor-LOO robustness (phase 19): median ρ = −0.107, IQR
[−0.138, −0.080] — consistent sign, no donor drives the result.

**R11. CPU-only environment.** torch 2.4.1+cpu; no CUDA available, so GPU paths are
unverified (code paths exist, device strings are parameterized). All performance numbers
in phase 23 are CPU numbers.

**R12. backbone.fc is a dead parameter set.** MONAI's ResNet exposes an fc module our
binary head replaces; the audit (phase 7) confirmed it receives no gradient and is
excluded from count/logic. Harmless, slightly inflates raw param lists; documented.

## 3. Explicit non-risks (checked, clean)

- Splits: train 128 / val 27 / test 27, disjoint at **subject** level, verified 0
  leakage (phase 4); session-vs-subject prefix handling verified in phase 3
  (0 duplicates / 0 missing).
- Determinism: same-seed re-run identical, different seed differs (phase 20).
- Preprocessing identical for train/val/test, no stochastic component; cache
  bit-identical to fresh compute (phase 5).
- Test AUC reproduced identically (0.817) after the affine fix with the checkpoint
  untouched — inference path unaffected by the M_CNN bug.

## 4. Bottom line

The codebase is clean of scaffolding, and every known risk is either (a) fixed and
re-verified (CAM affine), (b) mitigated operationally (checkpoint quarantine), or
(c) documented with its scientific consequence (null bridge, method sensitivity,
fallbacks). The three HIGH risks are: serving-layer fragility (R1), the obligation to
report the null genomics result as null (R2), and attribution-method sensitivity of
M_CNN (R3).
